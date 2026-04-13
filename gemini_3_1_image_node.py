import base64
import configparser
import io
import json
import time
import threading
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
import urllib3
from PIL import Image, ImageOps

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ComfyUI 中断检测
try:
    import comfy.model_management as model_management
    COMFY_INTERRUPT_AVAILABLE = True
except ImportError:
    COMFY_INTERRUPT_AVAILABLE = False
    print("[WARN] comfy.model_management not available, interrupt detection disabled")

# 线程本地存储,用于 Session 复用
thread_local = threading.local()

CATEGORY = "artsmcp"
CONFIG_SECTION = "Gemini31"  # 独立配置节
CONFIG_PATH = Path(__file__).parent / "config.ini"
CONFIG = configparser.ConfigParser()
if CONFIG_PATH.exists():
    CONFIG.read(CONFIG_PATH, encoding="utf-8")
else:
    CONFIG[CONFIG_SECTION] = {}  # 使用独立配置节
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        CONFIG.write(fp)

# 图像尺寸映射
IMAGE_SIZE_MAP = {
    "1:1": "1:1",
    "2:3": "2:3",
    "3:2": "3:2",
    "3:4": "3:4",
    "4:3": "4:3",
    "4:5": "4:5",
    "5:4": "5:4",
    "9:16": "9:16",
    "16:9": "16:9",
    "21:9": "21:9",
    "1:4": "1:4",
    "4:1": "4:1",
    "1:8": "1:8",
    "8:1": "8:1",
}

# 模型映射
MODEL_MAP = {
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
}

RESOLUTION_MAP = {
    "4K": "4K",
    "2K": "2K",
    "1K": "1K",
    "0.5K": "0.5K",
}


def tensor_to_base64(image_tensor):
    """将 ComfyUI tensor 转换为原始 base64 字符串（不含 data URI 前缀）"""
    if len(image_tensor.shape) > 3:
        image_tensor = image_tensor[0]
    
    array = np.clip(image_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array, mode='RGB')
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('utf-8')
    
    return base64_string


def get_session():
    """获取线程本地的 Session"""
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        thread_local.session = session
    
    return thread_local.session


def download_image_to_tensor(url: str, timeout: int = 60):
    """从 URL 下载图片并转换为 tensor"""
    response = None
    try:
        print(f"\n[DEBUG] 开始下载图片")
        print(f"[DEBUG] 图片URL: {url[:100]}...")
        print(f"[DEBUG] 下载超时: {timeout}秒")
        
        download_start_time = time.time()
        session = get_session()
        response = session.get(url, timeout=timeout, verify=False, stream=True)
        response.raise_for_status()
        
        download_duration = time.time() - download_start_time
        print(f"[DEBUG] 下载完成! 耗时: {download_duration:.2f} 秒")
        print(f"[DEBUG] 图片大小: {len(response.content)} 字节")
        
        pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        print(f"[DEBUG] 图片尺寸: {pil_image.size}")
        
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)
        
        print(f"[SUCCESS] ✅ 图片转换成功! Tensor shape: {tensor.shape}")
        return tensor
        
    except Exception as e:
        print(f"[ERROR] ❌ 下载图片失败: {type(e).__name__}")
        print(f"[ERROR] 错误详情: {e}")
        return None
        
    finally:
        try:
            if response is not None:
                response.close()
        except Exception as e:
            print(f"[WARN] 清理下载连接失败: {e}")


def make_api_request(url: str, headers: dict, payload: dict, timeout: int = 120, max_retries: int = 3, backoff: int = 2):
    """发送 API 请求（后台线程执行，主线程可随时响应中断）"""
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    
    print(f"\n{'='*60}")
    print(f"[DEBUG] 开始 API 请求")
    print(f"[DEBUG] 目标URL: {url}")
    print(f"[DEBUG] 超时设置: {timeout}秒")
    print(f"[DEBUG] 最大重试: {max_retries}次")
    print(f"[DEBUG] 请求头: {json.dumps({k: v[:20]+'...' if len(v)>20 else v for k, v in headers.items()}, ensure_ascii=False)}")
    print(f"[DEBUG] 请求体大小: {len(json.dumps(payload))} 字节")
    print(f"[DEBUG] 请求参数预览: {json.dumps(payload, ensure_ascii=False)[:300]}...")
    print(f"{'='*60}\n")
    
    def _check_interrupted():
        """检查 ComfyUI 中断信号"""
        if COMFY_INTERRUPT_AVAILABLE and model_management.processing_interrupted():
            error_msg = "用户在 ComfyUI 中中断了请求"
            print(f"\n{'='*60}")
            print(f"❌ {error_msg}")
            print(f"{'='*60}\n")
            raise RuntimeError(error_msg)
    
    def _do_post(session, url, headers, payload, timeout):
        """在后台线程中执行实际的 HTTP POST 请求"""
        return session.post(url, headers=headers, json=payload, timeout=timeout, verify=False)
    
    last_error = None
    response = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DEBUG] >>> 第 {attempt}/{max_retries} 次请求开始 <<<")
            request_start_time = time.time()
            
            # 检查中断信号(在重试前)
            _check_interrupted()
            
            if response is not None:
                response.close()
                response = None
            
            if attempt > 1:
                wait_time = min(backoff ** (attempt - 1), 20)
                print(f"[DEBUG] 重试等待: {wait_time} 秒...")
                # 分段等待,每0.5秒检查一次中断
                for i in range(int(wait_time * 2)):
                    _check_interrupted()
                    time.sleep(0.5)
                print(f"[DEBUG] 等待完成,继续重试")
            
            print(f"[DEBUG] 正在建立连接...")
            session = get_session()
            
            print(f"[DEBUG] 正在发送 POST 请求...")
            print(f"[DEBUG] 等待服务器响应 (最长 {timeout} 秒, 可随时中断)...")
            
            # 将 HTTP 请求放入后台线程，主线程轮询中断信号
            with ThreadPoolExecutor(max_workers=1) as req_executor:
                future = req_executor.submit(_do_post, session, url, headers, payload, timeout)
                
                while True:
                    # 每 0.5 秒检查一次中断信号
                    _check_interrupted()
                    try:
                        response = future.result(timeout=0.5)
                        break  # 请求完成
                    except FutureTimeoutError:
                        # future 尚未完成，继续等待并检查中断
                        elapsed = time.time() - request_start_time
                        # 每 10 秒打印一次等待状态
                        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                            print(f"[DEBUG] 已等待 {elapsed:.0f} 秒...")
                        continue
            
            request_duration = time.time() - request_start_time
            print(f"[DEBUG] 收到响应! 耗时: {request_duration:.2f} 秒")
            print(f"[DEBUG] HTTP 状态码: {response.status_code}")
            print(f"[DEBUG] 响应头: {dict(response.headers)}")
            print(f"[DEBUG] 响应大小: {len(response.content)} 字节")
            
            response.raise_for_status()
            
            print(f"[DEBUG] 正在解析 JSON 响应...")
            result = response.json()
            print(f"[DEBUG] JSON 解析成功!")
            print(f"[DEBUG] 响应结构: {list(result.keys())}")
            print(f"[SUCCESS] ✅ 请求成功! 总耗时: {request_duration:.2f} 秒")
            
            response.close()
            return result
            
        except Exception as exc:
            last_error = exc
            request_duration = time.time() - request_start_time if 'request_start_time' in locals() else 0
            
            # 如果是用户主动中断，直接抛出不重试
            if "中断" in str(exc):
                raise
            
            print(f"\n[ERROR] ❌ 请求失败 (尝试 {attempt}/{max_retries})")
            print(f"[ERROR] 错误类型: {type(exc).__name__}")
            print(f"[ERROR] 错误详情: {exc}")
            print(f"[ERROR] 失败耗时: {request_duration:.2f} 秒")
            
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_body = exc.response.json()
                    print(f"[ERROR] 服务器返回: {json.dumps(error_body, ensure_ascii=False)}")
                except:
                    error_text = exc.response.text[:500]
                    print(f"[ERROR] 服务器响应文本: {error_text}")
            
            # 判断是否是超时错误
            if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
                print(f"[ERROR] 🕐 检测到超时错误!")
                print(f"[ERROR] 当前超时设置: {timeout} 秒")
                print(f"[ERROR] 实际等待时间: {request_duration:.2f} 秒")
                print(f"[TIPS] 💡 建议: 增加'超时秒数'参数")
            
            if hasattr(exc, "response") and exc.response is not None:
                status_code = exc.response.status_code
                if 400 <= status_code < 500 and status_code != 429:
                    print(f"[ERROR] 客户端错误 ({status_code})，不进行重试")
                    if response:
                        response.close()
                    raise
        
        finally:
            try:
                if response is not None:
                    response.close()
            except Exception as e:
                pass
    
    print(f"\n{'='*60}")
    print(f"[ERROR] ❌ 请求最终失败，已重试 {max_retries} 次")
    print(f"{'='*60}\n")
    if last_error:
        raise last_error
    raise RuntimeError("未知请求失败")


def poll_task_status(task_id: str, api_key: str, query_base_url: str, max_retries: int = 60, delay: int = 5):
    """轮询任务状态"""
    # 如果用户没有输入路径,我们默认补充,如果是类似于直接提供的 endpoint prefix,也可以兼容
    base = query_base_url.rstrip('/')
    if not base.endswith('/v1/tasks') and 'task_0' not in base:
        # 如果是老版本的 base_url 逻辑,则拼接。目前用户提供的例子是 https://task.artsmcp.com/task_XXX
        status_url = f"{base}/{task_id}?language=zh"
    else:
        # 兜底处理
        status_url = f"{base}/{task_id}?language=zh"
    
    headers = {
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Accept": "*/*",
    }
    # Some endpoints may need authorization even for query, but from example it doesn't clearly show Bearer token for query,
    # Let's add it safely or check the example. Example GET has no Authorization header.
    # We will include it just in case, or not. The user example GET lacks auth header, we will stick to the example mostly,
    # but some APIs require it. We will add Authorization if user provided API key.
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    print(f"\n{'='*60}")
    print(f"[DEBUG] 开始轮询任务状态")
    print(f"[DEBUG] 任务ID: {task_id}")
    print(f"[DEBUG] 查询URL: {status_url}")
    print(f"[DEBUG] 最大轮询次数: {max_retries}")
    print(f"[DEBUG] 轮询间隔: {delay}秒")
    print(f"{'='*60}\n")
    
    for attempt in range(max_retries):
        # 检查 ComfyUI 中断信号
        if COMFY_INTERRUPT_AVAILABLE and model_management.processing_interrupted():
            error_msg = "用户在 ComfyUI 中中断了任务"
            print(f"\n{'='*60}")
            print(f"❌ {error_msg}")
            print(f"任务ID: {task_id}")
            print(f"{'='*60}\n")
            raise RuntimeError(error_msg)
        
        try:
            print(f"[DEBUG] 第 {attempt + 1}/{max_retries} 次查询...")
            query_start_time = time.time()
            
            session = get_session()
            response = session.get(status_url, headers=headers, verify=False, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            query_duration = time.time() - query_start_time
            print(f"[DEBUG] 查询响应耗时: {query_duration:.2f} 秒")
            
            if "code" in result and result["code"] == 200:
                data = result.get("data", {})
                status = data.get("status", "")
                
                print(f"[INFO] 任务状态: {status} ({attempt + 1}/{max_retries})")
                
                if status == "completed":
                    print(f"[SUCCESS] ✅ 任务完成!")
                    print(f"[DEBUG] 完整响应: {json.dumps(result, ensure_ascii=False)[:500]}...")
                    return data
                elif status == "failed":
                    error_msg = data.get("error", {}).get("message", "未知错误")
                    print(f"[ERROR] ❌ 任务失败!")
                    print(f"[ERROR] 失败原因: {error_msg}")
                    print(f"[DEBUG] 完整响应: {json.dumps(result, ensure_ascii=False)[:500]}...")
                    return None
                else:
                    print(f"[DEBUG] 任务进行中,{delay}秒后重试...")
            else:
                print(f"[WARN] 异常响应码: {result.get('code', 'unknown')}")
                print(f"[DEBUG] 响应内容: {json.dumps(result, ensure_ascii=False)[:300]}...")
                
        except Exception as e:
            print(f"[ERROR] 查询状态失败 ({attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            # 分段等待，每0.5秒检查一次中断信号
            for _ in range(int(delay * 2)):
                if COMFY_INTERRUPT_AVAILABLE and model_management.processing_interrupted():
                    error_msg = "用户在 ComfyUI 中中断了任务"
                    print(f"\n{'='*60}")
                    print(f"❌ {error_msg}")
                    print(f"任务ID: {task_id}")
                    print(f"{'='*60}\n")
                    raise RuntimeError(error_msg)
                time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"[WARN] ⚠️ 达到最大轮询次数 ({max_retries})")
    print(f"[WARN] 任务ID: {task_id}")
    print(f"[TIPS] 💡 建议增加'最大轮询次数'参数")
    print(f"{'='*60}\n")
    return None


class Gemini31ImageNode:
    """Gemini 3.1 图片生成节点 - 同步调用 Gemini generateContent API"""
    
    def __init__(self):
        self.verbose = False
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "请为黑客帝国设计一张高品质的3D海报...",
                    "display": "input"
                }),
                "API密钥": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_key", fallback="")
                }),
                "API地址": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback="https://api.artsmcp.com")
                }),
                "模型": (list(MODEL_MAP.keys()), {
                    "default": list(MODEL_MAP.keys())[0]
                }),
                "宽高比": (list(IMAGE_SIZE_MAP.keys()), {
                    "default": "1:1"
                }),
                "分辨率": (list(RESOLUTION_MAP.keys()), {
                    "default": "4K"
                }),
                "超时秒数": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 600,
                    "step": 10
                }),
                "最大重试次数": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "并发请求数": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "参考图片1": ("IMAGE", {}),
                "参考图片2": ("IMAGE", {}),
                "参考图片3": ("IMAGE", {}),
                "参考图片4": ("IMAGE", {}),
                "参考图片5": ("IMAGE", {}),
                "参考图片6": ("IMAGE", {}),
                "参考图片7": ("IMAGE", {}),
                "参考图片8": ("IMAGE", {}),
                "参考图片9": ("IMAGE", {}),
                "参考图片10": ("IMAGE", {}),
                "参考图片11": ("IMAGE", {}),
                "参考图片12": ("IMAGE", {}),
                "参考图片13": ("IMAGE", {}),
                "参考图片14": ("IMAGE", {}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片输出",)
    FUNCTION = "generate_image"
    CATEGORY = CATEGORY
    OUTPUT_NODE = False  # 标明这不是输出节点
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """强制每次都重新执行,不使用缓存(因为是外部API请求)"""
        import time
        return time.time()
    
    def generate_image(self, 提示词, API密钥, API地址, 模型, 宽高比,
                       分辨率, 超时秒数, 最大重试次数, 并发请求数,
                       参考图片1=None, 参考图片2=None, 参考图片3=None, 参考图片4=None,
                       参考图片5=None, 参考图片6=None, 参考图片7=None, 参考图片8=None,
                       参考图片9=None, 参考图片10=None, 参考图片11=None, 参考图片12=None,
                       参考图片13=None, 参考图片14=None):
        # 保存配置
        config_writer = configparser.ConfigParser()
        if CONFIG_PATH.exists():
            config_writer.read(CONFIG_PATH, encoding="utf-8")
        
        if not config_writer.has_section(CONFIG_SECTION):
            config_writer.add_section(CONFIG_SECTION)
        
        if API密钥.strip():
            config_writer.set(CONFIG_SECTION, "api_key", API密钥.strip())
        if API地址.strip():
            config_writer.set(CONFIG_SECTION, "api_url", API地址.strip())
            
        with CONFIG_PATH.open("w", encoding="utf-8") as fp:
            config_writer.write(fp)
            
        # 构建 Gemini generateContent API URL
        # 格式: {base_url}/v1beta/models/{model}:generateContent?key={api_key}
        base_url = API地址.rstrip('/')
        model_name = MODEL_MAP[模型]
        api_url = f"{base_url}/v1beta/models/{model_name}:generateContent?key={API密钥}"
            
        headers = {
            "Content-Type": "application/json",
        }
        
        # 构建 Gemini 请求体中的 parts
        parts = []
        
        # 处理参考图片 (支持多图) - 作为 inlineData parts
        ref_images = [
            参考图片1, 参考图片2, 参考图片3, 参考图片4,
            参考图片5, 参考图片6, 参考图片7, 参考图片8,
            参考图片9, 参考图片10, 参考图片11, 参考图片12,
            参考图片13, 参考图片14
        ]
        
        valid_refs = [img for img in ref_images if img is not None]
        
        if valid_refs:
            for img in valid_refs:
                b64_data = tensor_to_base64(img)
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": b64_data
                    }
                })
            print(f"[INFO] 接收到 {len(valid_refs)} 张有效参考图，已全部转换为 inlineData 格式。")
        else:
            print("[INFO] 本次生成没有传入参考图 (文生图模式)。")
        
        # 添加文本提示词
        parts.append({"text": 提示词})
        
        # 构建完整请求体 (Gemini generateContent 格式)
        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "imageConfig": {
                    "aspectRatio": IMAGE_SIZE_MAP[宽高比],
                    "imageSize": RESOLUTION_MAP[分辨率]
                }
            }
        }
            
        try:
            def _single_request():
                """执行单次 Gemini generateContent 请求并返回图片 tensor 列表"""
                result = make_api_request(api_url, headers, payload, timeout=超时秒数, max_retries=最大重试次数)
                
                # 解析 Gemini 响应，提取 inlineData 中的 base64 图片
                tensors = []
                candidates = result.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"API 响应中没有 candidates: {json.dumps(result, ensure_ascii=False)[:500]}")
                
                for candidate in candidates:
                    content = candidate.get("content", {})
                    resp_parts = content.get("parts", [])
                    for part in resp_parts:
                        # 跳过 thought 文本
                        if part.get("thought"):
                            thought_text = part.get("text", "")
                            if thought_text:
                                print(f"[INFO] 模型思考: {thought_text[:200]}...")
                            continue
                        
                        # 提取 inlineData 中的图片
                        inline_data = part.get("inlineData")
                        if inline_data and "data" in inline_data:
                            mime_type = inline_data.get("mimeType", "image/png")
                            b64_data = inline_data["data"]
                            print(f"[DEBUG] 收到内嵌图片, mimeType: {mime_type}, base64长度: {len(b64_data)}")
                            
                            # 解码 base64 -> PIL Image -> tensor
                            img_bytes = base64.b64decode(b64_data)
                            pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                            print(f"[DEBUG] 图片尺寸: {pil_image.size}")
                            
                            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
                            tensor = torch.from_numpy(numpy_image)
                            tensors.append(tensor)
                            print(f"[SUCCESS] 图片转换成功! Tensor shape: {tensor.shape}")
                        
                        # 非 thought 的普通文本
                        elif "text" in part:
                            text_content = part.get("text", "")
                            if text_content:
                                print(f"[INFO] 模型文本输出: {text_content[:300]}...")
                
                # 打印 usage 信息
                usage = result.get("usageMetadata", {})
                if usage:
                    print(f"[INFO] Token 使用: prompt={usage.get('promptTokenCount', '?')}, "
                          f"candidates={usage.get('candidatesTokenCount', '?')}, "
                          f"total={usage.get('totalTokenCount', '?')}, "
                          f"thoughts={usage.get('thoughtsTokenCount', '?')}")
                
                return tensors
            
            # 执行请求 (支持并发)
            output_tensors = []
            
            if 并发请求数 <= 1:
                output_tensors = _single_request()
            else:
                print(f"[INFO] 启动 {并发请求数} 个并发请求...")
                with ThreadPoolExecutor(max_workers=并发请求数) as executor:
                    futures = [executor.submit(_single_request) for _ in range(并发请求数)]
                    for future in as_completed(futures):
                        try:
                            tensors = future.result()
                            output_tensors.extend(tensors)
                        except Exception as e:
                            print(f"[ERROR] 并发请求失败: {e}")
                    
            if not output_tensors:
                raise RuntimeError("未从 API 响应中获取到任何图片")
                
            # 合并并返回 - 尺寸统一
            tensor_shapes = [(t.shape[0], t.shape[1]) for t in output_tensors]
            if len(set(tensor_shapes)) > 1:
                min_h = min([s[0] for s in tensor_shapes])
                min_w = min([s[1] for s in tensor_shapes])
                normalized = []
                for t in output_tensors:
                    h, w, c = t.shape
                    start_h = (h - min_h) // 2
                    start_w = (w - min_w) // 2
                    normalized.append(t[start_h:start_h+min_h, start_w:start_w+min_w, :])
                output_tensors = normalized
                
            batch_tensor = torch.stack(output_tensors, dim=0).contiguous()
            print(f"[SUCCESS] 成功生成 {len(output_tensors)} 张图片!")
            return (batch_tensor,)
            
        except Exception as e:
            print(f"[ERROR] 节点执行异常: {e}")
            import traceback
            traceback.print_exc()
            raise

# 供 __init__.py 导出使用
NODE_CLASS_MAPPINGS = {
    "Gemini31ImageNode": Gemini31ImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini31ImageNode": "artsmcp-gemini-3.1"
}
