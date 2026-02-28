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
    """将 ComfyUI tensor 转换为 base64 字符串"""
    if len(image_tensor.shape) > 3:
        image_tensor = image_tensor[0]
    
    array = np.clip(image_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array, mode='RGB')
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_string}"


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
        print(f"[INFO] 正在下载图片: {url}")
        session = get_session()
        response = session.get(url, timeout=timeout, verify=False, stream=True)
        response.raise_for_status()
        
        pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        print(f"[INFO] 图片尺寸: {pil_image.size}")
        
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)
        
        return tensor
        
    except Exception as e:
        print(f"[ERROR] 下载图片失败: {e}")
        return None
        
    finally:
        try:
            if response is not None:
                response.close()
        except Exception as e:
            print(f"[WARN] 清理下载连接失败: {e}")


def make_api_request(url: str, headers: dict, payload: dict, timeout: int = 120, max_retries: int = 3, backoff: int = 2):
    """发送 API 请求"""
    import time
    
    print(f"[INFO] 发送请求到: {url}")
    print(f"[INFO] 请求参数: {json.dumps(payload, ensure_ascii=False)[:300]}...")
    
    last_error = None
    response = None
    
    for attempt in range(1, max_retries + 1):
        try:
            if response is not None:
                response.close()
                response = None
            
            if attempt > 1:
                wait_time = min(backoff ** (attempt - 1), 20)
                print(f"[INFO] 第 {attempt} 次重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            
            session = get_session()
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=False
            )
            
            print(f"[INFO] HTTP 状态码: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            print(f"[SUCCESS] 请求成功！")
            
            response.close()
            return result
            
        except Exception as exc:
            last_error = exc
            print(f"[ERROR] 请求失败 (尝试 {attempt}/{max_retries}): {exc}")
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    print(f"[ERROR] 响应详情: {exc.response.json()}")
                except:
                    print(f"[ERROR] 响应文本: {exc.response.text[:500]}")
            
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
    
    print(f"\n[ERROR] ❌ 请求最终失败，已重试 {max_retries} 次")
    if last_error:
        raise last_error
    raise RuntimeError("未知请求失败")


def poll_task_status(task_id: str, api_key: str, query_base_url: str, max_retries: int = 60, delay: int = 5):
    """轮询任务状态"""
    # 如果用户没有输入路径，我们默认补充，如果是类似于直接提供的 endpoint prefix，也可以兼容
    base = query_base_url.rstrip('/')
    if not base.endswith('/v1/tasks') and 'task_0' not in base:
        # 如果是老版本的 base_url 逻辑，则拼接。目前用户提供的例子是 https://task.artsmcp.com/task_XXX
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
    
    print(f"[INFO] 开始轮询任务状态: {task_id}")
    
    for attempt in range(max_retries):
        try:
            session = get_session()
            response = session.get(status_url, headers=headers, verify=False, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "code" in result and result["code"] == 200:
                data = result.get("data", {})
                status = data.get("status", "")
                
                print(f"[INFO] 任务状态 ({attempt + 1}/{max_retries}): {status}")
                
                if status == "completed":
                    print("[SUCCESS] 任务完成!")
                    return data
                elif status == "failed":
                    error_msg = data.get("error", {}).get("message", "未知错误")
                    print(f"[ERROR] 任务失败! 详情: {error_msg}")
                    return None
            else:
                print(f"[WARN] 异常响应: {result}")
                
        except Exception as e:
            print(f"[错误] 查询状态失败 ({attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    print("[WARN] 达到最大重试次数")
    return None


class Gemini31ImageNode:
    """Gemini 3.1 图片生成节点 - 支持异步轮询创建"""
    
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
                "生成API地址": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback="https://apitt.cozex.cn")
                }),
                "查询API地址": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "query_url", fallback="https://task.artsmcp.com")
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
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "step": 10
                }),
                "最大重试次数": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "轮询间隔秒数": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
                "最大轮询次数": ("INT", {
                    "default": 120,
                    "min": 10,
                    "max": 600,
                    "step": 10
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
    
    def generate_image(self, 提示词, API密钥, 生成API地址, 查询API地址, 模型, 宽高比,
                       分辨率, 超时秒数, 最大重试次数, 轮询间隔秒数, 最大轮询次数, 并发请求数,
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
        if 生成API地址.strip():
            config_writer.set(CONFIG_SECTION, "api_url", 生成API地址.strip())
        if 查询API地址.strip():
            config_writer.set(CONFIG_SECTION, "query_url", 查询API地址.strip())
            
        with CONFIG_PATH.open("w", encoding="utf-8") as fp:
            config_writer.write(fp)
            
        # 准备创建任务URL
        create_url = 生成API地址
        if '/v1/images/generations' not in create_url:
            create_url = create_url.rstrip('/') + '/v1/images/generations'
            
        headers = {
            "Authorization": f"Bearer {API密钥}",
            "Content-Type": "application/json",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)"
        }
        
        payload = {
            "model": MODEL_MAP[模型],
            "prompt": 提示词,
            "size": IMAGE_SIZE_MAP[宽高比],
            "resolution": RESOLUTION_MAP[分辨率],
            "n": 并发请求数
        }
        
        # 处理参考图片 (支持多图)
        ref_images = [
            参考图片1, 参考图片2, 参考图片3, 参考图片4,
            参考图片5, 参考图片6, 参考图片7, 参考图片8,
            参考图片9, 参考图片10, 参考图片11, 参考图片12,
            参考图片13, 参考图片14
        ]
        
        valid_refs = [img for img in ref_images if img is not None]
        
        if valid_refs:
            image_b64_list = [tensor_to_base64(img) for img in valid_refs]
            print(f"[INFO] 接收到 {len(valid_refs)} 张有效参考图，已全部转换为 Base64 格式。")
            if len(image_b64_list) == 1:
                payload["image_urls"] = image_b64_list[0]
            else:
                payload["image_urls"] = image_b64_list
        else:
            print("[INFO] 本次生成没有传入参考图 (文生图模式)。")
            
        try:
            # 1. 提交任务
            create_result = make_api_request(create_url, headers, payload, timeout=超时秒数, max_retries=最大重试次数)
            
            task_ids = []
            if "data" in create_result and isinstance(create_result["data"], list):
                for item in create_result["data"]:
                    if "task_id" in item:
                        task_ids.append(item["task_id"])
            elif "data" in create_result and isinstance(create_result["data"], dict) and "task_id" in create_result["data"]:
                task_ids.append(create_result["data"]["task_id"])
                
            if not task_ids:
                raise RuntimeError(f"未从响应中找到任务ID: {create_result}")
                
            # 我们只处理第一个任务，因为如果有多个任务也只是同一个请求触发
            task_id = task_ids[0]
            print(f"[INFO] 成功提交任务，Task ID: {task_id}")
            
            # 2. 轮询任务
            task_data = poll_task_status(task_id, API密钥, 查询API地址, 最大轮询次数, 轮询间隔秒数)
            
            if not task_data:
                raise RuntimeError("任务轮询失败或超时")
                
            # 3. 解析结果图片URL
            image_urls = []
            if "result" in task_data and "images" in task_data["result"]:
                for img_data in task_data["result"]["images"]:
                    if "url" in img_data:
                        if isinstance(img_data["url"], list):
                            image_urls.extend(img_data["url"])
                        else:
                            image_urls.append(img_data["url"])
            
            if not image_urls:
                raise RuntimeError(f"任务已完成，但未找到图片URL: {task_data}")
                
            print(f"[INFO] 获取到 {len(image_urls)} 个图片URL")
            
            # 4. 下载图片
            output_tensors = []
            for url in image_urls:
                tensor = download_image_to_tensor(url, 超时秒数)
                if tensor is not None:
                    output_tensors.append(tensor)
                    
            if not output_tensors:
                raise RuntimeError("所有图片下载失败")
                
            # 5. 合并并返回
            # 根据原始逻辑, 需要进行尺寸统一
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
            print(f"[SUCCESS] ✅ 成功生成 {len(output_tensors)} 张图片!")
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
