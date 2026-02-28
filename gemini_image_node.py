import base64
import configparser
import io
import json
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
CONFIG_SECTION = "Gemini-banana"  # 独立配置节
CONFIG_PATH = Path(__file__).parent / "config.ini"
CONFIG = configparser.ConfigParser()
if CONFIG_PATH.exists():
    CONFIG.read(CONFIG_PATH, encoding="utf-8")
else:
    CONFIG[CONFIG_SECTION] = {}  # 使用独立配置节
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        CONFIG.write(fp)

# 图像尺寸映射（Gemini 支持多种宽高比）
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
}

# 模型映射
MODEL_MAP = {
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "gemini-3-pro-image-preview-2k": "gemini-3-pro-image-preview-2k",
    "gemini-3-pro-image-preview-4k": "gemini-3-pro-image-preview-4k",
}

# 响应格式映射
RESPONSE_FORMAT_MAP = {
    "URL": "url",
    "Base64": "b64_json",
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
    """获取线程本地的 Session (复用连接池,使用官方推荐的 HTTPAdapter 配置)"""
    if not hasattr(thread_local, "session"):
        # 创建 Session
        session = requests.Session()
        
        # 使用 HTTPAdapter 精细控制连接池
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,  # 连接池数量
            pool_maxsize=10,      # 每个连接池的最大连接数
            max_retries=0         # 重试由上层 make_api_request 控制
        )
        
        # 挂载到 http 和 https
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 存储到线程本地
        thread_local.session = session
    
    return thread_local.session


def download_image_to_tensor(url: str, timeout: int = 60):
    """从 URL 下载图片并转换为 tensor"""
    response = None
    
    try:
        print(f"[INFO] 正在下载图片: {url}")
        
        # 使用线程本地 Session (连接池复用)
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
        # 清理资源 (但保留 Session 供线程复用)
        try:
            if response is not None:
                response.close()
        except Exception as e:
            print(f"[WARN] 清理下载连接失败: {e}")


def base64_to_tensor(b64_string: str):
    """将 base64 字符串转换为 tensor (支持 data URI 格式)"""
    try:
        # 处理 data URI 格式 (如: data:image/png;base64,...)
        if b64_string.startswith("data:image"):
            # 提取实际的 base64 数据部分
            b64_string = b64_string.split(",", 1)[1]
        
        img_bytes = base64.b64decode(b64_string)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)
        return tensor
    except Exception as e:
        print(f"[ERROR] Base64 转换失败: {e}")
        return None


def make_api_request(url: str, headers: dict, payload: dict, timeout: int = 120, max_retries: int = 3, backoff: int = 2, verbose: bool = False):
    """发送 API 请求（支持重试）"""
    import time
    
    # 打印请求信息
    print(f"[INFO] 发送请求到: {url}")
    print(f"[INFO] 请求参数: {json.dumps(payload, ensure_ascii=False)[:200]}...")
    
    last_error = None
    response = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # 清理上次重试的 response
            if response is not None:
                response.close()
                response = None
            
            if attempt > 1:
                wait_time = min(backoff ** (attempt - 1), 20)  # 指数退避: 2s, 4s, 8s, 最大20s
                print(f"[INFO] 第 {attempt} 次重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            
            # 使用线程本地 Session (连接池复用)
            session = get_session()
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=False
            )
            
            # 详细的状态码处理
            print(f"[INFO] HTTP 状态码: {response.status_code}")
            
            # 检查是否成功
            response.raise_for_status()
            
            result = response.json()
            print(f"[SUCCESS] 请求成功！")
            if verbose:
                print(f"[DEBUG] 完整响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 成功后关闭 response (但保留 Session)
            response.close()
            return result
            
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            print(f"[ERROR] HTTP 错误 (尝试 {attempt}/{max_retries}): {exc}")
            
            # 打印响应内容用于调试
            try:
                if response is not None:
                    error_detail = response.json()
                    print(f"[ERROR] 错误详情: {json.dumps(error_detail, ensure_ascii=False)}")
            except:
                if response is not None:
                    print(f"[ERROR] 响应文本: {response.text[:500]}")
            
            # 4xx 客户端错误直接抛出，不重试（除了 429 限流）
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                print(f"[ERROR] 客户端错误 ({exc.response.status_code})，不进行重试")
                # 清理资源
                if response:
                    response.close()
                raise
                
        except requests.exceptions.Timeout as exc:
            last_error = exc
            print(f"[ERROR] 请求超时 (尝试 {attempt}/{max_retries}): {exc}")
            if verbose:
                print(f"[DEBUG] 超时类型: {type(exc).__name__}")
            
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            print(f"[ERROR] 连接失败 (尝试 {attempt}/{max_retries}): {exc}")
            
        except Exception as exc:
            last_error = exc
            print(f"[ERROR] 未知错误 (尝试 {attempt}/{max_retries}): {exc}")
        
        finally:
            # 确保 response 被关闭 (但保留线程本地 Session)
            try:
                if response is not None:
                    response.close()
            except Exception as e:
                print(f"[WARN] 清理连接失败: {e}")
        
        # 如果还有重试机会，继续循环
        if attempt < max_retries:
            continue
    
    # 所有重试都失败
    print(f"\n[ERROR] ❌ 请求最终失败，已重试 {max_retries} 次")
    print(f"[ERROR] 最后错误: {last_error}")
    print(f"\n💡 可能的解决方案:")
    print(f"   1. 检查 API 服务是否正常: {url}")
    print(f"   2. 确认 API Key 是否有效")
    print(f"   3. 稍后再试，可能是服务器临时过载")
    print(f"   4. 检查网络连接是否稳定")
    print(f"   5. 尝试增加 timeout 参数值")
    
    if last_error:
        raise last_error
    raise RuntimeError("未知请求失败")


class GeminiBananaNode:
    """Gemini Banana 图片生成节点 - 支持文生图、图生图、多图融合"""
    
    def __init__(self):
        self.verbose = False  # 默认关闭详细日志
    
    def log(self, message, level="INFO"):
        """统一日志输出 (支持分级)"""
        if level == "DEBUG" and not self.verbose:
            return  # DEBUG 日志只在 verbose 模式下打印
        print(message)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "星际穿越,黑洞,电影大片,超现实主义",
                    "display": "input"
                }),
                "API密钥": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_key", fallback="")
                }),
                "API地址": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback="https://api.openai.com")
                }),
                "模型": (list(MODEL_MAP.keys()), {
                    "default": list(MODEL_MAP.keys())[0]
                }),
                "宽高比": (list(IMAGE_SIZE_MAP.keys()), {
                    "default": "1:1"
                }),
                "响应格式": (list(RESPONSE_FORMAT_MAP.keys()), {
                    "default": "URL"
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
                "并发请求数": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "启用分行提示词": ("BOOLEAN", {
                    "default": False
                }),
                "匹配参考尺寸": ("BOOLEAN", {
                    "default": False,
                    "label_on": "开启",
                    "label_off": "关闭"
                }),
                "详细日志": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "参考图片1": ("IMAGE", {}),
                "参考图片2": ("IMAGE", {}),
                "参考图片3": ("IMAGE", {}),
                "参考图片4": ("IMAGE", {}),
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
    
    def _prepare_prompts(self, prompt: str, enable_multiline: bool, concurrent_requests: int):
        """准备提示词列表"""
        prompts = []
        if enable_multiline:
            prompts = [line.strip() for line in prompt.split('\n') if line.strip()]
            print("\n" + "="*60)
            print("[Gemini-Banana] 启用分行提示词模式")
            print(f"  - 拆分后提示词数量: {len(prompts)}")
            print(f"  - 每行并发请求数: {concurrent_requests}")
            print(f"  - 总任务数: {len(prompts)} 行 × {concurrent_requests} 并发 = {len(prompts) * concurrent_requests} 张图")
            for idx, p in enumerate(prompts, 1):
                print(f"  - 提示词{idx}: {p[:50]}...")
            print("="*60 + "\n")
        else:
            prompts = [prompt]
            print("\n" + "="*60)
            print("[Gemini-Banana] 完整提示词模式")
            print(f"  - 提示词: {prompt[:50]}...")
            print(f"  - 并发请求数: {concurrent_requests}")
            print(f"  - 总任务数: {concurrent_requests} 张图")
            print("="*60 + "\n")
        return prompts
    
    def _prepare_input_images(self, image1, image2, image3, image4):
        """收集和转换输入图片"""
        input_images = []
        for img in [image1, image2, image3, image4]:
            if img is not None:
                input_images.append(img)
        
        image_urls = []
        if input_images:
            for idx, img_tensor in enumerate(input_images):
                base64_url = tensor_to_base64(img_tensor)
                image_urls.append(base64_url)
                print(f"[INFO] 已转换图片{idx + 1}为 Base64")
        
        return image_urls
    
    def _build_request_tasks(self, prompts, model_value, size_value, 
                             response_format_value, concurrent_requests, image_urls):
        """构建请求任务列表"""
        request_tasks = []
        for prompt_idx, current_prompt in enumerate(prompts, 1):
            payload = {
                "model": model_value,
                "prompt": current_prompt,
                "size": size_value,
                "response_format": response_format_value,
                "n": concurrent_requests,
            }
            
            if image_urls:
                if len(image_urls) == 1:
                    payload["image"] = image_urls[0]
                else:
                    payload["image"] = image_urls
            
            request_tasks.append({
                "task_id": prompt_idx,
                "prompt": current_prompt,
                "payload": payload
            })
            
            self.log(f"[DEBUG] 任务 {prompt_idx} - 提示词: {current_prompt[:50]}...", "DEBUG")
        
        return request_tasks
    
    def _send_requests(self, request_tasks, base_url, headers, timeout, max_retries):
        """并发发送所有请求"""
        print(f"\n{'='*60}")
        print(f"[INFO] 开始并发发送请求...")
        print(f"[INFO] 提示词数量: {len(request_tasks)}")
        print(f"[INFO] 预计生成图片数: {sum(t['payload']['n'] for t in request_tasks)}")
        print(f"{'='*60}\n")
        
        def send_single_request(task):
            task_id = task["task_id"]
            task_prompt = task["prompt"]
            task_payload = task["payload"]
            
            print(f"\n[INFO] ▶ 任务 {task_id} 开始请求: {task_prompt[:30]}...")
            
            try:
                result = make_api_request(base_url, headers, task_payload, timeout, max_retries, 2, self.verbose)
                print(f"[SUCCESS] ✅ 任务 {task_id} 请求成功")
                return {
                    "task_id": task_id,
                    "prompt": task_prompt,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                print(f"[ERROR] ❌ 任务 {task_id} 请求失败: {e}")
                return {
                    "task_id": task_id,
                    "prompt": task_prompt,
                    "error": str(e),
                    "success": False
                }
        
        all_results = []
        max_workers = min(len(request_tasks), 10)
        print(f"\n[INFO] 使用线程池并发执行，最大并发数: {max_workers}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(send_single_request, task): task for task in request_tasks}
            
            for future in as_completed(future_to_task):
                task_result = future.result()
                all_results.append(task_result)
        
        print(f"\n{'='*60}")
        print(f"[INFO] 所有请求已完成！")
        print(f"[INFO] 成功: {sum(1 for r in all_results if r['success'])} 个")
        print(f"[INFO] 失败: {sum(1 for r in all_results if not r['success'])} 个")
        print(f"{'='*60}\n")
        
        all_results.sort(key=lambda x: x["task_id"])
        return all_results
    
    def _parse_results(self, all_results, response_format_value):
        """解析API响应,收集下载任务"""
        print(f"\n{'='*60}")
        print(f"[INFO] 开始统一解析响应并收集下载任务...")
        print(f"{'='*60}\n")
        
        download_tasks = []
        
        for task_result in all_results:
            if not task_result["success"]:
                print(f"[WARN] ⚠ 跳过失败的任务 {task_result['task_id']}: {task_result.get('error', 'Unknown')}")
                continue
            
            result = task_result["result"]
            task_id = task_result["task_id"]
            task_prompt = task_result["prompt"]
            
            self.log(f"\n[INFO] 解析任务 {task_id} 的响应 - 提示词: {task_prompt[:30]}...", "DEBUG")
            self.log(f"[DEBUG] 响应包含的键: {list(result.keys())}", "DEBUG")
            
            if "data" in result:
                data = result["data"]
                self.log(f"[DEBUG] data 类型: {type(data)}", "DEBUG")
                            
                if isinstance(data, list):
                    self.log(f"[DEBUG] data 是列表，长度: {len(data)}", "DEBUG")
                    for idx, item in enumerate(data, 1):
                        self.log(f"[DEBUG] 准备下载任务 {task_id} 的第 {idx}/{len(data)} 张图片", "DEBUG")
                        self.log(f"[DEBUG] 图片项包含的键: {list(item.keys()) if isinstance(item, dict) else 'N/A'}", "DEBUG")
                        
                        download_tasks.append({
                            "task_id": task_id,
                            "image_idx": idx,
                            "item": item,
                            "order": len(download_tasks)
                        })
                                        
                elif isinstance(data, dict):
                    self.log(f"[DEBUG] data 是字典", "DEBUG")
                    self.log(f"[DEBUG] 字典包含的键: {list(data.keys())}", "DEBUG")
                    
                    download_tasks.append({
                        "task_id": task_id,
                        "image_idx": 1,
                        "item": data,
                        "order": len(download_tasks)
                    })
                elif isinstance(data, str):
                    # 处理新的响应格式：data 是字符串 URL
                    self.log(f"[DEBUG] data 是字符串 URL", "DEBUG")
                    
                    download_tasks.append({
                        "task_id": task_id,
                        "image_idx": 1,
                        "item": {"url": data},  # 转换为字典格式以便后续处理
                        "order": len(download_tasks)
                    })
            else:
                print(f"[ERROR] 任务 {task_id} 响应中没有 'data' 字段！")
                print(f"[DEBUG] 完整响应内容: {result}")
                            
                if "created" in result and "usage" in result:
                    print(f"[INFO] 检测到可能是图像分析API的响应，没有图片数据")
        
        return download_tasks
    
    def _download_images(self, download_tasks, response_format_value, timeout):
        """并发下载所有图片"""
        if not download_tasks:
            raise RuntimeError("没有可下载的图片任务！API响应可能不包含图片数据")
        
        print(f"\n{'='*60}")
        print(f"[INFO] 开始并发下载 {len(download_tasks)} 张图片...")
        print(f"{'='*60}\n")
        
        def download_single_image(task):
            task_id = task["task_id"]
            image_idx = task["image_idx"]
            item = task["item"]
            order = task["order"]
            
            print(f"[INFO] ▶ 开始下载任务 {task_id} 的第 {image_idx} 张图片...")
            
            try:
                tensor = self._process_image_item(item, response_format_value, timeout)
                if tensor is not None:
                    print(f"[SUCCESS] ✅ 任务 {task_id} 第 {image_idx} 张图片下载成功")
                    return {
                        "order": order,
                        "task_id": task_id,
                        "image_idx": image_idx,
                        "tensor": tensor,
                        "success": True
                    }
                else:
                    print(f"[ERROR] ❌ 任务 {task_id} 第 {image_idx} 张图片下载失败")
                    return {
                        "order": order,
                        "task_id": task_id,
                        "image_idx": image_idx,
                        "success": False
                    }
            except Exception as e:
                print(f"[ERROR] ❌ 任务 {task_id} 第 {image_idx} 张图片下载异常: {e}")
                return {
                    "order": order,
                    "task_id": task_id,
                    "image_idx": image_idx,
                    "success": False,
                    "error": str(e)
                }
        
        download_results = []
        max_download_workers = min(len(download_tasks), 4)
        print(f"[INFO] 使用线程池并发下载，最大并发数: {max_download_workers}\n")
        
        with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
            future_to_download = {executor.submit(download_single_image, task): task for task in download_tasks}
            
            for future in as_completed(future_to_download):
                result = future.result()
                download_results.append(result)
        
        download_results.sort(key=lambda x: x["order"])
        
        output_tensors = []
        for result in download_results:
            if result["success"]:
                output_tensors.append(result["tensor"])
        
        print(f"\n{'='*60}")
        print(f"[INFO] 下载完成统计")
        print(f"[INFO] 成功: {len(output_tensors)} 张")
        print(f"[INFO] 失败: {len(download_results) - len(output_tensors)} 张")
        print(f"{'='*60}\n")
        
        # 如果没有一张成功,直接抛异常(防止缓存错误结果)
        if not output_tensors:
            raise RuntimeError(f"所有图片下载均失败！总计 {len(download_results)} 个任务")
        
        return output_tensors
    
    def _normalize_tensor_size(self, tensors):
        """归一化tensor尺寸,避免尺寸不一致导致stack崩溃"""
        if not tensors:
            return tensors
        
        # 获取所有tensor的尺寸
        shapes = [(t.shape[0], t.shape[1]) for t in tensors]
        heights = [s[0] for s in shapes]
        widths = [s[1] for s in shapes]
        
        # 检查是否所有尺寸都一致
        if len(set(shapes)) == 1:
            self.log(f"[DEBUG] 所有图片尺寸一致: {shapes[0]}", "DEBUG")
            return tensors
        
        # 尺寸不一致,需要归一化
        print(f"[WARN] ⚠️ 检测到图片尺寸不一致!")
        print(f"[WARN] 尺寸分布: {set(shapes)}")
        
        # 使用最小公共尺寸(裁剪策略)
        min_h = min(heights)
        min_w = min(widths)
        
        print(f"[INFO] 统一裁剪到最小公共尺寸: {min_h}×{min_w}")
        
        # 中心裁剪
        normalized = []
        for idx, t in enumerate(tensors):
            h, w, c = t.shape
            
            # 计算裁剪起始位置(中心对齐)
            start_h = (h - min_h) // 2
            start_w = (w - min_w) // 2
            
            # 裁剪
            cropped = t[start_h:start_h+min_h, start_w:start_w+min_w, :]
            normalized.append(cropped)
            
            if h != min_h or w != min_w:
                self.log(f"[DEBUG] 图片{idx+1}: {h}×{w} → {min_h}×{min_w} (裁剪)", "DEBUG")
        
        print(f"[SUCCESS] ✅ 已归一化 {len(normalized)} 张图片尺寸")
        return normalized
    
    def _merge_tensors(self, output_tensors, prompts, concurrent_requests, enable_multiline):
        """合并所有tensor为批次"""
        if not output_tensors:
            raise RuntimeError("未获取到任何图片数据！")
        
        # 归一化tensor尺寸(防止尺寸不一致导致stack崩溃)
        output_tensors = self._normalize_tensor_size(output_tensors)
        
        batch_tensor = torch.stack(output_tensors, dim=0).contiguous()
        print(f"\n{'='*60}")
        print(f"[SUCCESS] ✅ 成功生成 {len(output_tensors)} 张图片!")
        print(f"[INFO] 批次尺寸: {batch_tensor.shape}")
        if enable_multiline:
            print(f"[INFO] 提示词数量: {len(prompts)} 行")
            print(f"[INFO] 每行并发数: {concurrent_requests}")
            print(f"[INFO] 总图片数: {len(prompts)} × {concurrent_requests} = {len(output_tensors)}")
        else:
            print(f"[INFO] 并发请求数: {concurrent_requests}")
        print(f"{'='*60}\n")
        
        self.log(f"[DEBUG] 准备返回 tensor，确保数据完整性...", "DEBUG")
        self.log(f"[DEBUG] tensor 类型: {type(batch_tensor)}", "DEBUG")
        self.log(f"[DEBUG] tensor device: {batch_tensor.device}", "DEBUG")
        self.log(f"[DEBUG] tensor dtype: {batch_tensor.dtype}", "DEBUG")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[OUTPUT] 准备传递给下一个节点的数据详情:")
            print(f"{'='*60}")
            print(f"[OUTPUT] 数据类型: {type(batch_tensor).__name__}")
            print(f"[OUTPUT] 数据形状 (shape): {batch_tensor.shape}")
            print(f"  ├─ 批次大小 (batch): {batch_tensor.shape[0]}")
            print(f"  ├─ 图片高度 (height): {batch_tensor.shape[1]}")
            print(f"  ├─ 图片宽度 (width): {batch_tensor.shape[2]}")
            print(f"  └─ 通道数 (channels): {batch_tensor.shape[3]}")
            print(f"[OUTPUT] 数据维度 (ndim): {batch_tensor.ndim}")
            print(f"[OUTPUT] 元素总数: {batch_tensor.numel():,}")
            print(f"[OUTPUT] 数据类型 (dtype): {batch_tensor.dtype}")
            print(f"[OUTPUT] 存储设备 (device): {batch_tensor.device}")
            print(f"[OUTPUT] 是否需要梯度: {batch_tensor.requires_grad}")
            print(f"[OUTPUT] 内存大小: {batch_tensor.element_size() * batch_tensor.numel() / 1024 / 1024:.2f} MB")
            print(f"[OUTPUT] 数值范围: [{batch_tensor.min():.4f}, {batch_tensor.max():.4f}]")
            print(f"[OUTPUT] 数值均值: {batch_tensor.mean():.4f}")
            print(f"[OUTPUT] 数值标准差: {batch_tensor.std():.4f}")
            
            if batch_tensor.shape[0] <= 5:
                print(f"\n[OUTPUT] 各图片详细信息:")
                for i in range(batch_tensor.shape[0]):
                    img_tensor = batch_tensor[i]
                    print(f"  图片 {i+1}:")
                    print(f"    ├─ 尺寸: {img_tensor.shape[1]}×{img_tensor.shape[0]} ({img_tensor.shape[2]} 通道)")
                    print(f"    ├─ 数值范围: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
                    print(f"    ├─ 均值: {img_tensor.mean():.4f}")
                    print(f"    └─ 标准差: {img_tensor.std():.4f}")
            
            print(f"\n[OUTPUT] 返回值结构: tuple 包含 1 个元素")
            print(f"[OUTPUT] 返回值内容: (torch.Tensor,)")
            print(f"[OUTPUT] ComfyUI 将接收到类型为 'IMAGE' 的输出")
            print(f"{'='*60}\n")
        
        return batch_tensor
    
    def generate_image(self, 提示词, API密钥, API地址, 模型, 宽高比, 
                       响应格式, 超时秒数, 最大重试次数, 并发请求数,
                       启用分行提示词, 匹配参考尺寸, 详细日志,
                       参考图片1=None, 参考图片2=None, 参考图片3=None, 参考图片4=None):
        """主生成函数 - 重构为清晰的流程"""
        
        # 设置日志级别
        self.verbose = 详细日志
        
        # 重命名变量以便内部使用
        prompt = 提示词
        enable_multiline = 启用分行提示词
        concurrent_requests = 并发请求数
        match_reference_size = 匹配参考尺寸
        api_key = API密钥
        base_url = API地址
        model = 模型
        size = 宽高比
        response_format = 响应格式
        timeout = 超时秒数
        max_retries = 最大重试次数
        
        # 保存配置到独立配置节（重新读取确保不覆盖其他节点配置）
        config_writer = configparser.ConfigParser()
        if CONFIG_PATH.exists():
            config_writer.read(CONFIG_PATH, encoding="utf-8")
        
        if not config_writer.has_section(CONFIG_SECTION):
            config_writer.add_section(CONFIG_SECTION)
        
        if api_key.strip():
            config_writer.set(CONFIG_SECTION, "api_key", api_key.strip())
        if base_url.strip():
            config_writer.set(CONFIG_SECTION, "api_url", base_url.strip())
        
        with CONFIG_PATH.open("w", encoding="utf-8") as fp:
            config_writer.write(fp)
        
        # 打印其他参数
        print(f"[INFO] 模型: {model}")
        print(f"[INFO] 尺寸: {size}")
        print(f"[INFO] 响应格式: {response_format}")
        print(f"[INFO] 超时时间: {timeout}秒")
        print(f"[INFO] 最大重试次数: {max_retries}")
        
        # 准备请求基础参数
        model_value = MODEL_MAP[model]
        size_value = IMAGE_SIZE_MAP[size]
        response_format_value = RESPONSE_FORMAT_MAP[response_format]
        
        # 处理 API 地址：确保是完整的 URL
        if base_url.startswith('http://') or base_url.startswith('https://'):
            # 如果已经是完整 URL，检查是否包含路径
            if '/v1/images/generations' not in base_url:
                # 只有域名，需要添加路径
                base_url = base_url.rstrip('/') + '/v1/images/generations'
        else:
            # 如果只是域名，添加协议和路径
            base_url = f"https://{base_url}/v1/images/generations"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 步骤1: 准备提示词
            prompts = self._prepare_prompts(prompt, enable_multiline, concurrent_requests)
            
            # 步骤2: 准备输入图片
            input_images = [参考图片1, 参考图片2, 参考图片3, 参考图片4]
            input_images = [img for img in input_images if img is not None]
            image_urls = self._prepare_input_images(参考图片1, 参考图片2, 参考图片3, 参考图片4)
            
            # 步骤3: 构建请求任务
            request_tasks = self._build_request_tasks(
                prompts, model_value, size_value, 
                response_format_value, concurrent_requests, image_urls
            )
            
            # 步骤4: 并发发送请求
            all_results = self._send_requests(request_tasks, base_url, headers, timeout, max_retries)
            
            # 步骤5: 解析响应
            download_tasks = self._parse_results(all_results, response_format_value)
            
            # 步骤6: 下载图片
            output_tensors = self._download_images(download_tasks, response_format_value, timeout)
            
            # 步骤6.5: 如果启用"匹配参考尺寸"且有参考图片，则调整输出尺寸
            if match_reference_size and input_images:
                output_tensors = self._match_reference_size(output_tensors, input_images)
            
            # 步骤7: 合并 tensor
            batch_tensor = self._merge_tensors(output_tensors, prompts, concurrent_requests, enable_multiline)
            
            print(f"[INFO] ✅ 节点执行完毕，返回结果")
            return (batch_tensor,)
            
        except Exception as e:
            # 所有异常统一处理
            print(f"[ERROR] 生成失败: {e}")
            print(f"[DEBUG] 异常类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_image_item(self, item: dict, format_type: str, timeout: int):
        """处理单个图片数据项"""
        self.log(f"[DEBUG] _process_image_item 调用: format_type={format_type}", "DEBUG")
        self.log(f"[DEBUG] item 内容: {item}", "DEBUG")
        
        if format_type == "url" and "url" in item:
            self.log(f"[DEBUG] 匹配到 URL 格式，开始下载...", "DEBUG")
            return download_image_to_tensor(item["url"], timeout)
        elif format_type == "b64_json" and "b64_json" in item:
            self.log(f"[DEBUG] 匹配到 Base64 格式，开始解码...", "DEBUG")
            return base64_to_tensor(item["b64_json"])
        else:
            print(f"[ERROR] 未匹配到任何格式！")
            self.log(f"[DEBUG] 期望格式: {format_type}", "DEBUG")
            self.log(f"[DEBUG] item 包含的键: {list(item.keys()) if isinstance(item, dict) else 'N/A'}", "DEBUG")
            return None
    
    def _match_reference_size(self, output_tensors, input_images):
        """匹配参考图片尺寸 - 使用第一张参考图的尺寸作为目标"""
        if not output_tensors or not input_images:
            return output_tensors
        
        # 获取第一张参考图的尺寸 (tensor shape: [H, W, C])
        ref_tensor = input_images[0]
        if len(ref_tensor.shape) > 3:
            ref_tensor = ref_tensor[0]  # 如果是批次，取第一张
        
        target_h = ref_tensor.shape[0]
        target_w = ref_tensor.shape[1]
        
        print(f"\n{'='*60}")
        print(f"[INFO] 启用匹配参考尺寸功能")
        print(f"[INFO] 参考图尺寸: {target_w}×{target_h}")
        print(f"[INFO] 待处理图片数量: {len(output_tensors)}")
        print(f"{'='*60}\n")        
        matched_tensors = []
        for idx, tensor in enumerate(output_tensors):
            current_h, current_w = tensor.shape[0], tensor.shape[1]
            
            if current_h == target_h and current_w == target_w:
                self.log(f"[DEBUG] 图片{idx+1} 尺寸已匹配，跳过调整", "DEBUG")
                matched_tensors.append(tensor)
            else:
                print(f"[INFO] 图片{idx+1}: {current_w}×{current_h} → {target_w}×{target_h} (缩放+裁剪)")                
                # 转换为 PIL Image
                array = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
                pil_image = Image.fromarray(array, mode='RGB')
                
                # 使用 ImageOps.fit 进行智能缩放+居中裁剪
                resized_image = ImageOps.fit(pil_image, (target_w, target_h), method=Image.LANCZOS)
                
                # 转回 tensor
                resized_array = np.array(resized_image).astype(np.float32) / 255.0
                resized_tensor = torch.from_numpy(resized_array)
                
                matched_tensors.append(resized_tensor)
        
        print(f"[SUCCESS] ✅ 已将 {len(matched_tensors)} 张图片调整为参考尺寸 {target_w}×{target_h}\n")
        return matched_tensors

# ComfyUI 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiBananaNode": GeminiBananaNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiBananaNode": "artsmcp-gemini-banana"
}
