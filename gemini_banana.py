import base64
import configparser
import io
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import torch
import urllib3
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

# 图像尺寸映射（Gemini 支持 1K, 2K, 4K）
IMAGE_SIZE_MAP = {
    "1K": "1K",
    "2K": "2K",
    "4K": "4K",
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


def download_image_to_tensor(url: str, timeout: int = 60):
    """从 URL 下载图片并转换为 tensor"""
    session = None
    response = None
    
    try:
        print(f"[INFO] 正在下载图片: {url}")
        
        # 使用独立 Session
        session = requests.Session()
        response = session.get(url, timeout=timeout, verify=False)
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
        # 清理资源
        try:
            if response is not None:
                response.close()
            if session is not None:
                session.close()
        except Exception as e:
            print(f"[WARN] 清理下载连接失败: {e}")


def base64_to_tensor(b64_string: str):
    """将 base64 字符串转换为 tensor"""
    try:
        img_bytes = base64.b64decode(b64_string)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)
        return tensor
    except Exception as e:
        print(f"[ERROR] Base64 转换失败: {e}")
        return None


def make_api_request(url: str, headers: dict, payload: dict, timeout: int = 120, max_retries: int = 3, backoff: int = 2):
    """发送 API 请求（支持重试）"""
    import time
    
    # 打印请求信息
    print(f"[INFO] 发送请求到: {url}")
    print(f"[INFO] 请求参数: {json.dumps(payload, ensure_ascii=False)[:200]}...")
    
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        # 关键：每次重试都创建新的 Session，避免连接池污染
        session = requests.Session()
        response = None
        
        try:
            if attempt > 1:
                wait_time = min(backoff ** (attempt - 1), 20)  # 指数退避: 2s, 4s, 8s, 最大20s
                print(f"[INFO] 第 {attempt} 次重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            
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
            print(f"[SUCCESS] 请求成功！响应数据: {json.dumps(result, ensure_ascii=False)[:200]}...")
            
            # 成功后关闭
            response.close()
            session.close()
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
                session.close()
                raise
                
        except requests.exceptions.Timeout as exc:
            last_error = exc
            print(f"[ERROR] 请求超时 (尝试 {attempt}/{max_retries}): {exc}")
            print(f"[DEBUG] 超时类型: {type(exc).__name__}")
            
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            print(f"[ERROR] 连接失败 (尝试 {attempt}/{max_retries}): {exc}")
            
        except Exception as exc:
            last_error = exc
            print(f"[ERROR] 未知错误 (尝试 {attempt}/{max_retries}): {exc}")
        
        finally:
            # 关键：无论成功还是失败，都必须清理资源
            try:
                if response is not None:
                    response.close()
                session.close()
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
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "星际穿越，黑洞，电影大片，超现实主义",
                    "label": "💬 提示词"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_key", fallback=""),
                    "label": "🔑 API密钥"
                }),
                "base_url": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback="https://apitt.cozex.cn/v1/images/generations"),
                    "label": "🌐 API地址"
                }),
                "model": (list(MODEL_MAP.keys()), {
                    "default": list(MODEL_MAP.keys())[0],
                    "label": "🧠 模型"
                }),
                "size": (list(IMAGE_SIZE_MAP.keys()), {
                    "default": "2K",
                    "label": "📐 尺寸"
                }),
                "response_format": (list(RESPONSE_FORMAT_MAP.keys()), {
                    "default": "URL",
                    "label": "📦 响应格式"
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "step": 10,
                    "label": "⏱️ 超时(秒)"
                }),
                "max_retries": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "🔄 最大重试次数"
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "📊 生图数量"
                }),
            },
            "optional": {
                "image1": ("IMAGE", {"label": "🖼️ 图片1"}),
                "image2": ("IMAGE", {"label": "🖼️ 图片2"}),
                "image3": ("IMAGE", {"label": "🖼️ 图片3"}),
                "image4": ("IMAGE", {"label": "🖼️ 图片4"}),
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
    
    def generate_image(self, prompt, api_key, base_url, model, size, 
                       response_format, timeout, max_retries, n,
                       image1=None, image2=None, image3=None, image4=None):
        """主生成函数"""
        
        # 保存配置到独立配置节
        if not CONFIG.has_section(CONFIG_SECTION):
            CONFIG.add_section(CONFIG_SECTION)
        
        if api_key.strip():
            CONFIG.set(CONFIG_SECTION, "api_key", api_key.strip())
        if base_url.strip():
            CONFIG.set(CONFIG_SECTION, "api_url", base_url.strip())
        
        with CONFIG_PATH.open("w", encoding="utf-8") as fp:
            CONFIG.write(fp)
        
        # 打印输入参数（调试用）
        print("\n" + "="*60)
        print("[Gemini-Banana] 输入参数:")
        print(f"  - 提示词: {prompt[:50]}...")
        print(f"  - 模型: {model}")
        print(f"  - 尺寸: {size}")
        print(f"  - 响应格式: {response_format}")
        print(f"  - 生图数量: {n}")
        print("="*60 + "\n")
        
        # 收集输入图片
        input_images = []
        for img in [image1, image2, image3, image4]:
            if img is not None:
                input_images.append(img)
        
        # 构建请求参数（仅包含 Gemini 支持的参数）
        model_value = MODEL_MAP[model]
        size_value = IMAGE_SIZE_MAP[size]
        response_format_value = RESPONSE_FORMAT_MAP[response_format]
        
        payload = {
            "model": model_value,
            "prompt": prompt,
            "size": size_value,
            "response_format": response_format_value,
            "n": n,  # 添加生图数量参数
        }
        
        # 处理输入图片
        if input_images:
            image_urls = []
            for idx, img_tensor in enumerate(input_images):
                base64_url = tensor_to_base64(img_tensor)
                image_urls.append(base64_url)
                print(f"[INFO] 已转换图片{idx + 1}为 Base64")
            
            if len(image_urls) == 1:
                payload["image"] = image_urls[0]
                print("[INFO] 模式: 图生图")
            else:
                payload["image"] = image_urls
                print(f"[INFO] 模式: 多图融合 ({len(image_urls)}张)")
        else:
            print("[INFO] 模式: 文生图")
        
        # 发送请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            result = make_api_request(base_url, headers, payload, timeout, max_retries)
            
            # 解析响应
            output_tensors = []
            
            if "data" in result:
                data = result["data"]
                if isinstance(data, list):
                    for item in data:
                        tensor = self._process_image_item(item, response_format_value, timeout)
                        if tensor is not None:
                            output_tensors.append(tensor)
                elif isinstance(data, dict):
                    tensor = self._process_image_item(data, response_format_value, timeout)
                    if tensor is not None:
                        output_tensors.append(tensor)
            
            if not output_tensors:
                print("[WARN] 未获取到任何图片，返回默认黑色图片")
                return (torch.zeros((1, 512, 512, 3)),)
            
            # 合并所有 tensor
            batch_tensor = torch.stack(output_tensors, dim=0)
            print(f"[SUCCESS] 成功生成 {len(output_tensors)} 张图片! 尺寸: {batch_tensor.shape}")
            
            return (batch_tensor,)
            
        except Exception as e:
            # 关键:异常时直接抛出,不返回默认图片,避免缓存错误结果
            print(f"[ERROR] 生成失败: {e}")
            print(f"[DEBUG] 异常类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
                    
            # 直接抛出异常,让ComfyUI知道节点失败了
            raise e
    
    def _process_image_item(self, item: dict, format_type: str, timeout: int):
        """处理单个图片数据项"""
        if format_type == "url" and "url" in item:
            return download_image_to_tensor(item["url"], timeout)
        elif format_type == "b64_json" and "b64_json" in item:
            return base64_to_tensor(item["b64_json"])
        return None


# ComfyUI 节点映射
NODE_CLASS_MAPPINGS = {
    "GeminiBananaNode": GeminiBananaNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiBananaNode": "artsmcp-gemini-banana"
}
