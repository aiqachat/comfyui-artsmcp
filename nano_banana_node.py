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
CONFIG_SECTION = "Nano-banana"  # 独立配置节
CONFIG_PATH = Path(__file__).parent / "config.ini"
CONFIG = configparser.ConfigParser()
if CONFIG_PATH.exists():
    CONFIG.read(CONFIG_PATH, encoding="utf-8")
else:
    CONFIG[CONFIG_SECTION] = {}  # 使用独立配置节
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        CONFIG.write(fp)

# 宽高比映射
ASPECT_RATIO_MAP = {
    "1:1": "1:1",
    "4:3": "4:3",
    "3:4": "3:4",
    "16:9": "16:9",
    "9:16": "9:16",
    "2:3": "2:3",
    "3:2": "3:2",
    "4:5": "4:5",
    "5:4": "5:4",
    "21:9": "21:9",
}

# 图像尺寸映射(仅nano-banana-2支持)
IMAGE_SIZE_MAP = {
    "1K": "1K",
    "2K": "2K",
    "4K": "4K",
}

# 模型映射
MODEL_MAP = {
    "nano-banana": "gemini-2.5-flash-image-preview",
    "nano-banana-2": "gemini-3-pro-image-preview",
}

# 响应格式映射
RESPONSE_FORMAT_MAP = {
    "URL": "url",
    "Base64": "b64_json",
}


def get_config_value(section, key, fallback=None):
    """从配置文件获取配置值"""
    global CONFIG
    try:
        # 重新读取配置文件以确保获取最新值
        CONFIG.read(CONFIG_PATH, encoding="utf-8")
        return CONFIG.get(section, key, fallback=fallback)
    except Exception as e:
        print(f"[CONFIG] 读取配置失败: {e}")
        return fallback


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
        # 在每次重试前检查 ComfyUI 中断标志
        try:
            import comfy.model_management as mm
            if mm.interrupt_current_processing():
                print("[INFO] 检测到用户中断请求，停止重试")
                raise InterruptedError("用户中断了请求")
        except ImportError:
            pass  # 如果不是在 ComfyUI 环境下运行，忽略
        except Exception as e:
            pass  # 中断检测失败也继续
        # 关键：每次重试都创建新的 Session，避免连接池污染
        session = requests.Session()
        response = None
        
        try:
            if attempt > 1:
                wait_time = min(backoff ** (attempt - 1), 20)  # 指数退避: 2s, 4s, 8s, 最大20s
                print(f"[INFO] 第 {attempt} 次重试，等待 {wait_time} 秒...")
                
                # 分段 sleep，每 0.5 秒检查一次中断
                for _ in range(int(wait_time * 2)):
                    time.sleep(0.5)
                    try:
                        import comfy.model_management as mm
                        if mm.interrupt_current_processing():
                            print("[INFO] 等待重试时检测到用户中断，立即退出")
                            raise InterruptedError("用户中断了请求")
                    except (ImportError, AttributeError):
                        pass
            
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
            
            # 尝试解析 JSON 响应，如果失败打印原始文本便于调试
            try:
                result = response.json()
            except Exception as e:
                try:
                    print("[ERROR] 响应不是合法的 JSON，原始文本前500字符:")
                    print(response.text[:500])
                except Exception as e2:
                    print(f"[ERROR] 读取响应文本失败: {e2}")
                raise e
            
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


class NanoBananaNode:
    """Nano Banana 图片生成节点 - 支持文生图、图生图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的猫咪,卡通风格,高清",
                    "label": "💬 提示词"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_key", fallback=CONFIG.get("DEFAULT", "api_key", fallback="")),
                    "label": "🔑 API密钥"
                }),
                "base_url": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback=CONFIG.get("DEFAULT", "api_url", fallback="https://api.openai.com/v1/images/generations")),
                    "label": "🌐 API地址"
                }),
                "model": (list(MODEL_MAP.keys()), {
                    "default": list(MODEL_MAP.keys())[0],
                    "label": "🧠 模型"
                }),
                "aspect_ratio": (list(ASPECT_RATIO_MAP.keys()), {
                    "default": "1:1",
                    "label": "📐 宽高比"
                }),
                # 响应格式暂时写死为 Base64
                # "response_format": (list(RESPONSE_FORMAT_MAP.keys()), {
                #     "default": "URL",
                #     "label": "📦 响应格式"
                # }),
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
                "image_size": (list(IMAGE_SIZE_MAP.keys()) + ["none"], {
                    "default": "none",
                    "label": "📏 图像尺寸(仅nano-banana-2)"
                }),
                "image1": ("IMAGE", {"label": "🖼️ 参考图片1"}),
                "image2": ("IMAGE", {"label": "🖼️ 参考图片2"}),
                "image3": ("IMAGE", {"label": "🖼️ 参考图片3"}),
                "image4": ("IMAGE", {"label": "🖼️ 参考图片4"}),
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
    
    def generate_image(self, prompt, api_key, base_url, model, aspect_ratio, 
                       timeout, max_retries, n,
                       image_size="none",
                       image1=None, image2=None, image3=None, image4=None):
        """主生成函数"""
        
        # 写死响应格式为 Base64
        response_format = "Base64"
        
        # 保存配置到独立配置节（每次重新读取确保数据最新）
        config_writer = configparser.ConfigParser()
        if CONFIG_PATH.exists():
            config_writer.read(CONFIG_PATH, encoding="utf-8")
        
        if not config_writer.has_section(CONFIG_SECTION):
            config_writer.add_section(CONFIG_SECTION)
        
        # 只保存非空的配置项
        if api_key.strip():
            config_writer.set(CONFIG_SECTION, "api_key", api_key.strip())
            print(f"[CONFIG] 保存 api_key 到配置文件")
        if base_url.strip():
            config_writer.set(CONFIG_SECTION, "api_url", base_url.strip())
            print(f"[CONFIG] 保存 api_url 到配置文件: {base_url.strip()}")
        
        try:
            with CONFIG_PATH.open("w", encoding="utf-8") as fp:
                config_writer.write(fp)
            print(f"[CONFIG] 配置已成功写入: {CONFIG_PATH}")
        except Exception as e:
            print(f"[ERROR] 配置写入失败: {e}")
        
        # 打印输入参数（调试用）
        print("\n" + "="*60)
        print("[Nano-Banana] 输入参数:")
        print(f"  - 提示词: {prompt[:50]}...")
        print(f"  - 模型: {model}")
        print(f"  - 宽高比: {aspect_ratio}")
        print(f"  - 图像尺寸: {image_size}")
        print(f"  - 响应格式: {response_format}")
        print(f"  - 生图数量: {n}")
        print("="*60 + "\n")
        
        # 收集输入图片
        input_images = []
        for idx, img in enumerate([image1, image2, image3, image4], 1):
            if img is not None:
                input_images.append(img)
                print(f"[DEBUG] 检测到参考图片{idx}, 形状: {img.shape}")
        
        print(f"[DEBUG] 共收集到 {len(input_images)} 张参考图片")
        
        # 构建请求参数（Gemini 官方请求体）
        model_value = MODEL_MAP[model]
        aspect_ratio_value = ASPECT_RATIO_MAP[aspect_ratio]
        response_format_value = RESPONSE_FORMAT_MAP[response_format]
        
        # 组装文本部分，可以把宽高比等信息写进提示词，方便控制
        full_prompt = prompt
        if aspect_ratio_value:
            full_prompt += f"\nAspect ratio: {aspect_ratio_value}"
        if image_size != "none" and model == "nano-banana-2":
            full_prompt += f"\nImage size: {IMAGE_SIZE_MAP[image_size]}"
        
        parts = [{"text": full_prompt}]
        
        # 处理输入图片(支持多图) -> inline_data
        if input_images:
            for idx, img_tensor in enumerate(input_images):
                base64_url = tensor_to_base64(img_tensor)
                prefix = "data:image/jpeg;base64,"
                if base64_url.startswith(prefix):
                    b64_data = base64_url[len(prefix):]
                else:
                    b64_data = base64_url
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_data
                    }
                })
                print(f"[INFO] 已转换图片{idx + 1}为 inline_data")
            print(f"[INFO] 模式: 文本+参考图 ({len(input_images)} 张)")
        else:
            print("[INFO] 模式: 文生图")
        
        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ]
        }
        
        # 发送请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 打印完整的payload用于调试
        print(f"[DEBUG] 完整 payload 结构:")
        try:
            print(json.dumps(payload, ensure_ascii=False)[:500] + "...")
        except Exception as e:
            print(f"[WARN] payload 序列化失败: {e}")
        
        try:
            result = make_api_request(base_url, headers, payload, timeout, max_retries)
            
            # 解析响应
            output_tensors = []
            
            print(f"[DEBUG] 检查响应结构...")
            print(f"[DEBUG] 响应包含的键: {list(result.keys())}")
            
            if "data" in result:
                data = result["data"]
                print(f"[DEBUG] data 类型: {type(data)}")
                print(f"[DEBUG] data 内容: {data}")
                
                if isinstance(data, list):
                    print(f"[DEBUG] data 是列表，长度: {len(data)}")
                    for idx, item in enumerate(data):
                        print(f"[DEBUG] 处理第 {idx+1} 个图片项...")
                        print(f"[DEBUG] 图片项类型: {type(item)}")
                        print(f"[DEBUG] 图片项内容: {item}")
                        print(f"[DEBUG] 图片项包含的键: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
                        print(f"[DEBUG] 期望的响应格式: {response_format_value}")
                        
                        tensor = self._process_image_item(item, response_format_value, timeout)
                        if tensor is not None:
                            output_tensors.append(tensor)
                            print(f"[DEBUG] ✅ 第 {idx+1} 个图片转换成功")
                        else:
                            print(f"[DEBUG] ❌ 第 {idx+1} 个图片转换失败")
                            
                elif isinstance(data, dict):
                    print(f"[DEBUG] data 是字典")
                    print(f"[DEBUG] 字典包含的键: {list(data.keys())}")
                    print(f"[DEBUG] 期望的响应格式: {response_format_value}")
                    
                    tensor = self._process_image_item(data, response_format_value, timeout)
                    if tensor is not None:
                        output_tensors.append(tensor)
                        print(f"[DEBUG] ✅ 图片转换成功")
                    else:
                        print(f"[DEBUG] ❌ 图片转换失败")
            else:
                print(f"[ERROR] 响应中没有 'data' 字段！")
                print(f"[DEBUG] 完整响应内容: {result}")
                
                # 检查是否是图像分析API的响应格式
                if "created" in result and "usage" in result:
                    print(f"[INFO] 检测到可能是图像分析API的响应，没有图片数据")
                    print(f"[INFO] 该API可能用于图像分析而非图像生成")
                
            if not output_tensors:
                print("[ERROR] ❌ 未获取到任何图片数据！")
                print(f"[DEBUG] 输出 tensors 数量: {len(output_tensors)}")
                print("[WARN] 返回默认黑色图片")
                return (torch.zeros((1, 512, 512, 3)),)
            
            # 合并所有 tensor
            batch_tensor = torch.stack(output_tensors, dim=0)
            print(f"[SUCCESS] 成功生成 {len(output_tensors)} 张图片! 尺寸: {batch_tensor.shape}")
            
            return (batch_tensor,)
            
        except InterruptedError as e:
            # 用户主动中断
            print(f"[INFO] ℹ️ 用户已中断生成任务")
            raise e
            
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
    "NanoBananaNode": NanoBananaNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaNode": "artsmcp-nano-banana"
}
