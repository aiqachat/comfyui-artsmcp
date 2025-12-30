import http.client
import json
import base64
import io
import socket
import time
import torch
import numpy as np
from PIL import Image
import requests
import ssl
from urllib.parse import urlparse
import configparser
from pathlib import Path

# 加载配置文件
CATEGORY = "artsmcp"
CONFIG_SECTION = "Seedream"  # 独立配置节
CONFIG_PATH = Path(__file__).parent / "config.ini"
CONFIG = configparser.ConfigParser()
if CONFIG_PATH.exists():
    CONFIG.read(CONFIG_PATH, encoding="utf-8")
else:
    CONFIG[CONFIG_SECTION] = {}  # 使用独立配置节
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        CONFIG.write(fp)

class DoubaoSeedreamNode:
    """
    ComfyUI节点：使用Doubao Seedream API进行图片生成
    支持文生图、图生图、图生组图、多图融合
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车，抢视觉冲击力，电影大片，末日既视感，动感，对比色，oc渲染，光线追踪，动态模糊，景深，超现实主义，深蓝，画面通过细腻的丰富的色彩层次塑造主体与场景，质感真实，暗黑风背景的光影效果营造出氛围，整体兼具艺术幻想感，夸张的广角透视效果，耀光，反射，极致的光影，强引力，吞噬",
                    "description": "图片生成的提示词描述，详细描述你想要生成的图片内容",
                    "label": "提示词"
                }),
                "API密钥": ("STRING", {
                    "default": CONFIG.get(CONFIG_SECTION, "api_key", fallback="sk-your-api-key-here"),
                    "description": "API密钥，用于身份验证",
                    "label": "API密钥"
                }),
                "API地址": ("STRING", {
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback="https://api.openai.com"),
                    "description": "API服务地址，例如：api.openai.com",
                    "label": "API地址"
                }),
                "模型": (["doubao-seedream-4-0-250828", "doubao-seedream-4-5-251128"], {
                    "default": "doubao-seedream-4-0-250828",
                    "label": "模型"
                }),
                "宽度": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 64,
                    "description": "生成图片的宽度（像素），建议为64的倍数",
                    "label": "宽度"
                }),
                "高度": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 64,
                    "description": "生成图片的高度（像素），建议为64的倍数",
                    "label": "高度"
                }),
            },
            "optional": {
                "输入图片1": ("IMAGE", {
                    "description": "第一张输入图片，用于图生图或图生组图",
                    "label": "输入图片1"
                }),
                "输入图片2": ("IMAGE", {
                    "description": "第二张输入图片，用于多图融合或图生组图",
                    "label": "输入图片2"
                }),
                "最大图片数量": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "description": "最大生成图片数量，0=禁用组图生成，1-10=生成对应数量的图片",
                    "label": "最大图片数量"
                }),
                "水印": ("BOOLEAN", {
                    "default": False,
                    "description": "是否在生成的图片上添加水印",
                    "label": "水印"
                }),
                "返回格式": (["url", "b64_json"], {
                    "default": "url",
                    "label": "返回格式"
                }),
                "请求超时": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "description": "API请求超时时间（秒），范围：30-600秒",
                    "label": "请求超时"
                }),
                "调试模式": ("BOOLEAN", {
                    "default": False,
                    "description": "调试模式：输出完整的API请求和响应信息",
                    "label": "调试模式"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片输出",)
    FUNCTION = "generate_image"
    CATEGORY = CATEGORY
    OUTPUT_NODE = False
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """强制每次都重新执行(外部API请求)"""
        import time
        return time.time()
    
    def tensor_to_image_url(self, tensor):
        """
        将ComfyUI的tensor图像转换为base64 data URL格式
        """
        try:
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            if tensor.max() <= 1.0:
                tensor = tensor * 255.0
            
            tensor = tensor.clamp(0, 255).byte()
            numpy_image = tensor.cpu().numpy()
            pil_image = Image.fromarray(numpy_image, mode='RGB')
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            img_bytes = buffer.getvalue()
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/jpeg;base64,{base64_string}"
            
        except Exception as e:
            print(f"Error converting tensor to image URL: {e}")
            return None
    
    def url_to_tensor(self, image_url):
        """
        从URL下载图像并转换为ComfyUI tensor
        """
        try:
            print(f"Downloading image from URL: {image_url}")
            
            response = requests.get(image_url, timeout=30, verify=False)
            response.raise_for_status()
            
            pil_image = Image.open(io.BytesIO(response.content))
            pil_image = pil_image.convert('RGB')
            
            print(f"Downloaded image size: {pil_image.size}")
            
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(numpy_image).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"Error downloading/converting image from URL: {e}")
            return None
    
    def call_api(self, host, path, payload, headers, timeout, max_retries=3):
        """
        使用http.client调用API,支持指数退避重试机制
        """
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[尝试 {attempt}/{max_retries}] 正在调用API...")
                
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                conn = http.client.HTTPSConnection(host, timeout=timeout, context=context)
                conn.request("POST", path, payload, headers)
                
                res = conn.getresponse()
                data = res.read()
                conn.close()
                
                # 成功返回
                if res.status == 200:
                    print(f"[成功] API调用成功")
                    return res.status, data.decode("utf-8")
                
                # 服务端错误(5xx)可重试
                elif res.status >= 500:
                    error_msg = data.decode("utf-8")
                    print(f"[警告] 服务器错误 {res.status}: {error_msg[:100]}")
                    last_error = (res.status, error_msg)
                    
                    if attempt < max_retries:
                        wait_time = min(2 ** (attempt - 1), 30)  # 指数退避,最多30秒
                        print(f"[重试] 等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                else:
                    # 客户端错误(4xx)不重试
                    return res.status, data.decode("utf-8")
                    
            except socket.timeout as e:
                print(f"[超时] 请求超时: {e}")
                last_error = (None, f"Timeout: {e}")
                
                if attempt < max_retries:
                    wait_time = min(2 ** (attempt - 1), 30)
                    print(f"[重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                print(f"[错误] HTTP client error: {e}")
                last_error = (None, str(e))
                
                if attempt < max_retries:
                    wait_time = min(2 ** (attempt - 1), 30)
                    print(f"[重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
        
        # 所有重试都失败
        print(f"[失败] API调用失败,已重试 {max_retries} 次")
        if last_error:
            return last_error
        return None, "All retries failed"
    
    def validate_resolution(self, 模型, 宽度, 高度):
        """
        验证分辨率是否符合模型要求
        返回: (是否有效, 错误信息)
        """
        # 计算总像素和宽高比
        total_pixels = 宽度 * 高度
        aspect_ratio = 宽度 / 高度
        
        # 定义不同模型的限制
        model_limits = {
            "doubao-seedream-4-0-250828": {
                "min_pixels": 921600,      # 1280x720
                "max_pixels": 16777216,    # 4096x4096
                "min_ratio": 1/16,
                "max_ratio": 16,
                "name": "doubao-seedream-4-0"
            },
            "doubao-seedream-4-5-251128": {
                "min_pixels": 3686400,     # 2560x1440
                "max_pixels": 16777216,    # 4096x4096
                "min_ratio": 1/16,
                "max_ratio": 16,
                "name": "doubao-seedream-4-5"
            }
        }
        
        if 模型 not in model_limits:
            return True, ""  # 未知模型不验证
        
        limits = model_limits[模型]
        
        # 验证总像素范围
        if total_pixels < limits["min_pixels"]:
            return False, (
                f"❌ 分辨率验证失败：总像素数 {total_pixels:,} 低于 {limits['name']} 模型的最小要求 {limits['min_pixels']:,}\n"
                f"   建议：增加宽度或高度，使总像素数 ≥ {limits['min_pixels']:,}"
            )
        
        if total_pixels > limits["max_pixels"]:
            return False, (
                f"❌ 分辨率验证失败：总像素数 {total_pixels:,} 超过 {limits['name']} 模型的最大限制 {limits['max_pixels']:,}\n"
                f"   建议：减少宽度或高度，使总像素数 ≤ {limits['max_pixels']:,}"
            )
        
        # 验证宽高比范围
        if aspect_ratio < limits["min_ratio"]:
            return False, (
                f"❌ 分辨率验证失败：宽高比 {aspect_ratio:.4f} (宽/高={宽度}/{高度}) 低于模型最小要求 {limits['min_ratio']:.4f}\n"
                f"   建议：增加宽度或减少高度，使宽高比在 [{limits['min_ratio']:.4f}, {limits['max_ratio']:.1f}] 范围内"
            )
        
        if aspect_ratio > limits["max_ratio"]:
            return False, (
                f"❌ 分辨率验证失败：宽高比 {aspect_ratio:.4f} (宽/高={宽度}/{高度}) 超过模型最大限制 {limits['max_ratio']:.1f}\n"
                f"   建议：减少宽度或增加高度，使宽高比在 [{limits['min_ratio']:.4f}, {limits['max_ratio']:.1f}] 范围内"
            )
        
        return True, ""
    
    def generate_image(self, 提示词, API密钥, API地址, 模型, 宽度, 高度, 输入图片1=None, 输入图片2=None,
                      最大图片数量=0, 水印=False, 返回格式="url", 请求超时=120, 调试模式=False):
        """
        生成图片的主函数
        """
        try:
            # 验证分辨率
            is_valid, error_msg = self.validate_resolution(模型, 宽度, 高度)
            if not is_valid:
                print(f"\n{'='*60}")
                print(error_msg)
                print(f"\n💡 参考分辨率：")
                if 模型 == "doubao-seedream-4-0-250828":
                    print(f"   - 常用: 1280x720 (921K), 1920x1080 (2M), 2048x2048 (4M), 2560x1440 (3.6M)")
                    print(f"   - 最大: 4096x4096 (16M)")
                elif 模型 == "doubao-seedream-4-5-251128":
                    print(f"   - 最小: 2560x1440 (3.6M)")
                    print(f"   - 常用: 2048x2048 (4M), 3072x2048 (6M), 3750x1250 (4.6M)")
                    print(f"   - 最大: 4096x4096 (16M)")
                print(f"{'='*60}\n")
                raise ValueError(error_msg)
            # 保存配置到独立配置节
            if not CONFIG.has_section(CONFIG_SECTION):
                CONFIG.add_section(CONFIG_SECTION)
            
            if API密钥.strip():
                CONFIG.set(CONFIG_SECTION, "api_key", API密钥.strip())
            if API地址.strip():
                CONFIG.set(CONFIG_SECTION, "api_url", API地址.strip())
            
            with CONFIG_PATH.open("w", encoding="utf-8") as fp:
                CONFIG.write(fp)
            
            # 根据max_images自动判断sequential_image_generation
            if 最大图片数量 > 0:
                sequential_image_generation = "auto"
            else:
                sequential_image_generation = "disabled"
            
            # 准备请求数据
            # 将宽高转换为API要求的格式
            size_string = f"{宽度}x{高度}"
            
            request_data = {
                "model": 模型,
                "prompt": 提示词,
                "size": size_string,
                "sequential_image_generation": sequential_image_generation,
                "stream": False,
                "response_format": 返回格式,
                "watermark": 水印
            }
            
            # 处理图像输入
            images = []
            if 输入图片1 is not None:
                img_url = self.tensor_to_image_url(输入图片1)
                if img_url:
                    images.append(img_url)
            
            if 输入图片2 is not None:
                img_url = self.tensor_to_image_url(输入图片2)
                if img_url:
                    images.append(img_url)
            
            # 根据图像数量决定API参数
            if len(images) == 1:
                # 单图：图生图
                request_data["image"] = images[0]
            elif len(images) > 1:
                # 多图：图生组图或多图融合
                request_data["image"] = images
            
            # 如果启用了组图生成，添加配置
            if sequential_image_generation == "auto" and 最大图片数量 > 0:
                request_data["sequential_image_generation_options"] = {
                    "max_images": 最大图片数量
                }
            
            payload = json.dumps(request_data)
            
            headers = {
                'Authorization': f'Bearer {API密钥}',
                'Content-Type': 'application/json'
            }
            
            # 解析base_url
            if API地址.startswith('http://') or API地址.startswith('https://'):
                parsed_url = urlparse(API地址)
                host = parsed_url.netloc
                path = parsed_url.path if parsed_url.path else "/v1/images/generations"
            else:
                host = API地址
                path = "/v1/images/generations"
            
            print(f"\n{'='*60}")
            print(f"[Doubao-Seedream] 调用API")
            print(f"  - 地址: {host}{path}")
            print(f"  - 模型: {模型}")
            print(f"  - 分辨率: {宽度}x{高度} (总像素: {宽度*高度:,}, 宽高比: {宽度/高度:.2f})")
            print(f"  - 提示词: {提示词[:50]}...")
            print(f"  - 模式: {'文生图' if not images else ('图生图' if len(images) == 1 else '多图融合/组图')}")
            print(f"  - 组图生成: {'启用('+str(最大图片数量)+'张)' if 最大图片数量 > 0 else '禁用'}")
            print(f"  - 水印: {水印}")
            print(f"  - 返回格式: {返回格式}")
            print(f"={'='*60}\n")
            
            # Debug 模式：输出请求数据
            if 调试模式:
                print(f"\n{'='*60}")
                print(f"🐛 DEBUG: Request Data")
                print(f"{'='*60}")
                # 创建一个用于显示的请求数据副本（不包含base64图片）
                debug_request = request_data.copy()
                if 'image' in debug_request:
                    if isinstance(debug_request['image'], list):
                        debug_request['image'] = [f"<base64_image_{i+1}>" for i in range(len(debug_request['image']))]
                    else:
                        debug_request['image'] = "<base64_image>"
                print(json.dumps(debug_request, indent=2, ensure_ascii=False))
                print(f"{'='*60}\n")
            
            # 调用API
            status_code, response_text = self.call_api(host, path, payload, headers, 请求超时)
            
            if status_code == 200:
                try:
                    result = json.loads(response_text)
                    
                    # Debug 模式：输出完整响应
                    if 调试模式:
                        print(f"\n{'='*60}")
                        print(f"🐛 DEBUG: Full API Response")
                        print(f"{'='*60}")
                        # 创建一个用于显示的响应副本（不包含完整base64）
                        debug_result = json.loads(response_text)
                        if 'data' in debug_result:
                            data = debug_result['data']
                            if isinstance(data, list):
                                for item in data:
                                    if 'b64_json' in item and len(item['b64_json']) > 100:
                                        item['b64_json'] = item['b64_json'][:100] + '... (truncated)'
                            elif isinstance(data, dict) and 'b64_json' in data and len(data['b64_json']) > 100:
                                data['b64_json'] = data['b64_json'][:100] + '... (truncated)'
                        print(json.dumps(debug_result, indent=2, ensure_ascii=False))
                        print(f"{'='*60}\n")
                    
                    # 提取图像URL和base64数据
                    image_urls = []
                    base64_images = []
                    
                    # 处理不同的响应格式
                    if 'data' in result:
                        data = result['data']
                        if isinstance(data, list):
                            for item in data:
                                if 返回格式 == "url":
                                    url = item.get('url')
                                    if url:
                                        image_urls.append(url)
                                elif 返回格式 == "b64_json":
                                    b64_data = item.get('b64_json')
                                    if b64_data:
                                        base64_images.append(b64_data)
                        elif isinstance(data, dict):
                            if 返回格式 == "url":
                                url = data.get('url')
                                if url:
                                    image_urls.append(url)
                            elif 返回格式 == "b64_json":
                                b64_data = data.get('b64_json')
                                if b64_data:
                                    base64_images.append(b64_data)
                    elif 'url' in result:
                        image_urls.append(result['url'])
                    
                    # 处理base64格式的图像
                    if base64_images:
                        print(f"\n[INFO] 找到 {len(base64_images)} 张 base64 格式图片")
                        output_tensors = []
                        for b64_data in base64_images:
                            try:
                                # 解码base64图像
                                img_bytes = base64.b64decode(b64_data)
                                pil_image = Image.open(io.BytesIO(img_bytes))
                                pil_image = pil_image.convert('RGB')
                                numpy_image = np.array(pil_image).astype(np.float32) / 255.0
                                tensor = torch.from_numpy(numpy_image).unsqueeze(0)
                                output_tensors.append(tensor)
                            except Exception as e:
                                print(f"Error processing base64 image: {e}")
                        
                        if output_tensors:
                            # 将所有tensor合并成一个批次
                            batch_tensor = torch.cat(output_tensors, dim=0)
                            print(f"\n{'='*60}")
                            print(f"[SUCCESS] ✅ 成功生成 {len(output_tensors)} 张图片!")
                            print(f"[INFO] 批次尺寸: {batch_tensor.shape}")
                            print(f"{'='*60}\n")
                            return (batch_tensor,)
                    
                    # 处理URL格式的图像
                    if image_urls:
                        print(f"\n[INFO] 找到 {len(image_urls)} 张图片URL")
                        
                        # 下载所有图像并转换为tensor
                        output_tensors = []
                        for url in image_urls:
                            output_tensor = self.url_to_tensor(url)
                            if output_tensor is not None:
                                output_tensors.append(output_tensor)
                        
                        if output_tensors:
                            # 将所有tensor合并成一个批次
                            # 每个tensor的形状是 (1, height, width, 3)
                            # 使用torch.cat在batch维度（dim=0）上合并
                            batch_tensor = torch.cat(output_tensors, dim=0)
                            print(f"\n{'='*60}")
                            print(f"[SUCCESS] ✅ 成功生成 {len(output_tensors)} 张图片!")
                            print(f"[INFO] 批次尺寸: {batch_tensor.shape}")
                            print(f"{'='*60}\n")
                            return (batch_tensor,)
                        
                        print("[ERROR] 下载所有图片失败")
                    else:
                        print("[ERROR] API响应中未找到图片URL")
                        if 调试模式:
                            print(f"[DEBUG] 响应内容: {response_text[:1000]}")
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON response: {e}")
                    print("Raw response:", response_text[:500])
            else:
                print(f"\n[ERROR] API调用失败，状态码: {status_code}")
                print(f"[ERROR] 错误响应: {response_text[:500]}")
                print(f"\n💡 可能的解决方案:")
                print(f"   1. 检查 API Key 是否有效")
                print(f"   2. 确认 API 服务地址是否正确")
                print(f"   3. 查看错误信息，调整参数")
                print(f"   4. 检查网络连接是否正常")
            
            # 如果失败，返回默认图像或原始输入
            if 输入图片1 is not None:
                return (输入图片1,)
            else:
                # 创建一个默认的黑色图像
                default_tensor = torch.zeros((1, 512, 512, 3))
                return (default_tensor,)
            
        except Exception as e:
            print(f"Error in generate_image: {e}")
            import traceback
            traceback.print_exc()
            # 直接抛出异常,不返回默认图片
            raise e

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "DoubaoSeedreamNode": DoubaoSeedreamNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedreamNode": "artsmcp-seedream"
}

