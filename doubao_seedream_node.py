import http.client
import json
import base64
import io
import torch
import numpy as np
from PIL import Image
import requests
import ssl
from urllib.parse import urlparse

class DoubaoSeedreamNode:
    """
    ComfyUI节点：使用Doubao Seedream API进行图片生成
    支持文生图、图生图、图生组图、多图融合
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车，抢视觉冲击力，电影大片，末日既视感，动感，对比色，oc渲染，光线追踪，动态模糊，景深，超现实主义，深蓝，画面通过细腻的丰富的色彩层次塑造主体与场景，质感真实，暗黑风背景的光影效果营造出氛围，整体兼具艺术幻想感，夸张的广角透视效果，耀光，反射，极致的光影，强引力，吞噬",
                    "description": "图片生成的提示词描述，详细描述你想要生成的图片内容"
                }),
                "api_key": ("STRING", {
                    "default": "sk-your-api-key-here",
                    "description": "API密钥，用于身份验证"
                }),
                "base_url": ("STRING", {
                    "default": "api.cozex.cn",
                    "description": "API服务地址，例如：api.cozex.cn"
                }),
                "model": (["doubao-seedream-4-0-250828", "doubao-seedream-4-5-251128"], {
                    "default": "doubao-seedream-4-0-250828"
                }),
                "size": (["2K", "1K", "4K"], {
                    "default": "2K"
                }),
            },
            "optional": {
                "image1": ("IMAGE", {
                    "description": "第一张输入图片，用于图生图或图生组图"
                }),
                "image2": ("IMAGE", {
                    "description": "第二张输入图片，用于多图融合或图生组图"
                }),
                "max_images": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "description": "最大生成图片数量，0=禁用组图生成，1-10=生成对应数量的图片"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "description": "是否在生成的图片上添加水印"
                }),
                "response_format": (["url", "b64_json"], {
                    "default": "url"
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "description": "API请求超时时间（秒），范围：30-600秒"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "image/AI"
    
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
    
    def call_api(self, host, path, payload, headers, timeout):
        """
        使用http.client调用API
        """
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            conn = http.client.HTTPSConnection(host, timeout=timeout, context=context)
            conn.request("POST", path, payload, headers)
            
            res = conn.getresponse()
            data = res.read()
            conn.close()
            
            return res.status, data.decode("utf-8")
            
        except Exception as e:
            print(f"HTTP client error: {e}")
            return None, str(e)
    
    def generate_image(self, prompt, api_key, base_url, model, size, image1=None, image2=None,
                      max_images=0, watermark=False, response_format="url", timeout=120):
        """
        生成图片的主函数
        """
        try:
            # 根据max_images自动判断sequential_image_generation
            if max_images > 0:
                sequential_image_generation = "auto"
            else:
                sequential_image_generation = "disabled"
            
            # 准备请求数据
            request_data = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "sequential_image_generation": sequential_image_generation,
                "stream": False,
                "response_format": response_format,
                "watermark": watermark
            }
            
            # 处理图像输入
            images = []
            if image1 is not None:
                img_url = self.tensor_to_image_url(image1)
                if img_url:
                    images.append(img_url)
            
            if image2 is not None:
                img_url = self.tensor_to_image_url(image2)
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
            if sequential_image_generation == "auto" and max_images > 0:
                request_data["sequential_image_generation_options"] = {
                    "max_images": max_images
                }
            
            payload = json.dumps(request_data)
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # 解析base_url
            if base_url.startswith('http://') or base_url.startswith('https://'):
                parsed_url = urlparse(base_url)
                host = parsed_url.netloc
                path = parsed_url.path if parsed_url.path else "/v1/images/generations"
            else:
                host = base_url
                path = "/v1/images/generations"
            
            print(f"Calling Doubao Seedream API: {host}{path}")
            print(f"Model: {model}, Size: {size}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Image mode: {'文生图' if not images else ('图生图' if len(images) == 1 else '多图融合/组图')}")
            
            # 调用API
            status_code, response_text = self.call_api(host, path, payload, headers, timeout)
            
            if status_code == 200:
                try:
                    result = json.loads(response_text)
                    
                    # 提取图像URL和base64数据
                    image_urls = []
                    base64_images = []
                    
                    # 处理不同的响应格式
                    if 'data' in result:
                        data = result['data']
                        if isinstance(data, list):
                            for item in data:
                                if response_format == "url":
                                    url = item.get('url')
                                    if url:
                                        image_urls.append(url)
                                elif response_format == "b64_json":
                                    b64_data = item.get('b64_json')
                                    if b64_data:
                                        base64_images.append(b64_data)
                        elif isinstance(data, dict):
                            if response_format == "url":
                                url = data.get('url')
                                if url:
                                    image_urls.append(url)
                            elif response_format == "b64_json":
                                b64_data = data.get('b64_json')
                                if b64_data:
                                    base64_images.append(b64_data)
                    elif 'url' in result:
                        image_urls.append(result['url'])
                    
                    # 处理base64格式的图像
                    if base64_images:
                        print(f"Found {len(base64_images)} base64 image(s)")
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
                            print(f"Successfully processed {len(output_tensors)} base64 image(s)!")
                            return (batch_tensor,)
                    
                    # 处理URL格式的图像
                    if image_urls:
                        print(f"Found {len(image_urls)} image URL(s)")
                        
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
                            print(f"Successfully downloaded and merged {len(output_tensors)} image(s)!")
                            return (batch_tensor,)
                        
                        print("Failed to download any images")
                    else:
                        print("No image URLs found in API response")
                        print("Response:", response_text[:1000])
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON response: {e}")
                    print("Raw response:", response_text[:500])
            else:
                print(f"API call failed with status code: {status_code}")
                print(f"Error response: {response_text}")
            
            # 如果失败，返回默认图像或原始输入
            if image1 is not None:
                return (image1,)
            else:
                # 创建一个默认的黑色图像
                default_tensor = torch.zeros((1, 512, 512, 3))
                return (default_tensor,)
            
        except Exception as e:
            print(f"Error in generate_image: {e}")
            import traceback
            traceback.print_exc()
            
            if image1 is not None:
                return (image1,)
            else:
                default_tensor = torch.zeros((1, 512, 512, 3))
                return (default_tensor,)

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "DoubaoSeedreamNode": DoubaoSeedreamNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedreamNode": "Doubao Seedream Image"
}

