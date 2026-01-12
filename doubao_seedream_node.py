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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
                "并发请求数": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "description": "并发请求的数量，1=单次请求，2-10=并发多次请求",
                    "label": "并发请求数"
                }),
                "响应格式": (["url", "b64_json"], {
                    "default": "url",
                    "label": "响应格式"
                }),
                "超时秒数": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "description": "API请求超时时间（秒），范围：30-600秒",
                    "label": "超时秒数"
                }),
                "最大重试次数": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "description": "API请求失败时的最大重试次数,0=不重试,1-10=重试对应次数",
                    "label": "最大重试次数"
                }),
                "启用分行提示词": ("BOOLEAN", {
                    "default": False,
                    "description": "启用后,将提示词按行分割,每行作为独立提示词进行请求。配合并发请求数可实现:N行提示词×M并发=N×M张图片",
                    "label": "启用分行提示词"
                }),
                "水印": ("BOOLEAN", {
                    "default": False,
                    "description": "是否在生成的图片上添加水印",
                    "label": "水印"
                }),
                "详细日志": ("BOOLEAN", {
                    "default": False,
                    "description": "详细日志：输出完整的API请求和响应信息",
                    "label": "详细日志"
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
    
    def call_api(self, host, path, payload, headers, timeout, max_retries, request_id=None):
        """
        使用http.client调用API,支持指数退避重试机制
        """
        last_error = None
        prefix = f"[请求 {request_id}] " if request_id else ""
        
        # 如果max_retries为0,至少执行1次请求
        total_attempts = max(1, max_retries + 1)
        
        for attempt in range(1, total_attempts + 1):
            try:
                print(f"{prefix}[尝试 {attempt}/{max_retries}] 正在调用API...")
                
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
                    print(f"{prefix}[成功] API调用成功")
                    return res.status, data.decode("utf-8")
                
                # 服务端错误(5xx)可重试
                elif res.status >= 500:
                    error_msg = data.decode("utf-8")
                    print(f"{prefix}[警告] 服务器错误 {res.status}: {error_msg[:100]}")
                    last_error = (res.status, error_msg)
                    
                    if attempt < total_attempts:
                        wait_time = min(2 ** (attempt - 1), 30)  # 指数退避,最多30秒
                        print(f"{prefix}[重试] 等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                else:
                    # 客户端错误(4xx)不重试
                    return res.status, data.decode("utf-8")
                    
            except socket.timeout as e:
                print(f"{prefix}[超时] 请求超时: {e}")
                last_error = (None, f"Timeout: {e}")
                
                if attempt < total_attempts:
                    wait_time = min(2 ** (attempt - 1), 30)
                    print(f"{prefix}[重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                print(f"{prefix}[错误] HTTP client error: {e}")
                last_error = (None, str(e))
                
                if attempt < total_attempts:
                    wait_time = min(2 ** (attempt - 1), 30)
                    print(f"{prefix}[重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
        
        # 所有重试都失败
        retry_msg = f"已重试 {max_retries} 次" if max_retries > 0 else "未启用重试"
        print(f"{prefix}[失败] API调用失败,{retry_msg}")
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
    
    def parse_multiline_prompts(self, prompt_text, enable_multiline):
        """
        解析提示词,支持分行模式
        返回: [prompt1, prompt2, ...]
        """
        if not enable_multiline:
            # 单提示词模式,返回原始文本
            return [prompt_text.strip()] if prompt_text.strip() else []
        
        # 分行模式,按行分割并过滤空行
        lines = [line.strip() for line in prompt_text.split('\n')]
        valid_prompts = [line for line in lines if line]
        
        return valid_prompts
    
    def call_api_concurrent(self, host, path, payload, headers, timeout, 并发数, 最大重试次数, 调试模式=False):
        """
        并发调用API,等待所有请求完成或超时
        返回: [(status_code, response_text), ...]
        """
        print(f"\n{'='*60}")
        print(f"🚀 [并发模式] 启动 {并发数} 个并发请求")
        print(f"  - 最大重试次数: {最大重试次数}")
        print(f"{'='*60}\n")
        
        results = []
        lock = threading.Lock()
        
        def single_request(request_id):
            """单个请求的包装函数"""
            try:
                start_time = time.time()
                status_code, response_text = self.call_api(
                    host, path, payload, headers, timeout, 
                    max_retries=最大重试次数, request_id=request_id
                )
                elapsed = time.time() - start_time
                
                with lock:
                    print(f"✅ [请求 {request_id}] 完成，耗时: {elapsed:.2f}秒")
                
                return {
                    'request_id': request_id,
                    'status_code': status_code,
                    'response_text': response_text,
                    'elapsed_time': elapsed,
                    'success': status_code == 200
                }
            except Exception as e:
                with lock:
                    print(f"❌ [请求 {request_id}] 异常: {e}")
                return {
                    'request_id': request_id,
                    'status_code': None,
                    'response_text': str(e),
                    'elapsed_time': 0,
                    'success': False
                }
        
        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=并发数) as executor:
            # 提交所有任务
            futures = {executor.submit(single_request, i+1): i+1 for i in range(并发数)}
            
            # 等待所有任务完成
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        failed_count = 并发数 - success_count
        total_time = max([r['elapsed_time'] for r in results]) if results else 0
        avg_time = sum([r['elapsed_time'] for r in results]) / len(results) if results else 0
        
        print(f"\n{'='*60}")
        print(f"📊 [并发统计]")
        print(f"  - 总请求数: {并发数}")
        print(f"  - 成功: {success_count} | 失败: {failed_count}")
        print(f"  - 总耗时: {total_time:.2f}秒")
        print(f"  - 平均耗时: {avg_time:.2f}秒")
        print(f"{'='*60}\n")
        
        # 调试模式输出详细信息
        if 调试模式:
            print(f"\n{'='*60}")
            print(f"🐛 DEBUG: 并发请求详细结果")
            print(f"{'='*60}")
            for result in sorted(results, key=lambda x: x['request_id']):
                print(f"\n[请求 {result['request_id']}]")
                print(f"  状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
                print(f"  状态码: {result['status_code']}")
                print(f"  耗时: {result['elapsed_time']:.2f}秒")
                if not result['success']:
                    print(f"  错误: {result['response_text'][:200]}")
            print(f"{'='*60}\n")
        
        return results
    
    def generate_image(self, 提示词, API密钥, API地址, 模型, 宽度, 高度, 输入图片1=None, 输入图片2=None,
                      最大图片数量=0, 并发请求数=1, 响应格式="url", 超时秒数=120, 最大重试次数=3, 启用分行提示词=False, 水印=False, 详细日志=False):
        """
        生成图片的主函数
        """
        # 设置日志级别
        self.verbose = 详细日志
        
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
            # 保存配置到独立配置节（重新读取确保不覆盖其他节点配置）
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
            
            # 解析提示词(支持分行模式)
            prompts = self.parse_multiline_prompts(提示词, 启用分行提示词)
            
            if not prompts:
                print("[ERROR] 提示词为空,无法生成图片")
                default_tensor = torch.zeros((1, 512, 512, 3))
                return (default_tensor,)
            
            # 打印提示词信息
            print(f"\n{'='*60}")
            print(f"📝 [提示词解析]")
            print(f"  - 分行模式: {启用分行提示词}")
            print(f"  - 提示词数量: {len(prompts)}")
            if 启用分行提示词 and len(prompts) > 1:
                print(f"  - 提示词列表:")
                for idx, p in enumerate(prompts, 1):
                    preview = p[:50] + '...' if len(p) > 50 else p
                    print(f"    [{idx}] {preview}")
            else:
                preview = prompts[0][:50] + '...' if len(prompts[0]) > 50 else prompts[0]
                print(f"  - 提示词: {preview}")
            print(f"  - 总请求数: {len(prompts) * 并发请求数} (提示词×并发)")
            print(f"  - 预计生成图片数: {len(prompts) * 并发请求数}")
            print(f"{'='*60}\n")
            
            # 解析base_url
            if API地址.startswith('http://') or API地址.startswith('https://'):
                parsed_url = urlparse(API地址)
                host = parsed_url.netloc
                path = parsed_url.path if parsed_url.path else "/v1/images/generations"
            else:
                host = API地址
                path = "/v1/images/generations"
            
            # 准备所有请求的payload
            all_payloads = []
            
            for prompt_idx, single_prompt in enumerate(prompts, 1):
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
                    "prompt": single_prompt,
                    "size": size_string,
                    "sequential_image_generation": sequential_image_generation,
                    "stream": False,
                    "response_format": 响应格式,
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
                    # 单图:图生图
                    request_data["image"] = images[0]
                elif len(images) > 1:
                    # 多图:图生组图或多图融合
                    request_data["image"] = images
                            
                # 如果启用了组图生成,添加配置
                if sequential_image_generation == "auto" and 最大图片数量 > 0:
                    request_data["sequential_image_generation_options"] = {
                        "max_images": 最大图片数量
                    }
                            
                payload = json.dumps(request_data)
                            
                # 为每个提示词生成指定数量的并发请求
                for concurrent_idx in range(并发请求数):
                    all_payloads.append({
                        'payload': payload,
                        'prompt_id': prompt_idx,
                        'concurrent_id': concurrent_idx + 1,
                        'prompt_text': single_prompt
                    })
                        
            headers = {
                'Authorization': f'Bearer {API密钥}',
                'Content-Type': 'application/json'
            }
            
            print(f"\n{'='*60}")
            print(f"[Doubao-Seedream] 调用API")
            print(f"  - 地址: {host}{path}")
            print(f"  - 模型: {模型}")
            print(f"  - 分辨率: {宽度}x{高度} (总像素: {宽度*高度:,}, 宽高比: {宽度/高度:.2f})")
            if 输入图片1 is not None or 输入图片2 is not None:
                images_count = sum([1 for img in [输入图片1, 输入图片2] if img is not None])
                print(f"  - 模式: {'图生图' if images_count == 1 else '多图融合/组图'}")
            else:
                print(f"  - 模式: 文生图")
            print(f"  - 组图生成: {'启用('+str(最大图片数量)+'张)' if 最大图片数量 > 0 else '禁用'}")
            print(f"  - 分行提示词: {启用分行提示词}")
            print(f"  - 提示词数: {len(prompts)}")
            print(f"  - 每提示词并发数: {并发请求数}")
            print(f"  - 总请求数: {len(all_payloads)}")
            print(f"  - 最大重试次数: {最大重试次数}")
            print(f"  - 水印: {水印}")
            print(f"  - 响应格式: {响应格式}")
            print(f"={'='*60}\n")
            
            # Debug 模式:输出请求数据
            if self.verbose:
                self.log(f"\n{'='*60}", "DEBUG")
                self.log(f"🐛 DEBUG: Request Data Summary", "DEBUG")
                self.log(f"{'='*60}", "DEBUG")
                self.log(f"总请求数: {len(all_payloads)}", "DEBUG")
                for payload_info in all_payloads[:3]:  # 只显示前3个请求
                    debug_request = json.loads(payload_info['payload'])
                    if 'image' in debug_request:
                        if isinstance(debug_request['image'], list):
                            debug_request['image'] = [f"<base64_image_{i+1}>" for i in range(len(debug_request['image']))]
                        else:
                            debug_request['image'] = "<base64_image>"
                    self.log(f"\n[提示词 {payload_info['prompt_id']}-并发 {payload_info['concurrent_id']}]", "DEBUG")
                    self.log(json.dumps(debug_request, indent=2, ensure_ascii=False), "DEBUG")
                if len(all_payloads) > 3:
                    self.log(f"\n... 还有 {len(all_payloads)-3} 个请求(已省略)", "DEBUG")
                self.log(f"{'='*60}\n", "DEBUG")
                        
            # 批量并发调用API
            print(f"\n{'='*60}")
            print(f"🚀 [批量并发模式] 启动 {len(all_payloads)} 个请求")
            print(f"  - 提示词数量: {len(prompts)}")
            print(f"  - 每提示词并发数: {并发请求数}")
            print(f"  - 最大重试次数: {最大重试次数}")
            print(f"{'='*60}\n")
                        
            all_responses = []
            lock = threading.Lock()
                        
            def single_request(payload_info, request_id):
                """单个请求的包装函数"""
                try:
                    start_time = time.time()
                    prefix = f"[提示词{payload_info['prompt_id']}-并发{payload_info['concurrent_id']}]"
                    print(f"{prefix} 开始请求...")
                                
                    status_code, response_text = self.call_api(
                        host, path, payload_info['payload'], headers, 超时秒数,
                        max_retries=最大重试次数, request_id=request_id
                    )
                    elapsed = time.time() - start_time
                                
                    with lock:
                        if status_code == 200:
                            print(f"✅ {prefix} 完成,耗时: {elapsed:.2f}秒")
                        else:
                            print(f"❌ {prefix} 失败,状态码: {status_code}")
                                
                    return {
                        'request_id': request_id,
                        'prompt_id': payload_info['prompt_id'],
                        'concurrent_id': payload_info['concurrent_id'],
                        'status_code': status_code,
                        'response_text': response_text,
                        'elapsed_time': elapsed,
                        'success': status_code == 200,
                        'prompt_text': payload_info['prompt_text']
                    }
                except Exception as e:
                    with lock:
                        print(f"❌ [请求 {request_id}] 异常: {e}")
                    return {
                        'request_id': request_id,
                        'prompt_id': payload_info.get('prompt_id', 0),
                        'concurrent_id': payload_info.get('concurrent_id', 0),
                        'status_code': None,
                        'response_text': str(e),
                        'elapsed_time': 0,
                        'success': False,
                        'prompt_text': payload_info.get('prompt_text', '')
                    }
                        
            # 使用线程池并发执行所有请求
            max_workers = min(len(all_payloads), 10)  # 最多10个并发线程
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(single_request, payload_info, i+1): i 
                          for i, payload_info in enumerate(all_payloads)}
                            
                for future in as_completed(futures):
                    result = future.result()
                    all_responses.append(result)
                        
            # 统计结果
            success_count = sum(1 for r in all_responses if r['success'])
            failed_count = len(all_responses) - success_count
            total_time = max([r['elapsed_time'] for r in all_responses]) if all_responses else 0
            avg_time = sum([r['elapsed_time'] for r in all_responses]) / len(all_responses) if all_responses else 0
                        
            print(f"\n{'='*60}")
            print(f"📊 [批量请求统计]")
            print(f"  - 总请求数: {len(all_responses)}")
            print(f"  - 成功: {success_count} | 失败: {failed_count}")
            print(f"  - 总耗时: {total_time:.2f}秒")
            print(f"  - 平均耗时: {avg_time:.2f}秒")
                        
            # 按提示词分组统计
            if 启用分行提示词 and len(prompts) > 1:
                print(f"\n  按提示词统计:")
                for prompt_id in range(1, len(prompts) + 1):
                    prompt_results = [r for r in all_responses if r['prompt_id'] == prompt_id]
                    prompt_success = sum(1 for r in prompt_results if r['success'])
                    print(f"    [提示词{prompt_id}] 成功: {prompt_success}/{len(prompt_results)}")
                        
            print(f"{'='*60}\n")
                        
            # 收集所有成功的响应
            successful_responses = [r['response_text'] for r in all_responses if r['success']]
                        
            if not successful_responses:
                print(f"\n[ERROR] 所有请求都失败了")
                # 收集失败原因
                failed_responses = [r for r in all_responses if not r['success']]
                error_details = []
                for r in failed_responses[:3]:  # 只显示前3个错误
                    status = r.get('status_code', 'Unknown')
                    error_details.append(f"状态码: {status}")
                error_msg = f"API请求失败: {', '.join(error_details)}"
                raise ValueError(error_msg)
            
            # 处理所有响应,提取图片URL和base64数据
            all_image_urls = []
            all_base64_images = []
            
            for idx, response_text in enumerate(successful_responses):
                try:
                    result = json.loads(response_text)
                    
                    # Debug 模式：输出完整响应
                    if self.verbose and 并发请求数 <= 1:
                        self.log(f"\n{'='*60}", "DEBUG")
                        self.log(f"🐛 DEBUG: Full API Response", "DEBUG")
                        self.log(f"{'='*60}", "DEBUG")
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
                        self.log(json.dumps(debug_result, indent=2, ensure_ascii=False), "DEBUG")
                        self.log(f"{'='*60}\n", "DEBUG")
                    
                    # 处理不同的响应格式
                    if 'data' in result:
                        data = result['data']
                        if isinstance(data, list):
                            for item in data:
                                if 响应格式 == "url":
                                    url = item.get('url')
                                    if url:
                                        all_image_urls.append(url)
                                elif 响应格式 == "b64_json":
                                    b64_data = item.get('b64_json')
                                    if b64_data:
                                        all_base64_images.append(b64_data)
                        elif isinstance(data, dict):
                            if 响应格式 == "url":
                                url = data.get('url')
                                if url:
                                    all_image_urls.append(url)
                            elif 响应格式 == "b64_json":
                                b64_data = data.get('b64_json')
                                if b64_data:
                                    all_base64_images.append(b64_data)
                    elif 'url' in result:
                        all_image_urls.append(result['url'])
                        
                except json.JSONDecodeError as e:
                    print(f"[警告] 响应 {idx+1} JSON解析失败: {e}")
                    print(f"Raw response: {response_text[:500]}")
                    continue
                    
            # 处理base64格式的图像
            if all_base64_images:
                print(f"\n{'='*60}")
                print(f"📥 [下载] 开始处理 {len(all_base64_images)} 张 base64 格式图片")
                print(f"{'='*60}\n")
                
                output_tensors = []
                for idx, b64_data in enumerate(all_base64_images, 1):
                    try:
                        print(f"[处理] base64图片 {idx}/{len(all_base64_images)}...")
                        # 解码base64图像
                        img_bytes = base64.b64decode(b64_data)
                        pil_image = Image.open(io.BytesIO(img_bytes))
                        pil_image = pil_image.convert('RGB')
                        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
                        tensor = torch.from_numpy(numpy_image).unsqueeze(0)
                        output_tensors.append(tensor)
                        print(f"✅ [完成] base64图片 {idx}")
                    except Exception as e:
                        print(f"❌ [错误] 处理base64图片 {idx} 失败: {e}")
                
                if output_tensors:
                    # 将所有tensor合并成一个批次
                    batch_tensor = torch.cat(output_tensors, dim=0)
                    print(f"\n{'='*60}")
                    print(f"[SUCCESS] ✅ 成功生成 {len(output_tensors)} 张图片!")
                    print(f"[INFO] 批次尺寸: {batch_tensor.shape}")
                    if 并发请求数 > 1:
                        print(f"[INFO] 并发请求数: {并发请求数}")
                    print(f"{'='*60}\n")
                    return (batch_tensor,)
            
            # 处理URL格式的图像
            if all_image_urls:
                print(f"\n{'='*60}")
                print(f"📥 [下载] 开始下载 {len(all_image_urls)} 张图片")
                print(f"{'='*60}\n")
                
                # 使用线程池并发下载图片
                output_tensors = []
                
                def download_image(url, idx):
                    try:
                        print(f"[下载] 图片 {idx}/{len(all_image_urls)} - {url[:80]}...")
                        tensor = self.url_to_tensor(url)
                        if tensor is not None:
                            print(f"✅ [完成] 图片 {idx}")
                            return (idx, tensor)
                        else:
                            print(f"❌ [失败] 图片 {idx}")
                            return (idx, None)
                    except Exception as e:
                        print(f"❌ [错误] 图片 {idx} 下载异常: {e}")
                        return (idx, None)
                
                # 并发下载
                download_workers = min(len(all_image_urls), 5)  # 最多5个并发下载
                with ThreadPoolExecutor(max_workers=download_workers) as executor:
                    futures = {executor.submit(download_image, url, i+1): i for i, url in enumerate(all_image_urls)}
                    
                    results = [None] * len(all_image_urls)
                    for future in as_completed(futures):
                        idx, tensor = future.result()
                        if tensor is not None:
                            results[idx-1] = tensor
                
                # 过滤掉失败的下载
                output_tensors = [t for t in results if t is not None]
                
                if output_tensors:
                    # 将所有tensor合并成一个批次
                    batch_tensor = torch.cat(output_tensors, dim=0)
                    print(f"\n{'='*60}")
                    print(f"[SUCCESS] ✅ 成功生成 {len(output_tensors)}/{len(all_image_urls)} 张图片!")
                    print(f"[INFO] 批次尺寸: {batch_tensor.shape}")
                    if 并发请求数 > 1:
                        print(f"[INFO] 并发请求数: {并发请求数}")
                    print(f"{'='*60}\n")
                    return (batch_tensor,)
                
                print("[ERROR] 下载所有图片失败")
                raise ValueError("图片下载失败,请检查网络连接或响应格式")
            else:
                print("[ERROR] API响应中未找到图片URL或base64数据")
                raise ValueError("API响应格式错误,未找到图片数据")
            
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

