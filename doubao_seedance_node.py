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
import os
import tempfile
import folder_paths
import cv2
import configparser
from pathlib import Path

# 加载配置文件
CATEGORY = "artsmcp"
CONFIG_SECTION = "Seedance"  # 独立配置节
CONFIG_PATH = Path(__file__).parent / "config.ini"
CONFIG = configparser.ConfigParser()
if CONFIG_PATH.exists():
    CONFIG.read(CONFIG_PATH, encoding="utf-8")
else:
    CONFIG[CONFIG_SECTION] = {}  # 使用独立配置节
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        CONFIG.write(fp)

# ComfyUI 中断检测
try:
    import comfy.model_management as model_management
    COMFY_INTERRUPT_AVAILABLE = True
except ImportError:
    COMFY_INTERRUPT_AVAILABLE = False
    print("Warning: comfy.model_management not available, interrupt detection disabled")

# VIDEO 对象类，用于封装视频文件信息
class VideoObject:
    """
    封装视频文件的对象，提供 ComfyUI VIDEO 类型所需的接口
    """
    def __init__(self, filepath, is_placeholder=False):
        self.filepath = filepath
        self.is_placeholder = is_placeholder
        self._width = None
        self._height = None
        self._fps = None
        self._frame_count = None
        if not is_placeholder:
            self._load_metadata()
        else:
            # 占位符使用默认值
            self._width = 1920
            self._height = 1080
            self._fps = 24.0
            self._frame_count = 0
    
    def _load_metadata(self):
        """使用 OpenCV 加载视频元数据"""
        try:
            cap = cv2.VideoCapture(self.filepath)
            if cap.isOpened():
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._fps = cap.get(cv2.CAP_PROP_FPS)
                self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
        except Exception as e:
            print(f"Warning: Failed to load video metadata: {e}")
            self._width = 1920
            self._height = 1080
            self._fps = 24.0
            self._frame_count = 0
    
    def get_dimensions(self):
        """返回视频尺寸 (width, height)"""
        return (self._width, self._height)
    
    def get_fps(self):
        """返回视频帧率"""
        return self._fps
    
    def get_frame_count(self):
        """返回视频总帧数"""
        return self._frame_count
    
    def get_path(self):
        """返回视频文件路径"""
        return self.filepath
    
    def save_to(self, output_path, **kwargs):
        """
        保存视频到指定路径
        如果视频已经在目标位置，则不需要移动
        """
        import shutil
        
        # 如果是占位符视频，不执行保存操作
        if self.is_placeholder or not self.filepath:
            print(f"⚠️ Cannot save placeholder video (video generation failed)")
            return
        
        # 如果目标路径和源路径相同，不需要复制
        if os.path.abspath(self.filepath) == os.path.abspath(output_path):
            print(f"Video already at target location: {output_path}")
            return
        
        # 复制视频文件到目标位置
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(self.filepath, output_path)
        print(f"✓ Video saved to: {output_path}")
    
    def __str__(self):
        return f"VideoObject({self.filepath}, {self._width}x{self._height}, {self._fps}fps, {self._frame_count}frames)"

class DoubaoSeedanceNode:
    """
    ComfyUI节点：使用Doubao Seedance API进行视频生成
    支持文生视频、图生视频（单图或多图）
    """
    
    def create_placeholder_video(self):
        """
        【已废弃】创建占位符视频
        现在失败时直接抛出异常，不再创建占位符
        """
        raise RuntimeError("此方法已废弃，失败时应直接抛出异常")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "多个镜头。一名侦探进入一间光线昏暗的房间。他检查桌上的线索，手里拿起桌上的某个物品。镜头转向他正在思索。背景音乐低沉神秘。",
                    "description": "视频生成的文本提示词，详细描述场景、动作、镜头、氛围等。仅包含提示词内容，参数通过下方独立字段设置（新版API格式）",
                    "label": "提示词"
                }),
                "API密钥": ("STRING", {
                    "default": CONFIG.get(CONFIG_SECTION, "api_key", fallback="sk-your-api-key-here"),
                    "description": "API密钥，用于身份验证",
                    "label": "🔑 API密钥"
                }),
                "API地址": ("STRING", {
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback="https://api.openai.com"),
                    "description": "API服务地址，例如：api.openai.com",
                    "label": "🌐 API地址"
                }),
                "模型": (["doubao-seedance-1-5-pro-251215", "doubao-seedance-1-0-pro-fast-251015", "doubao-seedance-1-0-pro-250528"], {
                    "default": "doubao-seedance-1-5-pro-251215",
                    "label": "模型"
                }),
            },
            "optional": {
                "参考图片1": ("IMAGE", {
                    "description": "第一张输入图片，用于图生视频（单图）或多图生成视频的起始帧",
                    "label": "参考图片1"
                }),
                "参考图片2": ("IMAGE", {
                    "description": "第二张输入图片，用于多图生成视频的结束帧或中间帧",
                    "label": "参考图片2"
                }),
                "分辨率": (["480p", "720p", "1080p"], {
                    "default": "1080p",
                    "label": "分辨率"
                }),
                "宽高比": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {
                    "default": "adaptive",
                    "label": "宽高比"
                }),
                "时长": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 12,
                    "description": "生成视频时长（秒），范围：2-12秒",
                    "label": "时长(秒)"
                }),
                "帧率": ([24], {
                    "default": 24,
                    "label": "帧率"
                }),
                "随机种子": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "description": "种子整数，用于控制生成内容的随机性。-1表示随机（会使用随机数替代），固定值可生成类似结果",
                    "control_after_generate": False,
                    "label": "随机种子"
                }),
                "固定镜头": ("BOOLEAN", {
                    "default": False,
                    "description": "是否固定摄像头（参考图场景不支持）",
                    "label": "固定镜头"
                }),
                "水印": ("BOOLEAN", {
                    "default": False,
                    "description": "生成视频是否包含水印",
                    "label": "水印"
                }),
                "生成音频": ("BOOLEAN", {
                    "default": False,
                    "description": "是否生成包含画面同步音频的视频（仅 Seedance 1.5 pro 支持）",
                    "label": "生成音频"
                }),
                # "return_last_frame": ("BOOLEAN", {
                #     "default": False,
                #     "description": "是否返回视频尾帧图像（PNG格式，无水印），可用于生成连续视频"
                #     "注释原因": "上游中转站暂不支持此参数"
                # }),
                "调试模式": ("BOOLEAN", {
                    "default": False,
                    "description": "调试模式：输出完整的API响应信息",
                    "label": "调试模式"
                }),
                "请求超时": ("INT", {
                    "default": 60,
                    "min": 60,
                    "max": 600,
                    "description": "API初始请求超时时间（秒），用于创建视频生成任务，范围：60-600秒",
                    "label": "请求超时(秒)"
                }),
                "轮询间隔": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 30,
                    "description": "轮询间隔时间（秒），即每隔多少秒查询一次视频生成状态，范围：2-30秒",
                    "label": "轮询间隔(秒)"
                }),
                "最大等待时长": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 3600,
                    "description": "最大轮询时间（秒），即最多等待多长时间来获取视频结果，范围：60-3600秒（1分钟-1小时）",
                    "label": "最大等待时长(秒)"
                })
            }
        }
    
    # {{RIPER-5:
    #   Action: "Modified"
    #   Task_ID: "VIDEO type support"
    #   Timestamp: "2025-12-10"
    #   Authoring_Role: "LD"
    #   Principle_Applied: "Integration - 返回VIDEO类型，兼容SaveVideo节点"
    #   Quality_Check: "下载视频并返回VideoObject，完全兼容VHS等视频扩展"
    # }}
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频输出",)
    FUNCTION = "generate_video"
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
    
    def upload_image_to_url(self, image_url):
        """
        如果图像是base64 data URL，需要上传到可访问的URL
        这里简化处理，直接返回URL（实际项目中可能需要上传服务）
        """
        if image_url.startswith("data:image"):
            # 对于base64图像，在实际项目中需要上传到图床
            # 这里返回None，让API处理
            return None
        return image_url
    
    def call_api(self, host, path, payload, headers, timeout, max_retries=3):
        """
        使用http.client调用API,支持指数退避重试机制
        """
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"[INFO] 第 {attempt} 次重试...")
                else:
                    print(f"[INFO] 正在调用API...")
                
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
                    print(f"[SUCCESS] API调用成功")
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
    
    def download_last_frame(self, frame_url):
        """
        下载最后一帧图像并转换为 ComfyUI IMAGE tensor
        返回: tensor (1, H, W, 3) 或 None
        """
        try:
            print(f"Downloading last frame from: {frame_url[:80]}...")
            
            # 下载图像
            response = requests.get(frame_url, timeout=30, verify=False)
            response.raise_for_status()
            
            # 使用 PIL 打开图像
            pil_image = Image.open(io.BytesIO(response.content))
            pil_image = pil_image.convert('RGB')
            
            # 转换为 numpy 数组并归一化
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            
            # 转换为 torch tensor (1, H, W, 3)
            frame_tensor = torch.from_numpy(numpy_image).unsqueeze(0)
            
            return frame_tensor
            
        except Exception as e:
            print(f"Error downloading last frame: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_video(self, video_url):
        """
        从URL下载视频文件并返回 VideoObject
        优化：直接保存到 output/ 目录，避免二次复制
        """
        response = None
        try:
            print(f"正在下载视频: {video_url}")
            
            # 获取ComfyUI的output目录
            output_dir = folder_paths.get_output_directory()
            
            # 生成唯一的文件名
            timestamp = int(time.time() * 1000)
            filename = f"doubao_seedance_{timestamp}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            # 下载视频
            response = requests.get(video_url, timeout=120, verify=False, stream=True)
            response.raise_for_status()
            
            # 保存视频文件
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 关闭response连接
            response.close()
            
            print(f"✓ 视频下载成功: {filepath}")
            
            # 创建并返回 VideoObject
            video_obj = VideoObject(filepath)
            print(f"✓ 视频信息: {video_obj}")
            
            return video_obj
            
        except Exception as e:
            # 确保关闭连接
            if response:
                response.close()
            
            error_msg = f"视频下载失败: {e}"
            print(f"\n{'='*60}")
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            
            # 直接抛出异常，不返回占位符
            raise RuntimeError(error_msg)
    
    def query_video_status(self, task_id, api_key, base_url, timeout=30, max_retries=3):
        """
        查询视频生成状态,支持重试
        """
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                host = base_url if not base_url.startswith('http') else urlparse(base_url).netloc
                path = f"/v1/video/generations/{task_id}"
                
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                conn = http.client.HTTPSConnection(host, timeout=timeout, context=context)
                conn.request("GET", path, headers=headers)
                
                res = conn.getresponse()
                data = res.read()
                conn.close()
                
                if res.status == 200:
                    return json.loads(data.decode("utf-8"))
                else:
                    error_msg = data.decode('utf-8')
                    print(f"[警告] 查询失败 (status {res.status}): {error_msg[:100]}")
                    last_error = error_msg
                    
                    if attempt < max_retries:
                        wait_time = 2
                        time.sleep(wait_time)
                        continue
                    
            except socket.timeout as e:
                print(f"[超时] 查询状态超时: {e}")
                last_error = str(e)
                
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                    
            except Exception as e:
                print(f"[错误] 查询状态错误: {e}")
                last_error = str(e)
                
                if attempt < max_retries:
                    time.sleep(2)
                    continue
        
        print(f"[失败] 查询状态失败,已重试 {max_retries} 次")
        return None
    
    def generate_video(self, 提示词, API密钥, API地址, 模型, 参考图片1=None, 参考图片2=None,
                      分辨率="1080p", 宽高比="16:9", 时长=5, 帧率=24,
                      随机种子=-1, 固定镜头=False, 水印=False, 生成音频=False,
                      # return_last_frame=False,  # 注释：上游中转站暂不支持
                      调试模式=False, 请求超时=60, 轮询间隔=10, 最大等待时长=300):
        """
        生成视频的主函数
        """
        try:
            # 检查模型和分辨率的兼容性
            if 模型 == "doubao-seedance-1-5-pro-251215" and 分辨率 == "1080p":
                error_msg = (
                    "⚠️ 参数不兼容：模型 'doubao-seedance-1-5-pro-251215' 不支持 1080p 分辨率\n"
                    "📌 该模型仅支持: 480p, 720p\n"
                    "💡 请修改分辨率参数为 480p 或 720p"
                )
                print(f"\n{'='*60}")
                print(error_msg)
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
            
            # 打印输入参数（调试用）
            print("\n" + "="*60)
            print("[Doubao-Seedance] 输入参数:")
            print(f"  - 提示词: {提示词[:50]}...")
            print(f"  - 模型: {模型}")
            print(f"  - 分辨率: {分辨率}")
            print(f"  - 宽高比: {宽高比}")
            print(f"  - 时长: {时长}秒")
            print(f"  - 帧率: {帧率}fps")
            print(f"  - 种子: {随机种子 if 随机种子 >= 0 else '随机'}")
            print(f"  - 生成音频: {生成音频}")
            print(f"  - 水印: {水印}")
            if 参考图片1 is not None or 参考图片2 is not None:
                img_count = (1 if 参考图片1 is not None else 0) + (1 if 参考图片2 is not None else 0)
                print(f"  - 参考图片: {img_count}张")
            print("="*60 + "\n")
            
            # 准备请求数据 - 新版API格式
            # 新版API使用独立参数字段，不再拼接到prompt中
            request_data = {
                "model": 模型,
                "prompt": 提示词,  # 纯提示词内容，不包含参数
                "resolution": 分辨率,
                "ratio": 宽高比,
                "duration": 时长,
                "fps": 帧率,
                "watermark": 水印,
                "generate_audio": 生成音频
            }
            
            # seed参数处理：-1表示随机，>= 0表示固定种子
            if 随机种子 >= 0:
                request_data["seed"] = 随机种子
            
            # camerafixed参数：图生视频场景不支持
            if 固定镜头 and not (参考图片1 or 参考图片2):
                request_data["camerafixed"] = 固定镜头
            
            # 处理图像输入
            images = []
            if 参考图片1 is not None:
                img_url = self.tensor_to_image_url(参考图片1)
                if img_url:
                    # 注意：API可能需要实际的URL，而不是base64
                    # 这里需要根据实际API要求调整
                    # 如果API支持base64，可以直接使用
                    images.append(img_url)
            
            if 参考图片2 is not None:
                img_url = self.tensor_to_image_url(参考图片2)
                if img_url:
                    images.append(img_url)
            
            if images:
                # 使用images数组（支持单图或多图）
                request_data["images"] = images
            
            payload = json.dumps(request_data)
            
            headers = {
                'Authorization': f'Bearer {API密钥}',
                'Content-Type': 'application/json'
            }
            
            # 解析base_url
            if API地址.startswith('http://') or API地址.startswith('https://'):
                parsed_url = urlparse(API地址)
                host = parsed_url.netloc
                path = parsed_url.path if parsed_url.path else "/v1/video/generations"
            else:
                host = API地址
                path = "/v1/video/generations"
            
            print(f"[INFO] 调用 Doubao Seedance API: {host}{path}")
            print(f"[INFO] 模型: {模型}")
            
            # Debug 模式：输出请求数据
            if 调试模式:
                print(f"\n{'='*60}")
                print(f"🐛 DEBUG: Request Data")
                print(f"{'='*60}")
                print(json.dumps(request_data, indent=2, ensure_ascii=False))
                print(f"{'='*60}\n")
            
            # 调用API
            status_code, response_text = self.call_api(host, path, payload, headers, 请求超时)
            
            if status_code == 200:
                try:
                    result = json.loads(response_text)
                    
                    # 提取task_id
                    task_id = result.get('id') or result.get('task_id')
                    
                    if task_id:
                        print(f"[INFO] 视频生成任务已创建: {task_id}")
                        print(f"[INFO] 正在轮询视频生成状态...")
                        print(f"[INFO] 按 Ctrl+C 或点击 ComfyUI 的停止按钮可取消")
                        
                        # 轮询查询视频状态
                        start_time = time.time()
                        video_url = None
                        unknown_count = 0  # 连续未知状态计数
                        max_unknown_retries = 10  # 最大允许连续未知状态次数
                        
                        try:
                            while time.time() - start_time < 最大等待时长:
                                # 检查 ComfyUI 中断信号
                                if COMFY_INTERRUPT_AVAILABLE:
                                    if model_management.processing_interrupted():
                                        error_msg = "用户在 ComfyUI 中中断了视频生成"
                                        print(f"\n{'='*60}")
                                        print(f"❌ {error_msg}")
                                        print(f"{'='*60}\n")
                                        raise RuntimeError(error_msg)
                                
                                status_result = self.query_video_status(task_id, API密钥, API地址)
                                
                                if status_result:
                                    # Debug 模式：输出完整响应
                                    if 调试模式:
                                        print(f"\n{'='*60}")
                                        print(f"🐛 DEBUG: Full API Response")
                                        print(f"{'='*60}")
                                        print(json.dumps(status_result, indent=2, ensure_ascii=False))
                                        print(f"{'='*60}\n")
                                    
                                    # API 响应结构：{"code": "success", "data": {"data": {"status": "succeeded", "content": {...}}}}
                                    # 检查外层响应码
                                    response_code = status_result.get('code', '')
                                    
                                    if response_code == 'success' and 'data' in status_result:
                                        # 获取内层数据
                                        inner_data = status_result.get('data', {}).get('data', {})
                                        status = inner_data.get('status', 'unknown')
                                    else:
                                        # 兼容旧格式：直接从顶层获取 status
                                        status = status_result.get('status', 'unknown')
                                        inner_data = status_result
                                    
                                    elapsed = int(time.time() - start_time)
                                    print(f"[{elapsed}s] Task status: {status}")
                                    
                                    if status == 'succeeded' or status == 'completed':
                                        # 从 content 字段提取视频URL
                                        video_url = None
                                        if 'content' in inner_data:
                                            content = inner_data.get('content', {})
                                            video_url = content.get('video_url')
                                        
                                        # 兼容其他格式
                                        if not video_url:
                                            video_url = inner_data.get('video_url') or inner_data.get('url')
                                        
                                        if video_url:
                                            print(f"✓ Video generated successfully!")
                                            print(f"  URL: {video_url[:80]}...")
                                            
                                            # 下载视频文件并创建 VideoObject
                                            video_obj = self.download_video(video_url)
                                            if video_obj is None:
                                                error_msg = "视频下载失败"
                                                print(f"\n{'='*60}")
                                                print(f"❌ {error_msg}")
                                                print(f"{'='*60}\n")
                                                raise RuntimeError(error_msg)
                                            
                                            # ========== return_last_frame 功能已注释 ==========
                                            # 注释原因：上游中转站暂不支持 return_last_frame 参数
                                            # 如需启用，请取消以下代码的注释并修改返回类型
                                            # 
                                            # # 检查是否有最后一帧图像
                                            # last_frame_tensor = torch.zeros((1, 64, 64, 3))  # 默认占位符
                                            # 
                                            # print(f"📌 return_last_frame setting: {return_last_frame}")
                                            # 
                                            # if return_last_frame:
                                            #     print(f"🔍 Checking for last frame in response...")
                                            #     
                                            #     # 尝试多种可能的字段名和位置
                                            #     last_frame_url = None
                                            #     
                                            #     # 1. 从 content 中查找
                                            #     if content:
                                            #         last_frame_url = (content.get('last_frame_url') or 
                                            #                         content.get('lastFrameUrl') or
                                            #                         content.get('last_frame') or
                                            #                         content.get('tail_frame_url'))
                                            #     
                                            #     # 2. 从 inner_data 中查找
                                            #     if not last_frame_url and inner_data:
                                            #         last_frame_url = (inner_data.get('last_frame_url') or 
                                            #                         inner_data.get('lastFrameUrl') or
                                            #                         inner_data.get('last_frame') or
                                            #                         inner_data.get('tail_frame_url'))
                                            #     
                                            #     # 3. 从外层 data 中查找
                                            #     if not last_frame_url and 'data' in status_result:
                                            #         outer_data = status_result.get('data', {})
                                            #         last_frame_url = (outer_data.get('last_frame_url') or
                                            #                         outer_data.get('lastFrameUrl') or
                                            #                         outer_data.get('last_frame') or
                                            #                         outer_data.get('tail_frame_url'))
                                            #     
                                            #     if debug_mode:
                                            #         print(f"🔍 Available fields in content: {list(content.keys()) if content else 'None'}")
                                            #         print(f"🔍 Available fields in inner_data: {list(inner_data.keys())}")
                                            #         if 'data' in status_result:
                                            #             print(f"🔍 Available fields in outer data: {list(status_result.get('data', {}).keys())}")
                                            #     
                                            #     if last_frame_url:
                                            #         print(f"✓ Last frame URL found!")
                                            #         print(f"  URL: {last_frame_url[:80]}...")
                                            #         downloaded_frame = self.download_last_frame(last_frame_url)
                                            #         if downloaded_frame is not None:
                                            #             last_frame_tensor = downloaded_frame
                                            #             print(f"✓ Last frame loaded: {last_frame_tensor.shape}")
                                            #         else:
                                            #             print(f"⚠️ Failed to download last frame")
                                            #     else:
                                            #         print(f"\n{'!'*60}")
                                            #         print(f"⚠️ Last frame URL NOT found in API response")
                                            #         print(f"{'!'*60}")
                                            #         print(f"📝 Possible reasons:")
                                            #         print(f"   1. API may not support 'return_last_frame' parameter yet")
                                            #         print(f"   2. Parameter name might be different")
                                            #         print(f"   3. Feature may require specific model/plan")
                                            #         print(f"\n💡 Workaround: Extract last frame from video locally")
                                            #         print(f"   Will extract last frame from downloaded video...")
                                            #         
                                            #         # 备用方案：从下载的视频中提取最后一帧
                                            #         try:
                                            #             cap = cv2.VideoCapture(video_obj.get_path())
                                            #             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                            #             if total_frames > 0:
                                            #                 # 跳转到最后一帧
                                            #                 cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                                            #                 ret, frame = cap.read()
                                            #                 if ret:
                                            #                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            #                     frame_normalized = frame_rgb.astype(np.float32) / 255.0
                                            #                     last_frame_tensor = torch.from_numpy(frame_normalized).unsqueeze(0)
                                            #                     print(f"✓ Last frame extracted locally: {last_frame_tensor.shape}")
                                            #             cap.release()
                                            #         except Exception as e:
                                            #             print(f"⚠️ Failed to extract last frame locally: {e}")
                                            #         print(f"{'!'*60}\n")
                                            # else:
                                            #     if debug_mode:
                                            #         print(f"ℹ️ Last frame not requested (return_last_frame=False)")
                                            # 
                                            # return (video_obj, last_frame_tensor)
                                            # ========== 以上代码已注释 ==========
                                            
                                            return (video_obj,)
                                        else:
                                            print(f"⚠️ Video succeeded but no URL found")
                                            print(f"Response: {json.dumps(inner_data, ensure_ascii=False)[:200]}")
                                    
                                    elif status in ['failed', 'error']:
                                        error_msg = inner_data.get('error', {}).get('message', 'Unknown error')
                                        error_detail = f"视频生成失败: {error_msg}"
                                        print(f"\n{'='*60}")
                                        print(f"❌ {error_detail}")
                                        print(f"{'='*60}\n")
                                        raise RuntimeError(error_detail)
                                    
                                    elif status == 'queued':
                                        print(f"  ⏳ Task is queued, waiting...")
                                        unknown_count = 0  # 重置计数器
                                    
                                    elif status == 'running':
                                        print(f"  ⚙️ Task is running...")
                                        unknown_count = 0  # 重置计数器

                                    elif status == 'cancelled':
                                        error_msg = "任务已被取消"
                                        print(f"\n{'='*60}")
                                        print(f"❌ {error_msg}")
                                        print(f"{'='*60}\n")
                                        raise RuntimeError(error_msg)
                                    
                                    elif status == 'expired':
                                        error_msg = "任务已过期（超时）"
                                        print(f"\n{'='*60}")
                                        print(f"❌ {error_msg}")
                                        print(f"{'='*60}\n")
                                        raise RuntimeError(error_msg)
                                    
                                    elif status == 'unknown':
                                        unknown_count += 1
                                        print(f"  ⚠️ Unknown status (retry {unknown_count}/{max_unknown_retries})")
                                        if unknown_count >= max_unknown_retries:
                                            error_msg = f"连续 {max_unknown_retries} 次收到未知状态，任务可能异常"
                                            print(f"\n{'='*60}")
                                            print(f"❌ {error_msg}")
                                            print(f"原始响应: {json.dumps(status_result, ensure_ascii=False)[:300]}")
                                            print(f"{'='*60}\n")
                                            raise RuntimeError(error_msg)
                                    else:
                                        # 其他未知状态
                                        print(f"  ℹ️ Status: {status}")
                                        unknown_count = 0
                                else:
                                    print(f"⚠️ Failed to query status, will retry...")
                                    unknown_count += 1
                                
                                # 使用可中断的睡眠方式
                                # 将睡眠拆分成多个小睡眠，每0.5秒检查一次中断
                                for i in range(轮询间隔 * 2):
                                    if COMFY_INTERRUPT_AVAILABLE and model_management.processing_interrupted():
                                        error_msg = "用户在 ComfyUI 中中断了视频生成"
                                        print(f"\n{'='*60}")
                                        print(f"❌ {error_msg}")
                                        print(f"{'='*60}\n")
                                        raise RuntimeError(error_msg)
                                    time.sleep(0.5)
                        
                        except KeyboardInterrupt:
                            error_msg = f"用户通过 Ctrl+C 中断了视频生成\n任务ID: {task_id} (可稍后查询)"
                            print(f"\n{'='*60}")
                            print(f"❌ {error_msg}")
                            print(f"{'='*60}\n")
                            raise RuntimeError(error_msg)
                        
                        # 超时
                        error_msg = f"轮询超时，已等待 {最大等待时长} 秒"
                        print(f"\n{'='*60}")
                        print(f"❌ {error_msg}")
                        print(f"任务ID: {task_id}")
                        print(f"\n💡 可能的解决方案:")
                        print(f"   1. 增加'最大等待时长'参数值")
                        print(f"   2. 检查视频生成任务是否正常")
                        print(f"   3. 稍后使用任务ID查询")
                        print(f"{'='*60}\n")
                        raise RuntimeError(error_msg)
                    else:
                        error_msg = "API响应中未找到 task_id"
                        print(f"\n{'='*60}")
                        print(f"❌ {error_msg}")
                        print(f"响应内容: {response_text[:300]}...")
                        print(f"{'='*60}\n")
                        raise RuntimeError(error_msg)
                        
                except json.JSONDecodeError as e:
                    error_msg = f"JSON 解析失败: {e}"
                    print(f"\n{'='*60}")
                    print(f"❌ {error_msg}")
                    print(f"原始响应: {response_text[:500]}")
                    print(f"{'='*60}\n")
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"API调用失败 (状态码: {status_code})"
                print(f"\n{'='*60}")
                print(f"❌ {error_msg}")
                print(f"错误响应: {response_text[:500]}")
                print(f"\n💡 可能的解决方案:")
                print(f"   1. 检查 API Key 是否有效")
                print(f"   2. 确认 API 服务地址是否正确")
                print(f"   3. 查看错误信息，调整参数")
                print(f"   4. 检查网络连接是否正常")
                print(f"{'='*60}\n")
                raise RuntimeError(error_msg)
            
        except Exception as e:
            # 关键:异常时直接抛出,不返回占位符视频,避免缓存错误结果
            print(f"[ERROR] 生成失败: {e}")
            print(f"[DEBUG] 异常类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # 直接抛出异常,让ComfyUI知道节点失败了
            raise e

# ComfyUI节点映射
NODE_CLASS_MAPPINGS = {
    "DoubaoSeedanceNode": DoubaoSeedanceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoSeedanceNode": "artsmcp-seedance"
}

