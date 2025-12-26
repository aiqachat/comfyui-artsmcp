import base64
import configparser
import io
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
import torch
import urllib3
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CATEGORY = "artsmcp"
CONFIG_PATH = Path(__file__).parent / "config.ini"
CONFIG = configparser.ConfigParser()
if CONFIG_PATH.exists():
    CONFIG.read(CONFIG_PATH, encoding="utf-8")
else:
    CONFIG["DEFAULT"] = {}
    with CONFIG_PATH.open("w", encoding="utf-8") as fp:
        CONFIG.write(fp)

# Doubao 模型映射
DOUBAO_MODEL_MAP = {
    "Seedance Pro": "doubao-seedance-1-0-pro-250528",
    "Seedance Pro Fast": "doubao-seedance-1-0-pro-fast-251015",
}

# 即梦模型映射
JIMENG_MODEL_MAP = {
    "即梦 v3.0": "jimeng_v30",
    "即梦 v3.0 Pro": "jimeng_v30_pro",
}

# 分辨率映射
RESOLUTION_MAP = {
    "480p": "480p",
    "720p": "720p",
    "1080p": "1080p",
}

# 宽高比映射
RATIO_MAP = {
    "16:9": "16:9",
    "4:3": "4:3",
    "1:1": "1:1",
    "3:4": "3:4",
    "9:16": "9:16",
    "21:9": "21:9",
    "自适应": "adaptive",
}

# 运镜模板映射
CAMERA_TEMPLATE_MAP = {
    "无": "",
    "希区柯克推进": "hitchcock_dolly_in",
    "希区柯克拉远": "hitchcock_dolly_out",
    "机械臂": "robo_arm",
    "动感环绕": "dynamic_orbit",
    "中心环绕": "central_orbit",
    "起重机": "crane_push",
    "超级拉远": "quick_pull_back",
    "逆时针回旋": "counterclockwise_swivel",
    "顺时针回旋": "clockwise_swivel",
    "手持运镜": "handheld",
    "快速推拉": "rapid_push_pull",
}

# 运镜强度映射
CAMERA_STRENGTH_MAP = {
    "弱": "weak",
    "中": "medium",
    "强": "strong",
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


def download_video_to_path(url: str, output_dir: Path, timeout: int = 300):
    """下载视频到指定路径"""
    try:
        print(f"[INFO] 正在下载视频: {url}")
        response = requests.get(url, timeout=timeout, verify=False, stream=True)
        response.raise_for_status()
        
        # 生成文件名
        timestamp = int(time.time())
        filename = f"video_{timestamp}.mp4"
        filepath = output_dir / filename
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载视频
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r[INFO] 下载进度: {progress:.1f}%", end='')
        
        print(f"\n[SUCCESS] 视频已保存: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"[ERROR] 下载视频失败: {e}")
        return None


def make_api_request(url: str, headers: dict, payload: dict, timeout: int = 300, max_retries: int = 3):
    """发送 API 请求,支持指数退避重试"""
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[尝试 {attempt}/{max_retries}] 发送请求到: {url}")
            if attempt == 1:
                print(f"[INFO] 请求参数: {json.dumps(payload, ensure_ascii=False)[:300]}...")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=False
            )
            
            # 成功返回
            if response.status_code == 200:
                print(f"[成功] API调用成功")
                return response.json()
            
            # 服务端错误(5xx)可重试
            elif response.status_code >= 500:
                error_msg = response.text
                print(f"[警告] 服务器错误 {response.status_code}: {error_msg[:100]}")
                last_error = Exception(f"Server error {response.status_code}: {error_msg}")
                
                if attempt < max_retries:
                    wait_time = min(2 ** (attempt - 1), 30)  # 指数退避,最多30秒
                    print(f"[重试] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
            else:
                # 客户端错误(4xx)直接报错
                response.raise_for_status()
                
        except requests.exceptions.Timeout as e:
            print(f"[超时] 请求超时: {e}")
            last_error = e
            
            if attempt < max_retries:
                wait_time = min(2 ** (attempt - 1), 30)
                print(f"[重试] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
                
        except requests.exceptions.ConnectionError as e:
            print(f"[连接错误] 连接失败: {e}")
            last_error = e
            
            if attempt < max_retries:
                wait_time = min(2 ** (attempt - 1), 30)
                print(f"[重试] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
                
        except Exception as e:
            print(f"[错误] API请求失败: {e}")
            last_error = e
            
            if attempt < max_retries:
                wait_time = min(2 ** (attempt - 1), 30)
                print(f"[重试] 等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
    
    # 所有重试都失败
    print(f"[失败] API调用失败,已重试 {max_retries} 次")
    if last_error:
        raise last_error
    raise RuntimeError("All retries failed")


def poll_task_status(task_id: str, api_key: str, base_url: str, max_retries: int = 60, delay: int = 5):
    """轮询任务状态"""
    # 从 base_url 提取主机地址
    if "/v1/video/generations" in base_url:
        status_url = base_url.replace("/v1/video/generations", f"/v1/video/generations/{task_id}")
    else:
        status_url = f"{base_url.rstrip('/')}/v1/video/generations/{task_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"[INFO] 开始轮询任务状态: {task_id}")
    
    for attempt in range(max_retries):
        query_retries = 3  # 每次查询重试次数
        query_success = False
        
        for retry in range(query_retries):
            try:
                response = requests.get(status_url, headers=headers, verify=False, timeout=30)
                response.raise_for_status()
                result = response.json()
                query_success = True
                break
                
            except requests.exceptions.Timeout as e:
                print(f"[超时] 查询状态超时 (retry {retry + 1}/{query_retries}): {e}")
                if retry < query_retries - 1:
                    time.sleep(2)
                    continue
                    
            except Exception as e:
                print(f"[错误] 查询状态失败 (retry {retry + 1}/{query_retries}): {e}")
                if retry < query_retries - 1:
                    time.sleep(2)
                    continue
        
        if not query_success:
            print(f"[WARN] 查询状态失败，将在 {delay} 秒后重试...")
            if attempt < max_retries - 1:
                time.sleep(delay)
            continue
        
        # 提取状态
        status = None
        if "data" in result and isinstance(result["data"], dict):
            status = result["data"].get("status")
        elif "status" in result:
            status = result["status"]
        
        print(f"[INFO] 任务状态 ({attempt + 1}/{max_retries}): {status}")
        
        if status and str(status).upper() == "SUCCESS":
            print("[SUCCESS] 任务完成!")
            return result
        elif status and str(status).upper() == "FAILURE":
            print("[ERROR] 任务失败!")
            return result
        
        # 等待后重试
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    print("[WARN] 达到最大重试次数")
    return None


class VideoGenerationProNode:
    """视频生成节点 - 支持 Doubao 和即梦模型"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "清晨的海边，海浪轻轻拍打着沙滩，远处太阳缓缓升起",
                    "label": "💬 提示词"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": CONFIG["DEFAULT"].get("video_api_key", ""),
                    "label": "🔑 API密钥"
                }),
                "base_url": ("STRING", {
                    "multiline": False,
                    "default": CONFIG["DEFAULT"].get("video_api_url", "https://api.openai.com/v1/video/generations"),
                    "label": "🌐 API地址"
                }),
                "model_type": (["Doubao", "即梦"], {
                    "default": "Doubao",
                    "label": "🎬 模型类型"
                }),
                "doubao_model": (list(DOUBAO_MODEL_MAP.keys()), {
                    "default": list(DOUBAO_MODEL_MAP.keys())[1],
                    "label": "🧠 Doubao模型"
                }),
                "jimeng_model": (list(JIMENG_MODEL_MAP.keys()), {
                    "default": list(JIMENG_MODEL_MAP.keys())[0],
                    "label": "🧠 即梦模型"
                }),
                "resolution": (list(RESOLUTION_MAP.keys()), {
                    "default": "480p",
                    "label": "📺 分辨率"
                }),
                "ratio": (list(RATIO_MAP.keys()), {
                    "default": "16:9",
                    "label": "📐 宽高比"
                }),
                "duration": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 12,
                    "step": 1,
                    "label": "⏱️ 时长(秒)"
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 24,
                    "max": 30,
                    "step": 6,
                    "label": "🎞️ 帧率"
                }),
                "watermark": ("BOOLEAN", {
                    "default": True,
                    "label": "💧 添加水印"
                }),
                "camerafixed": ("BOOLEAN", {
                    "default": False,
                    "label": "📹 相机固定"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "label": "🎲 随机种子"
                }),
                "output_dir": ("STRING", {
                    "multiline": False,
                    "default": CONFIG["DEFAULT"].get("output_dir", "ComfyUI/output"),
                    "label": "📁 输出目录"
                }),
                "poll_interval": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 30,
                    "step": 1,
                    "label": "🔄 轮询间隔(秒)"
                }),
                "max_poll_time": ("INT", {
                    "default": 300,
                    "min": 60,
                    "max": 600,
                    "step": 30,
                    "label": "⏳ 最大等待(秒)"
                }),
            },
            "optional": {
                "first_frame_image": ("IMAGE", {"label": "🖼️ 首帧图片"}),
                "last_frame_image": ("IMAGE", {"label": "🖼️ 尾帧图片"}),
                "camera_template": (list(CAMERA_TEMPLATE_MAP.keys()), {
                    "default": "无",
                    "label": "🎥 运镜模板"
                }),
                "camera_strength": (list(CAMERA_STRENGTH_MAP.keys()), {
                    "default": "中",
                    "label": "💪 运镜强度"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频路径",)
    FUNCTION = "generate_video"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """强制每次都重新执行(外部API请求)"""
        import time
        return time.time()
    
    def generate_video(self, prompt, api_key, base_url, model_type, doubao_model, jimeng_model,
                       resolution, ratio, duration, fps, watermark, camerafixed, seed,
                       output_dir, poll_interval, max_poll_time,
                       first_frame_image=None, last_frame_image=None, 
                       camera_template="无", camera_strength="中"):
        """主生成函数"""
        
        # 保存配置
        if api_key.strip():
            CONFIG["DEFAULT"]["video_api_key"] = api_key.strip()
        if base_url.strip():
            CONFIG["DEFAULT"]["video_api_url"] = base_url.strip()
        if output_dir.strip():
            CONFIG["DEFAULT"]["output_dir"] = output_dir.strip()
        with CONFIG_PATH.open("w", encoding="utf-8") as fp:
            CONFIG.write(fp)
        
        # 打印输入参数（调试用）
        print("\n" + "="*60)
        print("[调试] 输入参数:")
        print(f"  - 提示词: {prompt[:50]}...")
        print(f"  - 模型类型: {model_type}")
        print(f"  - 分辨率: {resolution}")
        print(f"  - 宽高比: {ratio}")
        print(f"  - 时长: {duration}秒")
        print(f"  - 帧率: {fps}")
        print(f"  - 随机种子: {seed}")
        print(f"  - 首帧图片: {'有' if first_frame_image is not None else '无'}")
        print(f"  - 尾帧图片: {'有' if last_frame_image is not None else '无'}")
        print(f"  - 运镜模板: {camera_template}")
        print("="*60 + "\n")
        
        # 选择模型
        if model_type == "Doubao":
            model_value = DOUBAO_MODEL_MAP[doubao_model]
        else:
            model_value = JIMENG_MODEL_MAP[jimeng_model]
        
        # 构建请求参数
        payload = {
            "model": model_value,
            "prompt": prompt,
            "resolution": RESOLUTION_MAP[resolution],
            "ratio": RATIO_MAP[ratio],
            "duration": duration,
            "fps": fps,
            "watermark": watermark,
        }
        
        # 处理随机种子
        if seed >= 0:
            payload["seed"] = seed
        
        # Doubao 模型特有参数
        if model_type == "Doubao":
            payload["camerafixed"] = camerafixed
        
        # 处理输入图片
        if first_frame_image is not None or last_frame_image is not None:
            image_urls = []
            
            if first_frame_image is not None:
                base64_url = tensor_to_base64(first_frame_image)
                image_urls.append(base64_url)
                print("[INFO] 已转换首帧图片为 Base64")
            
            if last_frame_image is not None:
                base64_url = tensor_to_base64(last_frame_image)
                image_urls.append(base64_url)
                print("[INFO] 已转换尾帧图片为 Base64")
            
            payload["images"] = image_urls
            
            if len(image_urls) == 1:
                print("[INFO] 模式: 图生视频-首帧")
            else:
                print("[INFO] 模式: 图生视频-首尾帧")
        else:
            print("[INFO] 模式: 文生视频")
        
        # 处理运镜（仅即梦 v3.0 的 720p 支持）
        if model_type == "即梦" and jimeng_model == "即梦 v3.0" and resolution == "720p" and camera_template != "无":
            template_value = CAMERA_TEMPLATE_MAP[camera_template]
            strength_value = CAMERA_STRENGTH_MAP[camera_strength]
            payload["template_id"] = template_value
            payload["camera_strength"] = strength_value
            print(f"[INFO] 运镜模式: {camera_template} - {camera_strength}")
        
        # 发送请求
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            result = make_api_request(base_url, headers, payload, timeout=60)
            
            # 提取 task_id
            task_id = None
            if "data" in result and isinstance(result["data"], dict):
                task_id = result["data"].get("task_id")
            elif "task_id" in result:
                task_id = result["task_id"]
            
            if not task_id:
                print("[ERROR] 未获取到 task_id")
                return ("",)
            
            print(f"[INFO] 任务ID: {task_id}")
            
            # 轮询任务状态
            max_retries = max_poll_time // poll_interval
            final_result = poll_task_status(
                task_id, 
                api_key, 
                base_url, 
                max_retries=max_retries, 
                delay=poll_interval
            )
            
            if not final_result:
                print("[ERROR] 任务超时或失败")
                return ("",)
            
            # 提取视频 URL
            video_url = None
            if "data" in final_result and isinstance(final_result["data"], dict):
                video_url = final_result["data"].get("video_url")
            elif "video_url" in final_result:
                video_url = final_result["video_url"]
            
            if not video_url:
                print("[ERROR] 未获取到视频URL")
                return ("",)
            
            # 下载视频
            output_path = Path(output_dir)
            video_path = download_video_to_path(video_url, output_path, timeout=300)
            
            if video_path:
                return (video_path,)
            else:
                return ("",)
            
        except Exception as e:
            print(f"[ERROR] 生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 直接抛出异常,不返回空字符串
            raise e


# ComfyUI 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoGenerationProNode": VideoGenerationProNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGenerationProNode": "artsmcp-banana2(待上线)"
}
