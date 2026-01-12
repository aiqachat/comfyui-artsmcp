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
    # "nano-banana": "gemini-2.5-flash-image-preview",
    "nano-banana-2": "nano-banana-2",
    # "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
}

# 响应格式映射
RESPONSE_FORMAT_MAP = {
    "URL": "url",
    "Base64": "b64_json",
}


class ConfigManager:
    """配置管理单例类"""
    _instance = None
    _config = None
    _config_path = Path(__file__).parent / "config.ini"
    _config_section = "Nano-banana"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """加载配置文件"""
        cls._config = configparser.ConfigParser()
        if cls._config_path.exists():
            cls._config.read(cls._config_path, encoding="utf-8")
        else:
            cls._config[cls._config_section] = {}
            with cls._config_path.open("w", encoding="utf-8") as fp:
                cls._config.write(fp)
    
    def get_value(self, key, fallback=None):
        """获取配置值"""
        # 重新读取配置文件以确保获取最新值
        self._load_config()
        try:
            return self._config.get(self._config_section, key, fallback=fallback)
        except Exception as e:
            print(f"[CONFIG] 读取配置失败: {e}")
            return fallback
    
    def set_value(self, key, value):
        """设置配置值"""
        try:
            if not self._config.has_section(self._config_section):
                self._config.add_section(self._config_section)
            self._config.set(self._config_section, key, value)
            with self._config_path.open("w", encoding="utf-8") as fp:
                self._config.write(fp)
            print(f"[CONFIG] 保存 {key} 到配置文件")
        except Exception as e:
            print(f"[ERROR] 配置写入失败: {e}")

# 全局配置管理器实例
config_manager = ConfigManager()


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
            
            # 打印响应头信息（用于调试）
            if verbose:
                print(f"[DEBUG] 响应 Content-Type: {response.headers.get('Content-Type', 'unknown')}")
                print(f"[DEBUG] 响应 Content-Length: {response.headers.get('Content-Length', 'unknown')}")
            
            # 获取原始响应文本（用于调试）
            response_text = response.text
            if verbose:
                print(f"[DEBUG] 响应原始文本（前500字符）: {response_text[:500]}")
            
            # 检查响应是否为空
            if not response_text or response_text.strip() == "":
                print(f"[ERROR] ❌ API 返回空响应！")
                print(f"[ERROR] 这通常意味着 API 端点配置错误或 API 不支持当前请求格式")
                raise ValueError("API 返回空响应，请检查 API 地址和请求格式是否正确")
            
            # 尝试解析 JSON
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"[ERROR] ❌ JSON 解析失败: {e}")
                print(f"[ERROR] 响应不是有效的 JSON 格式")
                if verbose:
                    print(f"[DEBUG] 完整响应文本: {response_text}")
                raise ValueError(f"API 返回的内容不是有效的 JSON 格式，响应内容: {response_text[:200]}...")
            
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
            
        except ValueError as exc:
            # JSON 解析错误或空响应，不应该重试
            last_error = exc
            print(f"[ERROR] 数据格式错误: {exc}")
            print(f"[ERROR] 这不是临时错误，停止重试")
            if response:
                response.close()
            raise
            
        except Exception as exc:
            last_error = exc
            print(f"[ERROR] 未知错误 (尝试 {attempt}/{max_retries}): {exc}")
            if verbose:
                print(f"[DEBUG] 错误类型: {type(exc).__name__}")
                # 打印响应内容用于调试
                if response is not None:
                    try:
                        print(f"[DEBUG] 响应状态码: {response.status_code}")
                        print(f"[DEBUG] 响应头: {dict(response.headers)}")
                        print(f"[DEBUG] 响应文本: {response.text[:500]}")
                    except:
                        pass
        
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


class NanoBananaNode:
    """Nano Banana 图片生成节点 - 支持文生图、图生图、多图融合"""
    
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
                    "default": "一只可爱的猫咪,卡通风格,高清",
                    "label": "💬 提示词"
                }),
                "API密钥": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_key", fallback=CONFIG.get("DEFAULT", "api_key", fallback="")),
                    "label": "🔑 API密钥"
                }),
                "API地址": ("STRING", {
                    "multiline": False,
                    "default": CONFIG.get(CONFIG_SECTION, "api_url", fallback=CONFIG.get("DEFAULT", "api_url", fallback="https://api.openai.com")),
                    "label": "🌐 API地址"
                }),
                "模型": (list(MODEL_MAP.keys()), {
                    "default": list(MODEL_MAP.keys())[0],
                    "label": "🧠 模型"
                }),
                "宽高比": (list(ASPECT_RATIO_MAP.keys()), {
                    "default": "1:1",
                    "label": "📐 尺寸比例(size)"
                }),
                "分辨率": (list(IMAGE_SIZE_MAP.keys()) + ["none"], {
                    "default": "2K",
                    "label": "📏 分辨率"
                }),
                # 响应格式暂时写死为 Base64
                # "响应格式": (list(RESPONSE_FORMAT_MAP.keys()), {
                #     "default": "URL",
                #     "label": "📦 响应格式"
                # }),
                "超时秒数": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 600,
                    "step": 10,
                    "label": "⏱️ 超时(秒)"
                }),
                "最大重试次数": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "🔄 最大重试次数"
                }),
                "并发请求数": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "📊 并发请求数"
                }),
                "启用分行提示词": ("BOOLEAN", {
                    "default": False,
                    "label": "📝 启用分行提示词"
                }),
                "匹配参考尺寸": ("BOOLEAN", {
                    "default": False,
                    "label": "📸 匹配参考尺寸",
                    "label_on": "开启",
                    "label_off": "关闭"
                }),
                "详细日志": ("BOOLEAN", {
                    "default": False,
                    "label": "🔍 详细日志"
                }),
            },
            "optional": {
                "参考图片1": ("IMAGE", {"label": "🖼️ 参考图片1"}),
                "参考图片2": ("IMAGE", {"label": "🖼️ 参考图片2"}),
                "参考图片3": ("IMAGE", {"label": "🖼️ 参考图片3"}),
                "参考图片4": ("IMAGE", {"label": "🖼️ 参考图片4"}),
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
    
    def generate_image(self, 提示词, API密钥, API地址, 模型, 宽高比, 分辨率, 
                       超时秒数, 最大重试次数, 并发请求数, 启用分行提示词, 匹配参考尺寸, 详细日志,
                       参考图片1=None, 参考图片2=None, 参考图片3=None, 参考图片4=None):
        """主生成函数 - 重构为清晰的流程"""
        
        # 设置日志级别
        self.verbose = 详细日志
        
        # 写死响应格式为 Base64
        response_format = "Base64"
        
        # 保存配置到独立配置节（每次重新读取确保数据最新）
        config_writer = configparser.ConfigParser()
        if CONFIG_PATH.exists():
            config_writer.read(CONFIG_PATH, encoding="utf-8")
        
        if not config_writer.has_section(CONFIG_SECTION):
            config_writer.add_section(CONFIG_SECTION)
        
        # 只保存非空的配置项
        if API密钥.strip():
            config_writer.set(CONFIG_SECTION, "api_key", API密钥.strip())
            print(f"[CONFIG] 保存 api_key 到配置文件")
        if API地址.strip():
            config_writer.set(CONFIG_SECTION, "api_url", API地址.strip())
            print(f"[CONFIG] 保存 api_url 到配置文件: {API地址.strip()}")
        
        try:
            with CONFIG_PATH.open("w", encoding="utf-8") as fp:
                config_writer.write(fp)
            print(f"[CONFIG] 配置已成功写入: {CONFIG_PATH}")
        except Exception as e:
            print(f"[ERROR] 配置写入失败: {e}")
        
        # 打印输入参数（调试用）
        print("\n" + "="*60)
        print("[Nano-Banana] 输入参数:")
        print(f"  - 提示词: {提示词[:50]}...")
        print(f"  - 模型: {模型}")
        print(f"  - 宽高比: {宽高比}")
        print(f"  - 分辨率: {分辨率}")
        print(f"  - 响应格式: {response_format}")
        print(f"  - 并发请求数: {并发请求数}")
        print("="*60 + "\n")
        
        # 收集输入图片
        input_images = []
        for idx, img in enumerate([参考图片1, 参考图片2, 参考图片3, 参考图片4], 1):
            if img is not None:
                input_images.append(img)
                self.log(f"[DEBUG] 检测到参考图片{idx}, 形状: {img.shape}", "DEBUG")
        
        self.log(f"[DEBUG] 共收集到 {len(input_images)} 张参考图片", "DEBUG")
        
        # 按 Gemini demo 构建请求参数（contents + parts + inline_data）
        model_value = MODEL_MAP[模型]
        size_value = ASPECT_RATIO_MAP[宽高比]  # 仅用于日志
        response_format_value = RESPONSE_FORMAT_MAP[response_format]
        
        # 组装提示词
        if 启用分行提示词:
            # 每一行作为一个独立的提示词，分别生成图片
            prompt_lines = [line.strip() for line in 提示词.split('\n') if line.strip()]
            print(f"[INFO] 启用分行提示词，共 {len(prompt_lines)} 行")
            print(f"[INFO] 每行将各发送 {并发请求数} 个请求，总计: {len(prompt_lines) * 并发请求数} 个请求")
            self.log(f"[DEBUG] 分行提示词内容: {prompt_lines}", "DEBUG")
        else:
            # 单行提示词
            prompt_lines = [提示词]
        
        # 根据是否启用分行提示词，准备不同的 payload 列表
        payload_list = []
        
        for line_idx, prompt_text in enumerate(prompt_lines, 1):
            # 为每一行提示词构建独立的 payload
            contents_parts = [
                {"text": prompt_text}
            ]
            
            # 处理输入图片（图生图/多图融合模式）
            if input_images:
                print(f"[INFO] 检测到 {len(input_images)} 张参考图片，启用多图融合模式")
                # 将所有参考图片都添加到 parts 数组中
                for img_idx, img_tensor in enumerate(input_images, 1):
                    base64_image = tensor_to_base64(img_tensor)
                    # tensor_to_base64 返回 data URI，需要提取逗号后面的纯 Base64 数据
                    if isinstance(base64_image, str) and base64_image.startswith("data:image"):
                        base64_image = base64_image.split(",", 1)[1]
                    
                    contents_parts.append(
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    )
                    print(f"[INFO] 已添加参考图片 {img_idx}/{len(input_images)} 到 parts 数组")
            
            # 构建 Gemini 原生请求体
            payload = {
                "contents": [
                    {
                        "parts": contents_parts
                    }
                ]
            }

            # 根据模型和节点参数，按需注入尺寸 / 比例配置
            # 仅当确实设置了相关参数时才写入 payload，避免触发无效参数错误
            image_config = {}

            # 分辨率（原图像尺寸）
            if 分辨率 and 分辨率 != "none":
                image_size_value = IMAGE_SIZE_MAP.get(分辨率)
                if image_size_value:
                    image_config["imageSize"] = image_size_value

            # 宽高比: 始终注入（文生图和图生图模式都支持）
            if 宽高比 in ASPECT_RATIO_MAP:
                aspect_ratio_value = ASPECT_RATIO_MAP[宽高比]
                if aspect_ratio_value:
                    image_config["aspectRatio"] = aspect_ratio_value

            if image_config:
                payload["generationConfig"] = {
                    "imageConfig": image_config
                }
            
            # 每个 prompt 都发送 N 次请求（N = 并发请求数）
            for _ in range(并发请求数):
                payload_list.append((line_idx, prompt_text, payload.copy()))
        
        # 打印模式信息
        if input_images:
            print(f"[INFO] 模式: 图生图（参考图数量: {len(input_images)}）")
        else:
            print("[INFO] 模式: 文生图（仅文本提示词）")
        
        self.log(f"[DEBUG] 最终 payload 顶层字段: {list(payload_list[0][2].keys())}", "DEBUG")
        self.log(f"[DEBUG] 模型: {model_value}, 宽高比(仅日志): {size_value}", "DEBUG")
        
        # 按 Gemini demo 构建完整 URL:
        # {base_url}/v1beta/models/{model}:generateContent?key={API密钥}
        base_url = API地址.strip()
        if not base_url:
            base_url = "https://api.openai.com"
        
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            base_url = "https://" + base_url
        
        base_url = base_url.rstrip("/")
        final_url = f"{base_url}/v1beta/models/{model_value}:generateContent?key={API密钥.strip()}"
        
        print(f"[INFO] 解析后的完整 API 地址: {final_url}")
        
        # 发送请求（鉴权通过 URL 中的 key，Header 只需要 Content-Type）
        headers = {
            "Content-Type": "application/json"
        }
        
        # 打印完整的payload用于调试（只打印第一个）
        if payload_list and self.verbose:
            self.log(f"[DEBUG] 完整 payload 结构（示例）:", "DEBUG")
            try:
                self.log(json.dumps(payload_list[0][2], ensure_ascii=False)[:500] + "...", "DEBUG")
            except Exception as e:
                print(f"[WARN] payload 序列化失败: {e}")
        
        try:
            # 并发发送请求（支持分行提示词 + 生图数量）
            total_requests = len(payload_list)
            print(f"\n{'='*60}")
            print(f"[INFO] 开始并发生成 {total_requests} 张图片...")
            print(f"[INFO] 并发线程数: {min(total_requests, 5)}")
            print(f"{'='*60}\n")
            
            results = []
            with ThreadPoolExecutor(max_workers=min(total_requests, 5)) as executor:
                # 提交所有请求任务
                futures = [
                    executor.submit(
                        make_api_request, 
                        final_url, 
                        headers, 
                        payload_data,  # 已经是副本
                        超时秒数, 
                        最大重试次数,
                        2,  # backoff
                        详细日志  # 传递 verbose 参数
                    ) 
                    for line_idx, prompt_text, payload_data in payload_list
                ]
                
                # 等待所有请求完成并收集结果
                for idx, future in enumerate(as_completed(futures), 1):
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"[INFO] ✅ 第 {idx}/{total_requests} 个请求已完成")
                    except Exception as e:
                        print(f"[ERROR] ❌ 第 {idx}/{total_requests} 个请求失败: {e}")
                        # 继续处理其他请求，不中断
            
            # 检查是否至少有一个成功的结果
            if not results:
                raise RuntimeError(f"所有 {total_requests} 个请求均失败，未获取到任何图片数据")
            
            print(f"\n{'='*60}")
            print(f"[SUCCESS] ✅ 并发请求完成！")
            print(f"[INFO] 成功: {len(results)}/{total_requests} 个请求")
            if len(results) < total_requests:
                print(f"[WARN] ⚠️ 部分请求失败，仅返回成功的图片")
            print(f"{'='*60}\n")
            
            # 解析所有响应并合并输出
            output_tensors = []
            
            # 遍历所有请求的响应结果
            for result_idx, result in enumerate(results, 1):
                self.log(f"\n[DEBUG] ===== 处理第 {result_idx}/{len(results)} 个响应 =====", "DEBUG")
                self.log(f"[DEBUG] 响应包含的键: {list(result.keys())}", "DEBUG")
                
                # 优先处理 Gemini 原生格式: candidates -> content.parts
                if "candidates" in result:
                    candidates = result.get("candidates", [])
                    self.log(f"[DEBUG] 检测到 Gemini 响应格式，candidates 数量: {len(candidates)}", "DEBUG")
                    
                    for c_idx, candidate in enumerate(candidates):
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        self.log(f"[DEBUG] 处理第 {c_idx+1} 个 candidate，parts 数量: {len(parts)}", "DEBUG")
                        
                        for p_idx, part in enumerate(parts):
                            self.log(f"[DEBUG] 处理第 {c_idx+1} 个 candidate 的第 {p_idx+1} 个 part，keys: {list(part.keys())}", "DEBUG")
                            # 1. inlineData / inline_data（优先图片）
                            inline_data = part.get("inlineData") or part.get("inline_data")
                            if inline_data:
                                img_b64 = inline_data.get("data")
                                if img_b64:
                                    self.log(f"[DEBUG] 从 inline_data 中提取到图片 Base64，长度: {len(img_b64)}", "DEBUG")
                                    tensor = base64_to_tensor(img_b64)
                                    if tensor is not None:
                                        output_tensors.append(tensor)
                                        self.log(f"[DEBUG] ✅ 第 {len(output_tensors)} 张图片解码成功（来自响应 {result_idx}）", "DEBUG")
                                    else:
                                        self.log("[DEBUG] ❌ 图片 Base64 解码失败", "DEBUG")
                            # 2. 文本里可能塞了 data:image/base64,...
                            elif "text" in part:
                                text_content = part["text"]
                                self.log(f"[DEBUG] 文本 part 内容: {text_content[:100]}...", "DEBUG")
                                if "data:image" in text_content and "base64," in text_content:
                                    try:
                                        b64_part = text_content.split("base64,")[-1].strip()
                                        b64_part = b64_part.replace(")", "").replace("]", "")
                                        tensor = base64_to_tensor(b64_part)
                                        if tensor is not None:
                                            output_tensors.append(tensor)
                                            self.log(f"[DEBUG] ✅ 从文本中提取图片 Base64 并解码成功，当前总数: {len(output_tensors)}", "DEBUG")
                                    except Exception as e:
                                        print(f"[WARN] 从文本提取图片 Base64 失败: {e}")
                # 兼容旧的 OpenAI images/generations 风格: data + b64_json/url
                elif "data" in result:
                    data = result["data"]
                    self.log(f"[DEBUG] data 类型: {type(data)}", "DEBUG")
                    
                    if isinstance(data, list):
                        self.log(f"[DEBUG] data 是列表，长度: {len(data)}", "DEBUG")
                        for idx, item in enumerate(data):
                            self.log(f"[DEBUG] 处理第 {idx+1} 个图片项（来自响应 {result_idx}）...", "DEBUG")
                            self.log(f"[DEBUG] 图片项包含的键: {list(item.keys()) if isinstance(item, dict) else 'N/A'}", "DEBUG")
                            
                            tensor = self._process_image_item(item, response_format_value, 超时秒数)
                            if tensor is not None:
                                output_tensors.append(tensor)
                                self.log(f"[DEBUG] ✅ 第 {len(output_tensors)} 张图片转换成功", "DEBUG")
                            else:
                                self.log(f"[DEBUG] ❌ 图片转换失败", "DEBUG")
                                
                    elif isinstance(data, dict):
                        self.log(f"[DEBUG] data 是字典", "DEBUG")
                        self.log(f"[DEBUG] 字典包含的键: {list(data.keys())}", "DEBUG")
                        
                        tensor = self._process_image_item(data, response_format_value, 超时秒数)
                        if tensor is not None:
                            output_tensors.append(tensor)
                            self.log(f"[DEBUG] ✅ 图片转换成功（来自响应 {result_idx}）", "DEBUG")
                        else:
                            self.log(f"[DEBUG] ❌ 图片转换失败", "DEBUG")
                else:
                    print(f"[ERROR] 响应 {result_idx} 中既没有 'candidates' 也没有 'data' 字段！")
                    self.log(f"[DEBUG] 完整响应内容: {result}", "DEBUG")
            
            if not output_tensors:
                print("[ERROR] ❌ 未获取到任何图片数据！")
                self.log(f"[DEBUG] 输出 tensors 数量: {len(output_tensors)}", "DEBUG")
                # 直接抛出异常，不返回默认图片
                raise RuntimeError("未获取到任何图片数据")
            
            # 如果启用"匹配参考尺寸"且有参考图片，则调整输出尺寸
            if 匹配参考尺寸 and input_images:
                output_tensors = self._match_reference_size(output_tensors, input_images)
            
            # 归一化tensor尺寸(防止尺寸不一致导致stack崩溃)
            output_tensors = self._normalize_tensor_size(output_tensors)
            
            # 合并所有 tensor
            batch_tensor = torch.stack(output_tensors, dim=0).contiguous()
            print(f"\n{'='*60}")
            print(f"[SUCCESS] ✅ 成功生成 {len(output_tensors)} 张图片!")
            print(f"[INFO] 批次尺寸: {batch_tensor.shape}")
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
                print(f"\n[OUTPUT] 返回值结构: tuple 包含 1 个元素")
                print(f"[OUTPUT] 返回值内容: (torch.Tensor,)")
                print(f"[OUTPUT] ComfyUI 将接收到类型为 'IMAGE' 的输出")
                print(f"{'='*60}\n")
            
            print(f"[INFO] ✅ 节点执行完毕，返回结果")
            return (batch_tensor,)
            
        except InterruptedError as e:
            # 用户主动中断
            print(f"[INFO] ℹ️ 用户已中断生成任务")
            raise e
            
        except Exception as e:
            # 所有异常统一处理
            print(f"[ERROR] 生成失败: {e}")
            self.log(f"[DEBUG] 异常类型: {type(e).__name__}", "DEBUG")
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
    "NanoBananaNode": NanoBananaNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaNode": "artsmcp-nano-banana"
}
