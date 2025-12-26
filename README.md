ComfyUI 插件，集成多种 AI API 进行专业的图像和视频生成：
- **Nano Banana**: Google Nano Banana 图像生成（支持文生图、图生图）
- **Gemini Banana**: Google Gemini 3 Pro 图像生成（支持文生图、图生图、多图融合）
- **Doubao Seedance**: AI 视频生成（文生视频、图生视频）
- **Doubao Seedream**: AI 图像生成（文生图、图生图、图生组图、多图融合）

## 功能特性

### 🍌 Nano Banana（推荐）
- ✅ **文生图**：根据文本提示生成高质量图片
- ✅ **多图参考**：支持 1-4 张图片参考生成
- ✅ 支持 Nano Banana / Nano Banana 2 / Nano Banana Pro 模型
- ✅ 支持 10 种宽高比（1:1, 4:3, 3:4, 16:9, 9:16, 2:3, 3:2, 4:5, 5:4, 21:9）
- ✅ nano-banana-2 支持 1K/2K/4K 分辨率
- ✅ 支持 URL 和 Base64 两种响应格式
- ✅ 自动重试机制（智能错误处理）
- ✅ 独立配置管理
- ✅ 参数自动保存，下次使用更便捷

### 🎨 Gemini Banana（推荐）
- ✅ **文生图**：根据文本提示生成高质量图片
- ✅ **图生图**：单图输入生成新图片
- ✅ **多图融合**：融合多张图片生成新图片（最多4张）
- ✅ 支持 Gemini 3 Pro Image 模型（1K/2K/4K）
- ✅ 支持 1K/2K/4K 分辨率
- ✅ 支持 URL 和 Base64 两种响应格式
- ✅ 自动重试机制（智能错误处理）
- ✅ 独立配置管理
- ✅ 参数自动保存，下次使用更便捷

### 🎬 Doubao Seedance (AI 视频生成)

- ✅ **文生视频**：根据文本提示生成视频
- ✅ **图生视频**：单图或多图生成视频
- ✅ 支持视频状态查询和轮询
- ✅ 支持多种视频参数（分辨率、帧率、时长等）
- ✅ 自动轮询任务状态
- ✅ 自动下载视频到本地
- ✅ 返回标准 VIDEO 类型

### 🎨 Doubao Seedream (图片生成)
- ✅ **文生图**：根据文本提示生成图片
- ✅ **图生图**：单图输入生成新图片
- ✅ **图生组图**：生成多张相关图片
- ✅ **多图融合**：融合多张图片生成新图片
- ✅ 支持多种图片尺寸和质量选项

## 安装

1. 将插件文件夹复制到ComfyUI的`custom_nodes`目录
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 🍌 Nano Banana（推荐）

**请求示例**：
```json
{
  "model": "nano-banana-pro",
  "prompt": "一只可爱的猫咪，卡通风格，高清",
  "aspect_ratio": "1:1",
  "n": 1,
  "response_format": "url"
}
```

**参数说明**：

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| `提示词` | STRING | 图片生成的文本描述 | 支持中英文 |
| `API密钥` | STRING | API 身份验证密钥 | sk-xxx |
| `API地址` | STRING | API 服务端点 | 默认：https://api.openai.com/v1/images/generations |
| `模型` | ENUM | 选择模型 | nano-banana / nano-banana-2 / nano-banana-pro |
| `宽高比` | ENUM | 图片宽高比 | 1:1 / 4:3 / 3:4 / 16:9 / 9:16 / 2:3 / 3:2 / 4:5 / 5:4 / 21:9 |
| `图像尺寸` | ENUM | 图像分辨率(仅nano-banana-2) | none / 1K / 2K / 4K |
| `响应格式` | ENUM | 返回格式 | URL / Base64 |
| `超时(秒)` | INT | API 请求超时时间 | 30-600，默认120 |
| `最大重试次数` | INT | 失败后重试次数 | 1-10，默认3 |
| `生图数量` | INT | 一次生成图片数量 | 1-10，默认1 |
| `参考图片1-4` | IMAGE | 可选参考图片 | 用于多图参考 |

**功能模式**：
1. **文生图**：只填提示词，不提供参考图片
2. **多图参考**：提供 1-4 张参考图片

**使用示例**：
```
Nano Banana
  ├─ 提示词: "一只可爱的猫咪，卡通风格"
  ├─ 模型: nano-banana-pro
  ├─ 宽高比: 16:9
  ├─ 生图数量: 2
  ├─ 响应格式: URL
  └─ 图片输出 ──→ SaveImage
```

**多图参考示例**：
```
Load Image ──→ 参考图片1
Load Image ──→ 参考图片2
                 ↓
Nano Banana
  ├─ 提示词: "融合这两张图片的风格"
  ├─ image1: <connected>
  ├─ image2: <connected>
  └─ 图片输出 ──→ SaveImage
```

**重试机制说明**：
- ✅ **自动重试**：遇到 503/429 等错误自动重试
- ✅ **指数退避**：2秒、4秒、8秒递增等待
- ✅ **智能判断**：4xx 客户端错误（除429）直接失败不浪费重试
- ✅ **详细日志**：显示每次重试的详细信息

---

### 🎨 Gemini Banana（推荐）

**请求示例**：
```json
{
  "model": "gemini-3-pro-image-preview-2k",
  "prompt": "星际穿越，黑洞，电影大片，超现实主义",
  "size": "2K",
  "n": 1,
  "response_format": "url"
}
```

**参数说明**：

| 参数 | 类型 | 说明 | 可选值 |
|------|------|------|--------|
| `提示词` | STRING | 图片生成的文本描述 | 支持中英文 |
| `API密钥` | STRING | API 身份验证密钥 | sk-xxx |
| `API地址` | STRING | API 服务端点 | 默认：https://apitt.cozex.cn/v1/images/generations |
| `模型` | ENUM | 选择模型 | gemini-3-pro-image-preview / 2k / 4k |
| `尺寸` | ENUM | 图片分辨率 | 1K / 2K / 4K |
| `响应格式` | ENUM | 返回格式 | URL / Base64 |
| `超时(秒)` | INT | API 请求超时时间 | 30-600，默认120 |
| `最大重试次数` | INT | 失败后重试次数 | 1-10，默认3 |
| `生图数量` | INT | 一次生成图片数量 | 1-10，默认1 |
| `图片1-4` | IMAGE | 可选输入图片 | 用于图生图/多图融合 |

**功能模式**：
1. **文生图**：只填提示词，不提供输入图片
2. **图生图**：提供 1 张输入图片
3. **多图融合**：提供 2-4 张输入图片

**使用示例**：
```
Gemini Banana
  ├─ 提示词: "星际穿越，黑洞，电影大片"
  ├─ 模型: gemini-3-pro-image-preview-2k
  ├─ 尺寸: 2K
  ├─ 生图数量: 3
  ├─ 响应格式: URL
  └─ 图片输出 ──→ SaveImage
```

**图生图示例**：
```
Load Image ──→ 图片1
                 ↓
Gemini Banana
  ├─ 提示词: "将这张图片转换为油画风格"
  ├─ image1: <connected>
  └─ 图片输出 ──→ SaveImage
```

**多图融合示例**：
```
Load Image ──→ 图片1
Load Image ──→ 图片2
                 ↓
Gemini Banana
  ├─ 提示词: "融合这两张图片的风格"
  ├─ image1: <connected>
  ├─ image2: <connected>
  └─ 图片输出 ──→ SaveImage
```

**重试机制说明**：
- ✅ **自动重试**：遇到 503/429 等错误自动重试
- ✅ **指数退避**：2秒、4秒、8秒递增等待
- ✅ **智能判断**：4xx 客户端错误（除429）直接失败不浪费重试
- ✅ **详细日志**：显示每次重试的详细信息

---

### 🎬 Doubao Seedance (视频生成)

**功能亮点**：
- ✅ 文生视频、图生视频（单图/多图）
- ✅ 自动轮询任务状态
- ✅ 自动下载视频到 output 目录
- ✅ 返回标准 VIDEO 类型，可直接连接 SaveVideo
- ✅ 支持中断控制（Ctrl+C 或 Stop 按钮）
- ✅ Debug 模式查看完整 API 响应

#### 使用步骤

**1. 添加节点**
在 ComfyUI 中添加：`Doubao Seedance Video`

**重要说明**：本节点会自动将用户选择的参数（分辨率、比例、时长等）拼接到 prompt 中发送给 API。

例如，用户输入：
- `prompt`: "子弹时间效果的运镜画面..."
- `resolution`: 480p
- `ratio`: adaptive
- `duration`: 5
- `seed`: 3298453148

实际发送的 prompt 为：
```
子弹时间效果的运镜画面... --rt adaptive --dur 5 --rs 480p --fps 24 --wm false --seed 3298453148 --cf false
```

**2. 必需参数**
- `prompt` (STRING): 视频生成的提示词描述
  - 详细描述场景、动作、镜头、氛围等
  - 示例：`"多个镜头。一名侦探进入一间光线昏暗的房间。他检查桌上的线索，手里拿起桌上的某个物品。镜头转向他正在思索。背景音乐低沉神秘。"`
- `api_key` (STRING): API 密钥
- `base_url` (STRING): API 服务地址（默认：`api.cozex.cn`）
- `model` (下拉框): 模型选择
  - `doubao-seedance-1-5-pro-251215` ⭐ **(推荐，最新版，支持音频生成)**
  - `doubao-seedance-1-0-pro-fast-251015` (快速版)
  - `doubao-seedance-1-0-pro-250528` (标准版)

**3. 可选参数**

*输入图片：*
- `image1` (IMAGE): 第一张输入图片（图生视频的起始帧）
- `image2` (IMAGE): 第二张输入图片（结束帧或中间帧）

*视频配置：*
- `resolution` (下拉框): 视频分辨率
  - `480p`, `720p`, `1080p`（默认：1080p）
- `ratio` (下拉框): 视频宽高比
  - `16:9`, `4:3`, `1:1`, `3:4`, `9:16`, `21:9`, `adaptive`（默认：16:9）
- `duration` (INT): 视频时长，2-12秒（默认：5秒）
- `framespersecond` (下拉框): 帧率，仅支持 24

*高级参数：*
- `seed` (INT): 种子整数，用于控制生成内容的随机性
  - -1 = 随机（会使用随机数替代）
  - [-1, 2^32-1] 之间的整数 = 固定值可生成类似结果（但不保证完全一致）
  - ℹ️ **重要**：seed 值不会自动变化，保持用户设置的值不变
- `camerafixed` (BOOLEAN): 是否固定摄像头（仅文生视频支持）
- `watermark` (BOOLEAN): 是否添加水印（默认：False）
- `generate_audio` (BOOLEAN): 是否生成包含画面同步音频的视频（默认：True）
  - ⭐ **新功能**：仅 Seedance 1.5 pro 支持
  - `True`: 生成带有画面同步音频的视频
  - `False`: 生成无声视频

*系统参数：*
- `debug_mode` (BOOLEAN): 调试模式，输出完整 API 响应（默认：False）
- `timeout` (INT): API 请求超时时间，60-600秒（默认：60秒）
- `poll_interval` (INT): 轮询间隔，2-30秒（默认：10秒）
- `max_poll_time` (INT): 最大轮询时间，60-3600秒（默认：300秒）

**4. 输出**
- `video` (VIDEO): 生成的视频 (VideoObject 类型)
  - 自动下载到 `ComfyUI/output/` 目录
  - 文件名格式：`doubao_seedance_<timestamp>.mp4`
  - 可直接连接到 SaveVideo、Preview Video 等节点

**5. 使用示例**

*文生视频：*
```
Doubao Seedance Video
  ├─ prompt: "一个机器人在未来城市中行走..."
  ├─ resolution: 1080p
  ├─ ratio: 16:9
  ├─ duration: 10
  └─ video ──→ SaveVideo
```

*图生视频：*
```
Load Image ──→ image1
                 ↓
Doubao Seedance Video
  ├─ prompt: "图片中的场景开始动起来..."
  ├─ image1: <connected>
  └─ video ──→ SaveVideo
```

*Debug 调试：*
```
Doubao Seedance Video
  ├─ debug_mode: True ☑
  └─ 查看控制台输出完整的请求和响应
```

**6. 中断控制**
- 按 `Ctrl+C` 中断视频生成
- 点击 ComfyUI 的 `Stop` 按钮
- 每 0.5 秒检查一次中断信号

**7. VideoObject 类**
返回的 VIDEO 对象提供以下方法：
- `get_dimensions()`: 返回 (width, height)
- `get_fps()`: 返回帧率
- `get_frame_count()`: 返回总帧数
- `get_path()`: 返回文件路径
- `save_to(path)`: 保存视频到指定路径

### 🎨 Doubao Seedream (图片生成)
1. 在ComfyUI中添加节点：`Doubao Seedream Image`
2. 配置参数：
   - `api_key`: 你的API密钥（默认：`sk-your-api-key-here`）
   - `base_url`: API服务地址（默认：`api.cozex.cn`）
   - `model`: 模型名称（可选：`doubao-seedream-4-0-250828` 或 `doubao-seedream-4-5-251128`）
   - `prompt`: 图片生成提示词
   - `size`: 图片尺寸（可选：`2K`, `1K`, `4K`）
   - `image1`, `image2`: 可选的输入图片（用于图生图、多图融合）
   - `max_images`: 最大生成图片数（默认：0，0=禁用组图生成，1-10=自动启用组图模式生成对应数量的图片）
   - `watermark`: 是否添加水印（默认：False）
   - `response_format`: 响应格式（`url` 或 `b64_json`）
3. 输出：
   - `image`: 生成的图片（tensor格式）

## 版本说明

### V1版本特点
- 使用`requests`库发送HTTP请求
- 从API响应中提取base64图像数据
- 适合返回base64图像的API服务

### V2版本特点
- 使用`http.client`库（更底层）
- 支持从响应中提取图像URL并自动下载
- 更灵活的URL解析
- 更详细的错误日志
- 适合返回图像URL的API服务

## 依赖

### 核心依赖
- `requests>=2.31.0` - HTTP 请求库（API 调用、图片/视频下载）
- `Pillow>=10.0.0` - 图像处理库（格式转换、编码解码）
- `opencv-python>=4.8.0` - 视频处理库（视频帧提取、元数据读取）
- `torch>=2.0.0` - PyTorch（tensor 操作）*
- `numpy>=1.24.0` - 数值计算库*

\* 注：如果已安装 ComfyUI，通常已包含 torch 和 numpy，无需重复安装

### 快速安装
```bash
# 完整安装
pip install -r requirements.txt

# 或仅安装必需的（ComfyUI 环境）
pip install requests Pillow opencv-python
```

详见 `requirements.txt` 了解更多说明

## 注意事项

### Doubao Seedance 视频生成
1. **轮询机制**：视频生成是异步的，节点会自动轮询任务状态直到完成
2. **下载时间**：视频下载可能需要额外时间，取决于视频大小和网络速度
3. **存储空间**：视频文件保存在 `ComfyUI/output/` 目录，请确保有足够空间
4. **中断恢复**：中断后可通过 Task ID 查询任务状态（日志中会显示）
5. **参数限制**：
   - `resolution=1080p` 不支持参考图场景（图生视频）
   - `camerafixed=true` 仅文生视频支持
   - `ratio=adaptive` 仅图生视频支持

### API 配置
- 所有节点支持自定义 `base_url`，可使用中转站或自建服务
- SSL 验证已禁用（`verify=False`），适用于内网环境
- 建议使用环境变量管理 API 密钥

### 性能优化
- **Seedance 视频**：建议设置 `poll_interval=10` 避免频繁请求
- **Seedream 图片**：支持批量生成（`max_images`），比单独生成更高效
- **大视频生成**：增加 `max_poll_time` 避免超时


## 许可证

MIT License

