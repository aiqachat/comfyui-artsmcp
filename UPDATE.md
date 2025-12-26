# 更新日志

## v2.2.1 (2025-12-26)

### ⭐ Doubao Seedance - 新增模型和音频生成功能

#### 新增模型

新增支持 **Seedance 1.5 Pro** 模型，并设为默认模型。

**模型选项**：
- `doubao-seedance-1-5-pro-251215` ⭐ **(推荐，最新版，支持音频生成)**
- `doubao-seedance-1-0-pro-fast-251015` (快速版)
- `doubao-seedance-1-0-pro-250528` (标准版)

---

#### 新增参数：`generate_audio`

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 | 支持模型 |
|------|------|------|--------|-----------|
| generate_audio | BOOLEAN | 是否生成包含画面同步音频的视频 | True | Seedance 1.5 pro |

**功能特性**：
- ✅ **带音频视频**：设置为 `True` 时，生成带有与画面同步的背景音频
- ✅ **无声视频**：设置为 `False` 时，生成无声视频
- ✅ **仅 1.5 Pro 支持**：该功能仅在 Seedance 1.5 Pro 模型下可用

**请求示例**：
```json
{
  "model": "doubao-seedance-1-5-pro-251215",
  "prompt": "多个镜头。一名侦探进入一间光线昧暗的房间。他检查桌上的线索，手里拿起桌上的某个物品。镜头转向他正在思索。背景音乐低沉神秘。",
  "resolution": "1080p",
  "ratio": "16:9",
  "duration": 5,
  "fps": 24,
  "generate_audio": true,
  "watermark": false
}
```

**使用示例**：
```
Doubao Seedance Video
  ├─ 模型: doubao-seedance-1-5-pro-251215
  ├─ 提示词: "一个机器人在未来城市中行走..."
  ├─ 生成音频: True ☑  ← 新增
  ├─ 分辨率: 1080p
  ├─ 宽高比: 16:9
  ├─ 时长: 10秒
  └─ 视频输出 ──→ SaveVideo
```

---

### 🔧 节点优化

**对比 nano_banana_node.py 后的改进**：

1. **配置管理优化**
   - ✅ 添加 `CATEGORY` 和 `CONFIG_SECTION` 常量
   - ✅ 使用独立配置节 `[Seedance]`
   - ✅ 配置读写更加规范

2. **用户界面改进**
   - ✅ 为 `api_key` 和 `base_url` 添加 `label` 标签（🔑 和 🌐 图标）
   - ✅ 输出名称中文化：`"video"` → `"视频输出"`
   - ✅ 更友好的界面显示

3. **调试信息增强**
   - ✅ **新增参数打印**：提示词、模型、分辨率、宽高比、时长、帧率
   - ✅ 显示种子值（“随机”或具体数值）
   - ✅ 显示生成音频、水印状态
   - ✅ 显示参考图片数量

4. **日志输出标准化**
   - ✅ 统一日志格式：`[INFO]`、`[SUCCESS]`、`[ERROR]`、`[DEBUG]`
   - ✅ 中文化重要提示信息
   - ✅ 改进重试信息显示：“第 X 次重试...”

5. **错误提示优化**
   - ✅ **超时错误**：添加解决方案提示
   - ✅ **API调用失败**：添加4条解决建议
   - ✅ 明确说明异常处理原因（避免缓存错误结果）

6. **代码一致性**
   - ✅ 与 `nano_banana_node.py` 保持相似的代码风格
   - ✅ 统一使用 `CONFIG_SECTION` 变量
   - ✅ 统一的参数打印格式

---

### 📝 注意事项

**音频生成功能**：
- ❗ 仅 Seedance 1.5 Pro 模型支持 `generate_audio` 参数
- ❗ 带音频视频的文件体积可能较大，请确保存储空间充足
- ✅ 音频与画面自动同步，无需手动调整

---

## v2.2.0 (2025-12-25)

### 🎉 新增节点

#### Nano Banana（推荐）
Google Nano Banana 高质量图像生成节点。

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

| 参数 | 类型 | 说明 |
|------|------|------|
| prompt | STRING | 图片生成的文本描述 |
| api_key | STRING | API身份验证密钥 |
| base_url | STRING | API服务端点 |
| model | ENUM | nano-banana / nano-banana-2 / nano-banana-pro |
| aspect_ratio | ENUM | 1:1 / 4:3 / 3:4 / 16:9 / 9:16 / 2:3 / 3:2 / 4:5 / 5:4 / 21:9 |
| image_size | ENUM | 1K / 2K / 4K (仅nano-banana-2支持) |
| response_format | ENUM | URL / Base64 |
| timeout | INT | 超时时间 (30-600秒) |
| max_retries | INT | 最大重试次数 (1-10) |
| n | INT | 一次生成图片数量 (1-10) |
| image1-4 | IMAGE | 可选参考图片 (支持多图) |

**功能特性**：
- ✅ 文生图：纯文本生成图片
- ✅ 多图参考：支持 1-4 张图片参考生成
- ✅ 支持 10 种宽高比选择
- ✅ nano-banana-2 支持 1K/2K/4K 分辨率
- ✅ 支持 URL 和 Base64 响应格式
- ✅ 自动重试机制（指数退避）
- ✅ 智能错误处理（4xx 直接失败，5xx/429 重试）
- ✅ 独立配置节 [Nano-banana]
- ✅ 配置自动保存

**重试机制**：
- 指数退避：2秒、4秒、8秒，最大3次
- 4xx 客户端错误（除429）直接失败
- 5xx 服务器错误自动重试
- 429 限流错误自动重试
- 详细的错误日志和解决方案提示

---

### 📄 配置示例

**config.ini 示例**：
```ini
[Nano-banana]
api_key = sk-xxx...
api_url = https://api.openai.com/v1/images/generations

[Gemini-banana]
api_key = sk-xxx...
api_url = https://apitt.cozex.cn/v1/images/generations

[Seedance]
api_key = sk-xxx...
api_url = https://apitt.cozex.cn/v1/video/generations

[Seedream]
api_key = sk-xxx...
api_url = https://apitt.cozex.cn/v1/images/generations
```

---

### 🚀 使用方式

1. **重载节点**：在 ComfyUI 中右键 → Reload Custom Nodes
2. **搜索节点**：输入"Nano"或"Banana"
3. **配置 API**：首次使用填入 API 密钥和地址
4. **开始使用**：配置会自动保存

---

### 📝 注意事项

**Nano Banana**：
- 使用 `aspect_ratio` 参数控制宽高比，而非固定分辨率
- 支持多图参考（1-4张）
- `image_size` 仅在 nano-banana-2 模型下可用
- 建议使用 1:1 或 16:9 宽高比获得最佳效果
- URL格式需要网络连接

---

## v2.1.1 (2025-12-25)

### 🆕 新增参数

#### Gemini Banana - 生图数量控制

新增 `n` 参数，控制一次请求生成的图片数量。

**请求示例**：
```json
{
  "model": "gemini-3-pro-image-preview-2k",
  "prompt": "星际穿越，黑洞，电影大片，超现实主义",
  "size": "2K",
  "n": 3,
  "response_format": "url"
}
```

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 | 范围 |
|------|------|------|--------|------|
| n | INT | 一次生成图片数量 | 1 | 1-10 |

**功能特性**：
- ✅ 支持一次请求生成多张图片（1-10张）
- ✅ 自动合并为 batch tensor 输出
- ✅ 参数显示在调试日志中
- ✅ 与 API 文档完全兼容

**使用示例**：
```
Gemini Banana
  ├─ 提示词: "星际穿越，黑洞，电影大片"
  ├─ 模型: gemini-3-pro-image-preview-2k
  ├─ 尺寸: 2K
  ├─ 生图数量: 3  ← 新增
  ├─ 响应格式: URL
  └─ 图片输出 ──→ SaveImage (输出3张图片)
```

---

## v2.1.0 (2025-12-24)

### 🎉 新增节点

#### Gemini Banana（推荐）
Google Gemini 3 Pro Image 高质量图像生成节点。

**请求示例**：
```json
{
  "model": "gemini-3-pro-image-preview-2k",
  "prompt": "星际穿越，黑洞，电影大片，超现实主义",
  "size": "2K",
  "response_format": "url"
}
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| prompt | STRING | 图片生成的文本描述 |
| api_key | STRING | API身份验证密钥 |
| base_url | STRING | API服务端点 |
| model | ENUM | gemini-3-pro-image-preview / 2k / 4k |
| size | ENUM | 1K / 2K / 4K |
| response_format | ENUM | URL / Base64 |
| timeout | INT | 超时时间 (30-600秒) |
| max_retries | INT | 最大重试次数 (1-10) |
| image1-4 | IMAGE | 可选输入图片 |

**功能特性**：
- ✅ 文生图：纯文本生成图片
- ✅ 图生图：单图生成新图片
- ✅ 多图融合：融合多张图片（最多4张）
- ✅ 支持 1K/2K/4K 分辨率
- ✅ 支持 URL 和 Base64 响应格式
- ✅ 自动重试机制（指数退避）
- ✅ 智能错误处理（4xx 直接失败，5xx/429 重试）
- ✅ 独立配置节 [Gemini-banana]
- ✅ 配置自动保存

**重试机制**：
- 指数退避：2秒、4秒、8秒，最大3次
- 4xx 客户端错误（除429）直接失败
- 5xx 服务器错误自动重试
- 429 限流错误自动重试
- 详细的错误日志和解决方案提示

---

### 🔧 技术改进

1. **模块化重命名**
   - `image_generation_pro_node.py` → `gemini_banana.py`
   - 类名修改：`ImageGenerationProNode` → `GeminiBananaNode`
   - 节点显示名：`artsmcp-gemini-banana`

2. **配置管理优化**
   - 使用独立配置节 `[Gemini-banana]`
   - 配置键名简化：`api_key`, `api_url`
   - 避免与其他节点配置冲突

3. **API 重试机制**
   - 实现指数退避策略
   - 智能错误分类处理
   - 与 CY_Banana_Pro.py 的重试机制保持一致

---

### 📝 配置示例

**config.ini 示例**：
```ini
[Gemini-banana]
api_key = sk-xxx...
api_url = https://apitt.cozex.cn/v1/images/generations

[Seedance]
api_key = sk-xxx...
api_url = https://apitt.cozex.cn/v1/video/generations

[Seedream]
api_key = sk-xxx...
api_url = https://apitt.cozex.cn/v1/images/generations
```

---

### 🚀 使用方式

1. **重载节点**：在 ComfyUI 中右键 → Reload Custom Nodes
2. **搜索节点**：输入"Gemini"或"Banana"
3. **配置 API**：首次使用填入 API 密钥和地址
4. **开始使用**：配置会自动保存

---

### 📝 注意事项

**Gemini Banana**：
- 不支持序列生成（与 Seedream 不同）
- 不支持水印参数（Gemini 自带 SynthID 水印）
- 多图融合最多支持4张输入
- URL格式需要网络连接
- 建议使用 2K 或 4K 分辨率获得最佳效果

---

## v2.0.0 (2025-12-23)

### 🎉 新增节点

#### 1. 图片生成Pro
完整支持 API 文档中的所有图片生成功能。

**请求示例**：
```json
{
  "model": "doubao-seedream-4-0-250828",
  "prompt": "星际穿越，黑洞，电影大片",
  "size": "2K",
  "sequential_image_generation": "auto",
  "sequential_image_generation_options": {
    "max_images": 3
  },
  "response_format": "url",
  "watermark": true
}
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| prompt | STRING | 图片生成的文本描述 |
| api_key | STRING | API身份验证密钥 |
| base_url | STRING | API服务端点 |
| model | ENUM | Seedream 4.0 / 4.5 |
| size | ENUM | 1K / 2K / 4K |
| sequential_image_generation | ENUM | disabled / auto |
| max_images | INT | 最大生成数量 (1-15) |
| response_format | ENUM | URL / Base64 |
| watermark | BOOLEAN | 是否添加水印 |
| image1-4 | IMAGE | 可选输入图片 |

**功能特性**：
- ✅ 文生图：纯文本生成图片
- ✅ 图生图：单图生成新图片
- ✅ 图生组图：自动生成多张相关图片（最多15张）
- ✅ 多图融合：融合多张图片（最多4张）
- ✅ 支持 1K/2K/4K 分辨率
- ✅ 支持 URL 和 Base64 响应格式
- ✅ 配置自动保存

---

#### 2. 视频生成Pro
完整支持 Doubao Seedance 和即梦模型。

**Doubao 请求示例**：
```json
{
  "model": "doubao-seedance-1-0-pro-fast-251015",
  "prompt": "清晨的海边，海浪拍打沙滩",
  "resolution": "480p",
  "ratio": "16:9",
  "duration": 8,
  "fps": 24,
  "watermark": true,
  "seed": 11,
  "camerafixed": false
}
```

**即梦请求示例**：
```json
{
  "model": "jimeng_v30",
  "prompt": "一条小河流淌在森林中",
  "resolution": "720p",
  "ratio": "16:9",
  "duration": 5,
  "template_id": "dynamic_orbit",
  "camera_strength": "medium",
  "seed": -1
}
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| prompt | STRING | 视频生成的文本描述 |
| api_key | STRING | API身份验证密钥 |
| base_url | STRING | API服务端点 |
| model_type | ENUM | Doubao / 即梦 |
| doubao_model | ENUM | Pro / Pro Fast |
| jimeng_model | ENUM | v3.0 / v3.0 Pro |
| resolution | ENUM | 480p / 720p / 1080p |
| ratio | ENUM | 16:9 / 9:16 / 1:1 等 |
| duration | INT | 时长 (2-12秒) |
| fps | INT | 帧率 (24/30) |
| seed | INT | 随机种子 (-1=随机) |
| watermark | BOOLEAN | 是否添加水印 |
| camerafixed | BOOLEAN | 相机固定 (Doubao) |
| first_frame_image | IMAGE | 首帧图片 |
| last_frame_image | IMAGE | 尾帧图片 |
| camera_template | ENUM | 运镜模板 (即梦 v3.0 720p) |
| camera_strength | ENUM | 运镜强度 (弱/中/强) |

**功能特性**：
- ✅ 文生视频：纯文本生成视频
- ✅ 图生视频-首帧：单图作为首帧
- ✅ 图生视频-首尾帧：两图控制首尾
- ✅ 支持 Doubao Seedance (Pro / Pro Fast)
- ✅ 支持即梦 v3.0 和 v3.0 Pro
- ✅ 即梦 v3.0 支持 11 种运镜模板
- ✅ 支持 480p/720p/1080p
- ✅ 自动轮询任务状态
- ✅ 自动下载视频到本地
- ✅ 随机种子控制

**运镜模板**（即梦 v3.0 720p）：
- 希区柯克推进/拉远
- 机械臂
- 动感环绕 / 中心环绕
- 起重机
- 超级拉远
- 逆时针回旋 / 顺时针回旋
- 手持运镜
- 快速推拉

---

### 🔧 技术改进

1. **参数自动保存**
   - API 密钥和地址自动保存到 `config.ini`
   - 下次使用直接加载

2. **调试日志增强**
   - 结构化的参数输出
   - 详细的请求和响应日志
   - 进度显示

3. **错误处理**
   - 完整的异常捕获
   - 友好的错误提示
   - 失败时返回默认值

---

### 📁 新增文件

```
comfyui_api/
├── image_generation_pro_node.py  # 图片生成Pro节点
├── video_generation_pro_node.py  # 视频生成Pro节点
└── config.ini                    # 自动生成的配置文件
```

---

### 🚀 使用方式

1. **重载节点**：在 ComfyUI 中右键 → Reload Custom Nodes
2. **搜索节点**：输入"初阳"或"图片生成"、"视频生成"
3. **配置API**：首次使用填入 API 密钥和地址
4. **开始使用**：配置会自动保存

---

### 📝 注意事项

**图片生成**：
- 序列生成选择"auto"后需设置最大图片数
- 多图融合最多支持4张输入
- URL格式需要网络连接

**视频生成**：
- 视频生成为异步任务，自动轮询
- 运镜功能仅即梦 v3.0 的 720p 支持
- 首尾帧功能 Pro Fast 不支持
- 确保输出目录有足够空间

---

## v1.2.0 (2025-12-10)

- ✅ Doubao Seedance 改进：
  - 返回 VIDEO 类型（VideoObject）
  - 自动下载视频到 output 目录
  - 添加独立的视频参数配置
  - 支持中断控制
  - 添加 Debug 模式

---

## v1.1.0

- ✅ Doubao Seedream 优化：
  - 修复下拉选择框显示问题
  - 简化参数配置

---

## v1.0.0

- ✅ 初始发布
- ✅ 支持 Doubao Seedance 视频生成
- ✅ 支持 Doubao Seedream 图片生成
