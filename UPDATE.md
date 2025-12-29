# 更新日志

## v2.4.0 (2025-12-29)

### 🚀 Gemini Banana - 企业级生产优化

#### ✨ 核心优化

**1. 尺寸归一化（修复关键风险）**

修复了图片尺寸不一致导致 `torch.stack()` 崩溃的问题。

**问题场景**：
- API 返回的图片尺寸不一致（如 896×1200 和 1024×1024）
- `torch.stack()` 要求所有 tensor shape 完全一致
- 直接导致节点崩溃

**解决方案**：
```python
def _normalize_tensor_size(self, tensors):
    """归一化tensor尺寸,避免尺寸不一致导致stack崩溃"""
    # 使用最小公共尺寸(裁剪策略)
    min_h = min(heights)
    min_w = min(widths)
    # 中心裁剪保留主体内容
```

**功能特性**：
- ✅ 自动检测尺寸是否一致
- ✅ 不一致时自动中心裁剪到最小公共尺寸
- ✅ 保留图片主体内容
- ✅ 详细的裁剪日志输出

**输出示例**：
```
[WARN] ⚠️ 检测到图片尺寸不一致!
[WARN] 尺寸分布: {(896, 1200), (1024, 1024)}
[INFO] 统一裁剪到最小公共尺寸: 896×1024
[SUCCESS] ✅ 已归一化 2 张图片尺寸
```

---

**2. Session 连接池优化**

使用 requests 官方推荐的 `HTTPAdapter` 精细控制连接池。

**优化前**：
```python
session = requests.Session()  # 使用默认配置
```

**优化后**：
```python
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,  # 连接池数量
    pool_maxsize=10,      # 每个连接池的最大连接数
    max_retries=0         # 重试由上层控制
)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

**收益**：
- ✅ 显式配置，代码更清晰
- ✅ 重试职责分离，逻辑更统一
- ✅ 符合 requests 官方最佳实践

---

**3. Base64 格式兼容**

支持 Data URI 格式的 Base64 字符串。

**兼容格式**：
```python
# 纯 Base64
"iVBORw0KGgo..."

# Data URI 格式
data:image/png;base64,iVBORw0KGgo..."
data:image/jpeg;base64,/9j/4AA..."
"data:image/webp;base64,UklGR..."
```

**实现**：
```python
if b64_string.startswith("data:image"):
    b64_string = b64_string.split(",", 1)[1]
```

**收益**：
- ✅ 自动支持不同 API 的返回格式
- ✅ 零风险兼容，不影响纯 Base64
- ✅ 防止解码失败

---

**4. 异常处理规范**

失败时直接抛出异常，不返回占位图片。

**修改前**：
```python
if not output_tensors:
    return (torch.zeros((1, 512, 512, 3)),)  # 返回黑色占位图
```

**修改后**：
```python
if not output_tensors:
    raise RuntimeError("未获取到任何图片数据！")  # 直接抛异常
```

**原因**：
- ⚠️ 返回占位图会被 ComfyUI 缓存
- ⚠️ 下次执行会直接返回缓存的错误结果
- ✅ 直接抛异常避免缓存污染

---

**5. 线程池优化**

限制下载线程数，避免线程暴涨。

**修改**：
```python
# 之前: max_download_workers = min(len(download_tasks), 10)
max_download_workers = min(len(download_tasks), 4)  # 限制≤4
```

**原因**：
- 请求阶段：并发请求 (max=10)
- 下载阶段：并发下载 (max=4)
- 避免 Windows 线程抖动
- 防止 requests 连接池被打爆

---

**6. Tensor 内存优化**

添加 `.contiguous()` 确保内存连续性。

**修改**：
```python
# 之前
batch_tensor = torch.stack(output_tensors, dim=0)

# 现在
batch_tensor = torch.stack(output_tensors, dim=0).contiguous()
```

**收益**：
- ✅ 防止下游节点假设连续内存导致错误
- ✅ 提升内存访问效率

---

**7. 日志分级控制**

新增 `详细日志` 参数，支持两种模式。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 详细日志 | BOOLEAN | 是否显示详细调试信息 | True |

**日志级别**：
- `True`: 显示所有 DEBUG 日志，包括详细的 tensor 信息
- `False`: 只显示关键 INFO/ERROR 日志

**实现**：
```python
def log(self, message, level="INFO"):
    if level == "DEBUG" and not self.verbose:
        return
    print(message)
```

---

**8. 函数职责拆分**

将 `generate_image()` 从 600+ 行重构为 7 个清晰步骤。

**新增子函数**：
1. `_prepare_prompts()` - 准备提示词列表
2. `_prepare_input_images()` - 收集和转换输入图片
3. `_build_request_tasks()` - 构建请求任务列表
4. `_send_requests()` - 并发发送所有请求
5. `_parse_results()` - 解析API响应
6. `_download_images()` - 并发下载所有图片
7. `_merge_tensors()` - 合并所有tensor为批次
8. `_normalize_tensor_size()` - 归一化tensor尺寸

**收益**：
- ✅ 每个函数职责单一，易于理解
- ✅ 易于单元测试
- ✅ 易于维护和扩展
- ✅ 未来支持其他模型时只需修改对应子函数

---

#### 📊 性能对比

| 指标 | 优化前 | 优化后 | 提升 |
|-----|-------|--------|------|
| **下载并发数** | 10 | 4 | 🔽 更稳定 |
| **Session 管理** | 每次新建 | 线程复用 | ✅ 高效 |
| **Tensor 内存** | 可能不连续 | 强制连续 | ✅ 安全 |
| **日志控制** | 无 | 分级控制 | ✅ 灵活 |
| **尺寸处理** | 不一致崩溃 | 自动归一化 | ✅ 零风险 |
| **异常处理** | 返回占位图 | 直接抛异常 | ✅ 防缓存污染 |
| **代码可维护性** | 600+行单函数 | 7步骤+8子函数 | ✅✅✅ 极大提升 |

---

#### 📝 注意事项

**尺寸归一化**：
- ✅ 自动检测并处理，无需用户干预
- ✅ 使用中心裁剪，保留主体内容
- ⚠️ 裁剪后图片尺寸可能变小

**异常处理**：
- ✅ 失败时会抛出明确的错误信息
- ✅ 不会返回占位图片
- ✅ 避免 ComfyUI 缓存错误结果

**日志控制**：
- ✅ 调试时建议开启详细日志
- ✅ 生产环境可关闭详细日志减少输出

---

## v2.3.0 (2025-12-29)

### 🚀 Gemini Banana - 重大功能升级

#### ✨ 新增功能

**1. 分行提示词批量生成**

支持多行提示词，每行作为独立任务处理，大幅提升批量生成效率。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 启用分行提示词 | BOOLEAN | 是否按行拆分提示词 | False |
| 每行并发请求数 | INT | 每行提示词的并发请求数 | 1 (1-10) |

**功能特性**：
- ✅ 自动按行拆分提示词（过滤空行）
- ✅ 每行提示词独立请求 API
- ✅ 支持每行设置并发请求数
- ✅ 总图片数 = 提示词行数 × 每行并发数

**使用示例**：
```
Gemini Banana
  ├─ 提示词: "星际穿越，黑洞
             赛博朋克风格，未来城市
             海洋深处，神秘生物"
  ├─ 启用分行提示词: True ☑
  ├─ 每行并发请求数: 3
  └─ 图片输出 ──→ SaveImage (输出9张图片)
```

---

**2. 真正的并发请求与下载**

使用 `ThreadPoolExecutor` 实现真正的并发处理，性能提升 3-5 倍。

**技术亮点**：
- ✅ **并发 API 请求**：所有提示词同时发送请求
- ✅ **并发图片下载**：所有图片同时下载
- ✅ **智能线程池**：自动调整线程数（min(10, 任务数)）
- ✅ **顺序保持**：下载结果自动排序，保持原始顺序

**性能对比**：
```
串行模式：3行提示词 × 3并发 = 9张图片
  ├─ API请求：3次（串行）
  ├─ 图片下载：9次（串行）
  └─ 总耗时：~90秒

并发模式：3行提示词 × 3并发 = 9张图片
  ├─ API请求：3次（并发）
  ├─ 图片下载：9次（并发）
  └─ 总耗时：~25秒 ✅ （提升 3.6x）
```

---

**3. 中文参数界面**

所有参数名称均使用中文显示，更符合中文用户使用习惯。

**参数名称对照**：

| 英文名称 | 中文名称 |
|----------|----------|
| prompt | 提示词 |
| api_key | API密钥 |
| base_url | API地址 |
| model | 模型 |
| size | 宽高比 |
| response_format | 响应格式 |
| timeout | 超时时间秒 |
| max_retries | 最大重试次数 |
| enable_multiline | 启用分行提示词 |
| concurrent_requests | 每行并发请求数 |
| image1-4 | 参考图片1-4 |

---

#### 🔧 技术优化

**1. 移除中断检测逺辑**

修复了 "Processing interrupted" 误报问题，完全符合 ComfyUI 官方标准。

**修复内容**：
- ❌ 移除了 `download_image_to_tensor` 中的 `interrupt_current_processing()` 检查
- ❌ 移除了 `make_api_request` 中的分段 sleep 中断检测
- ❌ 移除了 return 前的中断标志检查
- ❌ 移除了 `InterruptedError` 的特殊捕获逻辑

**原因分析**：
- 调用 `interrupt_current_processing()` 会被 ComfyUI 识别为“非自然结束”
- 即使该函数返回 False，ComfyUI 也会标记为 "Processing interrupted"
- 正确做法：完全交由 ComfyUI 处理中断，节点不主动检测

**修复效果**：
- ✅ 成功生成不再显示 "Processing interrupted"
- ✅ 节点执行完全符合 ComfyUI 标准
- ✅ 用户中断（Stop 按钮）由 ComfyUI 自动处理

---

**2. 资源管理优化**

改进了 requests 连接池管理，避免连接污染。

**优化点**：
- ✅ 每次重试创建新的 Session
- ✅ 正确关闭 response 和 session
- ✅ 使用 finally 块确保资源释放
- ✅ 避免 socket 泄漏

---

**3. 详细的输出日志**

新增了非常详细的数据输出信息，方便调试和问题排查。

**输出信息包括**：
- ✅ Tensor 形状、类型、设备、内存大小
- ✅ 数值范围、均值、标准差
- ✅ 每张图片的详细信息（小于5张时）
- ✅ 返回值结构说明

**输出示例**：
```
============================================================
[OUTPUT] 准备传递给下一个节点的数据详情:
============================================================
[OUTPUT] 数据类型: Tensor
[OUTPUT] 数据形状 (shape): torch.Size([3, 1200, 896, 3])
  ├─ 批次大小 (batch): 3
  ├─ 图片高度 (height): 1200
  ├─ 图片宽度 (width): 896
  └─ 通道数 (channels): 3
[OUTPUT] 元素总数: 9,676,800
[OUTPUT] 数据类型 (dtype): torch.float32
[OUTPUT] 内存大小: 36.91 MB
[OUTPUT] 数值范围: [0.0000, 1.0000]
[OUTPUT] 数值均值: 0.4523
[OUTPUT] ComfyUI 将接收到类型为 'IMAGE' 的输出
============================================================
```

---

#### 📝 注意事项

**分行提示词功能**：
- ✅ 启用后，每行作为独立任务处理
- ✅ 空行会被自动过滤
- ✅ 总图片数 = 行数 × 每行并发数
- ⚠️ 大量图片生成注意内存占用

**并发请求**：
- ✅ 最大线程数自动调整，不超过10
- ✅ 下载结果自动排序保持顺序
- ✅ 单个任务失败不影响其他任务

**中文参数**：
- ✅ Python 3.x 原生支持中文变量名
- ✅ 内部自动映射为英文变量
- ✅ 与 API 完全兼容

---

## v2.2.2 (2025-12-26)

### 🎎 Gemini Banana - 参数优化与调试增强

#### 🔧 size 参数更新

**参数更改**：
- ✅ 从固定分辨率（1K/2K/4K）改为宽高比格式
- ✅ 支持 10 种宽高比：1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- ✅ 默认值改为 `1:1`
- ✅ UI 标签更新："📐 尺寸" → "📐 宽高比"

**请求示例**：
```json
{
  "model": "gemini-3-pro-image-preview",
  "prompt": "星际穿越，黑洞，电影大片",
  "size": "16:9",
  "n": 1,
  "response_format": "url"
}
```

**使用示例**：
```
Gemini Banana
  ├─ 模型: gemini-3-pro-image-preview-2k
  ├─ 提示词: "星际穿越，黑洞，电影大片"
  ├─ 宽高比: 16:9  ← 更新
  ├─ 生图数量: 1
  └─ 图片输出 ──→ SaveImage
```

---

#### 🔍 调试信息增强

**新增详细调试输出**：

1. **完整响应数据**
   - ✅ 显示完整的 JSON 响应（格式化输出）
   - ✅ 不再截断至 200 字符

2. **数据结构检查**
   - ✅ 显示响应包含的所有键
   - ✅ 显示 `data` 字段的类型和内容
   - ✅ 显示每个图片项的详细信息：
     - 图片项类型
     - 图片项完整内容
     - 图片项包含的键
     - 期望的响应格式

3. **转换过程跟踪**
   - ✅ 每个图片转换的成功/失败状态
   - ✅ 带 ✅/❌ 标识
   - ✅ 显示输出 tensors 数量

4. **格式匹配详情**
   - ✅ `_process_image_item` 的调用参数
   - ✅ 格式匹配过程（URL/Base64）
   - ✅ 未匹配时显示期望格式 vs 实际包含的键

**调试输出示例**：
```
[SUCCESS] 请求成功！
[DEBUG] 完整响应数据: {
  "created": 1766743019,
  "data": [
    {
      "url": "https://...",
      "revised_prompt": "..."
    }
  ]
}
[DEBUG] 检查响应结构...
[DEBUG] 响应包含的键: ['created', 'data']
[DEBUG] data 类型: <class 'list'>
[DEBUG] 图片项包含的键: ['url', 'revised_prompt']
[DEBUG] 期望的响应格式: url
[DEBUG] 匹配到 URL 格式，开始下载...
[DEBUG] ✅ 第 1 个图片转换成功
```

---

### 📝 注意事项

**宽高比参数**：
- ✅ 与 API 文档完全一致
- ✅ 支持更多画幅比选择
- ✅ 建议使用 16:9 或 1:1 获得最佳效果

**调试信息**：
- ✅ 详细的调试输出帮助快速定位问题
- ✅ 查看控制台输出了解完整的请求和响应流程
- ✅ 特别适用于 API 响应成功但无图片的情况

---

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
