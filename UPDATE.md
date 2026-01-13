 更新日志

## v2.10.1 (2026-01-13)

### 🎬 Doubao Seedance - 模型分辨率支持更新

#### ✨ 功能更新

**1. doubao-seedance-1-5-pro-251215 模型新增 1080p 支持**

`doubao-seedance-1-5-pro-251215` 模型现已支持 1080p 分辨率。

**变更内容**：
- ✅ 移除 `doubao-seedance-1-5-pro-251215` 模型的 1080p 限制
- ✅ 所有模型（1.5 pro / 1.0 pro fast / 1.0 pro）均支持 480p / 720p / 1080p
- ✅ 更新代码注释，说明限制已移除

**技术实现**：
```python
# 修改前
if 模型 == "doubao-seedance-1-5-pro-251215" and 分辨率 == "1080p":
    raise ValueError("模型 'doubao-seedance-1-5-pro-251215' 不支持 1080p")

# 修改后
# 检查模型和分辨率的兼容性（已移除限制：doubao-seedance-1-5-pro-251215 现已支持 1080p）
```

**影响范围**：
- ✅ `_generate_single_video()` 方法
- ✅ `_generate_videos_concurrent()` 方法

---

#### 📝 文档更新

**README.md**：
- ✅ 更新分辨率参数说明，注明所有模型均支持
- ✅ 移除注意事项中关于 1080p 不支持参考图的说明

**UPDATE.md**：
- ✅ 新增 v2.10.1 版本更新日志
- ✅ 记录模型分辨率支持变更

---

#### ✅ 向后兼容

**完全兼容**：
- ✅ 仅移除限制，无其他变更
- ✅ 现有工作流无需修改
- ✅ 其他参数和功能保持不变

---

## v2.10.0 (2026-01-07)

### 🚀 Doubao Seedance - 并发批量生成与多视频输出

#### ✨ 新增功能

**1. 并发批量生成视频**

支持同时创建多个视频生成任务，所有任务完成后统一下载。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 并发请求数 | INT | 同时生成的视频数量 | 1 (1-10) |
| 启用提示词分行 | BOOLEAN | 是否按行拆分提示词 | False |

**功能特性**：
- ✅ **三阶段并发流程**：
  1. 阶段 1：并发创建所有视频生成任务
  2. 阶段 2：轮询所有任务状态直到全部完成
  3. 阶段 3：所有任务完成后统一下载视频
- ✅ **智能线程池**：使用 `ThreadPoolExecutor` 并发创建任务
- ✅ **实时进度**：显示每个任务的状态和进度
- ✅ **错误容忍**：单个任务失败不影响其他任务
- ✅ **中断支持**：支持 Ctrl+C 和 ComfyUI Stop 按钮中断

**三种使用场景**：

1. **单任务模式**：
   - 并发请求数 = 1
   - 启用提示词分行 = False
   - 生成 1 个视频

2. **相同提示词重复模式**：
   - 并发请求数 = 3
   - 启用提示词分行 = False
   - 相同提示词重复 3 次，生成 3 个视频

3. **提示词分行模式**：
   - 启用提示词分行 = True
   - 按换行符拆分提示词，每行一个任务
   - 3 行提示词 = 3 个视频
   - 并发请求数被忽略

**技术实现**：

```python
# 阶段 1：并发创建任务
with ThreadPoolExecutor(max_workers=min(len(prompts), 5)) as executor:
    futures = [
        executor.submit(
            self._create_single_task,
            prompt, API密钥, API地址, 模型,
            参考图片1, 参考图片2, 分辨率, 宽高比,
            时长, 帧率, 随机种子, 固定镜头,
            水印, 生成音频, 超时秒数, 忽略SSL证书
        )
        for idx, prompt in enumerate(prompts)
    ]
    
    # 收集所有任务 ID
    for idx, future in enumerate(as_completed(futures)):
        task_id, video_url = future.result()
        task_infos.append({...})

# 阶段 2：轮询所有任务
while True:
    for task in task_infos:
        if task['status'] in ['pending', 'running']:
            # 查询任务状态
            status = self.query_task_status(task['task_id'])
            task['status'] = status
    
    # 检查是否所有任务完成
    if all(task['status'] in ['success', 'failed'] for task in task_infos):
        break

# 阶段 3：下载所有视频
for task in success_tasks:
    video_obj = self.download_video(task['video_url'])
    video_objects.append(video_obj)

return (video_objects, stats)  # 返回所有视频
```

**使用示例**：

*相同提示词重复生成*：
```
Doubao Seedance Video
  ├─ 提示词: "一个机器人在未来城市中行走..."
  ├─ 并发请求数: 3
  ├─ 启用提示词分行: False ☐
  └─ 视频输出 ──→ SaveVideo (输出3个视频)
```

*提示词分行批量生成*：
```
Doubao Seedance Video
  ├─ 提示词: "一个机器人在未来城市中行走
             赛博朋克风格的城市夜景
             海洋深处的神秘生物"
  ├─ 启用提示词分行: True ☑
  ├─ 并发请求数: 1 (自动忽略)
  └─ 视频输出 ──→ SaveVideo (输出3个视频)
```

---

**2. 多视频输出支持**

并发生成的所有视频自动保存到 ComfyUI。

**技术实现**：

```python
# 节点定义
RETURN_TYPES = ("VIDEO", "STRING")
RETURN_NAMES = ("视频输出", "生成统计")
OUTPUT_IS_LIST = (True, False)  # 视频输出为列表，统计不是

# 返回所有视频
return (video_objects, stats)  # video_objects 是 VideoObject 列表
```

**功能特性**：
- ✅ **单视频兼容**：单个视频也返回列表格式 `[video]`
- ✅ **多视频输出**：并发生成的所有视频都可保存
- ✅ **ComfyUI 自动处理**： SaveVideo 节点自动遍历所有视频
- ✅ **顺序保持**：视频按创建顺序排列

**效果对比**：

*修改前*：
```
并发生成 3 个视频
  ├─ 下载 3 个视频到 output/
  ├─ 返回第 1 个 VideoObject
  └─ ComfyUI 只能保存 1 个视频 ❌
```

*修改后*：
```
并发生成 3 个视频
  ├─ 下载 3 个视频到 output/
  ├─ 返回 [video1, video2, video3] 列表
  └─ ComfyUI 保存所有 3 个视频 ✅
```

---

**3. 生成统计信息**

并发模式下输出详细的统计信息。

**输出示例**：
```
[Seedance] ============================================================
[Seedance] 🚀 并发批量生成完成
[Seedance]   - 总任务数: 3
[Seedance]   - 创建成功: 3
[Seedance]   - 生成成功: 3
[Seedance]   - 下载成功: 3
[Seedance]   - 失败任务: 0
[Seedance]   - 总用时: 93.2s
[Seedance] ============================================================
```

**功能特性**：
- ✅ 显示总任务数和各阶段成功数
- ✅ 显示失败任务数
- ✅ 显示总耗时
- ✅ 单任务模式不显示统计信息

---

#### 🔧 参数调整

**新增参数**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 并发请求数 | INT | 同时生成的视频数量 | 1 (1-10) |
| 启用提示词分行 | BOOLEAN | 是否按行拆分提示词 | False |

**参数顺序调整**（从上到下）：

1. 提示词
2. API密钥
3. API地址
4. 模型
5. 分辨率
6. 宽高比
7. 时长
8. 帧率
9. 随机种子
10. 固定镜头
11. 水印
12. 生成音频
13. **并发请求数** (新增)
14. **启用提示词分行** (新增)
15. 轮询间隔
16. 超时秒数 (原“请求超时”)
17. 最大等待时长
18. 详细日志 (原“调试模式”)
19. 忽略SSL证书

**参数重命名**：
- `请求超时` → `超时秒数`
- `调试模式` → `详细日志`

---

#### 📊 性能优化

**1. 并发任务创建**

使用 `ThreadPoolExecutor` 并发创建多个视频生成任务。

**性能对比**：
```
串行模式：3个视频
  ├─ 创建任务：3次（串行）
  ├─ 等待生成：~180秒
  └─ 总耗时：~190秒

并发模式：3个视频
  ├─ 创建任务：3次（并发）
  ├─ 等待生成：~90秒 (同时进行)
  └─ 总耗时：~93秒 ✅ (提升 2x)
```

---

**2. 状态轮询优化**

所有任务共享相同的轮询周期，避免重复等待。

**实现**：
```python
while True:
    # 一次查询所有任务的状态
    for task in task_infos:
        if task['status'] in ['pending', 'running']:
            new_status = self.query_task_status(task['task_id'])
            task['status'] = new_status
    
    # 检查是否全部完成
    if all_completed:
        break
    
    # 统一等待
    time.sleep(轮询间隔)
```

**收益**：
- ✅ 所有任务同步轮询，不浪费时间
- ✅ 实时显示所有任务的进度
- ✅ 减少 API 请求次数

---

#### 🐞 Bug 修复

**1. 修复多视频输出问题**

*问题*：
- 并发生成多个视频后，只有第一个视频被保存
- ComfyUI 的 SaveVideo 节点只接收到一个 VideoObject

*解决方案*：
```python
# 设置 OUTPUT_IS_LIST
OUTPUT_IS_LIST = (True, False)

# 返回所有视频列表
return (video_objects, stats)  # 而不是 video_objects[0]
```

---

**2. 解耦并发请求数和提示词分行**

*问题*：
- 之前需要 `并发请求数 > 1` 且 `提示词行数 > 1` 才能进入并发模式
- 这两个功能被误认为是关联的

*解决方案*：
```python
# 提示词处理逻辑
if 启用提示词分行:
    # 按换行符拆分
    prompts = [line.strip() for line in 提示词.split('\n') if line.strip()]
else:
    # 根据并发请求数重复提示词
    prompts = [提示词] * 并发请求数

# 判断是否进入并发模式
if len(prompts) > 1:
    # 并发模式
    return self._generate_videos_concurrent(...)
else:
    # 单任务模式
    return self._generate_single_video(...)
```

**收益**：
- ✅ 两个功能完全独立，逻辑更清晰
- ✅ 用户可以自由选择使用哪种模式
- ✅ 支持相同提示词重复生成

---

#### 📝 文档更新

**README.md**：
- ✅ 更新功能特性列表
- ✅ 新增并发模式说明
- ✅ 新增多视频输出说明
- ✅ 更新所有参数说明（中文名称）
- ✅ 新增并发批量生成使用示例
- ✅ 新增三种使用场景说明
- ✅ 新增并发流程图

**UPDATE.md**：
- ✅ 新增 v2.10.0 版本更新日志
- ✅ 详细说明所有新功能
- ✅ 提供技术实现细节
- ✅ 列出所有 Bug 修复

---

#### ✅ 向后兼容

**完全兼容**：
- ✅ 所有新增参数都是 optional
- ✅ 默认行为与之前一致（单任务模式）
- ✅ 现有工作流无需修改
- ✅ 单视频输出仍然正常工作

---

#### 💡 最佳实践

**并发批量生成**：
- ✅ 合理设置并发请求数（建议 3-5）
- ✅ 使用提示词分行功能生成不同内容
- ✅ 注意 API 限流，避免过多并发
- ✅ 开启详细日志查看实时进度

**性能优化**：
- ✅ 利用并发模式提升 2-3 倍效率
- ✅ 所有视频同时生成，不浪费时间
- ✅ 轮询间隔设置为 10秒，平衡响应和请求

---

## v2.9.0 (2026-01-07)

### 🚀 Doubao Seedance - 生产级优化与稳定性提升

#### ✨ 核心优化

**1. TLS/SSL 证书校验安全升级**

默认开启证书校验，提升安全性，同时提供调试选项。

**技术实现**:
```python
def create_ssl_context(self, insecure=False):
    """创建 SSL 上下文，默认开启证书校验"""
    if insecure:
        # 调试模式：禁用证书校验
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    # 生产模式：启用证书校验
    return ssl.create_default_context()
```

**新增参数**:

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 忽略SSL证书 | BOOLEAN | 忽略 SSL 证书校验（仅用于调试） | False |

**功能特性**:
- ✅ **默认安全**: 生产环境自动开启证书校验
- ✅ **调试友好**: 可选禁用证书校验用于内网测试
- ✅ **防止 MITM**: 避免中间人攻击风险
- ✅ **稳定性提升**: 在正规环境更稳定可靠

**收益**:
- 🔒 提升安全性，符合生产级标准
- 🔧 保留调试灵活性，兼顾开发需求
- 🌐 在 Cloudflare 等代理环境更可靠

---

**2. API 路径解析重构（关键 Bug 修复）**

完美支持反代、中转站、网关等复杂部署场景。

**问题场景**:
```
用户配置: https://api.xxx.com/proxy/openai
之前行为: 实际请求 https://api.xxx.com/v1/video/generations
结果: 路径被截断，请求失败 ❌
```

**解决方案**:
```python
# 在 generate_video() 开始时统一解析
parsed_url = urlparse(API地址)
self.api_host = parsed_url.netloc
self.api_base_path = parsed_url.path.rstrip('/') if parsed_url.path else ''

# 后续所有请求使用统一路径
path = f"{self.api_base_path}/v1/video/generations"
path = f"{self.api_base_path}/v1/video/generations/{task_id}"  # 查询
```

**修复后**:
```
用户配置: https://api.xxx.com/proxy/openai
实际请求: https://api.xxx.com/proxy/openai/v1/video/generations
结果: 完美支持反代 ✅
```

**功能特性**:
- ✅ **统一解析**: 只在开始时解析一次，避免重复处理
- ✅ **路径完整**: 保留完整的 base_path
- ✅ **支持反代**: 完美兼容各种代理、网关、中转站
- ✅ **协议自动补全**: 自动添加 https:// 前缀

**收益**:
- 🌐 支持复杂部署架构
- 🔧 修复关键路径截断 Bug
- 📊 更好的企业级兼容性

---

**3. 轮询状态机归一化重构**

统一不同 API 返回的状态格式，提升健壮性。

**状态归一化函数**:
```python
def normalize_status(self, raw_status):
    """统一状态格式"""
    status_mapping = {
        # 进行中
        "queued": "running",
        "processing": "running",
        "running": "running",
        # 成功
        "succeeded": "success",
        "completed": "success",
        "success": "success",
        # 失败
        "failed": "failed",
        "error": "failed",
        "expired": "failed",
        # 取消
        "cancelled": "cancelled",
        "canceled": "cancelled",
    }
    return status_mapping.get(raw_status.lower(), 'unknown')
```

**轮询逻辑优化**:
```python
# 原始状态
raw_status = inner_data.get('status', 'unknown')

# 归一化
status = self.normalize_status(raw_status)

# 日志输出
self.log(f"[{elapsed}s] Task status: {raw_status} -> {status}", "INFO")

# 统一判断
if status == 'success':      # 成功
if status == 'failed':       # 失败
if status == 'running':      # 进行中
if status == 'cancelled':    # 取消
if status == 'unknown':      # 未知
```

**功能特性**:
- ✅ **状态统一**: 不同 API 格式自动归一化
- ✅ **易于扩展**: 新增状态只需修改映射表
- ✅ **调试友好**: 同时输出原始状态和归一化状态
- ✅ **逻辑清晰**: 只需判断 5 种归一化状态

**收益**:
- 🛡️ 更健壮，兼容多种 API 返回格式
- 🔧 更易维护，状态判断逻辑统一
- 📊 更清晰，调试信息更详细

---

**4. VideoObject 懒加载优化**

减少不必要的 IO 操作，提升性能。

**实现**:
```python
class VideoObject:
    def __init__(self, filepath, is_placeholder=False):
        self._metadata_loaded = False  # 懒加载标志
        # 不立即加载元数据
    
    def get_dimensions(self):
        """懒加载 - 只在需要时才读取"""
        if not self._metadata_loaded:
            self._load_metadata()
        return (self._width, self._height)
    
    def get_fps(self):
        if not self._metadata_loaded:
            self._load_metadata()
        return self._fps
```

**功能特性**:
- ✅ **按需加载**: 只在实际调用时才读取视频文件
- ✅ **避免重复**: 加载后设置标志，不重复读取
- ✅ **减少 IO**: 如果不调用获取方法，完全不读取
- ✅ **向后兼容**: API 接口不变

**收益**:
- ⚡ 减少 IO 阻塞，提升响应速度
- 💾 降低内存占用
- 📹 大视频场景性能提升明显

---

**5. 图像输入优化与警告增强**

自动压缩图片并警告用户使用公网 URL。

**实现**:
```python
def tensor_to_image_url(self, tensor, max_size=1024, quality=85):
    """转换图片并自动压缩"""
    # 限制尺寸
    if max(pil_image.size) > max_size:
        pil_image.thumbnail((max_size, max_size), Image.LANCZOS)
        self.log(f"⚠️ 图像已缩放: {original_size} -> {pil_image.size}", "INFO")
    
    # 降低质量
    pil_image.save(buffer, format='JPEG', quality=quality)
    
    # 警告大 base64
    size_mb = len(base64_string) / (1024 * 1024)
    if size_mb > 5:
        self.log(f"⚠️ 警告: base64 超过 5MB ({size_mb:.2f}MB)", "INFO")
        self.log("建议: 使用公网可访问的图片 URL", "INFO")
```

**功能特性**:
- ✅ **自动压缩**: 大图自动缩放到 1024px
- ✅ **质量控制**: JPEG 质量降至 85，减小体积
- ✅ **实时警告**: 超过 5MB 立即提示用户
- ✅ **明确建议**: 提示使用公网 URL 而非 base64

**收益**:
- 📦 减少 payload 体积，避免 413/502 错误
- ⚡ 提高 API 请求成功率
- 👤 提升用户体验，明确问题所在

---

**6. 日志系统统一优化**

所有日志添加统一前缀，支持分级控制。

**实现**:
```python
def log(self, message, level="INFO"):
    """统一日志输出"""
    if level == "DEBUG" and not self.verbose:
        return
    prefix = "[Seedance]" if not message.startswith("[") else ""
    print(f"{prefix} {message}" if prefix else message)
```

**日志输出示例**:
```
[Seedance] 解析 API 地址: host=api.xxx.com, base_path='/proxy'
[Seedance] 调用 Doubao Seedance API: api.xxx.com/proxy/v1/video/generations
[Seedance] 模型: doubao-seedance-1-5-pro-251215
[Seedance] API调用成功
[Seedance] 视频生成任务已创建: task_12345
[Seedance] [30s] Task status: succeeded -> success
[Seedance] ✓ 视频下载成功: output/doubao_seedance_1736234567890.mp4
```

**功能特性**:
- ✅ **统一前缀**: 所有日志带 [Seedance] 标识
- ✅ **分级控制**: DEBUG 日志只在 verbose 模式显示
- ✅ **易于检索**: 快速定位节点日志
- ✅ **专业规范**: 符合生产级日志标准

**收益**:
- 🔍 日志更易检索和过滤
- 📊 调试更高效
- 🎯 生产环境日志更清晰

---

**7. IS_CHANGED 优化**

使用更优雅的 `float("nan")` 替代 `time.time()`。

**修改**:
```python
# 之前
@classmethod
def IS_CHANGED(cls, **kwargs):
    import time
    return time.time()

# 现在
@classmethod
def IS_CHANGED(cls, **kwargs):
    return float("nan")
```

**收益**:
- ✅ 更优雅，不需要 import time
- ✅ 性能略优，不需要系统调用
- ✅ 语义更清晰（NaN 表示总是变化）
- ✅ ComfyUI 官方推荐做法

---

#### 📊 参数更新

**新增参数**:

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 忽略SSL证书 | BOOLEAN | 忽略 SSL 证书校验（仅调试） | False |

**完整参数列表**:

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 提示词 | STRING | 视频生成提示词 | "多个镜头..." |
| API密钥 | STRING | API 身份验证密钥 | sk-xxx |
| API地址 | STRING | API 服务地址 | https://api.openai.com |
| 模型 | ENUM | 选择模型 | doubao-seedance-1-5-pro-251215 |
| 参考图片1-2 | IMAGE | 可选参考图片 | - |
| 分辨率 | ENUM | 视频分辨率 | 1080p |
| 宽高比 | ENUM | 视频宽高比 | adaptive |
| 时长 | INT | 视频时长（秒） | 5 (2-12) |
| 帧率 | ENUM | 帧率 | 24 |
| 随机种子 | INT | 随机种子 | -1 |
| 固定镜头 | BOOLEAN | 固定摄像头 | False |
| 水印 | BOOLEAN | 添加水印 | False |
| 生成音频 | BOOLEAN | 生成画面同步音频 | False |
| 调试模式 | BOOLEAN | 输出完整响应 | False |
| 请求超时 | INT | API 请求超时（秒） | 60 (60-600) |
| 轮询间隔 | INT | 轮询间隔（秒） | 10 (2-30) |
| 最大等待时长 | INT | 最大等待时间（秒） | 300 (60-3600) |
| **忽略SSL证书** | **BOOLEAN** | **忽略证书校验（调试）** | **False** |

---

#### 🎯 使用示例

**标准使用（生产环境）**:
```
Doubao Seedance
  ├─ 提示词: "一个机器人在未来城市中行走..."
  ├─ API地址: https://api.openai.com
  ├─ 忽略SSL证书: False  ← 生产环境，开启证书校验
  └─ video ──→ SaveVideo
```

**反代/中转站场景**:
```
Doubao Seedance
  ├─ API地址: https://api.proxy.com/v1/doubao
  ├─ 提示词: "..." 
  └─ video ──→ SaveVideo
  
实际请求: https://api.proxy.com/v1/doubao/v1/video/generations ✅
```

**内网调试场景**:
```
Doubao Seedance
  ├─ API地址: http://192.168.1.100:8080
  ├─ 忽略SSL证书: True  ← 内网测试，禁用证书校验
  ├─ 调试模式: True  ← 输出详细日志
  └─ video ──→ SaveVideo
```

**图生视频（优化后）**:
```
Load Image ──→ 参考图片1
                 ↓
Doubao Seedance
  ├─ 提示词: "图片中的场景开始动起来..."
  ├─ 参考图片1: <connected>
  ├─ (自动压缩到1024px并警告)
  └─ video ──→ SaveVideo
```

---

#### 📝 技术亮点

**1. 架构改进**:
- ✅ 统一 API 路径解析（一次解析，多处使用）
- ✅ 状态机归一化（映射表驱动）
- ✅ SSL 上下文统一管理
- ✅ 日志系统标准化

**2. 性能优化**:
- ✅ VideoObject 懒加载（减少 IO）
- ✅ 图像自动压缩（减少 payload）
- ✅ 连接池复用（SSL 上下文）
- ✅ IS_CHANGED 优化（避免系统调用）

**3. 稳定性提升**:
- ✅ TLS 证书校验（防止 MITM）
- ✅ 路径解析修复（支持反代）
- ✅ 状态归一化（兼容多种格式）
- ✅ 异常处理规范（直接抛出异常）

**4. 用户体验**:
- ✅ 统一日志前缀（易于检索）
- ✅ base64 警告提示（明确问题）
- ✅ 调试信息增强（快速定位）
- ✅ 参数语义清晰（忽略SSL证书）

---

#### 🔧 向后兼容

**完全兼容**:
- ✅ 所有新增参数都是 optional
- ✅ 默认行为与优化前一致（除 SSL 校验）
- ✅ VideoObject API 接口不变
- ✅ 现有工作流无需修改

**唯一变化**:
- ⚠️ SSL 证书校验默认开启（更安全）
- ✅ 如需内网调试，手动启用"忽略SSL证书"

---

#### 💡 最佳实践

**生产环境**:
- ✅ 使用默认配置（SSL 校验开启）
- ✅ 关闭调试模式（减少日志）
- ✅ 合理设置超时时间
- ✅ 图片使用公网 URL 而非 base64

**调试环境**:
- ✅ 开启调试模式（查看详细日志）
- ✅ 内网测试时启用"忽略SSL证书"
- ✅ 观察状态归一化日志
- ✅ 检查路径解析是否正确

**反代场景**:
- ✅ API地址包含完整路径
- ✅ 检查日志确认实际请求路径
- ✅ 使用 SSL 证书校验（更安全）

---

## v2.8.0 (2026-01-06)

### 🚀 Doubao Seedream - 批量并发与重试优化

#### ✨ 新增功能

**1. 分行提示词批量生成**

支持多行提示词,每行作为独立任务处理,大幅提升批量生成效率。

**参数说明**:

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 分行提示词 | BOOLEAN | 是否按行拆分提示词 | False |
| 并发请求数 | INT | 每行提示词的并发请求数 | 1 (1-10) |

**功能特性**:
- ✅ 自动按行拆分提示词(过滤空行)
- ✅ 每行提示词独立请求 API
- ✅ 支持每行设置并发请求数
- ✅ 总图片数 = 提示词行数 × 每行并发数

**使用示例**:
```
Doubao Seedream
  ├─ 提示词: "星际穿越,黑洞,复古列车
             赛博朋克风格,未来城市
             海洋深处,神秘生物"
  ├─ 分行提示词: True ☑
  ├─ 并发请求数: 3
  └─ 图片输出 ──→ SaveImage (输出9张图片: 3行×3并发)
```

---

**2. 真正的并发请求**

使用 `ThreadPoolExecutor` 实现真正的并发处理,性能提升 3-5 倍。

**技术亮点**:
- ✅ **批量并发**: 所有提示词 × 并发数 = 全部并发执行
- ✅ **智能线程池**: 自动调整线程数(min(任务数, 10))
- ✅ **错误容忍**: 单个请求失败不影响其他请求
- ✅ **详细追踪**: 每个请求都有明确标识(提示词ID-并发ID)
- ✅ **实时统计**: 显示完成进度、成功率、耗时等

**性能对比**:
```
串行模式:3行提示词 × 3并发 = 9张图片
  ├─ API请求:9次(串行)
  └─ 总耗时:~90秒

并发模式:3行提示词 × 3并发 = 9张图片
  ├─ API请求:9次(并发)
  └─ 总耗时:~25秒 ✅ (提升 3.6x)
```

---

**3. 可调节的重试机制**

新增 `最大重试次数` 参数,用户可自定义重试策略。

**参数说明**:

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 最大重试次数 | INT | API请求失败时的最大重试次数 | 3 (0-10) |

**功能特性**:
- ✅ **灵活配置**: 0=不重试,1-10=重试对应次数
- ✅ **指数退避**: 2秒、4秒、8秒递增等待(最多30秒)
- ✅ **智能判断**: 仅对 5xx 服务器错误重试,4xx 客户端错误直接失败
- ✅ **详细日志**: 显示每次重试的详细信息和失败原因

**使用场景**:
- **快速失败**: `最大重试次数=0`,请求失败后立即返回
- **标准容错**: `最大重试次数=3`(默认),平衡性能和可靠性
- **高可靠性**: `最大重试次数=10`,适合网络不稳定场景

---

#### 🔧 技术优化

**1. 批量请求架构**

完全重构请求处理逻辑,支持 **提示词数 × 并发数** 的乘积效应。

**请求构建流程**:
```python
for 每个提示词 in prompts:
    构建该提示词的payload
    
    for i in range(并发请求数):
        创建请求任务 {
            prompt_id: 提示词序号
            concurrent_id: 并发序号
            payload: 请求数据
            prompt_text: 提示词文本
        }
        添加到任务队列

# 一次性并发执行所有任务
使用线程池执行所有请求
等待所有请求完成或超时
收集所有结果
```

**收益**:
- ✅ 清晰的任务标识,便于调试
- ✅ 真正的并发执行,最大化性能
- ✅ 统一的结果收集和处理
- ✅ 灵活的扩展性

---

**2. 调试信息增强**

根据用户偏好,大幅增强了调试信息的详细程度和可读性。

**提示词解析信息**:
```
============================================================
📝 [提示词解析]
  - 分行模式: True
  - 提示词数量: 3
  - 提示词列表:
    [1] 星际穿越,黑洞,复古列车...
    [2] 赛博朋克风格,未来城市...
    [3] 海洋深处,神秘生物...
  - 总请求数: 9 (提示词×并发)
  - 预计生成图片数: 9
============================================================
```

**批量请求统计**:
```
============================================================
📊 [批量请求统计]
  - 总请求数: 9
  - 成功: 9 | 失败: 0
  - 总耗时: 18.50秒
  - 平均耗时: 16.20秒

  按提示词统计:
    [提示词1] 成功: 3/3
    [提示词2] 成功: 3/3
    [提示词3] 成功: 3/3
============================================================
```

---

**3. 并发下载优化**

图片下载也采用并发方式,进一步提升性能。

**实现**:
```python
# 并发下载
download_workers = min(len(all_image_urls), 5)  # 最多5个并发下载
with ThreadPoolExecutor(max_workers=download_workers) as executor:
    futures = {executor.submit(download_image, url, i+1): i 
              for i, url in enumerate(all_image_urls)}
    
    results = [None] * len(all_image_urls)
    for future in as_completed(futures):
        idx, tensor = future.result()
        if tensor is not None:
            results[idx-1] = tensor
```

**收益**:
- ✅ 图片下载速度提升 3-5 倍
- ✅ 按顺序组织结果,保持图片顺序
- ✅ 错误容忍,单个下载失败不影响其他

---

#### 📊 参数完整列表

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `提示词` | STRING | 图片生成的文本描述(支持多行) | "星际穿越..." |
| `API密钥` | STRING | API 身份验证密钥 | 配置文件读取 |
| `API地址` | STRING | API 服务端点 | https://api.openai.com |
| `模型` | ENUM | 选择模型 | doubao-seedream-4-0-250828 |
| `宽度` | INT | 生成图片的宽度(像素) | 2048 (512-4096) |
| `高度` | INT | 生成图片的高度(像素) | 2048 (512-4096) |
| `输入图片1-2` | IMAGE | 可选输入图片 | - |
| `最大图片数量` | INT | 最大生成图片数量 | 0 (0-10) |
| `水印` | BOOLEAN | 是否添加水印 | False |
| `返回格式` | ENUM | 响应格式 | url |
| `请求超时` | INT | API请求超时时间(秒) | 120 (30-600) |
| `调试模式` | BOOLEAN | 输出完整API请求和响应 | False |
| `并发请求数` | INT | 并发请求的数量 | 1 (1-10) |
| `最大重试次数` | INT | 失败后重试次数 | 3 (0-10) |
| `分行提示词` | BOOLEAN | 是否按行拆分提示词 | False |

---

#### 🎯 使用示例

**普通模式(单提示词 + 并发)**:
```
Doubao Seedream
  ├─ 提示词: "星际穿越,黑洞,电影大片"
  ├─ 并发请求数: 3
  └─ 图片输出 ──→ SaveImage (3张)
```

**批量模式(多提示词 + 单次请求)**:
```
Doubao Seedream
  ├─ 提示词: "星际穿越,黑洞
             赛博朋克风格,未来城市
             海洋深处,神秘生物"
  ├─ 分行提示词: True ☑
  ├─ 并发请求数: 1
  └─ 图片输出 ──→ SaveImage (3张)
```

**终极模式(多提示词 + 并发)**:
```
Doubao Seedream
  ├─ 提示词: "星际穿越,黑洞
             赛博朋克风格,未来城市
             海洋深处,神秘生物"
  ├─ 分行提示词: True ☑
  ├─ 并发请求数: 3
  └─ 图片输出 ──→ SaveImage (9张: 3行×3并发)
```

**高可靠性配置**:
```
Doubao Seedream
  ├─ 提示词: "森林小屋,阳光洒落"
  ├─ 最大重试次数: 10
  ├─ 并发请求数: 1
  └─ 图片输出 ──→ SaveImage
```

---

#### 📝 注意事项

**分行提示词**:
- ✅ 启用后,每行作为独立任务处理
- ✅ 空行会被自动过滤
- ✅ 总图片数 = 行数 × 每行并发数
- ⚠️ 大量图片生成注意内存占用

**并发请求**:
- ✅ 最大线程数限制为 10,避免线程暴涨
- ✅ 单个任务失败不影响其他任务
- ✅ 实时显示每个请求的完成状态
- ✅ 按提示词分组统计成功率

**重试机制**:
- ✅ 仅对 5xx 服务器错误重试
- ✅ 4xx 客户端错误直接失败不浪费重试
- ✅ 使用指数退避策略,避免频繁请求
- ⚠️ 重试次数越多,总耗时越长

**性能优化**:
- ✅ API 请求和图片下载都采用并发
- ✅ 合理设置并发数,避免触发 API 限流
- ✅ 调试模式会显示详细信息,略微影响性能

---

## v2.7.0 (2026-01-06)

### 🚀 Nano Banana - 企业级重构与功能升级

#### ✨ 核心架构升级

**1. 迁移到 Gemini 原生 API 格式**

完全重构为 Gemini 原生 API 格式，使用 `contents → parts → inline_data` 结构。

**请求格式对比**：

**之前（OpenAI 风格）**：
```json
{
  "model": "nano-banana-pro",
  "prompt": "一只可爱的猫咪",
  "aspect_ratio": "1:1",
  "n": 1,
  "response_format": "url"
}
```

**现在（Gemini 原生格式）**：
```json
{
  "contents": [
    {
      "parts": [
        {"text": "一只可爱的猫咪"},
        {
          "inline_data": {
            "mime_type": "image/jpeg",
            "data": "<base64_encoded_image>"
          }
        }
      ]
    }
  ],
  "generationConfig": {
    "imageConfig": {
      "aspectRatio": "1:1",
      "imageSize": "2K"
    }
  }
}
```

**核心变化**：
- ✅ `prompt` → `contents[0].parts[0].text`
- ✅ 输入图片 → `contents[0].parts[1].inline_data`
- ✅ `aspect_ratio` → `generationConfig.imageConfig.aspectRatio`
- ✅ `image_size` → `generationConfig.imageConfig.imageSize`
- ✅ 完全兼容 Gemini 官方 API 格式

**API URL 构建**：
```python
# 新格式: {base_url}/v1beta/models/{model}:generateContent?key={api_key}
final_url = f"{base_url}/v1beta/models/{model_value}:generateContent?key={API密钥}"
```

**收益**：
- ✅ 完全符合 Gemini 官方标准
- ✅ 支持更多官方功能和参数
- ✅ 更好的长期兼容性
- ✅ 统一的多模态输入格式

---

**2. 响应格式固定为 Base64**

为了稳定性和一致性，响应格式固定为 Base64。

**实现**：
```python
# 写死响应格式为 Base64
response_format = "Base64"
response_format_value = RESPONSE_FORMAT_MAP[response_format]  # "b64_json"
```

**原因**：
- ✅ Base64 格式更稳定，不依赖外部下载
- ✅ 避免 URL 过期或下载失败问题
- ✅ 响应速度更快，无需额外网络请求
- ✅ 与 Gemini 原生格式完美兼容

**影响**：
- ❌ UI 中移除"响应格式"参数选项
- ✅ 用户无需手动选择格式
- ✅ 内部自动处理 Base64 解码

---

**3. 分辨率参数优化**

新增 `imageSize` 配置，支持 1K/2K/4K 分辨率选择。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 | 可选值 |
|------|------|------|--------|--------|
| 分辨率 | ENUM | 图像分辨率 | 2K | none / 1K / 2K / 4K |

**实现逻辑**：
```python
image_config = {}

# 分辨率（仅当设置了才注入）
if 分辨率 and 分辨率 != "none":
    image_size_value = IMAGE_SIZE_MAP.get(分辨率)
    if image_size_value:
        image_config["imageSize"] = image_size_value

# 宽高比：始终注入
if 宽高比 in ASPECT_RATIO_MAP:
    aspect_ratio_value = ASPECT_RATIO_MAP[宽高比]
    if aspect_ratio_value:
        image_config["aspectRatio"] = aspect_ratio_value

if image_config:
    payload["generationConfig"] = {
        "imageConfig": image_config
    }
```

**功能特性**：
- ✅ 支持 `none`（不指定，由 API 自动选择）
- ✅ 支持 `1K`、`2K`、`4K` 显式指定
- ✅ 按需注入，避免触发无效参数错误
- ✅ 与宽高比参数独立配置

---

#### 🔥 新增功能

**1. 分行提示词批量生成**

支持多行提示词，每行作为独立任务处理，大幅提升批量生成效率。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 启用分行提示词 | BOOLEAN | 是否按行拆分提示词 | False |
| 并发请求数 | INT | 每行提示词的并发请求数 | 1 (1-10) |

**功能特性**：
- ✅ 自动按行拆分提示词（过滤空行）
- ✅ 每行提示词独立请求 API
- ✅ 支持每行设置并发请求数
- ✅ 总图片数 = 提示词行数 × 每行并发数

**实现逻辑**：
```python
if 启用分行提示词:
    # 每一行作为一个独立的提示词，分别生成图片
    prompt_lines = [line.strip() for line in 提示词.split('\n') if line.strip()]
    print(f"[INFO] 启用分行提示词，共 {len(prompt_lines)} 行")
    print(f"[INFO] 每行将各发送 {并发请求数} 个请求，总计: {len(prompt_lines) * 并发请求数} 个请求")
else:
    # 单行提示词
    prompt_lines = [提示词]
```

**使用示例**：
```
Nano Banana
  ├─ 提示词: "星际穿越，黑洞
             赛博朋克风格，未来城市
             海洋深处，神秘生物"
  ├─ 启用分行提示词: True ☑
  ├─ 并发请求数: 3
  └─ 图片输出 ──→ SaveImage (输出9张图片: 3行×3并发)
```

---

**2. 并发请求优化**

使用 `ThreadPoolExecutor` 实现真正的并发处理，性能提升 3-5 倍。

**技术亮点**：
- ✅ **并发 API 请求**：所有提示词同时发送请求
- ✅ **智能线程池**：自动调整线程数（min(5, 任务数)）
- ✅ **错误容忍**：单个请求失败不影响其他请求
- ✅ **详细日志**：实时显示每个请求的完成状态

**实现**：
```python
with ThreadPoolExecutor(max_workers=min(total_requests, 5)) as executor:
    # 提交所有请求任务
    futures = [
        executor.submit(
            make_api_request, 
            final_url, 
            headers, 
            payload_data,
            超时秒数, 
            最大重试次数
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
```

**性能对比**：
```
串行模式：3行提示词 × 3并发 = 9张图片
  ├─ API请求：9次（串行）
  └─ 总耗时：~90秒

并发模式：3行提示词 × 3并发 = 9张图片
  ├─ API请求：9次（并发）
  └─ 总耗时：~25秒 ✅ （提升 3.6x）
```

---

**3. 匹配参考尺寸**

在图生图模式下自动将输出图片调整为与参考图片相同的尺寸。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 匹配参考尺寸 | BOOLEAN | 是否将输出图片调整为参考图尺寸 | False |

**功能特性**：
- ✅ 仅在有输入参考图片时生效（图生图/多图参考模式）
- ✅ 使用第一张参考图的尺寸作为目标
- ✅ 自动对所有 API 返回的图片进行智能缩放 + 居中裁剪
- ✅ 使用 `ImageOps.fit` + `Image.LANCZOS` 高质量重采样算法
- ✅ 确保输出图片尺寸与参考图完全一致

**实现**：
```python
def _match_reference_size(self, output_tensors, input_images):
    """匹配参考图片尺寸 - 使用第一张参考图的尺寸作为目标"""
    if not output_tensors or not input_images:
        return output_tensors
    
    # 获取第一张参考图的尺寸
    ref_tensor = input_images[0]
    target_h = ref_tensor.shape[0]  # 高度
    target_w = ref_tensor.shape[1]  # 宽度
    
    matched_tensors = []
    for idx, tensor in enumerate(output_tensors):
        # 使用 ImageOps.fit 进行智能缩放+居中裁剪
        resized_image = ImageOps.fit(pil_image, (target_w, target_h), method=Image.LANCZOS)
        matched_tensors.append(resized_tensor)
    
    return matched_tensors
```

**使用示例**：
```
Load Image ──→ 参考图片1 (1600×2848)
                 ↓
Nano Banana
  ├─ 提示词: "将这张图片转换为油画风格"
  ├─ 宽高比: 16:9  ← API 按此比例构图
  ├─ 匹配参考尺寸: True ☑
  ├─ 参考图片1: <connected>
  └─ 图片输出 ──→ SaveImage (输出 1600×2848)
```

---

**4. 日志分级控制**

新增 `详细日志` 参数，支持两种日志模式。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 详细日志 | BOOLEAN | 是否显示详细调试信息 | False |

**日志级别**：
- `False`: 只显示关键 INFO/ERROR 日志（推荐生产环境）
- `True`: 显示所有 DEBUG 日志，包括详细的 tensor 信息（推荐调试）

**实现**：
```python
def log(self, message, level="INFO"):
    """统一日志输出 (支持分级)"""
    if level == "DEBUG" and not self.verbose:
        return  # DEBUG 日志只在 verbose 模式下打印
    print(message)
```

**详细日志输出示例**：
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
[OUTPUT] 数据维度 (ndim): 4
[OUTPUT] 元素总数: 9,676,800
[OUTPUT] 数据类型 (dtype): torch.float32
[OUTPUT] 存储设备 (device): cpu
[OUTPUT] 是否需要梯度: False
[OUTPUT] 内存大小: 36.91 MB
[OUTPUT] 数值范围: [0.0000, 1.0000]
[OUTPUT] 数值均值: 0.4523
[OUTPUT] 数值标准差: 0.2891

[OUTPUT] 返回值结构: tuple 包含 1 个元素
[OUTPUT] 返回值内容: (torch.Tensor,)
[OUTPUT] ComfyUI 将接收到类型为 'IMAGE' 的输出
============================================================
```

---

#### 🔧 技术优化

**1. Session 连接池优化**

使用 requests 官方推荐的 `HTTPAdapter` 精细控制连接池。

**实现**：
```python
def get_session():
    """获取线程本地的 Session (复用连接池)"""
    if not hasattr(thread_local, "session"):
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
```

**收益**：
- ✅ 显式配置，代码更清晰
- ✅ 连接池复用，减少握手开销
- ✅ 重试职责分离，逻辑更统一
- ✅ 符合 requests 官方最佳实践

---

**2. 尺寸归一化（修复关键风险）**

修复了图片尺寸不一致导致 `torch.stack()` 崩溃的问题。

**问题场景**：
- API 返回的图片尺寸不一致（如 896×1200 和 1024×1024）
- `torch.stack()` 要求所有 tensor shape 完全一致
- 直接导致节点崩溃

**解决方案**：
```python
def _normalize_tensor_size(self, tensors):
    """归一化tensor尺寸,避免尺寸不一致导致stack崩溃"""
    # 检查是否所有尺寸都一致
    if len(set(shapes)) == 1:
        return tensors
    
    # 尺寸不一致,需要归一化
    # 使用最小公共尺寸(裁剪策略)
    min_h = min(heights)
    min_w = min(widths)
    
    # 中心裁剪
    normalized = []
    for t in tensors:
        h, w, c = t.shape
        start_h = (h - min_h) // 2
        start_w = (w - min_w) // 2
        cropped = t[start_h:start_h+min_h, start_w:start_w+min_w, :]
        normalized.append(cropped)
    
    return normalized
```

**功能特性**：
- ✅ 自动检测尺寸是否一致
- ✅ 不一致时自动中心裁剪到最小公共尺寸
- ✅ 保留图片主体内容
- ✅ 详细的裁剪日志输出

---

**3. 响应解析优化**

优先支持 Gemini 原生响应格式，兼容 OpenAI 格式。

**Gemini 原生格式**：
```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "inlineData": {
              "mimeType": "image/jpeg",
              "data": "<base64_encoded_image>"
            }
          }
        ]
      }
    }
  ]
}
```

**解析逻辑**：
```python
# 优先处理 Gemini 原生格式: candidates -> content.parts
if "candidates" in result:
    candidates = result.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            # 1. inlineData / inline_data（优先图片）
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data:
                img_b64 = inline_data.get("data")
                if img_b64:
                    tensor = base64_to_tensor(img_b64)
                    if tensor is not None:
                        output_tensors.append(tensor)
            # 2. 文本里可能塞了 data:image/base64,...
            elif "text" in part:
                text_content = part["text"]
                if "data:image" in text_content and "base64," in text_content:
                    # 提取并解析
                    ...
# 兼容旧的 OpenAI images/generations 风格: data + b64_json/url
elif "data" in result:
    ...
```

**收益**：
- ✅ 完美支持 Gemini 官方响应格式
- ✅ 自动兼容 `inlineData` 和 `inline_data` 两种写法
- ✅ 兼容文本中嵌入的 Base64 图片
- ✅ 向后兼容 OpenAI 格式

---

**4. Tensor 内存优化**

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
- ✅ 避免潜在的内存布局问题

---

**5. 异常处理规范**

失败时直接抛出异常，不返回占位图片。

**修改前**：
```python
if not output_tensors:
    return (torch.zeros((1, 512, 512, 3)),)  # 返回黑色占位图
```

**修改后**：
```python
if not output_tensors:
    raise RuntimeError("未获取到任何图片数据")  # 直接抛异常
```

**原因**：
- ⚠️ 返回占位图会被 ComfyUI 缓存
- ⚠️ 下次执行会直接返回缓存的错误结果
- ✅ 直接抛异常避免缓存污染

---

#### 📊 参数完整列表

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `提示词` | STRING | 图片生成的文本描述（支持多行） | "一只可爱的猫咪..." |
| `API密钥` | STRING | API 身份验证密钥 | 配置文件读取 |
| `API地址` | STRING | API 服务端点 | https://api.vectortara.com/gemini |
| `模型` | ENUM | 选择模型 | nano-banana-2 |
| `宽高比` | ENUM | 图片宽高比 | 1:1 |
| `分辨率` | ENUM | 图像分辨率 | 2K |
| `超时秒数` | INT | API 请求超时时间 | 120 (30-600) |
| `最大重试次数` | INT | 失败后重试次数 | 3 (1-10) |
| `并发请求数` | INT | 一次生成的图片数量 | 1 (1-10) |
| `启用分行提示词` | BOOLEAN | 是否按行拆分提示词 | False |
| `匹配参考尺寸` | BOOLEAN | 是否将输出图片调整为参考图尺寸 | False |
| `详细日志` | BOOLEAN | 是否显示详细调试信息 | False |
| `参考图片1-4` | IMAGE | 可选输入图片 | - |

---

#### 🎯 使用示例

**文生图**：
```
Nano Banana
  ├─ 提示词: "一只可爱的猫咪，卡通风格"
  ├─ 模型: nano-banana-2
  ├─ 宽高比: 16:9
  ├─ 分辨率: 2K
  ├─ 并发请求数: 2
  └─ 图片输出 ──→ SaveImage (2张)
```

**批量生成**：
```
Nano Banana
  ├─ 提示词: "星际穿越，黑洞
             赛博朋克风格，未来城市
             海洋深处，神秘生物"
  ├─ 启用分行提示词: True ☑
  ├─ 并发请求数: 3
  └─ 图片输出 ──→ SaveImage (9张: 3行×3并发)
```

**图生图 + 匹配尺寸**：
```
Load Image ──→ 参考图片1 (1600×2848)
                 ↓
Nano Banana
  ├─ 提示词: "将这张图片转换为油画风格"
  ├─ 匹配参考尺寸: True ☑
  └─ 图片输出 ──→ SaveImage (1600×2848)
```

**多图参考**：
```
Load Image ──→ 参考图片1
Load Image ──→ 参考图片2
                 ↓
Nano Banana
  ├─ 提示词: "融合这两张图片的风格"
  ├─ 参考图片1: <connected>
  ├─ 参考图片2: <connected>
  └─ 图片输出 ──→ SaveImage
```

---

#### 📝 注意事项

**API 格式变更**：
- ⚠️ 完全迁移到 Gemini 原生格式，与之前的 OpenAI 风格不兼容
- ⚠️ API 地址需要支持 `/v1beta/models/{model}:generateContent` 路径
- ✅ 默认地址已更新为 `https://api.vectortara.com/gemini`
- ✅ 兼容 Gemini 官方 API 端点

**响应格式固定**：
- ❌ 移除了"响应格式"参数选项
- ✅ 固定使用 Base64 格式，更稳定可靠
- ✅ 自动处理解码，用户无需关心

**分行提示词**：
- ✅ 启用后，每行作为独立任务处理
- ✅ 空行会被自动过滤
- ✅ 总图片数 = 行数 × 每行并发数
- ⚠️ 大量图片生成注意内存占用

**并发请求**：
- ✅ 最大线程数限制为 5，避免线程暴涨
- ✅ 单个任务失败不影响其他任务
- ✅ 实时显示每个请求的完成状态

**匹配参考尺寸**：
- ⚠️ `宽高比` 参数不受影响，API 仍按选择的比例构图
- ⚠️ 匹配参考尺寸仅在本地对输出进行后处理
- ✅ 使用居中裁剪，保留主体内容
- ⚠️ 裁剪可能会损失部分画面内容

**日志控制**：
- ✅ 调试时建议开启详细日志
- ✅ 生产环境可关闭详细日志减少输出
- ✅ 详细日志会显示完整的 tensor 信息

---

## v2.6.0 (2026-01-06)

### 🚀 Gemini Banana - 参数体验优化

#### 🔧 参数调整

**1. 参数重命名和默认值优化**

为了提供更直观的用户体验，对部分参数名称和默认值进行了调整。

**参数名称更改**：

| 原名称 | 新名称 | 说明 |
|----------|----------|------|
| `每行并发请求数` | `并发请求数` | 更简洁明了，适用于所有模式 |
| `超时时间秒` | `超时秒数` | 与其他节点保持一致 |

**默认值调整**：

| 参数 | 原默认值 | 新默认值 | 调整原因 |
|------|----------|----------|----------|
| `最大重试次数` | 1 | 3 | 与 Nano Banana 保持一致，增强稳定性 |
| `匹配参考尺寸` | True | False | 默认关闭，由用户主动开启 |
| `详细日志` | True | False | 减少日志噪音，提升性能 |

---

**2. 参数语义优化**

`并发请求数` 参数现在在两种模式下都能正常使用：

**未启用分行提示词时**：
- 语义：一次生成的图片数量
- 示例：`并发请求数=3` → 生成 3 张图片

**启用分行提示词时**：
- 语义：每行提示词的并发请求数
- 示例：`3行提示词 × 并发请求数=3` → 生成 9 张图片

---

#### 📄 最新参数表

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `提示词` | STRING | 图片生成的文本描述（支持多行） | "星际穿越..." |
| `API密钥` | STRING | API 身份验证密钥 | sk-xxx |
| `API地址` | STRING | API 服务端点 | https://api.openai.com |
| `模型` | ENUM | 选择模型 | gemini-3-pro-image-preview |
| `宽高比` | ENUM | 图片宽高比 | 1:1 |
| `响应格式` | ENUM | 返回格式 | URL |
| `超时秒数` | INT | API 请求超时时间 | 120 (30-600) |
| `最大重试次数` | INT | 失败后重试次数 | 3 (1-10) |
| `并发请求数` | INT | 一次生成的图片数量 | 1 (1-10) |
| `启用分行提示词` | BOOLEAN | 是否按行拆分提示词 | False |
| `匹配参考尺寸` | BOOLEAN | 是否将输出图片调整为参考图尺寸 | False |
| `详细日志` | BOOLEAN | 是否显示详细调试信息 | False |
| `参考图片1-4` | IMAGE | 可选输入图片 | - |

---

#### 📊 使用示例

**普通模式（未启用分行提示词）**：
```
Gemini Banana
  ├─ 提示词: "星际穿越，黑洞，电影大片"
  ├─ 并发请求数: 3  ← 生成 3 张图片
  └─ 图片输出 ──→ SaveImage (3张)
```

**批量模式（启用分行提示词）**：
```
Gemini Banana
  ├─ 提示词: "星际穿越，黑洞
             赛博朋克风格，未来城市
             海洋深处，神秘生物"
  ├─ 启用分行提示词: True ☑
  ├─ 并发请求数: 3  ← 每行 3 张
  └─ 图片输出 ──→ SaveImage (3行×3=9张)
```

**图生图 + 匹配尺寸**：
```
Load Image ──→ 参考图片1 (1600×2848)
                 ↓
Gemini Banana
  ├─ 提示词: "将这张图片转换为油画风格"
  ├─ 匹配参考尺寸: True ☑
  └─ 图片输出 ──→ SaveImage (1600×2848)
```

---

#### 📝 注意事项

**参数调整影响**：
- ✅ 对现有工作流无影响，旧参数名仍可使用
- ✅ 默认值调整只影响新建节点
- ✅ 已保存的配置不受影响

**推荐配置**：
- ✅ **普通用户**：使用默认配置即可
- ✅ **调试场景**：开启“详细日志”
- ✅ **图生图场景**：需要时才开启“匹配参考尺寸”

---

## v2.5.0 (2025-12-30)

### 🌟 Gemini Banana - 新增匹配参考尺寸功能

#### ✨ 新增功能

**匹配参考尺寸**

在图生图模式下自动将输出图片调整为与参考图片相同的尺寸。

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| 匹配参考尺寸 | BOOLEAN | 是否将输出图片调整为参考图尺寸 | True |

**功能特性**：
- ✅ 仅在有输入参考图片时生效（图生图/多图融合模式）
- ✅ 使用第一张参考图的尺寸作为目标
- ✅ 自动对所有 API 返回的图片进行智能缩放 + 居中裁剪
- ✅ 使用 `ImageOps.fit` + `Image.LANCZOS` 高质量重采样算法
- ✅ 确保输出图片尺寸与参考图完全一致

**请求示例**：
```json
{
  "model": "gemini-3-pro-image-preview-2k",
  "prompt": "将这张图片转换为油画风格",
  "size": "16:9",
  "response_format": "url",
  "image": "data:image/jpeg;base64,..."
}
```

**使用示例**：
```
Load Image ──→ 参考图片1 (1600×2848)
                 ↓
Gemini Banana
  ├─ 提示词: "将这张图片转换为油画风格"
  ├─ 宽高比: 16:9  ← API 按此比例构图
  ├─ 匹配参考尺寸: True ☑
  ├─ 参考图片1: <connected>
  └─ 图片输出 ──→ SaveImage (输出 1600×2848)
```

**输出示例**：
```
============================================================
[INFO] 启用匹配参考尺寸功能
[INFO] 参考图尺寸: 1600×2848
[INFO] 待处理图片数量: 3
============================================================

[INFO] 图片1: 1696×2528 → 1600×2848 (缩放+裁剪)
[INFO] 图片2: 1696×2528 → 1600×2848 (缩放+裁剪)
[INFO] 图片3: 1696×2528 → 1600×2848 (缩放+裁剪)

[SUCCESS] ✅ 已将 3 张图片调整为参考尺寸 1600×2848
```

---

#### 🔧 日志优化

**修复尺寸显示顺序**

修复了日志中尺寸显示为“高×宽”的问题，现在与 ComfyUI 界面保持一致。

**修复前**：
```python
print(f"[INFO] 参考图尺寸: {target_h}×{target_w}")  # 2848×1600
print(f"[INFO] 图片1: {current_h}×{current_w} → {target_h}×{target_w}")
```

**修复后**：
```python
print(f"[INFO] 参考图尺寸: {target_w}×{target_h}")  # 1600×2848
print(f"[INFO] 图片1: {current_w}×{current_h} → {target_w}×{target_h}")
```

**效果**：
- ✅ 日志显示“宽×高”，与 ComfyUI 节点尺寸显示一致
- ✅ 避免用户误解为图片被“旋转”
- ✅ 提升调试信息可读性

---

#### 📝 技术细节

**实现原理**：

1. **提取参考尺寸**：
   ```python
   ref_tensor = input_images[0]  # 使用第一张参考图
   if len(ref_tensor.shape) > 3:
       ref_tensor = ref_tensor[0]  # 如果是批次，取第一张
   
   target_h = ref_tensor.shape[0]  # 高度
   target_w = ref_tensor.shape[1]  # 宽度
   ```

2. **智能缩放 + 裁剪**：
   ```python
   # 转换为 PIL Image
   array = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
   pil_image = Image.fromarray(array, mode='RGB')
   
   # 使用 ImageOps.fit 进行智能缩放+居中裁剪
   resized_image = ImageOps.fit(pil_image, (target_w, target_h), method=Image.LANCZOS)
   
   # 转回 tensor
   resized_array = np.array(resized_image).astype(np.float32) / 255.0
   resized_tensor = torch.from_numpy(resized_array)
   ```

3. **执行时机**：
   - 在 API 图片下载完成后
   - 在 tensor 合并之前
   - 不影响 API 请求参数

**与 CY_Banana_Pro 的一致性**：
- ✅ 使用相同的 `ImageOps.fit` 逻辑
- ✅ 使用相同的 `Image.LANCZOS` 算法
- ✅ 相同的参考图选择逻辑（第一张）
- ✅ 相同的触发条件（开关 + 有参考图）

---

#### 📊 适用场景

**推荐使用**：
1. **风格转换**：保持原图尺寸，只改变风格
2. **批量处理**：确保所有输出图片尺寸一致
3. **后续处理**：需要固定尺寸的 ComfyUI 流程
4. **多图融合**：统一输出尺寸便于拼接

**不推荐使用**：
1. **纯文生图**：没有参考图，功能不会触发
2. **需要高分辆率**：API 返回的高分辆率图会被裁剪
3. **特殊构图需求**：不希望裁剪画面内容

---

#### 📝 注意事项

**API 参数**：
- ⚠️ `宽高比` 参数不受影响，API 仍按选择的比例构图
- ⚠️ 匹配参考尺寸仅在本地对输出进行后处理
- ✅ 不影响 model、response_format 等其他参数

**裁剪逻辑**：
- ✅ 使用居中裁剪，保留主体内容
- ✅ 如果 API 返回的图片比参考图小，会放大后裁剪
- ✅ 如果 API 返回的图片比参考图大，会直接裁剪
- ⚠️ 裁剪可能会损失部分画面内容

**性能影响**：
- ✅ 仅在有参考图且开启开关时执行
- ✅ 使用 LANCZOS 算法，质量高但较慢
- ✅ 处理时间与图片数量和尺寸成正比

---

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
