# ComfyUI Doubao API Plugin

ComfyUI 插件，集成 **Doubao（豆包）AI API** 进行专业的图像和视频生成：
- **Doubao Seedance**: AI 视频生成（文生视频、图生视频）
- **Doubao Seedream**: AI 图像生成（文生图、图生图、图生组图、多图融合）

## 功能特性

### Doubao Seedance (AI 视频生成)
- 文生视频：根据文本提示生成视频
- 图生视频：单图或多图生成视频
- 支持视频状态查询和轮询
- 支持多种视频参数（分辨率、帧率、时长等）

### Doubao Seedream (图片生成)
- 文生图：根据文本提示生成图片
- 图生图：单图输入生成新图片
- 图生组图：生成多张相关图片
- 多图融合：融合多张图片生成新图片
- 支持多种图片尺寸和质量选项

## 安装

1. 将插件文件夹复制到ComfyUI的`custom_nodes`目录
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### Doubao Seedance (视频生成)

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

### Doubao Seedream (图片生成)
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

## 故障排除

### 问题：节点无法连接
**解决方案**：
1. 检查 ComfyUI 是否已重新加载节点
2. 右键画布 → Refresh / Reload Custom Nodes
3. 完全重启 ComfyUI

### 问题：视频生成一直 "unknown" 状态
**解决方案**：
1. 启用 `debug_mode` 查看完整响应
2. 检查 API 密钥是否有效
3. 检查网络连接
4. 节点会在 10 次 unknown 后自动中止

### 问题：无法中断视频生成
**解决方案**：
1. 确保已更新到最新版本（包含中断检测）
2. 使用 Ctrl+C 或 ComfyUI Stop 按钮
3. 节点每 0.5 秒检查一次中断信号

### 问题：'list' / 'str' object has no attribute 'get_dimensions'
**解决方案**：
1. 确保返回的是 VideoObject 而不是字符串或列表
2. 检查是否正确安装了 opencv-python
3. 重新加载节点

### 问题：SSL 警告
**说明**：这是预期行为，代码中已禁用 SSL 验证以兼容中转站。如需启用验证，修改代码中的 `verify=False` 为 `verify=True`。

## 更新日志

### v1.2.0 (2025-12-10)
- ✅ Doubao Seedance 改进：
  - 返回 VIDEO 类型（VideoObject）而不是 URL 字符串
  - 自动下载视频到 output 目录
  - 添加独立的视频参数配置（resolution, ratio, duration 等）
  - 移除 prompt 中的命令行参数要求
  - 支持中断控制（Ctrl+C / Stop 按钮）
  - 添加 Debug 模式查看完整 API 响应
  - 修复嵌套 API 响应解析问题
  - 添加所有任务状态支持（queued, running, succeeded, failed, cancelled, expired）

### v1.1.0
- ✅ Doubao Seedream 优化：
  - 修复下拉选择框显示问题
  - 简化参数配置
  - 移除冗余参数

### v1.0.0
- ✅ 初始发布
- ✅ 支持 Doubao Seedance 视频生成
- ✅ 支持 Doubao Seedream 图片生成

## 许可证

MIT License

