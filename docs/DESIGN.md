# 技术与接口设计

状态：开发前设计，本文中的模块、接口和子命令尚未实现。开始任务 worktree 前，由主聊天完成最小依赖验证并提交共享代码；接口变化先修改本文，再同步相关任务。

## 识别流程

```text
注册照片 → 检测与对齐 → 预训练 CNN → 本人特征库
                                             ↓
图片或视频帧 → 相同特征提取流程 → 距离比对 → 昵称 / 未知
```

无脸和多脸在识别流程中返回独立状态。读取失败、库损坏和模型故障属于运行错误，不转换为“未知”。

DeepFace 提供特征提取能力；本项目维护注册库、匹配规则、命令行和视频展示。第一版使用简单 JSON 文件保存少量特征，不引入数据库或额外服务。

## 技术选择与待验证事项

| 项目 | 当前设计 | 确定方式 |
| --- | --- | --- |
| Python | 优先使用现有 Python 3.12 | 在两台电脑分别验证依赖 |
| 特征模型 | 首选候选 `Facenet512` | 本地样本推理通过后固定，当前未下载权重 |
| 检测与对齐 | 候选 `opencv`，开启对齐 | 验证无脸、多脸和本人样本 |
| 输入归一化 | 候选 `base`，交由 DeepFace 执行 | 与最终模型组合一起验证和固定 |
| 推理引擎 | 优先验证 ONNX Runtime 路线 | 确认安装版本实际支持的 DeepFace 接口与权重 |
| 距离 | 余弦距离，数值越小越相似 | 确定后保持注册与查询约定一致 |
| 阈值 | 由固定模型配置及独立样本验证后确定 | 不在开发前指定通用数值 |
| 图像与视频读取 | OpenCV | 验证图片解码、视频及摄像头输入 |
| 窗口展示 | M1 根据 headless 依赖约束确定 | 单独验证窗口和中文文字显示，不默认可调用 `cv2.imshow` |

截至 2026-11-12，PyPI 查询到的 DeepFace 发布版本为 `0.0.104`，仅作为候选记录，尚未安装验证。下方源码引用固定到上游提交 `ce9ad19055c5671258f20a0abb07bd2ca86f5f42`，源码快照与实际安装包的行为需要分别核对。[发布信息](https://pypi.org/project/deepface/0.0.104/)

上游 `represent` 接口支持 BGR 数组输入和人脸特征输出；具体检测、对齐与归一化参数以固定版本为准。[DeepFace 接口源码](https://github.com/serengil/deepface/blob/ce9ad19055c5671258f20a0abb07bd2ca86f5f42/deepface/DeepFace.py)

上游 ONNX 会话支持按 CUDA、CPU 顺序选择执行后端，配置涉及 `DEEPFACE_ONNX_PROVIDERS`；应用还需验证实际模型执行，不能只根据显卡存在或后端名称判断加速成功。[ONNX 适配源码](https://github.com/serengil/deepface/blob/ce9ad19055c5671258f20a0abb07bd2ca86f5f42/deepface/commons/onnx_utils.py)

GPU 包需要匹配驱动及 CUDA、cuDNN 运行环境。GTX 1060 和 RTX 4070 移动版分别测试，版本在验证后记录，不直接照搬最新安装命令。[ONNX Runtime 兼容性说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

## 最小模块划分

| 模块 | 职责 | 负责人 |
| --- | --- | --- |
| `contracts.py` | 共享数据类型与校验约定 | 主聊天 |
| `engine.py` | 配置运行设备，加载模型，统一提取特征 | 主聊天先完成 |
| `enrollment.py` | 注册库建立、校验、保存与读取 | 注册任务 |
| `recognition.py` | 特征距离计算与单帧识别 | 识别任务 |
| `camera.py` | 视频源读取、单帧调用、窗口展示与资源释放 | 视频任务 |
| `__main__.py` | 参数解析与模块串联 | 主聊天 |

复用 DeepFace 的模型缓存，模型在进程中预热并重复使用。每次查询只提取查询帧的特征，注册特征启动时读取一次。

## 共享数据类型

采用普通数据类；持久化时转换为 JSON。以下字段是各模块的共同约定，不是当前已有代码。

### `PipelineConfig`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `model_name` | `str` | 特征模型名称 |
| `detector_backend` | `str` | 人脸检测器 |
| `align` | `bool` | 是否对齐 |
| `normalization` | `str` | 输入归一化配置 |
| `backend_engine` | `str` | 推理引擎标识 |
| `distance_metric` | `str` | 第一版固定为 `cosine` |

CPU/GPU 选择单独作为进程运行参数，不改变模型配置。变更模型、权重或特征处理方式后重新注册；跨引擎复用特征需先验证一致性。

### `FaceEmbedding`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `embedding` | `list[float]` | 有限数值组成的非零特征向量 |
| `bbox` | `tuple[int, int, int, int]` | 原始输入帧坐标中的 `x, y, w, h` |

### `Gallery`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | `int` | 第一版固定为 `1` |
| `identity` | `str` | 非空登记昵称，例如 `WuL0og` |
| `pipeline` | `PipelineConfig` | 提取这些特征时的配置 |
| `embedding_dim` | `int` | 从模型实际输出确定的正整数维数 |
| `embeddings` | `list[list[float]]` | 至少一个向量，长度均等于 `embedding_dim` |

JSON 文件采用 UTF-8。保存前完整校验，写入同目录临时文件后替换目标文件；第一版每个库只允许一个注册进程写入。加载时校验结构、版本、向量维数、数值有效性和配置兼容性。

### `RecognitionResult`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `status` | `str` | `matched`、`unknown`、`no_face`、`multiple_faces` 之一 |
| `identity` | `str \| None` | 仅 `matched` 时为登记昵称 |
| `bbox` | `tuple[int, int, int, int] \| None` | 恰好一张脸时提供，其他情况为 `None` |
| `distance` | `float \| None` | 单脸时提供最小余弦距离，其他情况为 `None` |
| `threshold` | `float` | 本次使用的阈值 |
| `elapsed_ms` | `float` | 单帧完整识别耗时，不包含首次模型加载 |

## 函数接口约定

以下签名是计划中的公共接口；类型定义由主聊天放入 `contracts.py`，任务 agent 使用这些定义，不各自复制。

```python
# engine.py：在导入并初始化推理引擎前选择设备，返回诊断信息。
def configure_runtime(device: str) -> dict: ...

# engine.py：输入为非空的 uint8、H×W×3 BGR 数组；无脸返回空列表。
def extract_embeddings(
    frame_bgr, *, pipeline: PipelineConfig
) -> list[FaceEmbedding]: ...

# enrollment.py：读取文件夹内的图片，完整成功后才保存并返回特征库。
def enroll(
    identity: str,
    images_dir: Path,
    gallery_path: Path,
    *,
    pipeline: PipelineConfig,
) -> Gallery: ...

# enrollment.py：严格校验并加载现有特征库。
def load_gallery(gallery_path: Path) -> Gallery: ...

# recognition.py：不读取或改写特征库文件，只处理当前帧。
def recognize(
    frame_bgr, gallery: Gallery, *, threshold: float
) -> RecognitionResult: ...

# camera.py：source 为摄像头整数索引或视频 Path，按帧调用识别函数。
def run_stream(
    source: int | Path, gallery: Gallery, *, threshold: float
) -> None: ...
```

图片路径在入口层解码为 BGR 数组。引擎保持检测启用，将明确的无脸结果映射为空列表；其他异常继续上抛，不能通过关闭检测把整幅图片当成人脸。

### 匹配规则

对于查询向量 `q` 和每个注册向量 `g`：

```text
d(q, g) = 1 - dot(q, g) / (norm(q) × norm(g))
d_min = 所有注册向量距离的最小值
d_min <= threshold 时匹配，否则为未知
```

阈值必须是 `[0, 2]` 内的有限数值。校验维数、非零范数和有限数值，浮点误差导致的余弦值越界仅按数值误差截断。阈值依据该匹配规则验证，多参考照片时重新检查误识表现。

## 设备和视频状态

- `cpu`：明确选择 CPU；`cuda`：显式要求 GPU；`auto`：优先可验证的 CUDA，失败时说明原因并使用 CPU。
- 设备初始化发生在模型第一次加载前；切换设备使用新进程，避免模型缓存继续使用旧设备。
- CUDA 验证以真实特征提取为准，保留会话及算子执行证据；初始化名称和显卡利用率仅是辅助信息。
- 视频第一版使用同步循环，先保证正确。若采样跳过推理，画面应显示“待识别”或上次结果的时间，不能把旧结果标成当前帧的新判断。
- 缩放或镜像用于展示时同步变换人脸框坐标；第一版可以直接使用原始帧，减少坐标变换。
- OpenCV 负责视频源操作，视频模块在异常和退出时都应释放设备，并关闭所选展示组件的窗口。[VideoCapture 官方文档](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)

## 命令行设计

当前只有 `--help`、`--version` 可运行，下表是后续验收目标。

| 子命令 | 必要参数 | 行为 |
| --- | --- | --- |
| `doctor` | `--device auto/cpu/cuda` | 检查环境并运行真实模型预热，记录设备与版本 |
| `enroll` | `--name`、`--images`、`--gallery`、`--device` | 创建或替换本人特征库 |
| `recognize` | `--image`、`--gallery`、`--threshold`、`--device` | 输出单图识别结果 |
| `stream` | `--camera` 或 `--video` 二选一，另需 `--gallery`、`--threshold`、`--device` | 视频或摄像头识别 |

`--device` 默认 `auto`；阈值第一版显式传入，在共享基线实测后再决定是否增加已验证默认值。路径相对于调用命令时的工作目录解释。

识别的四种正常状态退出码均为 `0`；无效参数、注册失败、模型故障和设备故障使用非零退出码及中文说明。错误路径不得覆盖已有特征库。
