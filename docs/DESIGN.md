# 技术说明

操作步骤见 [使用指南](USAGE.md)，验证结果见 [测试与验收](VALIDATION.md)。实现主链路已完成，后续维护以代码和本文中的约定为准。

## 功能范围

系统面向一个已登记身份，处理单人画面。支持多张本人照片注册、单张图片识别、本地视频及摄像头展示。

注册保存预训练 CNN 提取的特征，不更新模型权重。单脸查询与注册库比对；无脸、多脸分别返回独立状态，多脸画面不执行身份判定。多身份管理、多人同时识别、活体检测、训练与微调不属于当前交付范围。

```text
注册照片 → 检测与对齐 → 预训练 CNN → 本人特征库
                                             ↓
图片或视频帧 → 相同特征提取流程 → 距离比对 → 昵称 / 未知
```

## 固定的模型配置

| 配置 | 当前值 |
| --- | --- |
| 特征模型 | `Facenet512`，输出 512 维 |
| 推理后端 | PyTorch，支持 CPU 和兼容的 CUDA 环境 |
| 检测器 / 对齐 | `opencv` / 开启 |
| 输入归一化 | `base` |
| 比对方式 | 余弦距离 |
| 视频读取 / 展示 | headless OpenCV / Tkinter 与 Pillow |

依赖以 [pyproject.toml](../pyproject.toml) 和 [环境锁定文件](../requirements-win-cu124.lock) 为准。CPU ONNX Runtime 保留为 DeepFace 发布依赖，当前 CNN 在 PyTorch 后端执行。

引擎固定校验官方 `facenet512_weights.pth` 的 SHA-256。更换模型、权重或特征处理方式时，应更新共享配置、重新注册并重新校准阈值；跨引擎复用特征需先验证一致性。

## 模块与接口

| 模块 | 主要职责与公共入口 |
| --- | --- |
| [contracts.py](../src/cnn_face/contracts.py) | `PipelineConfig`、`FaceEmbedding`、`Gallery`、`RecognitionResult` 及公共校验 |
| [engine.py](../src/cnn_face/engine.py) | `configure_runtime(device)`、`read_image(path)`、`extract_embeddings(frame_bgr, *, pipeline)` |
| [enrollment.py](../src/cnn_face/enrollment.py) | `enroll(identity, images_dir, gallery_path, *, pipeline)`、`load_gallery(gallery_path)` |
| [recognition.py](../src/cnn_face/recognition.py) | `recognize(frame_bgr, gallery, *, threshold)` |
| [camera.py](../src/cnn_face/camera.py) | `run_stream(source, gallery, *, threshold)`，整数源为摄像头，`Path` 为视频 |
| [__main__.py](../src/cnn_face/__main__.py) | 参数解析、统一初始化模型、串联业务模块及错误退出 |

入口每个进程调用一次 `configure_runtime`，选择设备、加载模型并预热。各模块复用引擎及共享类型，不自行复制定义或重复加载模型。

图像统一为非空的 `uint8`、`H×W×3` BGR 数组；显示副本转为 RGB。人脸框采用原始帧坐标 `(x, y, w, h)`，缩放或镜像展示时同步转换坐标。

## 特征库与状态约定

- 注册库是 UTF-8 JSON，字段为 `schema_version`、`identity`、`pipeline`、`embedding_dim`、`embeddings`；版本为 `1`，字段类型以 `Gallery` 定义为准。
- 注册目录只扫描第一层受支持的图片，按文件名排序，每张图片恰好一张脸。全部成功后，使用同目录临时文件与原子替换保存；重新注册整体替换，失败保留旧库。
- 加载时拒绝缺失或额外字段、重复 JSON 键、错误版本或配置、空向量、零向量、非有限数值及不一致维数。
- 读取失败、库损坏和模型故障通过异常报告，不能伪装成未知身份。

| 状态 | 含义 | 返回约定 |
| --- | --- | --- |
| `matched` | 单脸且距离不大于阈值 | 提供昵称、人脸框和距离 |
| `unknown` | 单脸且距离大于阈值 | 昵称为空，提供人脸框和距离 |
| `no_face` | 未检测到脸 | 昵称、人脸框和距离均为空 |
| `multiple_faces` | 检测到多张脸 | 昵称、人脸框和距离均为空 |

四种正常结果都带有本次阈值和识别耗时，退出码为 `0`；参数或运行错误使用非零退出码。注册摘要、图片结果及 doctor 报告输出 JSON；业务命令的设备证据输出到标准错误。

## 匹配与设备规则

查询向量与每个注册向量计算余弦距离，取最小值。距离越小越相似，`distance <= threshold` 时匹配，距离不是身份概率。阈值须为 `[0, 2]` 内的有限数值，并通过独立样本确定。

识别先校验输入、向量维数和数值，再进行比对。归一化采用数值稳定方式处理极端有限分量，仅截断浮点误差造成的余弦越界。

- `cpu`：明确选择 CPU。
- `cuda`：明确要求 GPU；不可用或模型无法在 GPU 执行时报错。
- `auto`：优先可验证的 CUDA，失败时说明原因并回退 CPU。

同一进程初始化后，切换设备需要启动新进程。GPU 证据包含网络参数、前向输入和输出的实际设备，不能只根据显卡存在判断成功。

视频逐帧同步识别，处理当前帧期间退出事件可能需等待。旧画面与旧状态一同保留至新结果就绪。所有退出路径释放视频源并关闭窗口；无可靠帧数元数据时，播放后返回读取失败无法可靠区分正常结尾与解码中断。

## 协作约定

- 先明确任务范围与共同提交，再为每个任务创建独立分支和 worktree；开始前核对目录、分支、基线与 Git 状态。
- 为任务约定文件归属，公共接口、依赖和命令入口由集成负责人协调；需要跨范围修改时先说明原因。
- 每个 worktree 使用自己的虚拟环境；可用 `DEEPFACE_HOME` 指向已校验的共享权重缓存，可写注册库和报告分别保存。
- worktree 隔离文件，GPU 及显存仍共享；当前 3GB 显存机器的真实 GPU 检查串行安排，普通测试可并行。
- 各任务检查通过并提交后，由集成负责人逐项审查、合并和验证，最后更新主分支。提交说明使用中文，身份沿用仓库本地配置。
- 已完成任务的分支与 worktree 在确认无未保存内容后再清理；应用管理的 worktree 使用对应归档工具管理。
