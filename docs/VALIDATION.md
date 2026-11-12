# 环境准备与验收

状态：2026-11-12 已在当前开发机完成共享基线验证。依赖、CPU/CUDA 真实图片特征提取及设备选择已通过；本人识别准确率、完整视频窗口和 4070 摄像头验收仍待后续功能完成。

## 已能运行的检查

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m cnn_face --help
.\.venv\Scripts\python.exe -m cnn_face --version
.\.venv\Scripts\python.exe -m cnn_face doctor --device cpu --report artifacts/reports/doctor-cpu.json
.\.venv\Scripts\python.exe -m cnn_face doctor --device cuda --report artifacts/reports/doctor-cuda.json
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git status --short --branch
```

新环境按 [README](../README.md) 安装；完整版本记录在 [Windows CUDA 环境锁定文件](../requirements-win-cu124.lock)。当前安装已包含 DeepFace 与 PyTorch CUDA 构建。

## 依赖验证关口

当前电脑已完成以下共享基线检查；迁移到目标电脑时重做设备、依赖与模型检查。

1. 检查目标电脑的操作系统、Python、GPU、显存和驱动。当前机已检测到 GTX 1060 3GB；目标机 RTX 4070 移动版的信息仍需到机验证。
2. 确认所选 DeepFace 发布包的接口和权重格式，完成一张图片的 CPU 特征提取，记录向量维数。
3. 按运行库兼容表选择 GPU 依赖，在独立进程完成同模型 CUDA 推理。记录配置的执行后端和实际执行证据；CUDA 不可用时显式 CUDA 模式应失败。
4. 保留发布包依赖的 headless OpenCV，展示层采用 Tkinter/Pillow。已通过隐藏窗口、图片对象及宋体字体初始化检查，完整视频界面待实现。headless 包不提供 `cv2.imshow` 所需桌面 GUI。[发布包信息](https://pypi.org/project/deepface/0.0.104/)、[OpenCV 包说明](https://pypi.org/project/opencv-contrib-python-headless/)
5. 记录权重下载位置、文件名与 SHA-256。需要离线演示时先准备缓存，再断网验证完整启动和推理。
6. 固定验证通过的直接依赖，在两台电脑分别记录运行环境。使用独立虚拟环境，不复制另一台电脑的 `.venv`。

当前固定 PyTorch `2.6.0+cu124` 运行 CNN，驱动 560.94 的 GTX 1060 已通过验证。官方构建安装方式见 [PyTorch 版本说明](https://pytorch.org/get-started/previous-versions/)。驱动报告的 CUDA 能力不等于运行库已安装；GPU 验收还应核对网络前向输入、输出与参数设备。

GPU 和 CPU 共用固定模型及配置。ONNX Runtime 仅作为 DeepFace 的发布依赖保留，当前 CNN 使用 PyTorch 后端。

### 环境记录表


| 项目                  | 当前开发机           | 最终演示机                        |
| ----------------------- | ---------------------- | ----------------------------------- |
| GPU                   | 已检测：GTX 1060 3GB | 用户提供：RTX 4070 移动版，待核实 |
| 摄像头                | 用户确认没有         | 用户确认有，待实测                |
| Python                | 已检测：3.12.7       | 待核实                            |
| 驱动                  | 已检测：560.94       | 待核实                            |
| DeepFace / 推理库版本 | 0.0.104 / PyTorch 2.6.0+cu124 | 待安装验证 |
| 权重文件及摘要 | 已准备，见下方权重记录 | 待迁移或下载 |
| CPU 图片推理 | 已通过，两张测试图片各提取一个 512 维特征 | 未验证 |
| CUDA 图片推理 | 已通过，前向张量与参数均在 `cuda:0` | 未验证 |
| 图形窗口 / 摄像头识别 | Tkinter/Pillow 初始化通过；完整界面待实现 | 未验证 |

## 共享基线实测记录

- `pip check` 通过，15 项自动检查通过，覆盖共享类型、中文路径解码、输入校验、无脸转换、错误传播与进程内设备切换限制。
- 两张上游测试图片分别检测到一张脸；拼接图片检测到两张脸；纯色图片检测到零张脸。CPU 与 CUDA 路径均通过这些场景。
- 同两张图片的 CPU/CUDA 特征余弦相似度均大于 `0.999999`，最大绝对分量差约为 `3.34e-6`；这是本轮一致性检查，不是身份识别准确率。
- 隐藏 CUDA 设备后，自动模式明确提示回退并完成 CPU 预热，显式 CUDA 模式返回退出码 `2`。
- CUDA 诊断记录网络参数、前向输入与输出张量的 `cuda:0` 位置，以及 CUDA Event 时间。当前 PyTorch 构建只提供 CPU profiler 活动，未采集到 CUDA kernel trace；不把这项分析能力写成已通过。
- 本轮是功能与执行设备验证，未形成可报告的摄像头帧率、稳定延迟分位数或加速倍数。

本地证据位于 `artifacts/reports/`：`doctor-cpu.json`、`doctor-cuda.json`、`doctor-auto-cpu.json`、`model-smoke-cpu.json`、`model-smoke-cuda.json` 与 `cpu-cuda-consistency.json`。这些报告和测试特征属于忽略产物，不进入 Git；关键结论保存在本文。

### 权重准备

- 文件：`artifacts/.deepface/weights/facenet512_weights.pth`，大小 `94299192` 字节。
- SHA-256：`5b0eb002a6033d3d770f4157783229aa1d7639c943353877e67ab06fa927d28d`。
- 校验依据：[DeepFace 官方模型发布](https://github.com/serengil/deepface_models/releases/tag/v1.0) 中对应资产的摘要。引擎每次初始化都会校验此值。
- 默认根目录是运行命令时的 `artifacts/`；设置 `DEEPFACE_HOME` 时，实际权重位于该根目录下的 `.deepface/weights/`。

当前网络下官方直连下载曾失败，使用镜像传输后与官方摘要核对一致。迁移时可以复制已校验的权重文件；自动下载不通时，可在项目根目录执行下列准备命令：

```powershell
New-Item -ItemType Directory -Force -Path artifacts/.deepface/weights | Out-Null
curl.exe --fail --location --output artifacts/.deepface/weights/facenet512_weights.pth 'https://hf-mirror.com/serengil/deepface/resolve/main/facenet512_weights.pth'
if ($LASTEXITCODE -ne 0) { throw '模型下载失败' }
$weightHash = (Get-FileHash artifacts/.deepface/weights/facenet512_weights.pth -Algorithm SHA256).Hash.ToLowerInvariant()
if ($weightHash -ne '5b0eb002a6033d3d770f4157783229aa1d7639c943353877e67ab06fa927d28d') { throw '模型校验失败' }
```

### 复现真实图片检查

测试图片来自 DeepFace 的上游测试目录，仅用于依赖和模型检查，不属于本人的注册或最终验收数据。[测试数据目录](https://github.com/serengil/deepface/tree/ce9ad19055c5671258f20a0abb07bd2ca86f5f42/tests/unit/dataset)

```powershell
New-Item -ItemType Directory -Force -Path data/smoke | Out-Null
curl.exe --fail --location --output data/smoke/img1.jpg 'https://raw.githubusercontent.com/serengil/deepface/ce9ad19055c5671258f20a0abb07bd2ca86f5f42/tests/unit/dataset/img1.jpg'
.\.venv\Scripts\python.exe -m cnn_face doctor --device cuda --image data/smoke/img1.jpg --report artifacts/reports/doctor-cuda.json
```

报告中的 `warmup_source=synthetic` 指网络预热；存在 `image_check` 时才表示额外执行了真实图片检查。无脸图片返回零张人脸是正常结果，需结合所用样本解释。

## 样本准备

建议第一轮准备几张不同角度和光照的本人注册照片，保持单人、清晰可见。数量是开发建议，最终样本数量如实记录。

```text
data/
├── enroll/WuL0og/       本人注册照片
├── validation/self/    本人阈值调整样本
├── validation/other/   他人阈值调整样本
├── test/self/          最终本人测试
├── test/other/         最终他人测试
├── test/no_face/       无脸图片
├── test/multiple/      多脸图片
└── videos/             本地测试视频

artifacts/
├── gallery.json        注册库
└── reports/            环境与实验记录
```

注册与测试使用不同原图和不同拍摄会话。最终测试之前锁定模型、注册库、距离度量和阈值；如根据测试结果调整配置，应另取样本重新测试。

## 功能验收矩阵

下表是完整应用的验收目标；设备路径及部分共享约定已在基线阶段覆盖，注册、身份判定、视频及摄像头仍待实现。测试替身可以验证模块逻辑，真实模型推理和摄像头不能由替身结果代替。


| 编号 | 场景                                   | 通过条件                                              |
| ------ | ---------------------------------------- | ------------------------------------------------------- |
| F01  | 合法单脸照片注册                       | 保存昵称、配置与完整向量，能重新加载                  |
| F02  | 注册文件夹为空、坏图、无脸或多脸       | 报告具体原因，已有库内容不变                          |
| F03  | 重新注册                               | 整体替换，不意外追加旧样本                            |
| F04  | 本人独立测试照片                       | 输出匹配或误拒记录，昵称仅在匹配时出现                |
| F05  | 他人单脸照片                           | 输出未知或误识记录，不掩盖失败样本                    |
| F06  | 无脸 / 多脸图片                        | 分别返回 `no_face` / `multiple_faces`，不进行身份判定 |
| F07  | 坏库、错版本、错维数、非有限值、零向量 | 明确拒绝；不返回未知掩盖错误                          |
| F08  | 距离恰好等于阈值及阈值两侧             | 严格符合 `distance <= threshold` 的匹配规则           |
| F09  | 本地视频                               | 按相同识别规则输出，读到结尾正常释放资源              |
| F10  | 摄像头不存在或被占用                   | 中文提示，非零退出，资源释放                          |
| F11  | 显式 CPU / CUDA / 自动回退             | 与设备选择语义一致，真实执行设备有记录                |
| F12  | 正常退出 / 运行异常                    | 摄像头释放，窗口关闭，无重复加载模型                  |
| F13  | 4070 电脑摄像头本人、他人和空画面      | 有真实运行记录，显示状态与当次识别对应                |

## 统计口径

每个最终测试样本只计入一次；连续视频按预先声明的采样规则统计，说明帧之间存在相关性。

- 本人识别成功率：输出本人昵称的本人样本数 / 本人样本总数。
- 本人端到端失败率：未输出本人昵称的本人样本数 / 本人样本总数，包含检测失败。
- 本人误拒率：成功检测为单脸但返回未知的本人样本数 / 成功检测为单脸的本人样本数。
- 他人误识率：被错误判为本人的他人样本数 / 他人样本总数。
- 单独报告无脸、多脸、坏图和模型错误计数，不把技术失败当作正确拒绝。
- 分母为零时记录“不适用”，同时保留原始计数；不填 0% 冒充已验证。

初期样本少时以完整结果表为主，不把少量测试结果写成模型通用准确率。数值目标在首轮基线后确定，目前不虚构达标比例。

## 性能与报告

- 将安装、下载、首次加载与稳定运行耗时分开记录。
- 同一批样本和同一流水线配置下比较 CPU、GPU。先预热，再测多次；GPU 测量应等待推理结果就绪。
- 报告单帧识别延迟的中位数和 P95，注明样本数、分辨率、模型及实际执行设备。
- 区分预览帧率、实际识别频率与端到端延迟；跳过推理的帧不算一次识别。
- 摄像头验收记录昵称、画面状态、运行参数、目标电脑环境和典型失败条件。

实验记录至少包含：日期、Git 提交哈希、设备、依赖与权重版本、注册和测试样本数量、配置、阈值确定方法、原始计数、性能结果和失败示例。
