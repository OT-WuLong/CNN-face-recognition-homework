# 测试与验收

本文保存当前代码的验证结论与复现方法。安装、照片准备和操作命令统一见 [使用指南](USAGE.md)。

## 已验证的环境和结果

2026-11-12 的真实集成技术检查对应实现提交 `6998f08`，后续提交同步了文档。原始证据保存在集成 worktree 的忽略目录中；本次文档整理未重新执行模型或桌面验收。

| 项目 | 记录 |
| --- | --- |
| 开发机 | Windows、Python 3.12.7、GTX 1060 3GB、驱动 560.94，无摄像头 |
| 模型与运行库 | DeepFace 0.0.104、Facenet512、PyTorch 2.6.0+cu124，固定依赖一致 |
| 自动检查 | 集成时 71 项通过；依赖一致性、编译、帮助和版本检查通过 |
| 真实 CPU/CUDA | 注册、保存、重载、整体重新注册及四种识别状态均通过 |
| 原库保护 | 空目录、坏图、无脸和多脸注册失败时，原库字节不变 |
| 设备选择 | 隐藏 CUDA 后 auto 回退 CPU；显式 cuda 返回错误 |
| 真实视频 | 640×480 MJPG 视频 48 帧，真实解码、识别和窗口循环到结尾，资源释放 |
| 目标机 | RTX 4070 移动版及摄像头由用户提供信息，到机验收待完成 |

自动检查使用替身隔离模型和桌面，覆盖类型与输入边界、严格库校验、最小距离和阈值、四种状态、视频颜色与框坐标、旧结果、资源清理及命令入口；数量不能作为真实硬件验收数量。

真实人脸前向的参数、输入与输出在 CPU 检查中均为 `cpu`，在 GPU 检查中均为 `cuda:0`。两张上游样本的 CPU/CUDA 特征余弦相似度均大于 `0.999999`，最大分量差约 `3.34e-6`，仅说明本批技术样本的执行一致性。

技术脚本使用阈值 `0` 和 `2` 覆盖未知与匹配分支，不用于本人判定。当前 PyTorch 构建未提供 CUDA profiler 活动；设备证据来自真实前向张量及 CUDA Event，不声称取得了 GPU kernel trace。

当前没有本人识别准确率、误拒率、误识率、摄像头稳定帧率或加速倍数的达标结论。

## 常规回归

在项目根目录执行，先准备使用指南中的环境变量：

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m cnn_face --help
.\.venv\Scripts\python.exe -m cnn_face enroll --help
.\.venv\Scripts\python.exe -m cnn_face recognize --help
.\.venv\Scripts\python.exe -m cnn_face stream --help
git diff --check
```

## 复现真实模型技术检查

使用上游 `img1.jpg`、`img2.jpg`，这些图片及其拼接、缩放和视频衍生物属于同一技术测试集合，不纳入本人的校准或最终测试。[上游测试数据](https://github.com/serengil/deepface/tree/ce9ad19055c5671258f20a0abb07bd2ca86f5f42/tests/unit/dataset)

```powershell
New-Item -ItemType Directory -Force -Path data/smoke | Out-Null
curl.exe --fail --location --output data/smoke/img1.jpg 'https://raw.githubusercontent.com/serengil/deepface/ce9ad19055c5671258f20a0abb07bd2ca86f5f42/tests/unit/dataset/img1.jpg'
if ($LASTEXITCODE -ne 0) { throw '第一张技术样本下载失败' }
curl.exe --fail --location --output data/smoke/img2.jpg 'https://raw.githubusercontent.com/serengil/deepface/ce9ad19055c5671258f20a0abb07bd2ca86f5f42/tests/unit/dataset/img2.jpg'
if ($LASTEXITCODE -ne 0) { throw '第二张技术样本下载失败' }

# 前一个进程成功退出后再运行下一个，避免显存争用。
.\.venv\Scripts\python.exe tests/manual_smoke.py --device cpu --samples data/smoke --output artifacts/integration/cpu
.\.venv\Scripts\python.exe tests/manual_smoke.py --device cuda --samples data/smoke --output artifacts/integration/cuda
```

增加 `--stream` 会打开真实窗口，检查合成技术视频到结尾与资源释放；需要操作者观察时再手动执行。本轮历史检查通过了视频自然结尾；不能替代真实按键退出和窗口目视验收。

原始集成证据位置：`C:\Users\Admin\.codex\worktrees\cnn-face-integration\worktree\artifacts`。其中 `integration/cpu/report.json`、`integration/cuda/report.json`、`integration/consistency.json` 记录真实模型结果，`integration/cli-results.json` 和 `cli-errors.json` 记录命令行检查，`reports/integration-tests.txt` 记录自动检查。该工作区若已清理，需通过上述命令重新生成；Git 保存代码及本文的结论摘要。

## 仍需完成的验收

| 场景 | 完成条件 |
| --- | --- |
| 本人注册 | 使用本人单脸照片建立并重载特征库，数量正确 |
| 阈值校准 | 本人和他人的独立校准样本有距离记录，确定并固定阈值 |
| 最终图片测试 | 使用独立测试集合记录正确匹配、误拒、误识、无脸、多脸及错误计数 |
| 窗口目视 | 中文、颜色、人脸框位置正确，Esc/Q/关闭窗口可退出并释放设备 |
| 4070 环境 | 重新安装依赖、检查权重摘要，并取得实际 CUDA 执行证据 |
| 真实摄像头 | 本人、他人、空画面及设备异常的真实运行记录 |
| 离线与性能 | 需要离线演示时断网复验；另测稳定延迟和识别频率 |

## 数据与统计口径

注册、阈值校准、最终测试使用不同原图和拍摄会话；同一照片及衍生图、同一连续视频的相邻帧保持在同一集合。测试前固定模型、注册库和阈值；根据测试结果调整后，应另取数据重新测试。

| 指标 | 计算方法 |
| --- | --- |
| 本人识别成功率 | 输出本人昵称的本人样本数 / 本人样本总数 |
| 本人端到端失败率 | 未输出本人昵称的本人样本数 / 本人样本总数，包含检测失败 |
| 本人误拒率 | 单脸检测成功但返回未知的本人样本数 / 单脸检测成功的本人样本数 |
| 他人误识率 | 错误判为本人的他人样本数 / 他人样本总数 |

单独报告检测失败、坏图和模型错误，不把技术失败当作正确拒绝；分母为零记录“不适用”。保留样本数量和原始计数，少量样本结果不能当作模型通用准确率。

性能检查应分开记录首次加载与稳定推理，统一样本和配置、预热后多次测量，并等待 GPU 结果就绪。区分预览帧率、实际识别频率与端到端延迟，报告中位数和 P95 及样本数。

实验记录包含日期、Git 提交、设备、依赖及权重版本、数据规模、流水线配置、阈值确定方法、原始计数、性能结果和失败示例。
