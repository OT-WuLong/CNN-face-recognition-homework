# CNN 人脸识别课程设计

使用预训练 CNN，通过 [DeepFace](https://github.com/serengil/deepface) 提取人脸特征，完成本人注册、图片识别与电脑摄像头识别。无需自行训练或微调模型。

当前阶段：已完成需求文档、共享数据类型、模型引擎和 `doctor` 诊断命令；当前开发机已通过 CPU/CUDA 图片特征提取验证。注册、身份判定和视频展示由后续任务实现。

## 文档导航

- [需求与项目范围](docs/REQUIREMENTS.md)：确定做什么、交付什么。
- [技术与接口设计](docs/DESIGN.md)：统一模块、特征库格式和匹配规则。
- [开发与 worktree 协作计划](docs/PLAN.md)：开发顺序、文件分工和任务说明模板。
- [环境准备与验收](docs/VALIDATION.md)：依赖验证、样本准备、测试与实验口径。

文档中标为计划的功能尚未实现；当前可运行帮助、版本与 `doctor` 命令。

## 开发环境

- Python 3.12（当前项目代码支持 Python 3.11 及以上）。
- 使用项目内 `.venv`，避免影响系统 Python。
- 使用 DeepFace 0.0.104、Facenet512 和 PyTorch 2.6.0；当前已验证 CUDA 12.4 构建，并保留 CPU 运行路径。
- 当前开发机为 NVIDIA GeForce GTX 1060 3GB，驱动 560.94，CPU 与 CUDA 图片推理均已验证。
- 目标演示机配有 RTX 4070 移动版和摄像头（用户提供），作为 GPU 摄像头演示的主要验收设备；显存、驱动和实际运行效果待在该电脑确认。
- 当前电脑没有摄像头，先用本地照片或视频开发；两台电脑分别建立虚拟环境并验证运行依赖。
- 模型首次运行需要下载约 94MB 权重；默认缓存位于当前工作目录的 `artifacts/.deepface/weights/`，可通过 `DEEPFACE_HOME` 指定缓存根目录。

在项目根目录运行以下 PowerShell 命令：

```powershell
python -m venv .venv
# 先安装官方 CUDA 构建，再复用已验证的 Windows / Python 3.12 依赖。
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-win-cu124.lock
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m cnn_face --help
.\.venv\Scripts\python.exe -m cnn_face --version
.\.venv\Scripts\python.exe -m cnn_face doctor --device cuda --report artifacts/reports/doctor-cuda.json
```

以上安装命令用于新环境；当前电脑已完成安装。其他电脑先检查驱动兼容性，再按相同流程验证。官方 CUDA 包同时支持 CPU 推理；使用 `--device cpu` 明确选择 CPU，使用 `--device auto` 允许自动回退。

`doctor` 默认使用合成输入检查模型执行。增加 `--image 本地图片路径` 可同时验证真实人脸检测与特征提取，报告不输出人脸特征向量。权重下载问题、样本复现和具体环境记录见 [环境准备与验收](docs/VALIDATION.md)。

## 目录约定

```text
src/cnn_face/       Python 包，后续实现注册、识别与摄像头处理
tests/              共享约定与引擎边界的自动检查
docs/              设计与协作说明
data/              本地数据集，不提交人脸图片
artifacts/         运行时生成的权重、日志和评估结果，不纳入 Git
AGENTS.md          agent 协作规则
pyproject.toml     包元数据和命令行入口
requirements-win-cu124.lock  已验证的 Windows / Python 3.12 CUDA 环境
```

`data/`、`artifacts/` 中的实际文件以及 `.venv/` 不会随提交进入其他 worktree，需要分别准备。

## 多 agent 开发

主分支为 `main`。先确定数据与函数约定，再按任务分配 worktree，由主聊天统一检查和合并。具体拆分见 [开发计划](docs/PLAN.md)。

仓库独立维护；借鉴或复制上游代码时记录来源并保留其许可要求。
