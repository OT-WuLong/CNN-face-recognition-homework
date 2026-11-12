# CNN 人脸识别课程设计

使用预训练 CNN，通过 [DeepFace](https://github.com/serengil/deepface) 提取人脸特征，完成本人注册、图片识别与电脑摄像头识别。无需自行训练或微调模型。

当前阶段：已完成项目初始化，已编写需求、技术与接口设计及验收计划，尚未实现人脸注册与识别。

## 文档导航

- [需求与项目范围](docs/REQUIREMENTS.md)：确定做什么、交付什么。
- [技术与接口设计](docs/DESIGN.md)：统一模块、特征库格式和匹配规则。
- [开发与 worktree 协作计划](docs/PLAN.md)：开发顺序、文件分工和任务说明模板。
- [环境准备与验收](docs/VALIDATION.md)：依赖验证、样本准备、测试与实验口径。

上述文档中标为计划的模块和子命令尚未实现；目前仅下方基础环境命令可运行。

## 开发环境

- Python 3.12（当前项目代码支持 Python 3.11 及以上）。
- 使用项目内 `.venv`，避免影响系统 Python。
- 计划支持兼容环境下的 GPU 加速，并保留 CPU 运行路径；具体模型和推理依赖在下一阶段测试后确定。
- 当前开发机检测到 NVIDIA GeForce GTX 1060 3GB，尚未安装和验证 GPU 推理环境；需根据显卡、驱动和运行库选择兼容版本。
- 目标演示机配有 RTX 4070 移动版和摄像头（用户提供），作为 GPU 摄像头演示的主要验收设备；显存、驱动和实际运行效果待在该电脑确认。
- 当前电脑没有摄像头，先用本地照片或视频开发；两台电脑分别建立虚拟环境并验证运行依赖。
- 初始化阶段不下载框架、模型权重或人脸数据；后续记录首次下载与本地缓存的准备方式。

在项目根目录运行以下 PowerShell 命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m cnn_face --help
.\.venv\Scripts\python.exe -m cnn_face --version
```

以上命令用于首次安装；已有 `.venv` 时直接使用其中的 Python。命令行目前仅提供帮助和版本信息。

## 目录约定

```text
src/cnn_face/       Python 包，后续实现注册、识别与摄像头处理
docs/              设计与协作说明
data/              本地数据集，不提交人脸图片
artifacts/         运行时生成的权重、日志和评估结果，不纳入 Git
AGENTS.md          agent 协作规则
pyproject.toml     包元数据和命令行入口
```

`data/`、`artifacts/` 中的实际文件以及 `.venv/` 不会随提交进入其他 worktree，需要分别准备。

## 多 agent 开发

主分支为 `main`。先确定数据与函数约定，再按任务分配 worktree，由主聊天统一检查和合并。具体拆分见 [开发计划](docs/PLAN.md)。

仓库独立维护；借鉴或复制上游代码时记录来源并保留其许可要求。
