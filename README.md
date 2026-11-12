# CNN 人脸识别课程设计

从零搭建并训练 CNN，完成受控人脸库中的身份分类；参考 [DeepFace](https://github.com/serengil/deepface) 的人脸识别流程，并在后续实验中作为预训练模型对照。

当前阶段：项目初始化。尚未实现数据处理、CNN、训练或人脸识别功能。

## 开发环境

- Python 3.12（当前项目代码支持 Python 3.11 及以上）。
- 使用项目内 `.venv`，避免影响系统 Python。
- 计划使用 PyTorch 实现自建 CNN；待确认硬件和数据集后配置训练依赖。
- DeepFace 对照实验后续独立配置，初始化阶段不下载框架、模型权重或人脸数据。

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
src/cnn_face/       Python 包，后续实现数据处理、模型、训练和评估
docs/              设计与协作说明
data/              本地数据集，不提交人脸图片
artifacts/         运行时生成的权重、日志和评估结果，不纳入 Git
AGENTS.md          agent 协作规则
pyproject.toml     包元数据和命令行入口
```

`data/`、`artifacts/` 中的实际文件以及 `.venv/` 不会随提交进入其他 worktree，需要分别准备。

## 课程目标与实验边界

1. 使用自己定义的 CNN 和训练循环，保留模型结构、训练配置与实验记录。
2. 第一版针对已登记身份进行分类。未知身份拒识、摄像头演示作为后续扩展。
3. 先按数据来源或拍摄会话划分训练、验证和测试数据，再做增强；同一原图及其衍生图属于同一集合。
4. 固定划分与随机种子，记录分类准确率、每类指标、混淆矩阵和推理耗时。
5. DeepFace 对照使用相同测试数据和明确的评估协议，区分闭集分类与 1:1 验证，单独报告结果。

## 多 agent 开发

主分支为 `main`。先确定数据与函数约定，再按任务分配 worktree，由主聊天统一检查和合并。具体拆分见 [开发计划](docs/PLAN.md)。

仓库独立维护；借鉴或复制上游代码时记录来源并保留其许可要求。
