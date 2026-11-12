# 开发与 worktree 协作计划

当前阶段：共享类型、模型引擎、诊断命令和环境记录已完成；当前开发机通过 CPU/CUDA 图片特征提取。共享基线提交后可开始任务 worktree，当前尚未创建。

## 文档职责

| 文档 | 内容 | 何时阅读 |
| --- | --- | --- |
| [需求](REQUIREMENTS.md) | 功能范围、设备分工和交付条件 | 开始任何功能任务前 |
| [技术与接口](DESIGN.md) | 模块签名、数据格式和匹配语义 | 编写或调整模块前 |
| [环境与验收](VALIDATION.md) | 环境关口、样本管理和验收方法 | 安装依赖、测试或演示前 |
| 本文 | 开发顺序、文件归属和 Git 操作 | 创建或合并任务 worktree 前 |

## 开发阶段

| 阶段 | 工作 | 完成条件 |
| --- | --- | --- |
| M0 文档 | 固定第一版范围和公共接口 | 已完成，提交 `51d3da7` |
| M1 共享基线 | 验证依赖与图片推理，实现公共类型和引擎 | 已实现并验证，取本阶段完成后的 `main` 提交为共同起点 |
| M2 并行开发 | 注册库、识别核心、视频展示 | 各任务相关检查通过并提交 |
| M3 集成 | 接通命令行，测试真实图片与本地视频 | 注册、加载、识别、展示形成闭环 |
| M4 目标机验收 | 配置 4070 电脑、摄像头与 GPU | 完成真实摄像头演示及设备证据记录 |
| M5 报告 | 整理原理、结果与复现说明 | 实验数据可追溯，交付资料与实际实现一致 |

M1 固定安装版本及共享类型后才从同一提交分出任务，避免各 agent 各自选择模型或数据格式。当前电脑 GPU 配置若需要单独处理，记录限制；最终 GPU 验收仍在 4070 电脑完成。

## 文件归属

| 任务 | 建议分支 | 可修改范围 | 验收重点 |
| --- | --- | --- | --- |
| 主聊天 | `main` | `contracts.py`、`engine.py`、`__main__.py`、`pyproject.toml`、公共文档与共享配置 | 类型与引擎、依赖、CLI 集成 |
| 注册任务 | `feat/enrollment` | `src/cnn_face/enrollment.py`、`tests/test_enrollment.py` | 单脸录入、整库替换、坏库校验、失败不覆盖 |
| 识别任务 | `feat/recognition` | `src/cnn_face/recognition.py`、`tests/test_recognition.py` | 距离边界、四种状态、维数与数值检查 |
| 视频任务 | `feat/camera` | `src/cnn_face/camera.py`、`tests/test_camera.py` | 视频与摄像头输入、展示、异常释放 |

共享模块已存在，三个任务模块及对应测试文件仍为计划路径。现有 `tests/test_contracts.py`、`tests/test_engine.py` 属于主聊天维护范围。只为真实逻辑编写必要检查；测试替身用于隔离硬件依赖，最终仍须实际模型和摄像头验收。

每个任务使用共享类型和引擎，不复制公共定义。需要变更接口、依赖或他人文件时，把原因与建议反馈主聊天，由主聊天协调更新。

## 创建 worktree 的操作示例

以下命令用于 M1 完成后的手动 Git 练习，当前不执行。以当前主目录 `E:\code\worktree` 为基准；开始前确认主目录干净，三个目标目录和分支尚未被占用。

```powershell
Set-Location E:\code\worktree
git status --short --branch
git log -1 --oneline

git worktree add -b feat/enrollment ..\cnn-face-enrollment main
git worktree add -b feat/recognition ..\cnn-face-recognition main
git worktree add -b feat/camera ..\cnn-face-camera main
git worktree list
```

这里的 `-b` 创建新分支，末尾 `main` 指定共同起点。同一个分支通常只在一个 worktree 检出；每个任务使用自己的目录和分支。[Git 官方文档](https://git-scm.com/docs/git-worktree)

应用管理的 worktree 通过应用提供的创建工具建立，实际位置以工具返回结果为准。不要同时创建同一任务的手动和应用管理目录。

`.venv`、数据、权重及注册库属于忽略文件，不会随着提交自动进入新 worktree。各目录独立安装环境；可通过显式绝对路径读取同一数据集，测试输出与可写注册库分别保存，避免任务互相覆盖。

可以设置 `DEEPFACE_HOME=E:\code\worktree\artifacts` 复用已校验的权重缓存。worktree 隔离文件，GPU 与显存仍由所有任务共享；当前 3GB 显存开发机的真实 GPU 检查串行安排，编码和不依赖硬件的测试可以并行。

## 给任务 agent 的说明模板

```text
任务：填写功能目标和验收编号。
工作目录：填写 worktree 的绝对路径。
预期分支：填写分支名；共同起点：填写实际提交哈希。
开始前确认目录、分支和 Git 状态，阅读 AGENTS.md 与三份设计文档。
只修改分配的模块及对应测试，按 DESIGN.md 使用共享接口。
代码和注释遵循中文、易读、规范的项目约定。
运行与修改相关的检查；真实推理或硬件条件不足时明确记录。
完成后提交修改，提交类型后的说明使用中文。
汇报修改文件、检查命令和结果、提交哈希、依赖事项及未完成项。
由主聊天统一合并。
```

## 集成顺序

1. 确认任务已提交、检查结果清楚，主目录没有未处理修改。
2. 检查注册分支，合并后验证注册库保存和读取。
3. 检查识别分支，合并后验证真实图片注册与查询闭环。
4. 检查视频分支，合并后验证本地视频，再到目标电脑验收摄像头。
5. 更新文档中的真实命令、依赖版本和验收结果。

注册分支的合并示例：

```powershell
Set-Location E:\code\worktree
git status --short --branch
git log --oneline main..feat/enrollment
git diff main...feat/enrollment
git merge --no-ff feat/enrollment -m "merge: 合并人脸注册与特征库功能"
```

发生冲突时先处理并检查，完成本次合并后再继续下一项。决定取消本次未完成合并时使用 `git merge --abort`。项目身份使用仓库级配置，昵称为 `WuL0og`；不更改全局 Git 身份。

## 收尾

任务全部合并、检查通过且 agent 已停止写入后，移除对应 worktree，再删除已合并分支。手动 worktree 使用 `git worktree remove` 和 `git branch -d`；应用管理的 worktree 使用对应归档工具管理。清理前确认数据与未提交文件已妥善保留。

M0、M1 和后续功能各自使用可审查的提交。已经完成的初始提交保留，后续文档和功能形成新提交。
