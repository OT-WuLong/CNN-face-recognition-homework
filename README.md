# CNN 人脸识别课程设计

使用 DeepFace 的预训练 CNN 提取人脸特征，完成本人照片注册、图片识别、本地视频与摄像头识别，无需自行训练模型。

目前主链路已集成，当前开发机通过 CPU/CUDA 技术验证；本人照片、阈值校准和 RTX 4070 移动版电脑的摄像头验收待完成。

## 从哪里开始

- **[使用指南](docs/USAGE.md)**：安装 → 准备照片 → 注册 → 校准阈值 → 图片、视频与摄像头识别。
- [技术说明](docs/DESIGN.md)：功能范围、模块接口、匹配规则与 worktree 协作约定。
- [测试与验收](docs/VALIDATION.md)：已有验证结论、复现命令、统计口径及待验收事项。

当前电脑已安装环境，直接从使用指南的“准备运行环境”开始，跳过“新电脑首次安装”。

## 项目结构

```text
src/cnn_face/                 注册、识别、视频展示与命令入口
tests/                        自动检查及手动真实模型检查
docs/                         使用、技术和验收文档
data/                         本地照片与视频
artifacts/                    权重缓存、注册库和运行报告
AGENTS.md                     开发约定
pyproject.toml                包元数据与直接依赖
requirements-win-cu124.lock   已验证的 Windows / Python 3.12 CUDA 环境
```

数据、特征库、权重、报告和虚拟环境已由 Git 忽略，迁移电脑或使用其他 worktree 时分别准备。

参考项目：[DeepFace](https://github.com/serengil/deepface)。引用或复制上游代码时保留来源和相应许可要求。
