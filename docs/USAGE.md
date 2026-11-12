# 使用指南

完整流程：准备环境 → 检查模型 → 放入本人照片 → 注册 → 校准阈值 → 图片测试 → 视频或摄像头识别。

当前电脑已安装环境并验证 CPU/GPU，使用时跳过第 2 节。首次在另一台电脑运行时，按顺序完成安装。下面各节命令在同一个 PowerShell 窗口执行；新开窗口后先重新执行第 1 节。

## 1. 准备运行环境

```powershell
# 另一台电脑请改成它的项目目录。
Set-Location 'E:\code\worktree'
$env:PYTHONUTF8 = '1'
$py = Join-Path (Get-Location).Path '.venv\Scripts\python.exe'
$env:DEEPFACE_HOME = Join-Path (Get-Location).Path 'artifacts'
$gallery = 'artifacts/gallery-wul0og.json'
$device = 'cuda'
```

`cuda` 明确使用 GPU，不能使用时报错；`cpu` 明确使用 CPU；`auto` 优先 GPU，失败时提示并回退 CPU。切换时修改 `$device`，再运行下一条命令即可，每次命令都是新进程。

所有相对路径以项目根目录为准。本文为本人注册单独使用 `gallery-wul0og.json`，与技术测试特征库分开。

## 2. 新电脑首次安装

先安装 Python 3.12 和适合本机显卡的 NVIDIA 驱动，检查 `python --version`、`nvidia-smi`。以下是当前已验证的 Windows x64 / CUDA 12.4 方案，4070 电脑仍需实际检查。

```powershell
python -m venv .venv
& $py -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install -r requirements-win-cu124.lock
& $py -m pip install -e .
& $py -m pip check
```

每条成功后再继续；依赖检查应显示 `No broken requirements found.`。安装包较大，可以复用本机 pip 缓存或已校验的同版本官方 wheel。每台电脑、每个 worktree 分别建立虚拟环境。

## 3. 检查模型与 GPU

```powershell
& $py -m cnn_face doctor --device $device --report artifacts/reports/doctor.json
```

成功后检查 JSON 中的 `device`、`parameter_device`、`forward_input_device`、`forward_output_device`。明确使用 GPU 时应为 `cuda:0`，并显示 `embedding_dim: 512`。

`doctor` 默认使用合成输入运行真实网络。需要同时检查真实图片时增加 `--image 图片路径`，报告中将增加 `image_check`；合成预热成功不代表已经录入本人。

首次运行会下载模型。默认权重位置为 `artifacts/.deepface/weights/facenet512_weights.pth`；下载失败的处理见第 10 节。已有缓存时会复用并校验。

## 4. 准备本人照片

用手机或带摄像头的电脑拍摄，推荐先准备 5～10 张注册照片，以清晰正脸为主，包含轻微左右转头和不同光照。每张只有一张脸，保存为 JPG 或 PNG，避免重度美颜、严重模糊及大幅遮挡。数量是起步建议，实际效果需验证。

```powershell
New-Item -ItemType Directory -Force -Path @(
    'data/enroll/WuL0og'
    'data/validation/self'
    'data/validation/other'
    'data/test/self'
    'data/test/other'
    'data/videos'
) | Out-Null
```

| 目录 | 放什么 |
| --- | --- |
| `data/enroll/WuL0og/` | 你的注册照片，直接放在这一层 |
| `data/validation/self/` | 另拍的本人照片，用于调阈值 |
| `data/validation/other/` | 可用于测试的其他人照片，用于检查误识 |
| `data/test/self/`、`data/test/other/` | 阈值确定后再使用的最终测试照片 |
| `data/videos/` | 本地演示视频 |

注册、校准、最终测试使用不同原图，最好分不同拍摄会话。裁剪、增强或连续视频相邻帧仍属于同一集合。手机 HEIC 格式请先转换为 JPG/PNG；仅改扩展名不会完成转换。

数据目录已被 Git 忽略，照片保存在本地。新建目录不会自动拍照或生成样本，下面命令中的 `01.jpg` 需要替换为实际文件名。

## 5. 注册本人

放好注册照片后执行：

```powershell
& $py -m cnn_face enroll --name WuL0og --images data/enroll/WuL0og --gallery $gallery --device $device
```

成功时输出昵称、特征库位置、`embedding_dim: 512` 和 `embedding_count`。检查数量是否与注册图片数量一致。注册提取并保存特征，不训练 CNN。

程序仅扫描目录第一层的受支持图片，按文件名排序。某张图片无法读取、无脸或有多张脸时，整次注册失败并指出文件；修正后重新执行。

重新执行会整体替换同一路径的特征库，失败保留原库。以后增加照片时，将旧照片和新照片一起保留在注册目录，再执行注册。

## 6. 用独立照片确定阈值

距离越小越相似，距离不大于阈值时判为本人。系统要求显式传入阈值，没有已经为你验证好的默认值。

先用校准照片查看距离：

```powershell
# 此处阈值 0 仅用于收集距离，不作为实际识别阈值。
& $py -m cnn_face recognize --image data/validation/self/01.jpg --gallery $gallery --threshold 0 --device $device
& $py -m cnn_face recognize --image data/validation/other/01.jpg --gallery $gallery --threshold 0 --device $device
```

替换文件名，检查多张本人和他人照片，记录每张的 `distance`。先处理 `no_face`、`multiple_faces` 和运行错误，不把它们记成有效比对距离。

如果本人距离均低于他人距离，可在两组距离之间选一个候选阈值，再用全部校准照片复核。若两组明显重叠，单个阈值难以兼顾误识和误拒，应先检查注册照片质量、姿态和光照覆盖，不只靠放宽阈值。

确定后，在当前窗口保存实际数值：

```powershell
$threshold = [double](Read-Host '输入经校准确定的阈值，范围 0 到 2')
```

阈值越大越容易匹配，同时可能增加误识。任何文档示例数值、技术脚本中的 0 或 2，都不能当作你已经校准好的阈值。更改注册照片后需复核阈值。

## 7. 识别独立测试图片

```powershell
& $py -m cnn_face recognize --image data/test/self/01.jpg --gallery $gallery --threshold $threshold --device $device
& $py -m cnn_face recognize --image data/test/other/01.jpg --gallery $gallery --threshold $threshold --device $device
```

| `status` | 含义 |
| --- | --- |
| `matched` | 匹配成功，`identity` 为登记昵称 |
| `unknown` | 检测到单脸，但未达到匹配条件 |
| `no_face` | 未检测到脸 |
| `multiple_faces` | 检测到多张脸，本版不作身份判定 |

单脸结果还包含人脸框、最小距离、阈值和耗时。设备信息输出到标准错误，识别结果输出为标准输出 JSON。四种状态都是正常业务结果，退出码为 0；参数或运行错误为非零退出码。

最终测试时固定注册库与阈值，并如实记录失败样本。如果看完测试结果又调整阈值，应另取样本重新测试。详细统计方法见 [测试与验收](VALIDATION.md)。

## 8. 本地视频与摄像头

当前无摄像头的电脑，先使用本地视频：

```powershell
& $py -m cnn_face stream --video data/videos/demo.mp4 --gallery $gallery --threshold $threshold --device $device
```

在带摄像头的电脑上，完成环境安装、权重和注册库准备及 GPU 检查后执行：

```powershell
& $py -m cnn_face stream --camera 0 --gallery $gallery --threshold $threshold --device $device
```

`--video` 与 `--camera` 二选一。摄像头 0 通常是默认设备，打不开时检查摄像头权限、是否被其他应用占用，以及实际设备索引。

窗口显示画面、状态和可用的人脸框。按 Esc、Q 或关闭窗口退出；同步推理时需等待当前帧完成。先检查本人、他人及空画面，再记录稳定运行效果。4070 摄像头的效果以目标机实测为准。

## 9. 迁移到另一台电脑

1. 获取最新项目代码和环境锁定文件，在新电脑重新建立 `.venv`。
2. 将 `artifacts/gallery-wul0og.json` 复制到新项目相同位置；它用于识别。保留或备份注册原图，便于以后更新。
3. 可复制已校验的 `artifacts/.deepface/weights/facenet512_weights.pth`，省去再次下载。
4. 根据需要迁移照片和视频，重新执行第 1 节，路径改为新电脑的项目位置。
5. 运行 `doctor --device cuda`，重新输入已确定的阈值，再启动摄像头；新设备光照和成像变化后复核效果。

Git 不会自动同步忽略目录中的照片、特征库和权重。虚拟环境分别安装，不复制另一台电脑的 `.venv`。需要离线演示时，在准备好缓存后断网验证一次启动与识别。

## 10. 常见问题

| 问题 | 处理 |
| --- | --- |
| 提示图片不存在 | 检查项目工作目录和实际文件名；有空格的路径加引号 |
| 注册目录没有图片 | 图片直接放在目录第一层，优先使用 JPG/PNG |
| 某张注册图无脸或多脸 | 替换该图或裁剪为单人照片，再重新注册 |
| 本人经常为未知 | 检查注册图质量、姿态和光照覆盖，再用校准样本复核阈值 |
| 他人被认成本人 | 检查两组距离，适当收紧阈值并重新验证 |
| CUDA 不可用 | 先检查驱动和 CUDA 版 PyTorch；临时将 `$device` 改为 `cpu` |
| 特征库无效或不兼容 | 使用当前模型重新注册，不手工拼接特征 JSON |
| 新窗口中 `$py` 等变量不存在 | 重新执行第 1 节，并重新设置阈值 |

权重自动下载不通时，可以从 [官方模型发布](https://github.com/serengil/deepface_models/releases/tag/v1.0) 获取文件，或使用镜像传输后核对官方摘要：

```powershell
# 默认缓存位置；只有需要准备权重时才执行。
New-Item -ItemType Directory -Force -Path artifacts/.deepface/weights | Out-Null
curl.exe --fail --location --output artifacts/.deepface/weights/facenet512_weights.pth 'https://hf-mirror.com/serengil/deepface/resolve/main/facenet512_weights.pth'
if ($LASTEXITCODE -ne 0) { throw '模型下载失败' }
$weightHash = (Get-FileHash artifacts/.deepface/weights/facenet512_weights.pth -Algorithm SHA256).Hash.ToLowerInvariant()
if ($weightHash -ne '5b0eb002a6033d3d770f4157783229aa1d7639c943353877e67ab06fa927d28d') { throw '模型校验失败' }
```

引擎初始化也会校验该摘要。排查其他问题可运行 `& $py -m cnn_face 子命令 --help` 查看实际参数，或使用 `doctor --report` 保存运行证据。
