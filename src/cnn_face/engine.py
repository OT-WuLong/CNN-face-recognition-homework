"""统一模型初始化、执行设备验证和人脸特征提取。"""

from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

import numpy as np

from cnn_face.contracts import FaceEmbedding, PipelineConfig, validate_embedding


_runtime: dict | None = None
_deepface = None
_client = None

# 固定官方权重，避免相同配置下混用不同模型参数。
_WEIGHTS_SHA256 = (
    "5b0eb002a6033d3d770f4157783229aa1d7639c943353877e67ab06fa927d28d"
)


def _warmup(client, torch, device: str) -> dict:
    """运行真实网络，记录模型参数、输入和输出张量的执行设备。"""
    selected = torch.device(device)
    client.model = client.model.to(selected)
    client.device = selected
    evidence = {"parameter_device": str(next(client.model.parameters()).device)}

    def record_devices(module, inputs, output) -> None:
        evidence["forward_input_device"] = str(inputs[0].device)
        evidence["forward_output_device"] = str(output.device)

    handle = client.model.register_forward_hook(record_devices)
    cuda_events = None
    if selected.type == "cuda":
        cuda_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        cuda_events[0].record()
    started = perf_counter()
    try:
        # 合成输入只验证网络能运行，不代表已识别到真实人脸。
        embedding = client.forward(np.zeros((160, 160, 3), dtype=np.float32))
    finally:
        handle.remove()
    validate_embedding(embedding, 512)
    if any(value != str(selected) for value in evidence.values()):
        raise RuntimeError("模型参数、输入或输出没有位于所选执行设备。")
    evidence["cuda_event_ms"] = None
    if cuda_events is not None:
        cuda_events[1].record()
        cuda_events[1].synchronize()
        evidence["cuda_event_ms"] = cuda_events[0].elapsed_time(cuda_events[1])
    evidence["warmup_ms"] = round((perf_counter() - started) * 1000, 3)
    evidence["embedding_dim"] = len(embedding)
    return evidence


def configure_runtime(device: str) -> dict:
    """每个进程初始化一次；显式 CUDA 失败时不静默回退。"""
    global _runtime, _deepface, _client

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("执行设备只能选择 auto、cpu 或 cuda。")
    if _runtime is not None:
        if _runtime["requested_device"] != device:
            raise RuntimeError("本进程已初始化模型，请在新进程中切换设备。")
        return _runtime.copy()

    import torch

    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("CUDA 不可用，请检查 GPU 驱动和 PyTorch CUDA 安装。")
    selected = "cuda:0" if available and device != "cpu" else "cpu"
    fallback_reason = None
    if device == "auto" and not available:
        fallback_reason = "未检测到可用 CUDA，使用 CPU。"

    os.environ["DEEPFACE_BACKEND_ENGINE"] = "pytorch"
    cache_root = Path(os.environ.setdefault(
        "DEEPFACE_HOME", str((Path.cwd() / "artifacts").resolve())
    ))
    with redirect_stdout(sys.stderr):
        from deepface import DeepFace

        if DeepFace.backend_engine != "pytorch":
            raise RuntimeError("DeepFace 已使用其他后端初始化，请启动新进程。")
        client = DeepFace.build_model("Facenet512")

    try:
        evidence = _warmup(client, torch, selected)
    except RuntimeError as error:
        if device != "auto" or selected == "cpu":
            raise RuntimeError(f"模型无法在 {selected} 上运行：{error}") from error
        fallback_reason = f"CUDA 模型预热失败，使用 CPU：{error}"
        selected = "cpu"
        evidence = _warmup(client, torch, selected)

    if fallback_reason:
        warnings.warn(fallback_reason, RuntimeWarning, stacklevel=2)

    weights = cache_root / ".deepface" / "weights" / "facenet512_weights.pth"
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != _WEIGHTS_SHA256:
        raise RuntimeError("模型权重与共享基线不一致，请核对官方文件及校验值。")
    _deepface = DeepFace
    _client = client
    _runtime = {
        "requested_device": device,
        "device": selected,
        "fallback_reason": fallback_reason,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            package: version(package)
            for package in (
                "deepface", "torch", "numpy", "onnxruntime",
                "opencv-contrib-python-headless", "Pillow",
            )
        },
        "backend_engine": "pytorch",
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "weights_file": str(weights.resolve()),
        "weights_sha256": digest,
        "warmup_source": "synthetic",
        **evidence,
    }
    return _runtime.copy()


def read_image(path: Path) -> np.ndarray:
    """解码本地图片，兼容 Windows 中文路径，返回 uint8 BGR 数组。"""
    import cv2

    if not path.is_file():
        raise ValueError(f"图片文件不存在：{path}")
    data = np.fromfile(path, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if frame is None:
        raise ValueError(f"无法解码图片：{path}")
    return frame


def extract_embeddings(
    frame_bgr: np.ndarray, *, pipeline: PipelineConfig
) -> list[FaceEmbedding]:
    """提取所有检测人脸的特征；仅无脸情况转换为空列表。"""
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or frame_bgr.size == 0
    ):
        raise ValueError("输入帧必须是非空的 uint8、H×W×3 BGR 数组。")
    if not isinstance(pipeline, PipelineConfig):
        raise ValueError("需要有效的流水线配置。")
    if _runtime is None:
        raise RuntimeError("请先调用 configure_runtime 初始化执行设备与模型。")

    from deepface.modules.exceptions import FaceNotDetected

    try:
        with redirect_stdout(sys.stderr):
            faces = _deepface.represent(
                img_path=frame_bgr,
                model_name=pipeline.model_name,
                detector_backend=pipeline.detector_backend,
                align=pipeline.align,
                normalization=pipeline.normalization,
                enforce_detection=True,
            )
    except FaceNotDetected:
        return []

    results = []
    for face in faces:
        embedding = face["embedding"]
        validate_embedding(embedding, _client.output_shape)
        area = face["facial_area"]
        bbox = tuple(int(area[key]) for key in ("x", "y", "w", "h"))
        results.append(FaceEmbedding(embedding=embedding, bbox=bbox))
    return results
