"""注册、识别和视频模块共用的数据约定。"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal


def validate_embedding(values: list[float], dimension: int | None = None) -> None:
    """在模型输出和特征库边界检查向量，拒绝无效值与零向量。"""
    if not isinstance(values, list) or not values:
        raise ValueError("特征向量必须是非空列表。")
    if dimension is not None and len(values) != dimension:
        raise ValueError("特征向量维数与约定不一致。")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("特征向量只能包含有限数值。")
    if not any(value != 0 for value in values):
        raise ValueError("特征向量不能是零向量。")


def validate_threshold(value: float) -> None:
    """余弦距离的阈值必须位于闭区间 [0, 2]。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 2
    ):
        raise ValueError("余弦距离阈值必须是 [0, 2] 内的有限数值。")


def _validate_bbox(bbox: tuple[int, int, int, int]) -> None:
    if (
        not isinstance(bbox, tuple)
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
        or min(bbox[:2]) < 0
        or min(bbox[2:]) <= 0
    ):
        raise ValueError("人脸框必须为有效的 (x, y, w, h) 整数坐标。")


@dataclass(frozen=True)
class PipelineConfig:
    """固定首版流水线，避免不同任务使用不兼容的特征。"""

    model_name: str = "Facenet512"
    detector_backend: str = "opencv"
    align: bool = True
    normalization: str = "base"
    backend_engine: str = "pytorch"
    distance_metric: str = "cosine"

    def __post_init__(self) -> None:
        expected = {
            "model_name": "Facenet512",
            "detector_backend": "opencv",
            "normalization": "base",
            "backend_engine": "pytorch",
            "distance_metric": "cosine",
        }
        if self.align is not True or any(
            getattr(self, name) != value for name, value in expected.items()
        ):
            raise ValueError("当前仅支持已固定的 Facenet512 / PyTorch 流水线。")


@dataclass(frozen=True)
class FaceEmbedding:
    """一张检测人脸的特征及其在原始帧中的位置。"""

    embedding: list[float]
    bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        validate_embedding(self.embedding)
        _validate_bbox(self.bbox)


@dataclass(frozen=True)
class Gallery:
    """单个登记身份的特征库，磁盘格式由注册模块负责。"""

    schema_version: int
    identity: str
    pipeline: PipelineConfig
    embedding_dim: int
    embeddings: list[list[float]]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("不支持的特征库版本。")
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("登记昵称不能为空。")
        if not isinstance(self.pipeline, PipelineConfig):
            raise ValueError("特征库缺少有效的流水线配置。")
        if type(self.embedding_dim) is not int or self.embedding_dim <= 0:
            raise ValueError("特征维数必须为正整数。")
        if not isinstance(self.embeddings, list) or not self.embeddings:
            raise ValueError("特征库至少需要一个向量。")
        for embedding in self.embeddings:
            validate_embedding(embedding, self.embedding_dim)


@dataclass(frozen=True)
class RecognitionResult:
    """四种正常识别状态；技术故障通过异常单独报告。"""

    status: Literal["matched", "unknown", "no_face", "multiple_faces"]
    identity: str | None
    bbox: tuple[int, int, int, int] | None
    distance: float | None
    threshold: float
    elapsed_ms: float

    def __post_init__(self) -> None:
        if self.status not in {"matched", "unknown", "no_face", "multiple_faces"}:
            raise ValueError("不支持的识别状态。")
        validate_threshold(self.threshold)
        if (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, Real)
            or not math.isfinite(self.elapsed_ms)
            or self.elapsed_ms < 0
        ):
            raise ValueError("识别耗时必须是非负有限数值。")
        if self.status in {"no_face", "multiple_faces"}:
            fields = (self.identity, self.bbox, self.distance)
            if any(value is not None for value in fields):
                raise ValueError("无脸和多脸状态不能携带身份判定。")
            return
        _validate_bbox(self.bbox)
        validate_threshold(self.distance)
        if self.status == "matched":
            if not isinstance(self.identity, str) or not self.identity.strip():
                raise ValueError("匹配结果必须包含登记昵称。")
            if self.distance > self.threshold:
                raise ValueError("匹配结果的距离不能大于阈值。")
        elif self.identity is not None or self.distance <= self.threshold:
            raise ValueError("未知结果必须没有身份且距离大于阈值。")
