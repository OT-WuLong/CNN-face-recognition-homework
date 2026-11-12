"""复用已初始化的特征引擎，对单帧执行余弦距离比对。"""

from dataclasses import replace
import math
from time import perf_counter

import numpy as np

from cnn_face import engine
from cnn_face.contracts import (
    Gallery,
    PipelineConfig,
    RecognitionResult,
    validate_embedding,
    validate_threshold,
)


def _normalize_embedding(values: list[float], dimension: int) -> list[float]:
    """先缩放再归一化，避免有限但极大或极小的分量导致溢出或下溢。"""
    validate_embedding(values, dimension)
    scale = max(abs(value) for value in values)
    scaled = [value / scale for value in values]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


def recognize(
    frame_bgr: np.ndarray, gallery: Gallery, *, threshold: float
) -> RecognitionResult:
    """比对内存库；调用方须先初始化引擎，技术错误直接向上传播。"""
    started = perf_counter()
    validate_threshold(threshold)
    if not isinstance(gallery, Gallery):
        raise ValueError("需要有效的内存特征库。")
    # 冻结数据类中的列表仍可变，沿用共享约定重新校验内存库。
    gallery = replace(gallery)
    if gallery.pipeline != PipelineConfig():
        raise ValueError("特征库与当前固定的特征提取流水线不兼容。")

    faces = engine.extract_embeddings(frame_bgr, pipeline=gallery.pipeline)
    if len(faces) != 1:
        return RecognitionResult(
            status="no_face" if not faces else "multiple_faces",
            identity=None,
            bbox=None,
            distance=None,
            threshold=threshold,
            elapsed_ms=(perf_counter() - started) * 1000,
        )

    face = faces[0]
    query = _normalize_embedding(face.embedding, gallery.embedding_dim)
    distance = 2.0
    for embedding in gallery.embeddings:
        reference = _normalize_embedding(embedding, gallery.embedding_dim)
        similarity = math.fsum(q * g for q, g in zip(query, reference))
        if (
            not math.isfinite(similarity)
            or not -1 - 1e-12 <= similarity <= 1 + 1e-12
        ):
            raise ValueError("余弦相似度超出有效数值范围。")
        # 只截断单位向量点积在端点附近的浮点误差，距离不是身份概率。
        current_distance = 1.0 - max(-1.0, min(1.0, similarity))
        distance = min(distance, current_distance)

    matched = distance <= threshold
    return RecognitionResult(
        status="matched" if matched else "unknown",
        identity=gallery.identity if matched else None,
        bbox=face.bbox,
        distance=distance,
        threshold=threshold,
        elapsed_ms=(perf_counter() - started) * 1000,
    )
