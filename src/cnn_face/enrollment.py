"""建立、原子保存并严格加载单个身份的人脸特征库。"""

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from cnn_face import engine
from cnn_face.contracts import (
    Gallery,
    PipelineConfig,
    validate_embedding,
)


_IMAGE_SUFFIXES = {
    ".bmp", ".dib", ".jpe", ".jpeg", ".jpg", ".jp2", ".png",
    ".tif", ".tiff", ".webp",
}


def enroll(
    identity: str,
    images_dir: Path,
    gallery_path: Path,
    *,
    pipeline: PipelineConfig,
) -> Gallery:
    """按文件名顺序注册单脸照片；调用方须提前初始化模型运行环境。"""
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("登记昵称不能为空。")
    if not isinstance(pipeline, PipelineConfig):
        raise ValueError("需要有效的流水线配置。")
    images_dir = Path(images_dir)
    gallery_path = Path(gallery_path)
    if not images_dir.is_dir():
        raise ValueError(f"注册图片目录不存在或不是目录：{images_dir}")
    image_paths = sorted(
        (
            path for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    if not image_paths:
        raise ValueError(f"注册目录中没有支持的图片：{images_dir}")

    embeddings = []
    for image_path in image_paths:
        try:
            frame = engine.read_image(image_path)
            faces = engine.extract_embeddings(frame, pipeline=pipeline)
            if not faces:
                raise ValueError("未检测到人脸，注册照片必须恰好包含一张脸。")
            if len(faces) != 1:
                raise ValueError(
                    f"检测到 {len(faces)} 张人脸，注册照片必须恰好包含一张脸。"
                )
            vector = faces[0].embedding
            dimension = len(embeddings[0]) if embeddings else None
            validate_embedding(vector, dimension)
            embeddings.append(vector.copy())
        except (OSError, ValueError, OverflowError) as error:
            raise ValueError(f"注册图片无效：{image_path}：{error}") from error
        except RuntimeError as error:
            raise RuntimeError(f"注册图片处理失败：{image_path}：{error}") from error

    gallery = Gallery(
        schema_version=1,
        identity=identity,
        pipeline=pipeline,
        embedding_dim=len(embeddings[0]),
        embeddings=embeddings,
    )
    content = json.dumps(
        asdict(gallery), ensure_ascii=False, allow_nan=False, indent=2,
    ) + "\n"
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # Windows 上须先关闭临时文件，再用同目录替换保证整库更新。
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=gallery_path.parent,
            prefix=f".{gallery_path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, gallery_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return gallery


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    """拒绝重复 JSON 字段，避免解码器悄悄覆盖损坏内容。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 字段重复：{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """JSON 不接受 NaN 和 Infinity 等非标准数值。"""
    raise ValueError(f"JSON 包含非有限数值：{value}")


def load_gallery(gallery_path: Path) -> Gallery:
    """校验完整字段和共享数据约定后加载 UTF-8 特征库。"""
    gallery_path = Path(gallery_path)
    try:
        with gallery_path.open(encoding="utf-8") as stream:
            data = json.load(
                stream, object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        if (
            not isinstance(data, dict)
            or set(data) != {field.name for field in fields(Gallery)}
        ):
            raise ValueError("特征库字段必须与 Gallery 格式完全一致。")
        pipeline = data["pipeline"]
        if (
            not isinstance(pipeline, dict)
            or set(pipeline) != {
                field.name for field in fields(PipelineConfig)
            }
        ):
            raise ValueError("流水线配置字段必须完整且不能包含未知字段。")
        data["pipeline"] = PipelineConfig(**pipeline)
        return Gallery(**data)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"特征库无效：{gallery_path}：{error}") from error
