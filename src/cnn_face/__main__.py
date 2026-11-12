"""课程设计项目的命令行入口。"""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cnn-face",
        description="基于预训练 CNN 的本人人脸识别课程设计。",
    )
    parser.add_argument(
        "--version", action="version", version=version("cnn-face-course")
    )
    commands = parser.add_subparsers(dest="command", title="可用命令")
    doctor = commands.add_parser("doctor", help="验证依赖、模型和实际执行设备")
    doctor.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    doctor.add_argument("--image", type=Path, help="额外验证一张真实图片")
    doctor.add_argument("--report", type=Path, help="将诊断结果保存为 JSON")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0

    try:
        from cnn_face.contracts import PipelineConfig
        from cnn_face.engine import (
            configure_runtime,
            extract_embeddings,
            read_image,
        )

        frame = read_image(args.image) if args.image else None
        report = configure_runtime(args.device)
        report["pipeline"] = asdict(PipelineConfig())
        if frame is not None:
            started = perf_counter()
            faces = extract_embeddings(frame, pipeline=PipelineConfig())
            report["image_check"] = {
                "source": str(args.image.resolve()),
                "face_count": len(faces),
                "embedding_dimensions": [len(face.embedding) for face in faces],
                "bounding_boxes": [face.bbox for face in faces],
                "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            }
        content = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(content + "\n", encoding="utf-8")
        print(content)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
