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
    doctor.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
    )
    doctor.add_argument("--image", type=Path, help="额外验证一张真实图片")
    doctor.add_argument("--report", type=Path, help="将诊断结果保存为 JSON")
    enrollment = commands.add_parser("enroll", help="创建或整体替换人脸特征库")
    enrollment.add_argument("--name", required=True, help="登记昵称")
    enrollment.add_argument(
        "--images", type=Path, required=True, help="注册照片目录",
    )
    enrollment.add_argument(
        "--gallery", type=Path, required=True, help="输出特征库",
    )
    recognition = commands.add_parser("recognize", help="识别一张本地图片")
    recognition.add_argument("--image", type=Path, required=True, help="查询图片")
    stream = commands.add_parser("stream", help="展示本地视频或摄像头识别")
    source = stream.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="摄像头索引，例如 0")
    source.add_argument("--video", type=Path, help="本地视频路径")
    for command in (recognition, stream):
        command.add_argument(
            "--gallery", type=Path, required=True, help="已有特征库",
        )
        command.add_argument(
            "--threshold", type=float, required=True, help="余弦距离阈值 [0, 2]",
        )
    for command in (enrollment, recognition, stream):
        command.add_argument(
            "--device", choices=("auto", "cpu", "cuda"), default="auto",
        )
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0

    try:
        from cnn_face.contracts import PipelineConfig, validate_threshold
        from cnn_face.engine import (
            configure_runtime,
            extract_embeddings,
            read_image,
        )

        pipeline = PipelineConfig()
        frame = None
        if args.command in {"doctor", "recognize"} and args.image:
            frame = read_image(args.image)
        if args.command in {"recognize", "stream"}:
            from cnn_face.enrollment import load_gallery

            validate_threshold(args.threshold)
            gallery = load_gallery(args.gallery)
        if args.command == "stream":
            if args.camera is not None and args.camera < 0:
                raise ValueError("摄像头索引不能为负数。")
            if args.video is not None and not args.video.is_file():
                raise ValueError(f"视频文件不存在：{args.video}")
        report = configure_runtime(args.device)
        report["pipeline"] = asdict(pipeline)
        if args.command != "doctor":
            # 标准输出仅保留业务 JSON，设备证据在启动时写入标准错误。
            print(
                "运行环境：" + json.dumps(report, ensure_ascii=False),
                file=sys.stderr,
            )
        if args.command == "enroll":
            from cnn_face.enrollment import enroll

            gallery = enroll(
                args.name, args.images, args.gallery, pipeline=pipeline,
            )
            print(json.dumps({
                "identity": gallery.identity,
                "gallery": str(args.gallery.resolve()),
                "embedding_dim": gallery.embedding_dim,
                "embedding_count": len(gallery.embeddings),
                "pipeline": asdict(gallery.pipeline),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "recognize":
            from cnn_face.recognition import recognize

            result = recognize(frame, gallery, threshold=args.threshold)
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return 0
        if args.command == "stream":
            from cnn_face.camera import run_stream

            video_source = (
                args.camera if args.camera is not None else args.video
            )
            run_stream(video_source, gallery, threshold=args.threshold)
            return 0
        if frame is not None:
            started = perf_counter()
            faces = extract_embeddings(frame, pipeline=pipeline)
            report["image_check"] = {
                "source": str(args.image.resolve()),
                "face_count": len(faces),
                "embedding_dimensions": [
                    len(face.embedding) for face in faces
                ],
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
