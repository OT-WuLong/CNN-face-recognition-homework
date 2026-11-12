"""手动运行真实模型技术检查；上游样本不用于本人准确率统计。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import cv2
import numpy as np

from cnn_face import camera, engine
from cnn_face.contracts import PipelineConfig
from cnn_face.enrollment import enroll, load_gallery
from cnn_face.recognition import recognize


def main():
    """每次仅检查一个设备；CPU 与 CUDA 必须在不同进程串行运行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", action="store_true", help="额外打开真实视频窗口")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / "注册技术样本"
    images.mkdir(exist_ok=True)
    shutil.copyfile(args.samples / "img1.jpg", images / "img1.jpg")
    first = engine.read_image(args.samples / "img1.jpg")
    second = engine.read_image(args.samples / "img2.jpg")
    blank = np.zeros_like(first)
    multiple = np.concatenate([first, first], axis=1)
    pipeline = PipelineConfig()
    runtime = engine.configure_runtime(args.device)
    evidence = set()

    def record(module, inputs, output):
        evidence.add((
            str(next(module.parameters()).device), str(inputs[0].device),
            str(output.device),
        ))

    # 验收脚本观察真实前向，生产接口无需暴露模型内部对象。
    hook = engine._client.model.register_forward_hook(record)
    try:
        gallery_path = args.output / "gallery.json"
        gallery = enroll("上游技术样本", images, gallery_path, pipeline=pipeline)
        assert load_gallery(gallery_path) == gallery
        gallery = load_gallery(gallery_path)
        results = {}
        for name, frame, threshold, expected in (
            ("matched", second, 2.0, "matched"),
            ("unknown", second, 0.0, "unknown"),
            ("no_face", blank, 0.3, "no_face"),
            ("multiple_faces", multiple, 0.3, "multiple_faces"),
        ):
            result = recognize(frame, gallery, threshold=threshold)
            assert result.status == expected, (name, result)
            results[name] = asdict(result)
        query = engine.extract_embeddings(
            second, pipeline=pipeline,
        )[0].embedding
        np.save(args.output / "embeddings.npy", [gallery.embeddings[0], query])

        original = gallery_path.read_bytes()
        failures = []
        for name, frame in (
            ("empty", None), ("bad_image", b"broken"),
            ("no_face", blank), ("multiple_faces", multiple),
        ):
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            if isinstance(frame, bytes):
                (directory / "样本.jpg").write_bytes(frame)
            elif frame is not None:
                ok, encoded = cv2.imencode(".png", frame)
                assert ok
                encoded.tofile(directory / "样本.png")
            try:
                enroll("失败不覆盖", directory, gallery_path, pipeline=pipeline)
            except ValueError as error:
                failures.append({"case": name, "error": str(error)})
            else:
                raise AssertionError(f"错误样本未被拒绝：{name}")
            assert gallery_path.read_bytes() == original
        replaced = enroll(
            "重新注册技术样本", images, gallery_path, pipeline=pipeline,
        )
        assert len(replaced.embeddings) == 1
        assert load_gallery(gallery_path).identity == "重新注册技术样本"

        # 原图和所有派生图均留在同一技术测试集合，不用于最终测试。
        video = args.output / "技术视频.avi"
        frames = [
            cv2.resize(frame, (640, 480))
            for frame in (first, second, blank, multiple)
        ]
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480),
        )
        assert writer.isOpened()
        try:
            for frame in frames:
                for _ in range(12):
                    writer.write(frame)
        finally:
            writer.release()
        stream_results = None
        if args.stream:
            captures = []
            create_capture = cv2.VideoCapture

            def open_capture(*arguments):
                capture = create_capture(*arguments)
                captures.append(capture)
                return capture

            # 仅观察调用和资源状态，识别、解码、窗口均执行真实实现。
            with (
                patch.object(
                    camera.cv2, "VideoCapture", side_effect=open_capture,
                ),
                patch(
                    "cnn_face.recognition.recognize", wraps=recognize,
                ) as calls,
            ):
                camera.run_stream(video, gallery, threshold=0.3)
            assert all(not capture.isOpened() for capture in captures)
            stream_results = {"frames": calls.call_count, "released": True}
            assert calls.call_count == 48
    finally:
        hook.remove()
    selected = runtime["device"]
    assert evidence == {(selected, selected, selected)}, evidence
    report = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "sample_kind": "上游技术测试样本及同集合派生图，非本人准确率测试",
        "runtime": runtime, "pipeline": asdict(pipeline),
        "real_forward_devices": sorted(evidence), "results": results,
        "registration_failures": failures, "reregistration_replaced": True,
        "stream": stream_results,
    }
    content = json.dumps(report, ensure_ascii=False, indent=2)
    (args.output / "report.json").write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
