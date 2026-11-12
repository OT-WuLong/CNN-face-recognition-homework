"""使用真实业务模块和引擎替身验证命令入口，不使用模型或硬件。"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cnn_face import camera, engine
from cnn_face.__main__ import main
from cnn_face.contracts import FaceEmbedding


class CliTests(unittest.TestCase):
    """检查入口的数据传递、一次初始化、输出与错误退出。"""

    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.images = self.root / "注册照片"
        self.images.mkdir()
        self.image = self.images / "单脸.png"
        ok, encoded = cv2.imencode(
            ".png", np.zeros((20, 20, 3), dtype=np.uint8),
        )
        self.assertTrue(ok)
        encoded.tofile(self.image)
        self.gallery = self.root / "本人库.json"
        self.face = FaceEmbedding([1.0, 0.0], (1, 2, 3, 4))
        self.configure = self.enterContext(patch.object(
            engine, "configure_runtime", return_value={"device": "cpu"},
        ))
        self.extract = self.enterContext(patch.object(
            engine, "extract_embeddings", return_value=[self.face],
        ))

    def invoke(self, *arguments):
        """隔离进程参数与两个输出流，保留入口的真实返回值。"""
        output, errors = StringIO(), StringIO()
        with (
            patch.object(sys, "argv", ["cnn-face", *map(str, arguments)]),
            redirect_stdout(output), redirect_stderr(errors),
        ):
            code = main()
        return code, output.getvalue(), errors.getvalue()

    def enroll(self):
        return self.invoke(
            "enroll", "--name", "中文昵称", "--images", self.images,
            "--gallery", self.gallery, "--device", "cpu",
        )

    def test_enroll_reload_and_recognize_use_real_business_modules(self):
        code, output, errors = self.enroll()
        self.assertEqual(code, 0)
        self.assertIn("运行环境", errors)
        self.assertEqual(json.loads(output)["embedding_count"], 1)
        self.assertNotIn("embeddings", json.loads(output))
        code, output, errors = self.invoke(
            "recognize", "--image", self.image, "--gallery", self.gallery,
            "--threshold", "0.5", "--device", "cpu",
        )
        result = json.loads(output)
        self.assertEqual((code, result["status"]), (0, "matched"))
        self.assertEqual(result["identity"], "中文昵称")
        self.assertEqual(result["bbox"], [1, 2, 3, 4])
        self.assertEqual(self.configure.call_count, 2)
        self.assertEqual(self.extract.call_count, 2)

    def test_all_normal_statuses_exit_zero(self):
        self.enroll()
        for faces, status in (
            ([], "no_face"), ([self.face] * 2, "multiple_faces"),
            ([self.face], "matched"),
            ([FaceEmbedding([0.0, 1.0], (1, 2, 3, 4))], "unknown"),
        ):
            with self.subTest(status=status):
                self.extract.return_value = faces
                code, output, _ = self.invoke(
                    "recognize", "--image", self.image,
                    "--gallery", self.gallery, "--threshold", "0.5",
                )
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(output)["status"], status)

    def test_stream_distinguishes_camera_index_and_video_path(self):
        self.enroll()
        video = self.root / "本地视频.avi"
        video.touch()
        for option, value in (("--camera", 0), ("--video", video)):
            with self.subTest(option=option), patch.object(
                camera, "run_stream",
            ) as stream:
                self.configure.reset_mock()
                code, _, _ = self.invoke(
                    "stream", option, value, "--gallery", self.gallery,
                    "--threshold", "0.25", "--device", "cuda",
                )
                self.assertEqual(code, 0)
                self.configure.assert_called_once_with("cuda")
                self.assertEqual(stream.call_args.args[0], value)
                self.assertEqual(stream.call_args.args[1].identity, "中文昵称")
                self.assertEqual(stream.call_args.kwargs, {"threshold": 0.25})

    def test_invalid_threshold_and_gallery_fail_before_model_loading(self):
        self.enroll()
        self.configure.reset_mock()
        for threshold in ("nan", "inf", "-0.1", "2.1"):
            code, output, errors = self.invoke(
                "recognize", "--image", self.image,
                "--gallery", self.gallery, "--threshold", threshold,
            )
            self.assertEqual((code, output), (2, ""))
            self.assertIn("错误", errors)
        self.gallery.write_text("{}", encoding="utf-8")
        code, _, errors = self.invoke(
            "recognize", "--image", self.image,
            "--gallery", self.gallery, "--threshold", "0.3",
        )
        self.assertEqual(code, 2)
        self.assertIn("特征库无效", errors)
        self.configure.assert_not_called()

    def test_model_failure_reports_error_without_overwriting_gallery(self):
        self.enroll()
        original = self.gallery.read_bytes()
        self.extract.side_effect = RuntimeError("模拟模型故障")
        code, output, errors = self.enroll()
        self.assertEqual((code, output), (2, ""))
        self.assertIn("模拟模型故障", errors)
        self.assertEqual(self.gallery.read_bytes(), original)

    def test_doctor_preserves_image_report_without_features(self):
        report = self.root / "诊断.json"
        code, output, _ = self.invoke(
            "doctor", "--image", self.image, "--report", report,
        )
        self.assertEqual(code, 0)
        data = json.loads(output)
        self.assertEqual(data["image_check"]["face_count"], 1)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8")), data)
        self.configure.assert_called_once_with("auto")

    def test_stream_requires_exactly_one_source(self):
        for sources in ((), ("--camera", "0", "--video", "video.avi")):
            with (
                self.subTest(sources=sources),
                self.assertRaises(SystemExit) as caught,
            ):
                self.invoke(
                    "stream", *sources, "--gallery", self.gallery,
                    "--threshold", "0.3",
                )
            self.assertEqual(caught.exception.code, 2)
        self.configure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
