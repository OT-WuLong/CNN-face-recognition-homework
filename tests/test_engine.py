"""检查输入与错误边界；模型和 GPU 的实测由 doctor 命令完成。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
from deepface.modules.exceptions import FaceNotDetected
import numpy as np

from cnn_face.contracts import PipelineConfig
from cnn_face import engine


class EngineTests(unittest.TestCase):
    """使用替身隔离模型下载，验证提取接口不会吞掉运行错误。"""

    def setUp(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_chinese_image_path(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "人脸测试.png"
            ok, encoded = cv2.imencode(".png", self.frame)
            self.assertTrue(ok)
            encoded.tofile(path)
            self.assertTrue(np.array_equal(engine.read_image(path), self.frame))

    def test_invalid_image_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "损坏.png"
            path.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                engine.read_image(path)

    def test_invalid_frame_is_rejected(self):
        for frame in (None, [], np.zeros((10, 10)), self.frame.astype(float)):
            with self.subTest(frame_type=type(frame)), self.assertRaises(ValueError):
                engine.extract_embeddings(frame, pipeline=PipelineConfig())

    def test_initialization_is_required(self):
        with patch.object(engine, "_runtime", None), self.assertRaises(RuntimeError):
            engine.extract_embeddings(self.frame, pipeline=PipelineConfig())

    def test_detection_and_bgr_input_remain_enabled(self):
        face = {"embedding": [1.0, 0.0], "facial_area": {
            "x": 10, "y": 20, "w": 30, "h": 40,
        }}
        deepface = SimpleNamespace(represent=Mock(return_value=[face]))
        with (
            patch.object(engine, "_runtime", {"device": "cpu"}),
            patch.object(engine, "_deepface", deepface),
            patch.object(engine, "_client", SimpleNamespace(output_shape=2)),
        ):
            result = engine.extract_embeddings(self.frame, pipeline=PipelineConfig())
        self.assertEqual(result[0].bbox, (10, 20, 30, 40))
        arguments = deepface.represent.call_args.kwargs
        self.assertIs(arguments["img_path"], self.frame)
        self.assertIs(arguments["enforce_detection"], True)

    def test_only_missing_face_becomes_empty_result(self):
        deepface = SimpleNamespace(represent=Mock(side_effect=FaceNotDetected()))
        with (
            patch.object(engine, "_runtime", {"device": "cpu"}),
            patch.object(engine, "_deepface", deepface),
        ):
            self.assertEqual(
                engine.extract_embeddings(self.frame, pipeline=PipelineConfig()), []
            )
            deepface.represent.side_effect = RuntimeError("模拟模型故障")
            with self.assertRaisesRegex(RuntimeError, "模拟模型故障"):
                engine.extract_embeddings(self.frame, pipeline=PipelineConfig())

    def test_runtime_cannot_switch_devices_in_same_process(self):
        with patch.object(engine, "_runtime", {"requested_device": "cpu"}):
            with self.assertRaisesRegex(RuntimeError, "新进程"):
                engine.configure_runtime("cuda")


if __name__ == "__main__":
    unittest.main()
