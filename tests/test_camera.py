"""用接口替身与合成视频验证展示、退出及错误路径，不运行模型。"""

from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from cnn_face import camera
from cnn_face.contracts import Gallery, PipelineConfig, RecognitionResult


class CameraTests(unittest.TestCase):
    """隔离桌面与摄像头，同时保留真实的颜色转换和框绘制。"""

    def setUp(self):
        self.gallery = Gallery(1, "测试昵称", PipelineConfig(), 2, [[1.0, 0.0]])
        self.frame = np.full((24, 32, 3), (10, 20, 30), dtype=np.uint8)
        self.original = self.frame.copy()
        self.capture = Mock()
        self.capture.isOpened.return_value = True
        self.capture.get.return_value = 1
        self.capture.read.side_effect = [(True, self.frame), (False, None)]
        self.window = Mock()
        self.preview = Mock()
        self.status = Mock()
        self.recognize = Mock(return_value=self.result("no_face"))
        self.real_capture = cv2.VideoCapture
        recognition = ModuleType("cnn_face.recognition")
        recognition.recognize = self.recognize
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.dict(sys.modules, {
            "cnn_face.recognition": recognition,
        }))
        self.open_capture = stack.enter_context(patch.object(
            camera.cv2, "VideoCapture", return_value=self.capture,
        ))
        self.create_window = stack.enter_context(patch.object(
            camera.tk, "Tk", return_value=self.window,
        ))
        self.families = stack.enter_context(patch.object(
            camera.tkfont, "families", return_value=("Arial", "SimSun"),
        ))
        self.labels = stack.enter_context(patch.object(
            camera.tk, "Label",
            side_effect=[self.preview, self.status, Mock()],
        ))
        self.photo = stack.enter_context(patch.object(
            camera.ImageTk, "PhotoImage",
            side_effect=lambda image, **kwargs: image,
        ))
        self.directory = self.enterContext(TemporaryDirectory())
        self.video = Path(self.directory) / "合成视频.avi"
        self.video.touch()

    @staticmethod
    def result(status):
        """构造遵守共享类型约定的四种识别结果。"""
        single_face = status in {"matched", "unknown"}
        distance = {"matched": 0.2, "unknown": 0.8}.get(status)
        return RecognitionResult(
            status, "测试昵称" if status == "matched" else None,
            (4, 6, 12, 10) if single_face else None,
            distance, 0.3, 1.0,
        )

    def assert_released(self):
        """每个已创建资源都必须恰好清理一次。"""
        self.capture.release.assert_called_once_with()
        self.window.destroy.assert_called_once_with()

    def test_video_states_bgr_boxes_and_no_stale_results(self):
        self.capture.get.return_value = 4
        self.capture.read.side_effect = (
            [(True, self.frame)] * 4 + [(False, None)]
        )
        self.recognize.side_effect = [
            self.result(state)
            for state in ("matched", "unknown", "no_face", "multiple_faces")
        ]

        camera.run_stream(self.video, self.gallery, threshold=0.3)

        self.open_capture.assert_called_once_with(str(self.video))
        self.assertEqual(self.recognize.call_count, 4)
        for arguments in self.recognize.call_args_list:
            self.assertIs(arguments.args[0], self.frame)
            self.assertIs(arguments.args[1], self.gallery)
            self.assertEqual(arguments.kwargs, {"threshold": 0.3})
        texts = [
            call.kwargs["text"]
            for call in self.status.configure.call_args_list
        ]
        self.assertEqual(texts, [
            "匹配：测试昵称", "未知", "未检测到人脸", "检测到多张人脸，暂不判定身份",
        ])
        images = [call.args[0] for call in self.photo.call_args_list]
        self.assertEqual(images[0].size, (32, 24))
        self.assertEqual(images[0].getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(images[0].getpixel((4, 6)), (0, 128, 0))
        self.assertEqual(images[0].getpixel((15, 15)), (0, 128, 0))
        self.assertEqual(images[0].getpixel((16, 16)), (30, 20, 10))
        self.assertEqual(images[1].getpixel((4, 6)), (255, 0, 0))
        self.assertEqual(images[2].getpixel((4, 6)), (30, 20, 10))
        self.assertEqual(images[3].getpixel((4, 6)), (30, 20, 10))
        self.assertTrue(np.array_equal(self.frame, self.original))
        self.assertEqual(
            self.labels.call_args_list[1].kwargs["font"], ("SimSun", 12),
        )
        self.assert_released()

    def test_camera_index_and_read_failure(self):
        with self.assertRaisesRegex(RuntimeError, "摄像头 2读取失败"):
            camera.run_stream(2, self.gallery, threshold=0.3)
        self.open_capture.assert_called_once_with(2)
        self.assertEqual(self.recognize.call_count, 1)
        self.assert_released()

    def test_camera_first_read_failure(self):
        self.capture.read.side_effect = [(False, None)]
        with self.assertRaisesRegex(RuntimeError, "摄像头 0读取失败"):
            camera.run_stream(0, self.gallery, threshold=0.3)
        self.recognize.assert_not_called()
        self.assert_released()

    def test_close_window_stops_before_reading(self):
        def close_window():
            self.window.protocol.call_args.args[1]()

        self.window.update.side_effect = close_window
        camera.run_stream(0, self.gallery, threshold=0.3)
        self.capture.read.assert_not_called()
        self.recognize.assert_not_called()
        self.assert_released()

    def test_exit_keys_stop_after_current_frame(self):
        for key in ("<Escape>", "<KeyPress-q>", "<KeyPress-Q>"):
            with self.subTest(key=key):
                self.capture.reset_mock()
                self.window.reset_mock()
                self.labels.side_effect = [self.preview, self.status, Mock()]
                self.capture.read.side_effect = [(True, self.frame)]

                def press_key():
                    if self.capture.read.called:
                        bindings = dict(
                            call.args
                            for call in self.window.bind.call_args_list
                        )
                        bindings[key](Mock())

                self.window.update.side_effect = press_key
                camera.run_stream(0, self.gallery, threshold=0.3)
                self.capture.read.assert_called_once_with()
                self.assert_released()

    def test_open_failure_releases_capture_without_window(self):
        self.capture.isOpened.return_value = False
        for source in (0, self.video):
            self.capture.reset_mock()
            with self.subTest(source=source), self.assertRaisesRegex(
                RuntimeError, "无法打开",
            ):
                camera.run_stream(source, self.gallery, threshold=0.3)
            self.capture.release.assert_called_once_with()
        self.create_window.assert_not_called()

    def test_recognition_failure_preserves_cause_and_releases(self):
        failure = ValueError("模拟模型故障")
        self.recognize.side_effect = failure
        with self.assertRaisesRegex(
            RuntimeError, "当前帧识别失败.*模拟模型故障",
        ) as caught:
            camera.run_stream(0, self.gallery, threshold=0.3)
        self.assertIs(caught.exception.__cause__, failure)
        self.photo.assert_not_called()
        self.status.configure.assert_not_called()
        self.assert_released()

    def test_read_exception_releases_resources(self):
        self.capture.read.side_effect = cv2.error("模拟解码错误")
        with self.assertRaisesRegex(
            RuntimeError, "读取或画面处理失败.*模拟解码错误",
        ):
            camera.run_stream(self.video, self.gallery, threshold=0.3)
        self.assert_released()

    def test_video_empty_or_truncated_is_error(self):
        self.capture.read.side_effect = [(False, None)]
        with self.assertRaisesRegex(RuntimeError, "未正常读到结尾"):
            camera.run_stream(self.video, self.gallery, threshold=0.3)
        self.assert_released()

    def test_video_failure_before_declared_end_is_error(self):
        self.capture.get.return_value = 3
        with self.assertRaisesRegex(RuntimeError, "未正常读到结尾"):
            camera.run_stream(self.video, self.gallery, threshold=0.3)
        self.assert_released()

    def test_empty_successful_frame_is_error(self):
        self.capture.read.side_effect = [(True, self.frame), (True, None)]
        with self.assertRaisesRegex(RuntimeError, "读取失败，返回空画面"):
            camera.run_stream(self.video, self.gallery, threshold=0.3)
        self.assert_released()

    def test_video_without_frame_count_can_finish(self):
        self.capture.get.return_value = float("nan")
        camera.run_stream(self.video, self.gallery, threshold=0.3)
        self.assert_released()

    def test_missing_recognition_module_reports_error(self):
        with patch.dict(sys.modules, {"cnn_face.recognition": None}):
            with self.assertRaisesRegex(RuntimeError, "无法加载识别模块"):
                camera.run_stream(0, self.gallery, threshold=0.3)
        self.capture.release.assert_called_once_with()
        self.create_window.assert_not_called()

    def test_window_creation_failure_releases_capture(self):
        self.create_window.side_effect = camera.tk.TclError("模拟窗口故障")
        with self.assertRaisesRegex(RuntimeError, "视频窗口运行失败"):
            camera.run_stream(0, self.gallery, threshold=0.3)
        self.capture.release.assert_called_once_with()

    def test_window_update_failure_releases_resources(self):
        self.window.update.side_effect = camera.tk.TclError("模拟事件故障")
        with self.assertRaisesRegex(RuntimeError, "视频窗口运行失败"):
            camera.run_stream(0, self.gallery, threshold=0.3)
        self.assert_released()

    def test_display_failure_releases_resources(self):
        self.photo.side_effect = OSError("模拟图片故障")
        with self.assertRaisesRegex(OSError, "模拟图片故障"):
            camera.run_stream(0, self.gallery, threshold=0.3)
        self.assert_released()

    def test_release_failure_still_destroys_window(self):
        self.capture.release.side_effect = RuntimeError("模拟释放故障")

        def close_window():
            self.window.protocol.call_args.args[1]()

        self.window.update.side_effect = close_window
        with self.assertRaisesRegex(RuntimeError, "模拟释放故障"):
            camera.run_stream(0, self.gallery, threshold=0.3)
        self.assert_released()

    def test_chinese_font_fallback(self):
        self.families.return_value = ("Arial", "Microsoft YaHei")
        camera.run_stream(self.video, self.gallery, threshold=0.3)
        self.assertEqual(
            self.labels.call_args_list[1].kwargs["font"],
            ("Microsoft YaHei", 12),
        )
        self.assert_released()

    def test_missing_chinese_font_releases_resources(self):
        self.families.return_value = ("Arial",)
        with self.assertRaisesRegex(RuntimeError, "未找到支持中文的字体"):
            camera.run_stream(0, self.gallery, threshold=0.3)
        self.assert_released()

    def test_invalid_inputs_do_not_open_resources(self):
        for source in (
            True, -1, "video.avi", Path(self.directory) / "不存在.avi",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                camera.run_stream(source, self.gallery, threshold=0.3)
        for threshold in (-1, float("nan"), True):
            with (
                self.subTest(threshold=threshold),
                self.assertRaises(ValueError),
            ):
                camera.run_stream(0, self.gallery, threshold=threshold)
        with self.assertRaises(ValueError):
            camera.run_stream(0, None, threshold=0.3)
        self.open_capture.assert_not_called()
        self.create_window.assert_not_called()

    def test_real_synthetic_video_finishes(self):
        # VideoCapture 使用真实实现；识别和窗口继续隔离硬件。
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24),
        )
        self.assertTrue(writer.isOpened())
        try:
            for value in (20, 60, 100):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        capture = self.real_capture(str(self.video))
        self.open_capture.return_value = capture
        camera.run_stream(self.video, self.gallery, threshold=0.3)
        self.assertFalse(capture.isOpened())
        self.window.destroy.assert_called_once_with()
        self.assertEqual(self.recognize.call_count, 3)
        for arguments, value in zip(
            self.recognize.call_args_list, (20, 60, 100), strict=True,
        ):
            self.assertTrue(np.allclose(arguments.args[0], value, atol=2))


if __name__ == "__main__":
    unittest.main()
