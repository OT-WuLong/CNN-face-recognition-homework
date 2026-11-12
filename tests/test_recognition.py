"""用替身与人工向量验证识别逻辑；测试阈值不代表实际模型阈值。"""

import math
import unittest
from unittest.mock import patch

import numpy as np

from cnn_face import engine, recognition
from cnn_face.contracts import FaceEmbedding, Gallery, PipelineConfig


class RecognitionTests(unittest.TestCase):
    """隔离模型加载和硬件，检查状态、距离边界及技术错误。"""

    def setUp(self):
        self.frame = np.zeros((60, 80, 3), dtype=np.uint8)
        self.bbox = (10, 20, 30, 40)
        self.gallery = Gallery(
            1, "登记昵称", PipelineConfig(), 2, [[1.0, 0.0]]
        )
        patcher = patch.object(engine, "extract_embeddings")
        self.extract = patcher.start()
        self.addCleanup(patcher.stop)

    def test_matched_returns_identity_bbox_and_cosine_distance(self):
        self.extract.return_value = [FaceEmbedding([3.0, 4.0], self.bbox)]
        result = recognition.recognize(self.frame, self.gallery, threshold=0.5)
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.identity, "登记昵称")
        self.assertEqual(result.bbox, self.bbox)
        self.assertAlmostEqual(result.distance, 0.4)
        self.assertEqual(result.threshold, 0.5)

    def test_unknown_keeps_distance_without_identity(self):
        self.extract.return_value = [FaceEmbedding([0.0, 1.0], self.bbox)]
        result = recognition.recognize(self.frame, self.gallery, threshold=0.5)
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.identity)
        self.assertEqual(result.bbox, self.bbox)
        self.assertEqual(result.distance, 1.0)

    def test_no_face_and_multiple_faces_skip_identity_comparison(self):
        # 多脸使用不同维数，若错误地进入身份比对便会触发维数异常。
        face = FaceEmbedding([1.0, 0.0, 0.0], self.bbox)
        for count, status in (
            (0, "no_face"), (2, "multiple_faces"), (3, "multiple_faces"),
        ):
            with self.subTest(count=count):
                self.extract.return_value = [face] * count
                result = recognition.recognize(
                    self.frame, self.gallery, threshold=0.5
                )
                self.assertEqual(result.status, status)
                self.assertIsNone(result.identity)
                self.assertIsNone(result.bbox)
                self.assertIsNone(result.distance)
                self.assertEqual(result.threshold, 0.5)

    def test_minimum_distance_uses_all_reference_vectors(self):
        gallery = Gallery(
            1, "登记昵称", PipelineConfig(), 2,
            [[-1.0, 0.0], [0.0, 1.0], [3.0, 4.0]],
        )
        self.extract.return_value = [FaceEmbedding([1.0, 0.0], self.bbox)]
        result = recognition.recognize(self.frame, gallery, threshold=0.5)
        self.assertEqual(result.status, "matched")
        self.assertAlmostEqual(result.distance, 0.4)

    def test_threshold_equality_and_adjacent_floats(self):
        self.extract.return_value = [FaceEmbedding([0.0, 1.0], self.bbox)]
        cases = (
            (math.nextafter(1.0, 0.0), "unknown"),
            (1.0, "matched"),
            (math.nextafter(1.0, 2.0), "matched"),
        )
        for threshold, status in cases:
            with self.subTest(threshold=threshold):
                result = recognition.recognize(
                    self.frame, self.gallery, threshold=threshold
                )
                self.assertEqual(result.status, status)
                self.assertEqual(result.distance, 1.0)
                self.assertEqual(result.threshold, threshold)

    def test_cosine_endpoints_accept_threshold_zero_and_two(self):
        for vector, threshold in (([1.0, 0.0], 0.0), ([-1.0, 0.0], 2.0)):
            with self.subTest(threshold=threshold):
                self.extract.return_value = [FaceEmbedding(vector, self.bbox)]
                result = recognition.recognize(
                    self.frame, self.gallery, threshold=threshold
                )
                self.assertEqual(result.status, "matched")
                self.assertEqual(result.distance, threshold)

    def test_roundoff_at_cosine_endpoints_is_clipped(self):
        gallery = Gallery(1, "登记昵称", PipelineConfig(), 3, [[1.0] * 3])
        for vector, distance in (([1.0] * 3, 0.0), ([-1.0] * 3, 2.0)):
            with self.subTest(vector=vector):
                self.extract.return_value = [FaceEmbedding(vector, self.bbox)]
                result = recognition.recognize(
                    self.frame, gallery, threshold=0.0
                )
                self.assertEqual(result.distance, distance)
                self.assertEqual(
                    result.status, "matched" if distance == 0.0 else "unknown"
                )

    def test_extreme_finite_vectors_do_not_overflow_or_underflow(self):
        for query_scale, reference_scale in (
            (1e308, 1e308), (1e-308, 1e-308),
            (1e308, 1e-308), (math.ulp(0.0), math.ulp(0.0)),
        ):
            with self.subTest(query=query_scale, reference=reference_scale):
                gallery = Gallery(
                    1, "登记昵称", PipelineConfig(), 2,
                    [[reference_scale, 0.0]],
                )
                self.extract.return_value = [
                    FaceEmbedding([query_scale, 0.0], self.bbox)
                ]
                result = recognition.recognize(
                    self.frame, gallery, threshold=0.0
                )
                self.assertEqual(result.status, "matched")
                self.assertEqual(result.distance, 0.0)

    def test_elapsed_time_is_recorded_for_every_normal_status(self):
        cases = (
            ([], "no_face"),
            ([FaceEmbedding([1.0, 0.0], self.bbox)] * 2, "multiple_faces"),
            ([FaceEmbedding([1.0, 0.0], self.bbox)], "matched"),
            ([FaceEmbedding([0.0, 1.0], self.bbox)], "unknown"),
        )
        for faces, status in cases:
            with (
                self.subTest(status=status),
                patch.object(
                    recognition, "perf_counter", return_value=10.0
                ) as clock,
            ):
                def extract(frame, *, pipeline):
                    clock.return_value = 10.125
                    return faces

                self.extract.side_effect = extract
                result = recognition.recognize(
                    self.frame, self.gallery, threshold=0.5
                )
                self.assertEqual(result.status, status)
                self.assertEqual(result.elapsed_ms, 125.0)

    def test_passes_frame_and_pipeline_without_initializing_runtime(self):
        self.extract.return_value = []
        with patch.object(engine, "configure_runtime") as configure:
            for _ in range(2):
                recognition.recognize(self.frame, self.gallery, threshold=0.5)
            configure.assert_not_called()
        self.assertEqual(self.extract.call_count, 2)
        for call in self.extract.call_args_list:
            self.assertIs(call.args[0], self.frame)
            self.assertIs(call.kwargs["pipeline"], self.gallery.pipeline)

    def test_invalid_threshold_is_rejected_before_extraction(self):
        for threshold in (
            -0.1, 2.1, float("nan"), float("inf"), -float("inf"),
            True, None, "0.5",
        ):
            with (
                self.subTest(threshold=threshold),
                self.assertRaises(ValueError),
            ):
                recognition.recognize(
                    self.frame, self.gallery, threshold=threshold
                )
        self.extract.assert_not_called()

    def test_query_dimension_must_match_gallery(self):
        self.extract.return_value = [FaceEmbedding([1.0, 0.0, 0.0], self.bbox)]
        with self.assertRaisesRegex(ValueError, "维数"):
            recognition.recognize(self.frame, self.gallery, threshold=0.5)

    def test_mutated_query_vectors_are_revalidated(self):
        for vector in (
            [], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0],
            [-float("inf"), 1.0], [True, 1.0], ["1", 0.0],
        ):
            with self.subTest(vector=vector):
                face = FaceEmbedding([1.0, 0.0], self.bbox)
                face.embedding[:] = vector
                self.extract.return_value = [face]
                with self.assertRaises(ValueError):
                    recognition.recognize(
                        self.frame, self.gallery, threshold=0.5
                    )

    def test_mutated_gallery_vectors_are_revalidated(self):
        for vector in (
            [1.0], [], [0.0, 0.0], [float("nan"), 1.0],
            [float("inf"), 1.0], [-float("inf"), 1.0], [True, 1.0],
            ["1", 0.0],
        ):
            with self.subTest(vector=vector):
                # 即使首个参考向量匹配，后续坏向量也不能被忽略。
                self.gallery.embeddings[:] = [[1.0, 0.0], vector]
                self.extract.return_value = [
                    FaceEmbedding([1.0, 0.0], self.bbox)
                ]
                with self.assertRaises(ValueError):
                    recognition.recognize(
                        self.frame, self.gallery, threshold=0.5
                    )
        self.extract.assert_not_called()

    def test_empty_or_invalid_gallery_is_rejected(self):
        self.gallery.embeddings.clear()
        for gallery in (self.gallery, None):
            with self.subTest(gallery=gallery), self.assertRaises(ValueError):
                recognition.recognize(self.frame, gallery, threshold=0.5)
        self.extract.assert_not_called()

    def test_incompatible_pipeline_is_rejected_before_extraction(self):
        # 模拟绕过构造校验的配置，检查识别入口仍拒绝混用特征。
        object.__setattr__(self.gallery.pipeline, "normalization", "raw")
        with self.assertRaisesRegex(ValueError, "不兼容"):
            recognition.recognize(self.frame, self.gallery, threshold=0.5)
        self.extract.assert_not_called()

    def test_engine_errors_propagate_unchanged(self):
        for error in (ValueError("模拟输入错误"), RuntimeError("模拟模型故障")):
            with self.subTest(error=error):
                self.extract.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    recognition.recognize(
                        self.frame, self.gallery, threshold=0.5
                    )
                self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
