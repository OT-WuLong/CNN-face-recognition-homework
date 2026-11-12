"""验证共享约定，阻止错误特征和不一致的识别结果流入其他模块。"""

from dataclasses import asdict
import unittest

from cnn_face.contracts import (
    FaceEmbedding,
    Gallery,
    PipelineConfig,
    RecognitionResult,
    validate_threshold,
)


class ContractTests(unittest.TestCase):
    """使用小向量检查格式，真实模型维数由引擎检查。"""

    def test_gallery_round_trip(self):
        gallery = Gallery(1, "WuL0og", PipelineConfig(), 2, [[1.0, 0.0]])
        data = asdict(gallery)
        data["pipeline"] = PipelineConfig(**data["pipeline"])
        self.assertEqual(Gallery(**data), gallery)

    def test_gallery_rejects_invalid_vectors(self):
        for vector in ([1.0], [0.0, 0.0], [float("nan"), 1.0], [True, 1.0]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                Gallery(1, "WuL0og", PipelineConfig(), 2, [vector])

    def test_gallery_rejects_invalid_metadata(self):
        for changes in (
            {"schema_version": True},
            {"identity": " "},
            {"embedding_dim": 0},
            {"embeddings": []},
            {"pipeline": {}},
        ):
            data = dict(
                schema_version=1, identity="WuL0og", pipeline=PipelineConfig(),
                embedding_dim=2, embeddings=[[1.0, 0.0]],
            )
            data.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Gallery(**data)

    def test_pipeline_rejects_incompatible_configuration(self):
        with self.assertRaises(ValueError):
            PipelineConfig(backend_engine="onnx")

    def test_invalid_bbox_is_rejected(self):
        for bbox in ((-1, 0, 10, 10), (0, 0, 0, 10), (0, 0, True, 10)):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                FaceEmbedding([1.0], bbox)

    def test_result_obeys_threshold_boundary(self):
        result = RecognitionResult(
            "matched", "WuL0og", (0, 0, 10, 10), 0.3, 0.3, 1.0
        )
        self.assertEqual(result.identity, "WuL0og")
        with self.assertRaises(ValueError):
            RecognitionResult("unknown", None, (0, 0, 10, 10), 0.3, 0.3, 1.0)

    def test_no_face_cannot_carry_identity(self):
        with self.assertRaises(ValueError):
            RecognitionResult("no_face", "WuL0og", None, None, 0.3, 1.0)
        result = RecognitionResult("no_face", None, None, None, 0.3, 1.0)
        self.assertIsNone(result.identity)

    def test_threshold_rejects_invalid_values(self):
        for value in (-0.1, 2.1, float("inf"), float("nan"), True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_threshold(value)


if __name__ == "__main__":
    unittest.main()
