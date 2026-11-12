"""用模型替身验证注册、严格加载和失败时的原库保护。"""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from cnn_face import engine, enrollment
from cnn_face.contracts import FaceEmbedding, Gallery, PipelineConfig


class EnrollmentTests(unittest.TestCase):
    """真实解码本地图片，仅替换特征提取，不加载模型或使用 GPU。"""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images_dir = self.root / "注册照片"
        self.images_dir.mkdir()
        self.gallery_path = self.root / "输出" / "本人特征库.json"
        self.pipeline = PipelineConfig()
        self.face = FaceEmbedding([1.0, 0.0], (0, 0, 2, 2))

    def write_image(self, name="本人.png", value=10):
        frame = np.full((4, 4, 3), value, dtype=np.uint8)
        success, encoded = cv2.imencode(".png", frame)
        self.assertTrue(success)
        path = self.images_dir / name
        encoded.tofile(path)
        return path

    def write_existing_gallery(self):
        gallery = Gallery(
            1, "原昵称", self.pipeline, 2, [[1.0, 0.0], [0.0, 1.0]],
        )
        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        self.gallery_path.write_text(
            json.dumps(asdict(gallery), ensure_ascii=False), encoding="utf-8",
        )
        return self.gallery_path.read_bytes()

    def enroll(self, identity="本人昵称"):
        return enrollment.enroll(
            identity, self.images_dir, self.gallery_path,
            pipeline=self.pipeline,
        )

    def test_registration_is_sorted_and_round_trips_utf8(self):
        paths = [
            self.write_image("乙.png", 30),
            self.write_image("02.JPG", 20),
            self.write_image("01.png", 10),
        ]
        (self.images_dir / "说明.txt").write_text("忽略非图片", encoding="utf-8")
        (self.images_dir / "子目录.png").mkdir()

        def extract(frame, *, pipeline):
            self.assertIs(pipeline, self.pipeline)
            self.assertEqual(frame.dtype, np.uint8)
            self.assertEqual(frame.shape, (4, 4, 3))
            return [FaceEmbedding([float(frame[0, 0, 0]), 1.0], (0, 0, 2, 2))]

        with (
            patch.object(
                engine, "read_image", wraps=engine.read_image,
            ) as read,
            patch.object(engine, "extract_embeddings", side_effect=extract),
            patch.object(engine, "configure_runtime") as configure,
        ):
            gallery = self.enroll()
        self.assertEqual(
            read.call_args_list, [call(path) for path in paths[::-1]],
        )
        configure.assert_not_called()
        self.assertEqual(gallery.embedding_dim, 2)
        self.assertEqual(
            gallery.embeddings, [[10.0, 1.0], [20.0, 1.0], [30.0, 1.0]],
        )
        self.assertEqual(enrollment.load_gallery(self.gallery_path), gallery)
        content = self.gallery_path.read_bytes().decode("utf-8")
        self.assertIn("本人昵称", content)
        self.assertNotIn("\\u", content)
        self.assertEqual(set(json.loads(content)), set(asdict(gallery)))
        self.assertEqual(
            list(self.gallery_path.parent.iterdir()), [self.gallery_path],
        )

    def test_reregistration_replaces_whole_gallery_atomically(self):
        original = self.write_existing_gallery()
        self.write_image()
        replace = enrollment.os.replace

        def extract(frame, *, pipeline):
            self.assertEqual(self.gallery_path.read_bytes(), original)
            return [self.face]

        def replace_checked(source, target):
            self.assertEqual(Path(source).parent, self.gallery_path.parent)
            self.assertEqual(target, self.gallery_path)
            self.assertEqual(self.gallery_path.read_bytes(), original)
            self.assertEqual(
                enrollment.load_gallery(Path(source)).identity, "新昵称",
            )
            replace(source, target)

        with (
            patch.object(engine, "extract_embeddings", side_effect=extract),
            patch.object(
                enrollment.os, "replace", side_effect=replace_checked,
            ) as save,
        ):
            gallery = self.enroll("新昵称")
        save.assert_called_once()
        self.assertEqual(gallery.embeddings, [self.face.embedding])
        self.assertEqual(enrollment.load_gallery(self.gallery_path), gallery)

    def test_invalid_metadata_does_not_read_images_or_replace_gallery(self):
        original = self.write_existing_gallery()
        self.write_image()
        for changes in (
            {"identity": ""}, {"identity": " \t"}, {"identity": None},
            {"identity": 123}, {"pipeline": {}}, {"pipeline": None},
        ):
            arguments = dict(
                identity="本人", images_dir=self.images_dir,
                gallery_path=self.gallery_path, pipeline=self.pipeline,
            )
            arguments.update(changes)
            with (
                self.subTest(changes=changes),
                patch.object(engine, "read_image") as read,
                self.assertRaises(ValueError),
            ):
                enrollment.enroll(**arguments)
            read.assert_not_called()
            self.assertEqual(self.gallery_path.read_bytes(), original)

    def test_missing_non_directory_and_empty_inputs_are_rejected(self):
        original = self.write_existing_gallery()
        ordinary_file = self.root / "普通文件"
        ordinary_file.write_text("不是目录", encoding="utf-8")
        empty_directory = self.root / "空目录"
        empty_directory.mkdir()
        (self.images_dir / "说明.txt").write_text("没有图片", encoding="utf-8")
        for directory, reason in (
            (self.root / "不存在", "不存在或不是目录"),
            (ordinary_file, "不存在或不是目录"),
            (empty_directory, "没有支持的图片"),
            (self.images_dir, "没有支持的图片"),
        ):
            with self.subTest(directory=directory), self.assertRaisesRegex(
                ValueError, reason,
            ) as raised:
                enrollment.enroll(
                    "本人", directory, self.gallery_path, pipeline=self.pipeline,
                )
            self.assertIn(str(directory), str(raised.exception))
            self.assertEqual(self.gallery_path.read_bytes(), original)

    def test_invalid_images_and_face_counts_preserve_original_gallery(self):
        original = self.write_existing_gallery()
        path = self.write_image()
        for results, reason in (([], "未检测到人脸"), ([self.face] * 2, "2 张人脸")):
            with (
                self.subTest(reason=reason),
                patch.object(
                    engine, "extract_embeddings", return_value=results,
                ),
                self.assertRaisesRegex(ValueError, reason) as raised,
            ):
                self.enroll()
            self.assertIn(str(path), str(raised.exception))
            self.assertEqual(self.gallery_path.read_bytes(), original)
        path.write_bytes(b"broken image")
        with self.assertRaisesRegex(ValueError, "无法解码图片") as raised:
            self.enroll()
        self.assertIn(str(path), str(raised.exception))
        self.assertEqual(self.gallery_path.read_bytes(), original)

    def test_later_model_failure_does_not_create_or_replace_gallery(self):
        self.write_image("01.png")
        failed_path = self.write_image("02.png")
        for exists in (False, True):
            original = self.write_existing_gallery() if exists else None
            with (
                self.subTest(existing_gallery=exists),
                patch.object(
                    engine, "extract_embeddings",
                    side_effect=[[self.face], RuntimeError("模拟模型故障")],
                ),
                self.assertRaisesRegex(RuntimeError, "模拟模型故障") as raised,
            ):
                self.enroll()
            self.assertIn(str(failed_path), str(raised.exception))
            if exists:
                self.assertEqual(self.gallery_path.read_bytes(), original)
            else:
                self.assertFalse(self.gallery_path.exists())
                self.assertFalse(self.gallery_path.parent.exists())

    def test_invalid_extracted_vectors_are_rejected_before_saving(self):
        original = self.write_existing_gallery()
        self.write_image("01.png")
        failed_path = self.write_image("02.png")
        for vector in ([1.0], [0.0, 0.0], [float("inf"), 1.0], [True, 1.0]):
            with (
                self.subTest(vector=vector),
                patch.object(
                    engine, "extract_embeddings",
                    side_effect=[
                        [self.face], [SimpleNamespace(embedding=vector)],
                    ],
                ),
                self.assertRaises(ValueError) as raised,
            ):
                self.enroll()
            self.assertIn(str(failed_path), str(raised.exception))
            self.assertEqual(self.gallery_path.read_bytes(), original)
        self.assertEqual(
            list(self.gallery_path.parent.iterdir()), [self.gallery_path],
        )

    def test_save_failures_clean_only_own_temporary_files(self):
        original = self.write_existing_gallery()
        self.write_image()
        unrelated = self.gallery_path.parent / ".other-task.tmp"
        unrelated.write_text("保留其他任务文件", encoding="utf-8")
        factory = enrollment.NamedTemporaryFile

        def fail_write(**arguments):
            stream = factory(**arguments)
            stream.write = Mock(side_effect=OSError("模拟写入失败"))
            return stream

        failures = (
            (
                enrollment, "NamedTemporaryFile",
                {"side_effect": OSError("创建失败")},
            ),
            (enrollment, "NamedTemporaryFile", {"side_effect": fail_write}),
            (enrollment.os, "fsync", {"side_effect": OSError("同步失败")}),
            (enrollment.os, "replace", {"side_effect": OSError("替换失败")}),
        )
        for target, name, arguments in failures:
            with (
                self.subTest(stage=arguments),
                patch.object(
                    engine, "extract_embeddings", return_value=[self.face],
                ),
                patch.object(target, name, **arguments),
                self.assertRaises(OSError),
            ):
                self.enroll()
            self.assertEqual(self.gallery_path.read_bytes(), original)
            self.assertEqual(
                set(self.gallery_path.parent.iterdir()),
                {self.gallery_path, unrelated},
            )
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "保留其他任务文件")

    def test_load_rejects_invalid_metadata_structure_and_vectors(self):
        self.write_existing_gallery()
        valid = json.loads(self.gallery_path.read_text(encoding="utf-8"))
        invalid = [None, [], "特征库", 1, {}, {**valid, "unknown": 1}]
        invalid.extend(
            {key: value for key, value in valid.items() if key != missing}
            for missing in valid
        )
        for key, values in {
            "schema_version": [0, 2, True, "1", 1.0, None],
            "identity": ["", " \t", None, 123, []],
            "pipeline": [None, [], "默认", {}],
            "embedding_dim": [0, -1, True, 2.0, "2", None],
            "embeddings": [
                None, {}, [], [None], [[1.0]], [[0.0, 0.0]],
                [[True, 1.0]], [["1", 0.0]], [[None, 1.0]],
                [[float("nan"), 1.0]], [[float("inf"), 1.0]],
                [[float("-inf"), 1.0]], [[10 ** 400, 1.0]],
                [[1.0, 0.0], [1.0]],
            ],
        }.items():
            invalid.extend({**valid, key: value} for value in values)
        for key in valid["pipeline"]:
            pipeline = valid["pipeline"]
            invalid.append({**valid, "pipeline": {
                name: value for name, value in pipeline.items() if name != key
            }})
            invalid.append({**valid, "pipeline": {**pipeline, key: None}})
        invalid.append({
            **valid, "pipeline": {**valid["pipeline"], "unknown": 1},
        })
        for key, value in (
            ("model_name", "Facenet"), ("detector_backend", "skip"),
            ("align", 1), ("normalization", "Facenet"),
            ("backend_engine", "onnx"), ("distance_metric", "euclidean"),
        ):
            invalid.append({
                **valid, "pipeline": {**valid["pipeline"], key: value},
            })
        for data in invalid:
            with self.subTest(data=data):
                self.gallery_path.write_text(
                    json.dumps(data), encoding="utf-8",
                )
                with self.assertRaises(ValueError) as raised:
                    enrollment.load_gallery(self.gallery_path)
                self.assertIn(str(self.gallery_path), str(raised.exception))

    def test_load_rejects_malformed_duplicate_and_non_utf8_json(self):
        self.write_existing_gallery()
        valid = self.gallery_path.read_text(encoding="utf-8")
        for content in (
            b"", b"{broken", b"\xff\xfe", valid.encode("utf-16"),
            (valid + " {}").encode("utf-8"),
            valid.replace(
                '"schema_version": 1', '"schema_version": 1e999',
            ).encode(),
            valid.replace(
                '"identity":', '"identity": "重复", "identity":',
            ).encode(),
            valid.replace(
                '"align": true', '"align": false, "align": true',
            ).encode(),
        ):
            with self.subTest(content=content):
                self.gallery_path.write_bytes(content)
                with self.assertRaises(ValueError):
                    enrollment.load_gallery(self.gallery_path)

    def test_load_does_not_hide_filesystem_errors(self):
        with self.assertRaises(FileNotFoundError):
            enrollment.load_gallery(self.gallery_path)
        with self.assertRaises(OSError):
            enrollment.load_gallery(self.images_dir)


if __name__ == "__main__":
    unittest.main()
