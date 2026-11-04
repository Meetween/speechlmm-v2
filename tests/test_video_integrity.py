"""CPU tests for corrupted-video filtering (no decoder backends required)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from speechlmm.data.video_integrity import (
    MISSING_FILE_REASON,
    VideoCheck,
    check_video,
    check_videos,
)


class PathCheckTests(unittest.TestCase):
    def test_missing_empty_and_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing = check_video(tmp_path / "nope.mp4")
            self.assertFalse(missing.ok)
            self.assertEqual(missing.reason, MISSING_FILE_REASON)

            empty = tmp_path / "empty.mp4"
            empty.write_bytes(b"")
            empty_check = check_video(empty)
            self.assertFalse(empty_check.ok)
            self.assertEqual(empty_check.reason, "file is empty")

            dir_check = check_video(tmp_path)
            self.assertFalse(dir_check.ok)
            self.assertEqual(dir_check.reason, "not a regular file")

    def test_check_videos_skips_nulls_and_dedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = check_videos(
                [None, "", "a.mp4", "a.mp4", None],
                media_root=Path(tmp),
            )
            self.assertEqual(list(results), ["a.mp4"])
            self.assertFalse(results["a.mp4"].ok)
            self.assertEqual(results["a.mp4"].reason, MISSING_FILE_REASON)


@unittest.skipUnless(HAS_PYARROW, "pyarrow is required")
class FilterConditionTests(unittest.TestCase):
    def test_keeps_audio_only_and_rejects_listed_paths(self):
        import pyarrow as pa

        from speechlmm.data.filter_parquet import video_integrity_condition

        table = pa.table(
            {
                "video_input_path": [None, "", "good.mp4", "bad.mp4", "good.mp4"],
            }
        )
        fake = {
            "good.mp4": VideoCheck(Path("good.mp4"), ok=True),
            "bad.mp4": VideoCheck(Path("bad.mp4"), ok=False, reason="truncated"),
        }
        with patch("speechlmm.data.filter_parquet.check_videos", return_value=fake):
            masks = video_integrity_condition(table, media_root=Path("/data"))
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].to_pylist(), [True, True, True, False, True])

    def test_all_paths_missing_raises(self):
        import pyarrow as pa

        from speechlmm.data.filter_parquet import video_integrity_condition

        table = pa.table({"video_input_path": ["a.mp4", "b.mp4"]})
        fake = {
            "a.mp4": VideoCheck(Path("a.mp4"), ok=False, reason=MISSING_FILE_REASON),
            "b.mp4": VideoCheck(Path("b.mp4"), ok=False, reason=MISSING_FILE_REASON),
        }
        with patch("speechlmm.data.filter_parquet.check_videos", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "none of the 2 videos exist"):
                video_integrity_condition(table, media_root=Path("/wrong"))

    def test_genuinely_corrupted_shard_does_not_raise(self):
        import pyarrow as pa

        from speechlmm.data.filter_parquet import video_integrity_condition

        table = pa.table({"video_input_path": ["only.mp4"]})
        fake = {"only.mp4": VideoCheck(Path("only.mp4"), ok=False, reason="truncated")}
        with patch("speechlmm.data.filter_parquet.check_videos", return_value=fake):
            masks = video_integrity_condition(table, media_root=Path("/data"))
        self.assertEqual(masks[0].to_pylist(), [False])


@unittest.skipUnless(HAS_PYARROW, "pyarrow is required")
class MediaRootAndCliTests(unittest.TestCase):
    def test_media_root_is_datasets_root_not_parent_parent(self):
        from speechlmm.data.corpus_preparer import CorpusPreparer

        class Dummy(CorpusPreparer):
            name = "lipcrops"

            def to_parquet(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["DATASETS_DIR"] = tmp
            self.addCleanup(os.environ.pop, "DATASETS_DIR", None)
            version_dir = Path(tmp) / "lipcrops" / "v1.0"
            preparer = Dummy(dataset_root=version_dir)
            self.assertEqual(preparer._media_root(), Path(tmp))
            # The old parent.parent climb would be tmp's parent when dataset_root
            # is datasets_dir()/name without a version component.
            shallow = Dummy(dataset_root=Path(tmp) / "lipcrops")
            self.assertEqual(shallow._media_root(), Path(tmp))
            self.assertNotEqual(shallow._media_root(), Path(tmp).parent)

    def test_prepare_honors_no_check_flag(self):
        from speechlmm.data.corpus_preparer import (
            CorpusPreparer,
            add_video_integrity_args,
            apply_video_integrity_args,
        )

        parser = argparse.ArgumentParser()
        add_video_integrity_args(parser)
        args = parser.parse_args(["--no-check-video-integrity"])

        class Dummy(CorpusPreparer):
            name = "dummy"

            def to_parquet(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["DATASETS_DIR"] = tmp
            self.addCleanup(os.environ.pop, "DATASETS_DIR", None)
            preparer = Dummy(dataset_root=Path(tmp) / "dummy" / "v1.0")
            self.assertTrue(preparer.check_video_integrity)
            apply_video_integrity_args(preparer, args)
            self.assertFalse(preparer.check_video_integrity)
            self.assertFalse(preparer.deep_video_check)


if __name__ == "__main__":
    unittest.main()
