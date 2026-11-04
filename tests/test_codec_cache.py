import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pyarrow.parquet as pq  # noqa: E402

from speechlmm.data.codec_cache import SHARD_SIZE, CacheRecord, CodecCache  # noqa: E402


class CodecCacheTests(unittest.TestCase):
    def test_lookup_after_flush_and_reopen(self):
        with TemporaryDirectory() as raw:
            cache = CodecCache(Path(raw), "qwen3_omni", "qwen3-distill-v1")
            frames = [[t] * 16 for t in range(4)]
            cache.put(CacheRecord("utt", frames, "ethan"))
            self.assertEqual(cache.lookup("utt").codes, frames)
            cache.flush()
            reopened = CodecCache.from_dir(Path(raw) / "qwen3_omni" / "qwen3-distill-v1")
            record = reopened.lookup("utt")
            self.assertEqual(record.codes, frames)
            self.assertEqual(record.speaker, "ethan")
            self.assertIsNone(reopened.lookup("other"))

    def test_put_rejects_the_wrong_codebook_count(self):
        cache = CodecCache(Path("/unused"), "qwen2_5_omni", "qwen25-distill-v1")
        with self.assertRaisesRegex(ValueError, "16 codebooks"):
            cache.put(CacheRecord("utt", [[1] * 16]))
        with self.assertRaisesRegex(ValueError, "no codec frames"):
            cache.put(CacheRecord("utt", []))
        with self.assertRaisesRegex(ValueError, "Unknown backbone"):
            CodecCache(Path("/unused"), "mimi", "tag")

    def test_flush_fills_shards_in_order(self):
        with TemporaryDirectory() as raw:
            cache = CodecCache(Path(raw), "qwen2_5_omni", "qwen25-distill-v1")
            for i in range(SHARD_SIZE + 3):
                cache.put(CacheRecord(f"utt-{i}", [i, i + 1]))
            cache.flush()
            self.assertTrue((cache.dir / "shards" / "000000.parquet").is_file())
            self.assertTrue((cache.dir / "shards" / "000001.parquet").is_file())
            reopened = CodecCache.from_dir(cache.dir)
            self.assertEqual(reopened.lookup("utt-0").codes, [0, 1])
            self.assertEqual(reopened.lookup(f"utt-{SHARD_SIZE + 2}").codes, [SHARD_SIZE + 2, SHARD_SIZE + 3])

    def test_lookup_reuses_the_open_shard(self):
        with TemporaryDirectory() as raw:
            cache = CodecCache(Path(raw), "qwen2_5_omni", "qwen25-distill-v1")
            cache.put(CacheRecord("a", [1, 2, 3]))
            cache.put(CacheRecord("b", [4, 5, 6]))
            cache.flush()
            reader = CodecCache.from_dir(cache.dir)
            real_read = pq.read_table
            reads = []

            def counting_read(path, *args, **kwargs):
                reads.append(Path(path).name)
                return real_read(path, *args, **kwargs)

            with patch("speechlmm.data.codec_cache.pq.read_table", counting_read):
                self.assertEqual(reader.lookup("a").codes, [1, 2, 3])
                self.assertEqual(reader.lookup("b").codes, [4, 5, 6])
            self.assertEqual(reads.count("000000.parquet"), 1)


if __name__ == "__main__":
    unittest.main()
