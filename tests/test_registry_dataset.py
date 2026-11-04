"""dataset_info column inference for ShareGPT rows."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speechlmm.data.registry_dataset import columns_for_entries  # noqa: E402


class RegistryDatasetTests(unittest.TestCase):
    def test_always_declares_messages(self):
        cols = columns_for_entries([{"messages": []}])
        self.assertEqual(cols, {"messages": "messages"})

    def test_declares_speaker_and_sample_id(self):
        cols = columns_for_entries(
            [
                {
                    "messages": [],
                    "speaker": "ethan",
                    "sample_id": "utt-1",
                    "codec_tokens": [1, 2],
                }
            ]
        )
        self.assertEqual(cols["speaker"], "speaker")
        self.assertEqual(cols["sample_id"], "sample_id")
        self.assertEqual(cols["codec_tokens"], "codec_tokens")

    def test_skips_keys_absent_from_rows(self):
        cols = columns_for_entries([{"messages": [], "audios": ["a.wav"]}])
        self.assertIn("audios", cols)
        self.assertNotIn("speaker", cols)
        self.assertNotIn("sample_id", cols)
        self.assertNotIn("codec_tokens", cols)


if __name__ == "__main__":
    unittest.main()
