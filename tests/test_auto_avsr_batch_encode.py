"""Ragged AutoAVSR batches match per-sample encode in eval mode."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AutoAVSRBatchEncodeTests(unittest.TestCase):
    def test_ragged_batch_matches_single_samples(self):
        import torch

        from speechlmm.models.encoders.auto_avsr import AutoAVSR

        torch.manual_seed(0)
        model = AutoAVSR().eval()
        frames = 16
        lengths = [16, 9]
        videos = torch.randn(2, frames, 1, 88, 88)
        videos[1, lengths[1] :] = 0

        with torch.no_grad():
            batched, projected = model.encode(videos, lengths=lengths)
            alone0, _ = model.encode(videos[0:1, : lengths[0]], lengths=[lengths[0]])
            alone1, _ = model.encode(videos[1:2, : lengths[1]], lengths=[lengths[1]])
            raw, _ = model._encode_raw(videos, lengths=lengths)

        self.assertIsNotNone(projected)
        self.assertEqual(tuple(batched.shape), (2, frames, 768))
        self.assertLess((batched[0, : lengths[0]] - alone0[0]).abs().max().item(), 1e-3)
        self.assertLess((batched[1, : lengths[1]] - alone1[0]).abs().max().item(), 1e-3)
        self.assertEqual(batched[1, lengths[1] :].abs().max().item(), 0.0)
        self.assertGreater((raw[1, : lengths[1]] - alone1[0]).abs().max().item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
