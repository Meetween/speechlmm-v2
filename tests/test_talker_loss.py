"""Talker teacher-forcing helper tests (no GPU, no checkpoints)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speechlmm.models.talker_loss import (  # noqa: E402
    build_streaming_text,
    codec_targets,
    continue_position_ids,
    dummy_talker_forward_loss,
    exclude_trailing_im_end,
    pad_codec_batch,
    reduce_talker_losses,
    rvq_codec_embeds,
    token_ce_sum,
)


class TalkerLossHelperTests(unittest.TestCase):
    def test_target_alignment_lm(self):
        codes = torch.tensor([10, 11, 12])
        target = codec_targets(context_len=5, prefill_len=2, codes=codes, eos_id=8294)
        self.assertEqual(target.shape[0], 5 + 2 + 3)
        self.assertEqual(int((target != -100).sum()), 4)
        self.assertEqual(int(target[6]), 10)
        self.assertEqual(int(target[-1]), 8294)

    def test_target_alignment_xl(self):
        codes = torch.tensor([1, 2, 3, 4])
        target = codec_targets(context_len=7, prefill_len=9, codes=codes, eos_id=2150)
        self.assertEqual(int((target != -100).nonzero(as_tuple=True)[0][0]), 7 + 8)
        self.assertEqual(int(target[-1]), 2150)
        self.assertEqual(target.shape[0], 7 + 9 + 4)

    def test_no_double_shift_zero_loss(self):
        vocab = 32
        codes = torch.tensor([3, 5, 7])
        target = codec_targets(2, 2, codes, eos_id=9)
        logits = torch.zeros(1, target.numel(), vocab)
        for index, token in enumerate(target.tolist()):
            if token != -100:
                logits[0, index, token] = 20.0
        loss, n_valid = token_ce_sum(logits, target.unsqueeze(0))
        self.assertEqual(n_valid, 4)
        self.assertLess(float(loss / n_valid), 0.01)

    def test_ignore_index_excludes_context(self):
        vocab = 16
        target = torch.tensor([[-100, -100, 3, 4]])
        logits = torch.zeros(1, 4, vocab)
        logits[0, 2, 3] = 10.0
        logits[0, 3, 4] = 10.0
        base, _ = token_ce_sum(logits, target)
        logits[0, 0, 1] = 50.0
        perturbed, _ = token_ce_sum(logits, target)
        self.assertEqual(float(base), float(perturbed))

    def test_token_mean_vs_sample_mean(self):
        short = torch.tensor(2.0)
        long = torch.tensor(10.0)
        token = reduce_talker_losses([short, long], [1, 9], "token")
        sample = reduce_talker_losses([short, long], [1, 9], "sample")
        self.assertAlmostEqual(float(token), 12.0 / 10.0)
        self.assertAlmostEqual(float(sample), (2.0 / 1 + 10.0 / 9) / 2)
        self.assertNotAlmostEqual(float(token), float(sample))

    def test_streaming_pad_hold(self):
        trailing = torch.arange(6, dtype=torch.float).view(2, 3)
        pad = torch.full((1, 3), 9.0)
        out = build_streaming_text(trailing, pad, 4)
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertTrue(torch.equal(out[2:], pad.expand(2, -1)))

    def test_pad_single_codebook(self):
        labels, mask = pad_codec_batch([[1, 2], None, [3]], num_codebooks=1, ignore_index=-100)
        self.assertEqual(tuple(labels.shape), (3, 2))
        self.assertEqual(labels[1].tolist(), [-100, -100])
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_pad_multi_codebook(self):
        sample = [[1, 2, 3], [4, 5, 6]]
        labels, mask = pad_codec_batch([sample, None], num_codebooks=3)
        self.assertEqual(tuple(labels.shape), (2, 2, 3))
        self.assertTrue(bool(mask[0]))
        self.assertFalse(bool(mask[1]))

    def test_codebook_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pad_codec_batch([[[1, 2]]], num_codebooks=16)

    def test_dummy_batch(self):
        labels, mask = pad_codec_batch([None, None], num_codebooks=16, dummy=True)
        self.assertEqual(tuple(labels.shape), (2, 1, 16))
        self.assertFalse(bool(mask.any()))

    def test_continue_position_ids(self):
        prefix = torch.arange(4).view(1, 1, 4).expand(3, 1, 4).clone()
        out = continue_position_ids(prefix, 2)
        self.assertEqual(tuple(out.shape), (3, 1, 6))
        self.assertEqual(int(out[0, 0, -1]), 5)

    def test_dummy_talker_zero_loss(self):
        talker = torch.nn.Module()
        talker.codec_head = torch.nn.Linear(4, 8)
        talker.model = torch.nn.Identity()
        talker.thinker_to_talker_proj = torch.nn.Linear(6, 4)

        def _forward(inputs_embeds=None, attention_mask=None, **kwargs):
            return (inputs_embeds,)

        talker.model.forward = _forward
        loss = dummy_talker_forward_loss(talker, torch.device("cpu"), torch.float32)
        self.assertIsNotNone(loss)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(talker.codec_head.weight.grad)
        self.assertTrue(torch.count_nonzero(talker.codec_head.weight.grad) == 0)

    def test_exclude_trailing_im_end(self):
        ids = torch.tensor([10, 11, 12, 99])
        self.assertEqual(exclude_trailing_im_end(ids, 4, 99), 3)
        self.assertEqual(exclude_trailing_im_end(ids, 4, 0), 4)
        self.assertEqual(exclude_trailing_im_end(ids, 0, 99), 0)

    def test_rvq_codec_embeds_sums_books(self):
        talker = torch.nn.Module()
        talker.model = torch.nn.Module()
        talker.model.codec_embedding = torch.nn.Embedding(8, 4)
        torch.nn.init.ones_(talker.model.codec_embedding.weight)
        talker.code_predictor = torch.nn.Module()
        talker.code_predictor.model = torch.nn.Module()
        residual = torch.nn.ModuleList([torch.nn.Embedding(8, 4) for _ in range(2)])
        for index, embedding in enumerate(residual):
            torch.nn.init.constant_(embedding.weight, float(index + 2))
        talker.code_predictor.model.codec_embedding = residual
        codes = torch.tensor([[1, 2, 3], [4, 5, 6]])
        hidden = rvq_codec_embeds(talker, codes)
        expected = talker.model.codec_embedding(codes[:, 0])
        expected = expected + residual[0](codes[:, 1]) + residual[1](codes[:, 2])
        self.assertTrue(torch.equal(hidden, expected))
        book0 = rvq_codec_embeds(talker, codes[:, 0])
        self.assertTrue(torch.equal(book0, talker.model.codec_embedding(codes[:, 0])))


if __name__ == "__main__":
    unittest.main()
