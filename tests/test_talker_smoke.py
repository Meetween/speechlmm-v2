"""Tiny Talker CE smoke with a stand-in module (inline codec tokens)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speechlmm.models.talker_loss import (  # noqa: E402
    build_streaming_text,
    codec_targets,
    dummy_talker_forward_loss,
    first_codebook,
    pad_codec_batch,
    reduce_talker_losses,
    token_ce_sum,
    valid_codec_codes,
)


class _TalkerModel(nn.Module):
    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, **kwargs):
        return (inputs_embeds,)


class TinyTalker(nn.Module):
    def __init__(self, in_dim=8, hidden=8, vocab=32):
        super().__init__()
        self.thinker_to_talker_proj = nn.Linear(in_dim, hidden)
        self.model = _TalkerModel()
        self.model.embed_tokens = nn.Embedding(vocab, in_dim)
        self.codec_head = nn.Linear(hidden, vocab)


class TalkerSmokeTests(unittest.TestCase):
    def test_inline_codec_ce_finite_and_talker_grads(self):
        torch.manual_seed(0)
        talker = TinyTalker()
        labels, mask = pad_codec_batch([[3, 5, 7, 9]], num_codebooks=1)
        self.assertTrue(bool(mask[0]))
        codes = first_codebook(valid_codec_codes(labels[0]))
        prompt = torch.randn(1, 6, 8)
        speaker = torch.randn(1, 1, 8)
        y1 = torch.randn(1, 1, 8)
        trailing = torch.randn(1, 5, 8)
        pad = torch.zeros(1, 1, 8)
        stream = build_streaming_text(trailing, pad, codes.numel())
        codec_bos = talker.model.embed_tokens(torch.tensor([1]))
        first_slot = y1 + codec_bos.unsqueeze(0)
        rest = (talker.model.embed_tokens(codes) + stream).unsqueeze(0)
        seq = torch.cat([prompt, speaker, first_slot, rest], dim=1)
        target = codec_targets(6, 2, codes, eos_id=2)
        self.assertEqual(seq.shape[1], target.shape[0])
        hidden = talker.model(inputs_embeds=talker.thinker_to_talker_proj(seq))[0]
        logits = talker.codec_head(hidden)
        loss_sum, n_valid = token_ce_sum(logits, target.unsqueeze(0))
        loss = reduce_talker_losses([loss_sum], [n_valid], "token")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(talker.codec_head.weight.grad)
        self.assertGreater(float(talker.codec_head.weight.grad.abs().sum()), 0.0)

    def test_dummy_path_zero_grads(self):
        talker = TinyTalker()
        loss = dummy_talker_forward_loss(talker, torch.device("cpu"), torch.float32)
        loss.backward()
        self.assertEqual(float(talker.codec_head.weight.grad.abs().sum()), 0.0)

    def test_perfect_logits_near_zero(self):
        vocab = 16
        target = torch.tensor([[-100, 2, 4, 7]])
        logits = torch.zeros(1, 4, vocab)
        logits[0, 1, 2] = 30
        logits[0, 2, 4] = 30
        logits[0, 3, 7] = 30
        loss, n_valid = token_ce_sum(logits, target)
        self.assertEqual(n_valid, 3)
        self.assertLess(float(loss / n_valid), 1e-4)
        uniform = torch.zeros(1, 4, vocab)
        u_loss, _ = token_ce_sum(uniform, target)
        self.assertAlmostEqual(float(u_loss / 3), float(torch.log(torch.tensor(float(vocab)))), places=4)


if __name__ == "__main__":
    unittest.main()
