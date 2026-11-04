"""Vocoder-free Talker weight load from prefixed Hugging Face shards."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speechlmm.models.modeling_speechlmm import load_prefixed_checkpoint  # noqa: E402


class TalkerLoadTests(unittest.TestCase):
    def test_load_prefixed_single_file(self):
        try:
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("safetensors is required")

        module = torch.nn.Linear(3, 2, bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            save_file({"talker.weight": torch.ones(2, 3)}, str(path / "model.safetensors"))
            load_prefixed_checkpoint(module, path, "talker.")
        self.assertTrue(torch.equal(module.weight, torch.ones(2, 3)))

    def test_load_prefixed_index(self):
        try:
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("safetensors is required")

        module = torch.nn.Linear(3, 2, bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            shard = "model-00001-of-00001.safetensors"
            save_file({"talker.weight": torch.full((2, 3), 4.0), "code2wav.weight": torch.zeros(1)}, str(path / shard))
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"talker.weight": shard, "code2wav.weight": shard}})
            )
            load_prefixed_checkpoint(module, path, "talker.")
        self.assertTrue(torch.equal(module.weight, torch.full((2, 3), 4.0)))

    def test_fuse_qwen_moe_expert_tensors(self):
        from speechlmm.models.modeling_speechlmm import _fuse_qwen_moe_expert_tensors

        tensors = {
            "model.norm.weight": torch.ones(4),
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.full((2, 3), 1.0),
            "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((2, 3), 2.0),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((2, 3), 3.0),
            "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((2, 3), 4.0),
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((3, 2), 5.0),
            "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((3, 2), 6.0),
        }
        fused = _fuse_qwen_moe_expert_tensors(tensors)
        self.assertEqual(fused["model.norm.weight"].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(tuple(fused["model.layers.0.mlp.experts.gate_up_proj"].shape), (2, 4, 3))
        self.assertEqual(tuple(fused["model.layers.0.mlp.experts.down_proj"].shape), (2, 3, 2))
        self.assertTrue(torch.equal(fused["model.layers.0.mlp.experts.gate_up_proj"][0, :2], torch.full((2, 3), 1.0)))
        self.assertTrue(torch.equal(fused["model.layers.0.mlp.experts.gate_up_proj"][0, 2:], torch.full((2, 3), 3.0)))
        self.assertNotIn("model.layers.0.mlp.experts.0.gate_proj.weight", fused)

    def test_talker_is_loaded_uses_ds_numel_when_numel_is_zero(self):
        from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration

        class Host:
            talker_is_loaded = SpeechLMMForConditionalGeneration.talker_is_loaded

        empty = torch.nn.Parameter(torch.empty(0))
        talker = torch.nn.Module()
        talker.register_parameter("weight", empty)
        wrap = Host()
        wrap.talker = talker
        self.assertFalse(wrap.talker_is_loaded())

        empty.ds_numel = 64
        self.assertTrue(wrap.talker_is_loaded())

        missing = Host()
        missing.talker = None
        self.assertFalse(missing.talker_is_loaded())

    def test_load_talker_weights_skips_when_native_talker_exists(self):
        from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration

        class Host:
            talker_is_loaded = SpeechLMMForConditionalGeneration.talker_is_loaded
            load_talker_weights = SpeechLMMForConditionalGeneration.load_talker_weights

        wrap = Host()
        wrap.talker = torch.nn.Linear(2, 2, bias=False)
        wrap.has_talker = False
        wrap.load_talker_weights("/does-not-exist")
        self.assertTrue(wrap.has_talker)
        self.assertIsInstance(wrap.talker, torch.nn.Linear)

    def test_load_talker_weights_skips_when_ds_numel_is_set(self):
        from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration

        class Host:
            talker_is_loaded = SpeechLMMForConditionalGeneration.talker_is_loaded
            load_talker_weights = SpeechLMMForConditionalGeneration.load_talker_weights

        empty = torch.nn.Parameter(torch.empty(0))
        empty.ds_numel = 64
        talker = torch.nn.Module()
        talker.register_parameter("weight", empty)
        wrap = Host()
        wrap.talker = talker
        wrap.has_talker = False
        wrap.load_talker_weights("/does-not-exist")
        self.assertTrue(wrap.has_talker)
        self.assertIs(wrap.talker, talker)

    def test_native_load_turns_audio_output_on_for_talker_or_vocoder(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration

        def audio_output(enable_talker, enable_code2wav):
            captured = {}

            def fake_from_pretrained(path, config=None, **kwargs):
                captured.update(config=config, kwargs=kwargs)
                return "omni"

            with patch("transformers.AutoModelForTextToWaveform.from_pretrained", fake_from_pretrained):
                out = SpeechLMMForConditionalGeneration._load_native_omni(
                    "/ckpt",
                    SimpleNamespace(enable_audio_output=None),
                    SimpleNamespace(enable_talker=enable_talker, enable_code2wav=enable_code2wav),
                    {"pretrained_model_name_or_path": "/ckpt", "torch_dtype": "auto"},
                )
            self.assertEqual(out, "omni")
            self.assertEqual(captured["kwargs"], {"torch_dtype": "auto"})
            return captured["config"].enable_audio_output

        self.assertTrue(audio_output(True, False))
        self.assertTrue(audio_output(False, True))
        self.assertFalse(audio_output(False, False))


if __name__ == "__main__":
    unittest.main()
