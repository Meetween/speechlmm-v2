"""Thinker LoRA scope validation for dense and MoE SpeechLMM."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speechlmm.training.hparams import SpeechLMMComponentLoraArguments  # noqa: E402


@dataclass
class _LoraArgs(SpeechLMMComponentLoraArguments):
    finetuning_type: str = "lora"
    lora_target: list[str] = field(default_factory=lambda: ["all"])
    lora_rank: int = 8
    lora_alpha: int = 16
    freeze_audio_encoder: bool = True
    freeze_audio_adapters: bool = True
    freeze_lipread_adapter: bool = True
    trainable_module_paths: list[str] | None = None

    def __post_init__(self):
        self._validate_component_lora()


class LoraLanguageModelScopeTests(unittest.TestCase):
    def test_attention_enables_language_model_without_experts(self):
        args = _LoraArgs(lora_language_model_scope="attention")
        self.assertTrue(args.lora_language_model)
        self.assertFalse(args.lora_language_model_experts)

    def test_attention_and_ffn_enables_language_model_without_experts(self):
        args = _LoraArgs(lora_language_model_scope="attention_and_ffn")
        self.assertTrue(args.lora_language_model)
        self.assertFalse(args.lora_language_model_experts)

    def test_attention_and_experts_enables_both_flags(self):
        args = _LoraArgs(lora_language_model_scope="attention_and_experts")
        self.assertTrue(args.lora_language_model)
        self.assertTrue(args.lora_language_model_experts)

    def test_none_disables_language_model(self):
        args = _LoraArgs(lora_language_model_scope="none")
        self.assertFalse(args.lora_language_model)
        self.assertFalse(args.lora_language_model_experts)

    def test_rejects_unknown_scope(self):
        with self.assertRaises(ValueError) as ctx:
            _LoraArgs(lora_language_model_scope="all")
        self.assertIn("attention_and_ffn", str(ctx.exception))

    def test_experts_flag_conflicts_with_attention_and_ffn(self):
        with self.assertRaises(ValueError) as ctx:
            _LoraArgs(
                lora_language_model_scope="attention_and_ffn",
                lora_language_model_experts=True,
            )
        self.assertIn("lora_language_model_experts", str(ctx.exception))
