#!/usr/bin/env python3
"""GPU smoke for Omni family wrap / one step / export.

Expects MODELS_DIR and TRAINING_CHECKPOINTS_DIR (set inside the training SIF).
"""

from __future__ import annotations

import os
import sys
import traceback
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_omni_families import DispatchTests, DummyScatterTests, ExportHelperTests, Qwen3DebugIntegrationTests  # noqa: E402


def _models() -> Path:
    raw = os.environ.get("MODELS_DIR")
    if not raw:
        raise RuntimeError("MODELS_DIR is not set")
    return Path(raw)


def _out() -> Path:
    raw = os.environ.get("TRAINING_CHECKPOINTS_DIR")
    if not raw:
        raise RuntimeError("TRAINING_CHECKPOINTS_DIR is not set")
    path = Path(raw) / "tier2_families"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _one_step(model) -> None:
    model.train()
    device = next(model.parameters()).device
    vocab = int(model.config.thinker_config.text_config.vocab_size)
    input_ids = torch.randint(0, min(vocab, 1000), (1, 8), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    if outputs.loss is None:
        raise RuntimeError("forward returned no loss")
    outputs.loss.backward()
    print(f"  loss={float(outputs.loss.detach().cpu()):.4f}", flush=True)


def _export(model, base: Path, dest: Path, fallback: Path | None) -> None:
    from transformers import AutoModel, AutoTokenizer

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_checkpoint", ROOT / "scripts" / "export_checkpoint.py"
    )
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    export._save_speechlmm(
        model,
        str(base),
        dest,
        max_shard_size="5GB",
        tokenizer=tokenizer,
        inference_fallback=fallback,
    )
    export.verify_exported_inference_assets(dest)
    loaded = AutoModel.from_pretrained(str(dest), trust_remote_code=True)
    print(f"  AutoModel class={type(loaded).__name__}", flush=True)


def run_unit() -> None:
    print("=== unit tests ===", flush=True)
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(DispatchTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ExportHelperTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(DummyScatterTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def run_qwen3_debug() -> None:
    print("=== Qwen3 debug wrap/export/AutoModel ===", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Qwen3DebugIntegrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def run_qwen25_7b() -> None:
    print("=== Qwen2.5-Omni-7B wrap + step + export ===", flush=True)
    from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration
    from speechlmm.models.modeling_speechlmm_qwen2_5_omni import (
        SpeechLMMQwen2_5OmniForConditionalGeneration,
    )

    src = _models() / "Qwen" / "Qwen2.5-Omni-7B"
    if not src.is_dir():
        raise FileNotFoundError(src)
    model = SpeechLMMForConditionalGeneration.from_qwen2_5_omni(
        str(src),
        config_overrides={"enable_talker": False, "enable_code2wav": False},
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    if not isinstance(model, SpeechLMMQwen2_5OmniForConditionalGeneration):
        raise TypeError(type(model))
    print(f"  wrapped class={type(model).__name__} backbone={model.config.backbone_type}", flush=True)
    _one_step(model)
    _export(model, src, _out() / "qwen25_7b_export_avsr", fallback=src)


def run_qwen3_30b_wrap() -> None:
    print("=== Qwen3-Omni-30B wrap + one step (no export) ===", flush=True)
    from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration
    from speechlmm.models.modeling_speechlmm_qwen3_omni import (
        SpeechLMMQwen3OmniForConditionalGeneration,
    )

    src = _models() / "Qwen" / "Qwen3-Omni-30B-A3B-Instruct"
    if not src.is_dir():
        raise FileNotFoundError(src)
    model = SpeechLMMForConditionalGeneration.from_qwen3_omni(
        str(src),
        config_overrides={"enable_talker": False, "enable_code2wav": False},
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not isinstance(model, SpeechLMMQwen3OmniForConditionalGeneration):
        raise TypeError(type(model))
    print(f"  wrapped class={type(model).__name__} backbone={model.config.backbone_type}", flush=True)
    _one_step(model)


def main() -> None:
    run_unit()
    run_qwen3_debug()
    run_qwen25_7b()
    run_qwen3_30b_wrap()
    print("=== all GPU smoke checks passed ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
