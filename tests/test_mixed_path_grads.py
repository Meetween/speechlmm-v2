#!/usr/bin/env python3
"""1-GPU probe: real audio vs real lipread batches, adapter grads on dummy vs live paths.

Audio-only batches inject dummy lipread (no lipread tokens). Lipread-only batches
inject dummy audio (no audio tokens). Dummy scatter is `feat.sum() * 0`, so the
idle tower must stay in the graph with ~zero grad; the live tower must not.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "llamafactory" / "src"))


def _skip_reason() -> str | None:
    if not os.environ.get("SPEECHLMM_TEST_MIXED_DATA"):
        return "set SPEECHLMM_TEST_MIXED_DATA to a mixed audio/lipread dataset dir"
    if not os.environ.get("MODELS_DIR"):
        return "MODELS_DIR is not set"
    return None


def _models() -> Path:
    raw = os.environ.get("MODELS_DIR")
    if not raw:
        raise RuntimeError("MODELS_DIR is not set")
    return Path(raw)


def _dataset_dir() -> str:
    raw = os.environ.get("SPEECHLMM_TEST_MIXED_DATA")
    if not raw:
        raise RuntimeError("SPEECHLMM_TEST_MIXED_DATA is not set")
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"SPEECHLMM_TEST_MIXED_DATA is not a directory: {path}")
    return str(path)


def _out_dir() -> str:
    raw = os.environ.get("TRAINING_CHECKPOINTS_DIR", "/tmp")
    path = Path(raw) / "tier2_grad_probe"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _train_args(dataset: str) -> dict:
    models = _models()
    return {
        "model_name_or_path": str(models / "speechlmm-v2" / "speechlmm_debug_model"),
        "trust_remote_code": True,
        "use_speechlmm_wrapper": True,
        "lipread_encoder_weights": str(models / "autoavsr" / "vsr_trlrs2lrs3vox2avsp_base.pth"),
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "freeze_vision_tower": True,
        "freeze_audio_encoder": True,
        "freeze_audio_adapters": False,
        "freeze_lipread_encoder": True,
        "freeze_lipread_adapter": False,
        "freeze_language_model": True,
        "freeze_talker": True,
        "freeze_code2wav": True,
        "freeze_code_predictor": True,
        "trainable_module_paths": (
            "audio_tower.proj1,audio_tower.proj2,audio_tower.ln_post,"
            "lipread_adapter.ln_post,lipread_adapter.linear_1,lipread_adapter.linear_2"
        ),
        "dataset": dataset,
        "dataset_dir": _dataset_dir(),
        "template": "speechlmm",
        "cutoff_len": 512,
        "max_samples": 2,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "overwrite_cache": True,
        "output_dir": _out_dir(),
        "overwrite_output_dir": True,
        "report_to": "none",
        "bf16": False,
        "fp16": False,
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "learning_rate": 1.0e-4,
        "logging_steps": 1,
        "save_steps": 1000,
        "plot_loss": False,
    }


def _grad_norm(module) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for param in module.parameters():
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().float().norm().item() ** 2)
    return total ** 0.5


def _count_tokens(batch: dict, token_id: int | None) -> int:
    if token_id is None or "input_ids" not in batch:
        return 0
    return int((batch["input_ids"] == token_id).sum().item())


def _load(dataset: str):
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_model, load_tokenizer
    from speechlmm.tokens import build_speechlmm_special_tokens

    model_args, data_args, training_args, finetuning_args, _generating_args = get_train_args(_train_args(dataset))
    extra = build_speechlmm_special_tokens(model_args)
    current = getattr(model_args, "add_special_tokens", None) or []
    if isinstance(current, str):
        current = [part for part in current.split(",") if part]
    model_args.add_special_tokens = list(current) + extra
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template, model_args, data_args, training_args, stage="sft", **tokenizer_module
    )
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
    model.float().cuda()
    model.train()
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    return model, dataset_module["train_dataset"], collator


def _step(model, dataset, collator, n: int = 1) -> dict:
    import torch

    model.zero_grad(set_to_none=True)
    raw = collator([dataset[i] for i in range(n)])
    batch = {}
    for key, value in raw.items():
        if key in {"pixel_values_videos"}:
            continue
        if torch.is_tensor(value):
            value = value.cuda()
            if value.is_floating_point():
                value = value.float()
        batch[key] = value
    outputs = model(**batch)
    if outputs.loss is None:
        raise RuntimeError("forward returned no loss")
    outputs.loss.backward()
    thinker = getattr(model.config, "thinker_config", None)
    audio_token_id = getattr(model.config, "audio_token_id", None) or getattr(thinker, "audio_token_id", None)
    lipread_token_id = getattr(model.config, "lipread_pad_token_id", None)
    return {
        "loss": float(outputs.loss.detach().cpu()),
        "n_audio_tokens": _count_tokens(batch, audio_token_id),
        "n_lipread_tokens": _count_tokens(batch, lipread_token_id),
        "has_audio_features": batch.get("input_features") is not None,
        "has_lipread": batch.get("lipread") is not None,
        "audio_proj_grad": _grad_norm(getattr(model, "audio_tower", None)),
        "lipread_adapter_grad": _grad_norm(getattr(model, "lipread_adapter", None)),
        "lipread_encoder_grad": _grad_norm(getattr(model, "lipread_encoder", None)),
    }


def main() -> None:
    import torch

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

    print("=== audio-only (dummy lipread expected) ===", flush=True)
    audio_model, audio_data, audio_collator = _load("pr62_audio")
    audio_stats = _step(audio_model, audio_data, audio_collator)
    print(audio_stats, flush=True)
    del audio_model, audio_data, audio_collator
    torch.cuda.empty_cache()

    print("=== lipread-only (dummy audio expected) ===", flush=True)
    lip_model, lip_data, lip_collator = _load("pr62_lipread")
    lip_stats = _step(lip_model, lip_data, lip_collator)
    print(lip_stats, flush=True)

    eps = 1e-8
    live = 1e-6
    failures = []
    if audio_stats["n_audio_tokens"] <= 0:
        failures.append(f"audio-only had no audio tokens: {audio_stats}")
    if audio_stats["n_lipread_tokens"] != 0:
        failures.append(f"audio-only injected lipread tokens: {audio_stats}")
    if audio_stats["audio_proj_grad"] <= live:
        failures.append(f"audio-only audio_tower grad too small: {audio_stats}")
    if audio_stats["lipread_adapter_grad"] > eps:
        failures.append(f"audio-only dummy lipread adapter grad should be ~0: {audio_stats}")
    if audio_stats["lipread_encoder_grad"] > eps:
        failures.append(f"audio-only frozen lipread encoder got grad: {audio_stats}")

    if lip_stats["n_lipread_tokens"] <= 0:
        failures.append(f"lipread-only had no lipread tokens: {lip_stats}")
    if lip_stats["n_audio_tokens"] != 0:
        failures.append(f"lipread-only injected audio tokens: {lip_stats}")
    if lip_stats["lipread_adapter_grad"] <= live:
        failures.append(f"lipread-only lipread_adapter grad too small: {lip_stats}")
    if lip_stats["audio_proj_grad"] > live:
        failures.append(f"lipread-only dummy audio_tower grad should be ~0: {lip_stats}")
    if lip_stats["lipread_encoder_grad"] > eps:
        failures.append(f"lipread-only frozen lipread encoder got grad: {lip_stats}")

    if failures:
        raise SystemExit("GRAD PATH CHECK FAILED:\n- " + "\n- ".join(failures))
    print("=== mixed-path gradient checks passed ===", flush=True)


class MixedPathGradTests(unittest.TestCase):
    def test_audio_vs_lipread_dummy_paths(self):
        reason = _skip_reason()
        if reason:
            self.skipTest(reason)
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        main()


if __name__ == "__main__":
    reason = _skip_reason()
    if reason:
        print(f"skip: {reason}", flush=True)
        raise SystemExit(0)
    main()
