#!/usr/bin/env python3
"""Merge a LoRA checkpoint (with optional trainable_module_paths weights) into
a full Qwen3-Omni / SpeechLMM model ready for inference.

Training with ``finetuning_type: lora`` + ``trainable_module_paths`` produces
two kinds of trained weights:

1. **LoRA adapters** -- saved by PEFT in ``adapter_model.safetensors``
2. **Full-weight modules** (e.g. ``audio_tower.proj1``, ``proj2``,
   ``ln_post``) -- only in the DeepSpeed checkpoint shards, unless the
   ``SaveTrainableModulesCallback`` wrote them as
   ``trainable_modules.safetensors``.

This script handles both scenarios transparently:

* If ``trainable_modules.safetensors`` exists alongside the adapter, it
  is loaded directly.
* Otherwise, the script attempts to consolidate the DeepSpeed ZeRO
  checkpoint (``checkpoint-*/global_step*/``) via ``zero_to_fp32.py``
  and extracts the relevant keys.

Usage examples
--------------
Merge the final adapter (new-format checkpoint with the callback):

    python scripts/merge_lora_checkpoint.py \
        --base-model /path/to/Qwen3-Omni-30B-A3B-Instruct \
        --adapter-path saves/audio_encoder_lora/  \
        --output-dir saves/merged_model/

Merge an intermediate checkpoint (old-format, DeepSpeed shards only):

    python scripts/merge_lora_checkpoint.py \
        --base-model /path/to/Qwen3-Omni-30B-A3B-Instruct \
        --adapter-path saves/audio_encoder_lora/checkpoint-500/ \
        --trainable-module-paths audio_tower.proj1,audio_tower.proj2,audio_tower.ln_post \
        --output-dir saves/merged_checkpoint_500/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForTextToWaveform, AutoTokenizer


TRAINABLE_MODULES_FILENAME = "trainable_modules.safetensors"


def find_deepspeed_checkpoint(adapter_path: Path) -> Path | None:
    """Locate the global_step* dir inside a checkpoint directory."""
    for candidate in sorted(adapter_path.glob("global_step*")):
        if candidate.is_dir() and any(candidate.glob("*_model_states.pt")):
            return candidate
    return None


def consolidate_deepspeed(checkpoint_dir: Path, output_file: Path) -> dict[str, torch.Tensor]:
    """Run zero_to_fp32.py to consolidate ZeRO shards and return the state dict."""
    z2fp32 = checkpoint_dir.parent / "zero_to_fp32.py"
    if not z2fp32.exists():
        z2fp32 = checkpoint_dir / "zero_to_fp32.py"
    if not z2fp32.exists():
        raise FileNotFoundError(
            f"Cannot find zero_to_fp32.py in {checkpoint_dir} or its parent. "
            "Make sure save_only_model=false was used during training."
        )

    print(f"Consolidating DeepSpeed checkpoint from {checkpoint_dir} ...")
    subprocess.check_call(
        [sys.executable, str(z2fp32), str(checkpoint_dir.parent), str(output_file)],
    )
    state = torch.load(output_file, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "module" in state:
        state = state["module"]
    return state


def extract_trainable_modules(
    state_dict: dict[str, torch.Tensor],
    trainable_paths: list[str],
) -> dict[str, torch.Tensor]:
    """Filter a consolidated state dict to keep only trainable_module_paths keys."""
    result = {}
    for key, val in state_dict.items():
        clean = key.replace("base_model.model.", "", 1) if key.startswith("base_model.model.") else key
        if any(clean == p or clean.startswith(p + ".") for p in trainable_paths):
            result[clean] = val
    return result


def load_trainable_modules(
    adapter_path: Path,
    trainable_paths: list[str] | None,
) -> dict[str, torch.Tensor]:
    """Load trainable_module_paths weights from the best available source."""
    safetensors_path = adapter_path / TRAINABLE_MODULES_FILENAME
    if safetensors_path.exists():
        print(f"Loading trainable modules from {safetensors_path}")
        return dict(load_file(str(safetensors_path)))

    if not trainable_paths:
        adapter_cfg_path = adapter_path / "adapter_config.json"
        if adapter_cfg_path.exists():
            with open(adapter_cfg_path) as f:
                cfg = json.load(f)
            modules_to_save = cfg.get("modules_to_save")
            if modules_to_save:
                return {}
        print(
            "WARNING: No trainable_modules.safetensors found and no --trainable-module-paths given.\n"
            "  If you trained with trainable_module_paths, those weights will be MISSING from the merge.\n"
            "  Pass --trainable-module-paths to extract them from the DeepSpeed checkpoint."
        )
        return {}

    ds_ckpt = find_deepspeed_checkpoint(adapter_path)
    if ds_ckpt is None:
        print(
            f"WARNING: No DeepSpeed checkpoint found in {adapter_path}.\n"
            "  trainable_module_paths weights will use base-model values."
        )
        return {}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        full_state = consolidate_deepspeed(ds_ckpt, tmp_path)
        modules = extract_trainable_modules(full_state, trainable_paths)
        print(f"Extracted {len(modules)} trainable module weights from DeepSpeed checkpoint")
        return modules
    finally:
        tmp_path.unlink(missing_ok=True)


def merge(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    trainable_module_paths: list[str] | None = None,
    dtype: str = "auto",
    max_shard_size: str = "5GB",
) -> None:
    """Delegate to export_checkpoint so expert-LoRA load/remap/merge stay single-sourced."""
    from export_checkpoint import export_checkpoint

    export_checkpoint(
        base_model_path=base_model_path,
        checkpoint_path=adapter_path,
        output_dir=output_dir,
        trainable_module_paths=trainable_module_paths,
        dtype=dtype,
        max_shard_size=max_shard_size,
        output_format="speechlmm",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA + trainable_module_paths into a full model"
    )
    parser.add_argument(
        "--base-model", required=True,
        help="Path to the base Qwen3-Omni model"
    )
    parser.add_argument(
        "--adapter-path", required=True,
        help="Path to the LoRA checkpoint directory (contains adapter_model.safetensors)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save the merged model"
    )
    parser.add_argument(
        "--trainable-module-paths", default=None,
        help="Comma-separated module path prefixes trained with trainable_module_paths "
             "(e.g. audio_tower.proj1,audio_tower.proj2,audio_tower.ln_post). "
             "Only needed for old checkpoints without trainable_modules.safetensors"
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
        help="Output dtype (default: auto, keeps original)"
    )
    parser.add_argument(
        "--max-shard-size", default="5GB",
        help="Max shard size for safetensors output (default: 5GB)"
    )
    args = parser.parse_args()

    trainable_paths = None
    if args.trainable_module_paths:
        trainable_paths = [p.strip() for p in args.trainable_module_paths.split(",")]

    merge(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        trainable_module_paths=trainable_paths,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
