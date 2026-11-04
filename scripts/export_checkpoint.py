#!/usr/bin/env python3
"""Export a training checkpoint (LoRA or full-finetuning) into a standalone
HF model directory.

Works on intermediate checkpoints (``checkpoint-N/``) and on the final output
dir. Each checkpoint should contain ``trainable_modules.safetensors`` when
``trainable_module_paths`` was used; LoRA runs also need ``adapter_config.json``
and ``adapter_model.safetensors``. DeepSpeed ``global_step*/`` shards are only
needed if ``trainable_modules.safetensors`` is missing (pass
``--trainable-module-paths`` to extract from ZeRO).

``--base-model`` must be the same full-weight model used as ``model_name_or_path``
during training (typically the cached Qwen3-Omni path). The script loads that
base, merges LoRA if present, then overlays ``trainable_modules.safetensors``.

When training used ``use_speechlmm_wrapper: true`` on a Qwen3-Omni base, LoRA
keys are stored in the flat SpeechLMM layout (``audio_tower.*``). Export
detects that and remaps the adapter to ``thinker.*`` before merging.

--use-speechlmm-wrapper  When producing ``--format speechlmm`` from a raw
    Qwen3-Omni base, wrap it with SpeechLMM before applying a checkpoint
    trained with use_speechlmm_wrapper=true. Omit this flag for
    ``--format qwen3-omni``; that path remaps flat keys to ``thinker.*``.

--lipread-encoder-weights  Path to pretrained AutoAVSR weights (.pth) loaded
    into the lipread encoder at wrap time (required for non-random lipread).

``--format`` controls the output layout:

  speechlmm   (default) Flat SpeechLMM layout with all modules preserved
              (lipread, adapters, etc.). For resumed training, KD, Triton, or
              direct Python loading. Writes auto_map so AutoConfig/AutoModel
              resolve under trust_remote_code without installing speechlmm.
              Lipread exports bake <|lipread_*|> tokens into tokenizer.json and
              resize embeddings so training can use resize_vocab=false.
              NOT compatible with vLLM-Omni serving.

  qwen3-omni  Re-nests weights under ``thinker.*`` with a Qwen3OmniMoe config.
              For native Qwen3-Omni audio/text serving via vLLM-Omni.
              SpeechLMM-only modules such as lipread are not represented;
              use ``speechlmm`` for lipread inference.

See ``examples/export/`` for runnable usage examples.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForTextToWaveform, AutoTokenizer

from llamafactory.extras.file_perms import install_safetensors_umask

import speechlmm.models  # noqa: F401 — registers SpeechLMMConfig with AutoConfig
from speechlmm.models.configuration_speechlmm import SpeechLMMConfig
from speechlmm.tokens import lipread_special_tokens

TRAINABLE_MODULES_FILENAME = "trainable_modules.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_MODEL_FILENAME = "adapter_model.safetensors"

# Module paths that SpeechLMM exposes at the top level but Qwen3-Omni nests under thinker.*
_SPEECHLMM_FLAT_PREFIXES = ("audio_tower.", "visual.", "model.", "lm_head.")
_SPEECHLMM_FLAT_ROOTS = ("audio_tower", "visual", "model", "lm_head")


def bake_lipread_special_tokens(model: torch.nn.Module, tokenizer: AutoTokenizer) -> int:
    """Add lipread tokens to *tokenizer*, resize embeddings, sync config token ids."""
    if not hasattr(model, "lipread_encoder"):
        return 0

    num_added = tokenizer.add_tokens(lipread_special_tokens(), special_tokens=True)
    embed_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embed_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    if getattr(model.config, "model_type", None) == "speechlmm":
        SpeechLMMConfig.sync_lipread_token_ids_from_tokenizer(model.config, tokenizer)
    return num_added


def has_lora_adapter(checkpoint_path: Path) -> bool:
    """Check whether the checkpoint contains a PEFT LoRA adapter."""
    return (checkpoint_path / ADAPTER_CONFIG_FILENAME).exists()


def find_deepspeed_checkpoint(checkpoint_path: Path) -> Path | None:
    """Locate the global_step* dir inside a checkpoint directory."""
    for candidate in sorted(checkpoint_path.glob("global_step*")):
        if candidate.is_dir() and any(candidate.glob("*_model_states.pt")):
            return candidate
    return None


def _uses_speechlmm_flat_keys(checkpoint_path: Path) -> bool:
    """Return True when the checkpoint was saved with use_speechlmm_wrapper=True.

    Training with the SpeechLMM wrapper stores LoRA / trainable-module keys
    in the flat SpeechLMM namespace (e.g. ``audio_tower.*``) instead of the
    Qwen3-Omni ``thinker.audio_tower.*`` nesting.  We detect this by looking
    for known flat prefixes in the adapter config or the trainable-modules file.
    """
    adapter_cfg_path = checkpoint_path / ADAPTER_CONFIG_FILENAME
    if adapter_cfg_path.exists():
        with open(adapter_cfg_path) as fh:
            adapter_cfg = json.load(fh)
        tm = adapter_cfg.get("target_modules", "")
        if isinstance(tm, str):
            # target_modules may be a regex string where dots are escaped as \\.
            # Strip backslashes for prefix detection so both ``audio_tower.``
            # and ``audio_tower\.`` (regex literal) are matched correctly.
            tm_clean = tm.replace("\\", "")
            return any(tm_clean.startswith(p) for p in _SPEECHLMM_FLAT_PREFIXES) and "thinker." not in tm_clean
        if isinstance(tm, (list, set)):
            return any(t.startswith(_SPEECHLMM_FLAT_PREFIXES) for t in tm)

    tm_path = checkpoint_path / TRAINABLE_MODULES_FILENAME
    if tm_path.exists():
        state = load_file(str(tm_path))
        return any(k.startswith(_SPEECHLMM_FLAT_PREFIXES) for k in state)

    return False


def _patch_lora_for_qwen3(checkpoint_path: Path) -> tempfile.TemporaryDirectory:
    """Copy the adapter to a temp dir with keys remapped to thinker.* namespace.

    When a LoRA was trained with use_speechlmm_wrapper=True on a Qwen3-Omni
    base, all adapter target paths use SpeechLMM's flat layout.  Applying that
    adapter directly to a raw Qwen3-Omni model fails because PEFT performs a
    fullmatch against each module name, and ``audio_tower.layers.0.*`` does
    not match ``thinker.audio_tower.layers.0.*``.

    This helper patches both adapter_config.json (target_modules regex) and
    adapter_model.safetensors (weight keys) so the adapter applies cleanly to
    the raw Qwen3-Omni model.
    """
    from safetensors.torch import save_file as st_save_file

    tmpdir = tempfile.TemporaryDirectory(prefix="speechlmm_export_lora_")
    tmp_p = Path(tmpdir.name)

    for f in checkpoint_path.iterdir():
        if f.is_file():
            shutil.copy2(str(f), str(tmp_p / f.name))

    # --- patch adapter_config.json ------------------------------------------------
    cfg_path = tmp_p / ADAPTER_CONFIG_FILENAME
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    def _is_flat_speechlmm_path(cleaned: str) -> bool:
        if cleaned.startswith("thinker."):
            return False
        return cleaned in _SPEECHLMM_FLAT_ROOTS or any(cleaned.startswith(prefix) for prefix in _SPEECHLMM_FLAT_PREFIXES)

    def _nest_flat_module_path(path: str) -> str:
        """Prefix a SpeechLMM flat module path with thinker.; leave nested paths alone."""
        cleaned = path.replace("\\", "")
        if not _is_flat_speechlmm_path(cleaned):
            return path
        if "\\" in path:
            return r"thinker\." + path.lstrip("^")
        return f"thinker.{path}"

    def _nest_flat_regex_alternation(pattern: str) -> str:
        """Prefix each regex alternative by anchored flat prefix, never thinker.thinker."""
        nested_parts = []
        for part in pattern.split("|"):
            stripped = part
            anchor = ""
            if stripped.startswith("^"):
                anchor = "^"
                stripped = stripped[1:]
            cleaned = stripped.replace("\\", "")
            if not _is_flat_speechlmm_path(cleaned):
                nested_parts.append(part)
                continue
            if "\\" in stripped:
                nested_parts.append(f"{anchor}thinker\\.{stripped}")
            else:
                nested_parts.append(f"{anchor}thinker.{stripped}")
        return "|".join(nested_parts)

    tm = cfg.get("target_modules", "")
    if isinstance(tm, str):
        cfg["target_modules"] = _nest_flat_regex_alternation(tm)
    elif isinstance(tm, (list, set)):
        cfg["target_modules"] = [_nest_flat_module_path(t) for t in tm]

    for pattern_key in ("rank_pattern", "alpha_pattern"):
        pattern = cfg.get(pattern_key)
        if isinstance(pattern, dict) and pattern:
            cfg[pattern_key] = {_nest_flat_module_path(k): v for k, v in pattern.items()}

    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    # --- patch adapter_model.safetensors keys -------------------------------------
    # PEFT key format: ``base_model.model.<module_path>.lora_{A,B}.*``
    adapter_p = tmp_p / ADAPTER_MODEL_FILENAME
    if adapter_p.exists():
        state = load_file(str(adapter_p))
        new_state = {}
        for key, val in state.items():
            new_key = key
            # Already nested under thinker — leave unchanged.
            if key.startswith("base_model.model.thinker."):
                new_state[new_key] = val
                continue
            for prefix in _SPEECHLMM_FLAT_PREFIXES:
                flat = f"base_model.model.{prefix}"
                nested = f"base_model.model.thinker.{prefix}"
                if key.startswith(flat):
                    new_key = nested + key[len(flat) :]
                    break
            new_state[new_key] = val
        st_save_file(new_state, str(adapter_p))

    return tmpdir


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
    checkpoint_path: Path,
    trainable_paths: list[str] | None,
) -> dict[str, torch.Tensor]:
    """Load trainable_module_paths weights from the best available source."""
    safetensors_path = checkpoint_path / TRAINABLE_MODULES_FILENAME
    if safetensors_path.exists():
        print(f"Loading trainable modules from {safetensors_path}")
        return dict(load_file(str(safetensors_path)))

    if not trainable_paths:
        print(
            "WARNING: No trainable_modules.safetensors found and no --trainable-module-paths given.\n"
            "  If you trained with trainable_module_paths, those weights will be MISSING.\n"
            "  Pass --trainable-module-paths to extract them from the DeepSpeed checkpoint."
        )
        return {}

    ds_ckpt = find_deepspeed_checkpoint(checkpoint_path)
    if ds_ckpt is None:
        print(
            f"WARNING: No DeepSpeed checkpoint found in {checkpoint_path}.\n"
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


def apply_trainable_modules(
    model: torch.nn.Module,
    trainable_state: dict[str, torch.Tensor],
) -> None:
    """Overlay trainable_module_paths weights onto the model's state dict."""
    if not trainable_state:
        return

    print(f"Applying {len(trainable_state)} trainable_module_paths weights ...")
    model_state = model.state_dict()
    applied, skipped = 0, []
    for key, val in trainable_state.items():
        target_key = key
        if key not in model_state:
            suffix = "." + key
            candidates = [k for k in model_state if k.endswith(suffix)]
            if len(candidates) == 1:
                target_key = candidates[0]

        if target_key in model_state:
            model_state[target_key].copy_(val.to(model_state[target_key].dtype))
            applied += 1
        else:
            skipped.append(key)

    if skipped:
        print(f"  WARNING: {len(skipped)} keys not found in model: {skipped[:5]}...")
    print(f"  Applied {applied} trainable module weights.")


def load_base_model(
    base_model_path: str,
    use_speechlmm_wrapper: bool = False,
    lipread_encoder_weights: str | None = None,
) -> torch.nn.Module:
    """Load the base model, auto-detecting SpeechLMM vs Qwen3-Omni.

    When *use_speechlmm_wrapper* is True and the base model is Qwen3-Omni,
    wrap it with SpeechLMM so the module hierarchy matches LoRA targets
    that were trained under the wrapper.
    """
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)

    if model_type == "speechlmm":
        from speechlmm.models import SpeechLMMForConditionalGeneration

        return SpeechLMMForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    qwen3_model = AutoModelForTextToWaveform.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if use_speechlmm_wrapper and model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
        from speechlmm.models import SpeechLMMForConditionalGeneration
        from speechlmm.models.configuration_speechlmm import SpeechLMMConfig

        overrides = {}
        if lipread_encoder_weights:
            overrides["lipread_encoder_weights"] = lipread_encoder_weights
            print(f"Loading pretrained lipread encoder from {lipread_encoder_weights}")
        if model_type == "qwen2_5_omni":
            speechlmm_config = SpeechLMMConfig.from_qwen2_5_omni_config(config, **overrides)
            model = SpeechLMMForConditionalGeneration._wrap_qwen2_5_omni(qwen3_model, speechlmm_config)
        else:
            speechlmm_config = SpeechLMMConfig.from_qwen3_omni_config(config, **overrides)
            model = SpeechLMMForConditionalGeneration._wrap_qwen3_omni(qwen3_model, speechlmm_config)
        print(f"Wrapped Omni model with SpeechLMM (model_type: {model_type} → speechlmm)")
        return model

    return qwen3_model


THINKER_PREFIXES = ("audio_tower.", "visual.", "model.", "lm_head.")


def _renest_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename SpeechLMM flat keys back to Qwen3-Omni ``thinker.*`` nesting.

    Keys already under ``thinker.*`` (e.g. when base model is Qwen3-Omni
    rather than SpeechLMM) are left unchanged.
    """
    out = {}
    for key, val in state_dict.items():
        if key.startswith("thinker."):
            out[key] = val
        elif any(key == p.rstrip(".") or key.startswith(p) for p in THINKER_PREFIXES):
            out[f"thinker.{key}"] = val
        else:
            out[key] = val
    return out


def _build_qwen3_config_dict(config_path: Path) -> dict:
    """Build a Qwen3OmniMoe config dict from the base model config.

    Handles both SpeechLMM configs (converts) and Qwen3-Omni configs (pass-through).
    """
    with open(config_path) as f:
        cfg = json.load(f)

    if cfg.get("model_type") == "qwen3_omni_moe":
        cfg["architectures"] = ["Qwen3OmniMoeForConditionalGeneration"]
        return cfg

    qwen3_cfg = {
        "model_type": "qwen3_omni_moe",
        "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
        "thinker_config": cfg["thinker_config"],
        "talker_config": cfg.get("talker_config"),
        "code2wav_config": cfg.get("code2wav_config"),
        "enable_audio_output": cfg.get("enable_talker", True),
        "torch_dtype": cfg.get("torch_dtype", "bfloat16"),
        "transformers_version": cfg.get("transformers_version"),
    }
    for tok_key in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
    ):
        if tok_key in cfg:
            qwen3_cfg[tok_key] = cfg[tok_key]

    return qwen3_cfg


def save_exported_model(
    model: torch.nn.Module,
    base_model_path: str,
    output_dir: str,
    max_shard_size: str = "5GB",
    output_format: str = "speechlmm",
    tokenizer: AutoTokenizer | None = None,
    inference_fallback: Path | None = None,
) -> None:
    """Save the model + tokenizer + auxiliary files for inference."""
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    if output_format == "qwen3-omni":
        _save_qwen3_omni(model, base_model_path, output_p, max_shard_size, inference_fallback)
    else:
        _save_speechlmm(
            model,
            base_model_path,
            output_p,
            max_shard_size,
            tokenizer=tokenizer,
            inference_fallback=inference_fallback,
        )

    if output_format == "speechlmm":
        verify_exported_inference_assets(output_p)


TOKENIZER_COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "chat_template.jinja",
)

PROCESSOR_COPY_FILES = (
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
)

SHARED_INFERENCE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "preprocessor_config.json",
    "generation_config.json",
    "configuration_speechlmm.py",
    "modeling_speechlmm.py",
    "talker_generate.py",
    "talker_loss.py",
    "auto_avsr.py",
)

BACKBONE_MODELING_FILES = {
    "qwen2_5_omni": "modeling_speechlmm_qwen2_5_omni.py",
    "qwen3_omni": "modeling_speechlmm_qwen3_omni.py",
}

BACKBONE_AUTO_MODEL = {
    "qwen2_5_omni": ("modeling_speechlmm_qwen2_5_omni.SpeechLMMQwen2_5OmniForConditionalGeneration"),
    "qwen3_omni": ("modeling_speechlmm_qwen3_omni.SpeechLMMQwen3OmniForConditionalGeneration"),
}

SERVING_IMPORT_ALLOWLIST = frozenset(sys.stdlib_module_names) | {
    "torch",
    "transformers",
    "numpy",
    "safetensors",
}


def _backbone_type_from_config(cfg: dict) -> str:
    return cfg.get("backbone_type") or "qwen3_omni"


def _backbone_modeling_file(backbone_type: str) -> str:
    modeling = BACKBONE_MODELING_FILES.get(backbone_type)
    if modeling is None:
        raise RuntimeError(f"Unsupported backbone_type={backbone_type!r} for SpeechLMM export")
    return modeling


def _required_inference_files(backbone_type: str) -> tuple[str, ...]:
    return SHARED_INFERENCE_FILES + (_backbone_modeling_file(backbone_type),)


def _speechlmm_auto_map(backbone_type: str) -> dict[str, str]:
    auto_model = BACKBONE_AUTO_MODEL.get(backbone_type)
    if auto_model is None:
        raise RuntimeError(f"Unsupported backbone_type={backbone_type!r} for SpeechLMM export")
    return {
        "AutoConfig": "configuration_speechlmm.SpeechLMMConfig",
        "AutoModel": auto_model,
    }


LIPREAD_WEIGHT_PREFIXES = ("lipread_encoder.", "lipread_adapter.")


def _inference_file_sources(
    base_p: Path,
    inference_fallback: Path | None = None,
) -> list[Path]:
    sources = [base_p]
    if inference_fallback is not None and inference_fallback != base_p and inference_fallback.is_dir():
        sources.append(inference_fallback)
    return sources


def _copy_inference_files(
    output_p: Path,
    sources: list[Path],
    names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> list[str]:
    copied: list[str] = []
    for name in names:
        dest = output_p / name
        if dest.exists() and not overwrite:
            continue
        for src_root in sources:
            src = src_root / name
            if src.exists():
                shutil.copy2(str(src), str(dest))
                copied.append(name)
                break
    return copied


def _export_spk_dict(model, output_p: Path, sources: list[Path]) -> None:
    dest = output_p / "spk_dict.pt"
    speaker_map = getattr(model, "speaker_map", None)
    if speaker_map:
        torch.save(dict(speaker_map), dest)
        print("  Wrote spk_dict.pt from in-memory speaker_map.")
        return
    copied = _copy_inference_files(output_p, sources, ("spk_dict.pt",))
    if copied:
        print("  Copied spk_dict.pt from base/fallback.")


def verify_exported_inference_assets(output_p: Path) -> None:
    """Fail fast if the export is missing files or lipread weights in shards."""
    config_path = output_p / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Export at {output_p} is missing config.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    backbone_type = _backbone_type_from_config(cfg)
    required = _required_inference_files(backbone_type)
    missing = [name for name in required if not (output_p / name).exists()]
    if missing:
        raise RuntimeError(f"Export at {output_p} is missing required inference files: {', '.join(missing)}")

    index_path = output_p / "model.safetensors.index.json"
    single_path = output_p / "model.safetensors"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        weight_keys = list(weight_map)
    elif single_path.is_file():
        from safetensors import safe_open

        with safe_open(str(single_path), framework="pt") as handle:
            weight_keys = list(handle.keys())
    else:
        raise RuntimeError(f"Export at {output_p} has neither model.safetensors.index.json nor model.safetensors")
    lipread_keys = [key for key in weight_keys if key.startswith(LIPREAD_WEIGHT_PREFIXES)]
    if not lipread_keys:
        raise RuntimeError(f"Export at {output_p} has no lipread_encoder./lipread_adapter. weights in shards.")

    encoder_keys = [key for key in lipread_keys if key.startswith("lipread_encoder.")]
    adapter_keys = [key for key in lipread_keys if key.startswith("lipread_adapter.")]
    if cfg.get("lipread_encoder_weights"):
        raise RuntimeError("config.json still references lipread_encoder_weights; weights must be baked into shards.")
    expected_auto_map = _speechlmm_auto_map(backbone_type)
    if cfg.get("auto_map") != expected_auto_map:
        raise RuntimeError(
            "config.json is missing auto_map for SpeechLMMConfig / "
            f"{expected_auto_map['AutoModel']}; trust_remote_code cannot resolve the classes."
        )

    unexpected = [name for name in BACKBONE_MODELING_FILES.values() if name not in required and (output_p / name).exists()]
    if unexpected:
        raise RuntimeError(f"Export at {output_p} ships unused backbone files: {', '.join(unexpected)}")

    _verify_remote_code_imports(output_p)

    print("Inference asset check:")
    print(f"  Required files: {len(required)}/{len(required)} present")
    print(f"  auto_map: AutoConfig/AutoModel -> {expected_auto_map['AutoModel']}")
    print(f"  Lipread encoder tensors in shards: {len(encoder_keys)}")
    print(f"  Lipread adapter tensors in shards: {len(adapter_keys)}")
    print(f"  Total weight tensors: {len(weight_keys)}")


def _verify_remote_code_imports(output_p: Path) -> None:
    """Fail if any copied .py has a top-level import a bare serving env cannot provide.

    Uses transformers' own scanner (try/except stripped) against a fixed
    allowlist, not the local environment — export runs where speechlmm and
    deepspeed already import fine.
    """
    from transformers.dynamic_module_utils import get_imports

    forbidden: list[str] = []
    for py_file in sorted(output_p.glob("*.py")):
        pkgs = {imp.split(".")[0] for imp in get_imports(str(py_file))}
        extra = sorted(pkgs - SERVING_IMPORT_ALLOWLIST)
        if extra:
            forbidden.append(f"{py_file.name}: {', '.join(extra)}")
    if forbidden:
        raise RuntimeError(
            f"Export at {output_p} has remote-code imports that a bare serving "
            f"environment cannot provide: {'; '.join(forbidden)}"
        )


def _embed_chat_template_file(output_p: Path) -> None:
    """Ensure AutoTokenizer.from_pretrained picks up the chat template."""
    for name in ("chat_template.jinja", "chat_template.json"):
        src = output_p / name
        if not src.exists():
            continue
        if name.endswith(".jinja"):
            template = src.read_text(encoding="utf-8")
        else:
            data = json.loads(src.read_text(encoding="utf-8"))
            template = data["chat_template"] if isinstance(data, dict) else data
        tok_cfg_path = output_p / "tokenizer_config.json"
        tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        if tok_cfg.get("chat_template"):
            return
        tok_cfg["chat_template"] = template
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n", encoding="utf-8")
        print("  Embedded chat_template into tokenizer_config.json.")
        return


def _copy_tokenizer_and_processor_files(
    base_p: Path,
    output_p: Path,
    inference_fallback: Path | None = None,
) -> None:
    """Copy tokenizer, processor, and related files from the base model.

    Copies files directly instead of using save_pretrained to preserve
    custom fields like extra_special_tokens that AutoTokenizer strips.
    Missing tokenizer assets (e.g. chat_template.json on a SpeechLMM base)
    are filled from ``--inference-fallback-model`` when provided.
    """
    sources = _inference_file_sources(base_p, inference_fallback)
    copied = _copy_inference_files(
        output_p,
        sources,
        TOKENIZER_COPY_FILES + PROCESSOR_COPY_FILES,
    )
    print(f"  Copied {len(copied)} tokenizer/processor files: {', '.join(copied) or '(none)'}")


def _save_speechlmm(
    model: torch.nn.Module,
    base_model_path: str,
    output_p: Path,
    max_shard_size: str,
    tokenizer: AutoTokenizer | None = None,
    inference_fallback: Path | None = None,
) -> None:
    print(f"Saving exported model (speechlmm format) to {output_p} ...")
    model.save_pretrained(str(output_p), max_shard_size=max_shard_size, safe_serialization=True)

    # Lipread weights are in the safetensors shards; drop .pth path so training does not
    # reload AutoAVSR under ZeRO-3 during __init__.
    config_path = output_p / "config.json"
    backbone_type = getattr(getattr(model, "config", None), "backbone_type", None) or "qwen3_omni"
    if config_path.is_file():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cleared_pth = bool(cfg.pop("lipread_encoder_weights", None))
        backbone_type = cfg.get("backbone_type") or backbone_type
        cfg["auto_map"] = _speechlmm_auto_map(backbone_type)
        config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        if cleared_pth:
            print("  Cleared lipread_encoder_weights from config.json (weights are in shards).")
        print("  Wrote auto_map for AutoConfig/AutoModel under trust_remote_code.")

    base_p = Path(base_model_path)
    sources = _inference_file_sources(base_p, inference_fallback)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_p))
        copied = _copy_inference_files(output_p, sources, PROCESSOR_COPY_FILES)
        print(f"  Saved tokenizer with lipread special tokens (vocab={len(tokenizer)}).")
        if copied:
            print(f"  Copied processor files: {', '.join(copied)}")
        tokenizer_copied = _copy_inference_files(
            output_p,
            sources,
            TOKENIZER_COPY_FILES,
        )
        if tokenizer_copied:
            print(f"  Filled tokenizer files: {', '.join(tokenizer_copied)}")
    else:
        _copy_tokenizer_and_processor_files(base_p, output_p, inference_fallback)

    import speechlmm.models

    pkg_dir = Path(speechlmm.models.__file__).resolve().parent
    aux_files = [
        "configuration_speechlmm.py",
        "modeling_speechlmm.py",
        "talker_generate.py",
        "talker_loss.py",
        _backbone_modeling_file(backbone_type),
    ]
    for aux_file in aux_files:
        src = pkg_dir / aux_file
        if not src.exists():
            src = base_p / aux_file
        if src.exists():
            shutil.copy2(str(src), str(output_p / aux_file))
    flattened = _flatten_nested_modules(output_p, pkg_dir)
    if flattened:
        print(f"  Flattened nested modules to export root: {', '.join(flattened)}")

    _export_spk_dict(model, output_p, sources)

    _embed_chat_template_file(output_p)


def _flatten_nested_modules(output_p: Path, pkg_dir: Path) -> list[str]:
    """Copy encoders/*.py to the export root and rewrite nested relative imports.

    Hugging Face remote-code loading looks for a literal file named
    ``encoders.auto_avsr.py`` for ``from .encoders.auto_avsr import ...``,
    so nested packages cannot be served as-is.
    """
    copied: list[str] = []
    src_dir = pkg_dir / "encoders"
    if not src_dir.is_dir():
        return copied
    for src in sorted(src_dir.glob("*.py")):
        if src.name == "__init__.py":
            continue
        dest = output_p / src.name
        if dest.exists() and dest.resolve() != src.resolve():
            raise RuntimeError(f"Export flatten collision: {src.name} already exists at {dest} " f"(from {src})")
        shutil.copy2(str(src), str(dest))
        copied.append(src.name)
        stem = src.stem
        old_import = f"from .encoders.{stem} import"
        new_import = f"from .{stem} import"
        for py_file in output_p.glob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if old_import in text:
                py_file.write_text(text.replace(old_import, new_import), encoding="utf-8")
    return copied


def _save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_p: Path,
    max_shard_size: str = "5GB",
) -> None:
    """Save a state dict as sharded safetensors with an index file."""
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file as st_save

    plan = split_torch_state_dict_into_shards(
        state_dict,
        max_shard_size=max_shard_size,
    )

    for filename, shard_tensors in plan.filename_to_tensors.items():
        shard = {k: state_dict[k] for k in shard_tensors}
        st_save(shard, str(output_p / filename))

    if plan.is_sharded:
        index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
        with open(output_p / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Saved {len(plan.filename_to_tensors)} shards.")
    else:
        print("  Saved as single file.")


def _save_qwen3_omni(
    model: torch.nn.Module,
    base_model_path: str,
    output_p: Path,
    max_shard_size: str,
    inference_fallback: Path | None = None,
) -> None:
    print(f"Saving exported model (qwen3-omni format) to {output_p} ...")
    base_p = Path(base_model_path)

    model_type = getattr(model.config, "model_type", None)
    is_native_qwen3 = model_type in ("qwen3_omni_moe", "qwen3_omni_moe_thinker")

    if is_native_qwen3:
        # save_pretrained handles MoE weight conversion (fused 3-D experts ->
        # individual per-expert tensors) automatically via revert_weight_conversion.
        model.save_pretrained(
            str(output_p),
            max_shard_size=max_shard_size,
            safe_serialization=True,
        )
        # Overwrite config.json with the original: save_pretrained modernises fields
        # (e.g. merges rope_theta/rope_scaling into rope_parameters) which breaks
        # vLLM-Omni's config parser.
        shutil.copy2(str(base_p / "config.json"), str(output_p / "config.json"))
    else:
        # SpeechLMM base: unfuse MoE weights, then re-nest under thinker.*.
        from transformers.core_model_loading import revert_weight_conversion

        state_dict = revert_weight_conversion(model, model.state_dict())
        state_dict = _renest_state_dict(state_dict)
        n_thinker = sum(1 for k in state_dict if k.startswith("thinker."))
        print(f"  Re-nested {n_thinker} thinker keys.")
        _save_sharded_safetensors(state_dict, output_p, max_shard_size)

        qwen3_cfg = _build_qwen3_config_dict(base_p / "config.json")
        with open(output_p / "config.json", "w") as f:
            json.dump(qwen3_cfg, f, indent=2)

    _copy_tokenizer_and_processor_files(base_p, output_p, inference_fallback)


def export_checkpoint(
    base_model_path: str,
    checkpoint_path: str,
    output_dir: str,
    trainable_module_paths: list[str] | None = None,
    dtype: str = "auto",
    max_shard_size: str = "5GB",
    output_format: str = "speechlmm",
    use_speechlmm_wrapper: bool = False,
    lipread_encoder_weights: str | None = None,
    inference_fallback: str | None = None,
) -> None:
    ckpt_p = Path(checkpoint_path)
    is_lora = has_lora_adapter(ckpt_p)
    fallback_p = Path(inference_fallback) if inference_fallback else None

    base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    base_model_type = getattr(base_config, "model_type", None)

    # When the base is a raw Qwen3-Omni model but the checkpoint was produced
    # with use_speechlmm_wrapper=True, the LoRA adapter uses SpeechLMM's flat
    # key layout (audio_tower.* etc.).  PEFT fullmatches module names, so we
    # remap the adapter to the thinker.* namespace before loading -- unless we
    # re-wrap the base with --use-speechlmm-wrapper, in which case the wrapped
    # model already exposes the flat layout and no remap is needed.
    needs_qwen3_remap = (
        is_lora
        and not use_speechlmm_wrapper
        and base_model_type in ("qwen3_omni_moe", "qwen2_5_omni")
        and _uses_speechlmm_flat_keys(ckpt_p)
    )

    print(f"Loading base model from {base_model_path} ...")
    model = load_base_model(
        base_model_path,
        use_speechlmm_wrapper=use_speechlmm_wrapper,
        lipread_encoder_weights=lipread_encoder_weights,
    )

    _tmp_adapter_dir = None
    try:
        if is_lora:
            from llamafactory.model.model_utils.qwen3_omni_moe_lora import (
                load_peft_model_maybe_expert_lora,
            )

            if needs_qwen3_remap:
                print("Detected SpeechLMM wrapper keys in checkpoint — " "remapping LoRA to thinker.* namespace ...")
                _tmp_adapter_dir = _patch_lora_for_qwen3(ckpt_p)
                lora_path = _tmp_adapter_dir.name
            else:
                lora_path = str(ckpt_p)

            print(f"Loading LoRA adapter from {lora_path} ...")
            # Conditionally re-register Experts LoRA mapping (not in adapter_config).
            model = load_peft_model_maybe_expert_lora(model, lora_path)
            model = model.merge_and_unload()
            if any("lora_" in key for key in model.state_dict()):
                raise RuntimeError("LoRA merge left residual lora_* keys in the exported model.")
            print("LoRA adapters merged successfully.")
        else:
            print("No LoRA adapter found — full-finetuning checkpoint.")

        trainable_state = load_trainable_modules(ckpt_p, trainable_module_paths)
        apply_trainable_modules(model, trainable_state)

        # PEFT merge_and_unload can reset modules outside the LoRA target; reload AutoAVSR
        # after merge so lipread_encoder weights are correct in the exported checkpoint.
        if lipread_encoder_weights and hasattr(model, "load_auto_avsr_weights"):
            print(f"Reloading pretrained lipread encoder from {lipread_encoder_weights} ...")
            model.load_auto_avsr_weights(lipread_encoder_weights)

        baked_tokenizer = None
        if output_format == "speechlmm" and hasattr(model, "lipread_encoder"):
            print("Baking lipread special tokens into tokenizer and embeddings ...")
            baked_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            num_added = bake_lipread_special_tokens(model, baked_tokenizer)
            print(
                f"  Lipread tokenizer bake: {num_added} new token(s), "
                f"vocab={len(baked_tokenizer)}, embed={model.get_input_embeddings().weight.shape[0]}"
            )

        if dtype != "auto":
            target_dtype = getattr(torch, dtype)
            model = model.to(target_dtype)
            print(f"Converted model to {target_dtype}")

        if hasattr(model, "_fix_lipread_bn_buffers"):
            model._fix_lipread_bn_buffers()

        save_exported_model(
            model,
            base_model_path,
            output_dir,
            max_shard_size,
            output_format,
            tokenizer=baked_tokenizer,
            inference_fallback=fallback_p,
        )
        print(f"Done. Exported model ({output_format} format) saved to {output_dir}")
    finally:
        if _tmp_adapter_dir is not None:
            _tmp_adapter_dir.cleanup()


def main():
    install_safetensors_umask()
    parser = argparse.ArgumentParser(
        description="Export a training checkpoint (LoRA or full-finetuning) into a full model for inference."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to the original model with full weights (model_name_or_path used during training)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory (LoRA adapter or full-finetuning checkpoint)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the exported model",
    )
    parser.add_argument(
        "--trainable-module-paths",
        default=None,
        help="Comma-separated module path prefixes trained with trainable_module_paths "
        "(e.g. audio_tower.proj1,audio_tower.proj2,audio_tower.ln_post). "
        "Only needed for old checkpoints without trainable_modules.safetensors",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Output dtype (default: auto, keeps original)",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Max shard size for safetensors output (default: 5GB)",
    )
    parser.add_argument(
        "--format",
        default="speechlmm",
        choices=["speechlmm", "qwen3-omni"],
        help="Output model format: 'speechlmm' (flat layout) or "
        "'qwen3-omni' (native audio/text layout for vLLM-Omni; no lipread)",
    )
    parser.add_argument(
        "--use-speechlmm-wrapper",
        action="store_true",
        help="Wrap a Qwen3-Omni base model with SpeechLMM before applying LoRA. "
        "Use for speechlmm export of a checkpoint trained with "
        "use_speechlmm_wrapper=true; omit for qwen3-omni export.",
    )
    parser.add_argument(
        "--lipread-encoder-weights",
        default=None,
        help="Path to pretrained AutoAVSR weights (.pth) for the lipread encoder.",
    )
    parser.add_argument(
        "--inference-fallback-model",
        default=None,
        help="Optional model directory for tokenizer/processor files missing on the base.",
    )
    args = parser.parse_args()

    trainable_paths = None
    if args.trainable_module_paths:
        trainable_paths = [p.strip() for p in args.trainable_module_paths.split(",")]

    export_checkpoint(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        trainable_module_paths=trainable_paths,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        output_format=args.format,
        use_speechlmm_wrapper=args.use_speechlmm_wrapper,
        lipread_encoder_weights=args.lipread_encoder_weights,
        inference_fallback=args.inference_fallback_model,
    )


if __name__ == "__main__":
    main()
