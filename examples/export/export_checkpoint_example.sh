#!/bin/bash
# Export a training checkpoint into a standalone model directory.
#
# Run from the speechlmm-v2 repo root. Set BASE_MODEL to the same path as
# model_name_or_path in your training YAML (absolute path, not a Hub ID).
#
# --format qwen3-omni  → native Qwen / vLLM-Omni audio-text inference (no lipread)
# --format speechlmm   → resume training or inference with custom modules (e.g. lipread)
#
# Intermediate checkpoints work: use checkpoint-N/ (e.g. checkpoint-200), not only the last step.
# For wrapper-trained checkpoints, qwen3-omni export remaps flat LoRA keys to
# thinker.*; speechlmm export from a raw Qwen base uses --use-speechlmm-wrapper.
#
# ━━━ Expected input checkpoint directory ━━━
#
#   Adapter-only finetuning (finetuning_type: full + trainable_module_paths):
#     checkpoint-500/
#     ├── trainable_modules.safetensors   ← proj1, proj2, ln_post, …
#     ├── trainer_state.json
#     └── training_args.bin
#
#   LoRA + adapter finetuning (typical ASR/ST with use_speechlmm_wrapper):
#     checkpoint-500/
#     ├── adapter_config.json
#     ├── adapter_model.safetensors
#     ├── trainable_modules.safetensors
#     ├── trainer_state.json
#     └── training_args.bin
#
#   LoRA only (adapters frozen, no trainable_module_paths):
#     checkpoint-500/
#     ├── adapter_config.json
#     ├── adapter_model.safetensors
#     └── trainer_state.json
#
#   global_step*/ (+ zero_to_fp32.py) only if trainable_modules.safetensors is missing;
#   then pass --trainable-module-paths (see legacy example below).
#
# ━━━ Generated output (--format qwen3-omni) ━━━
#
#   output/exported/
#   ├── config.json
#   ├── model-00001-of-00015.safetensors
#   ├── ...
#   ├── model.safetensors.index.json
#   ├── tokenizer.json
#   ├── tokenizer_config.json
#   ├── preprocessor_config.json
#   └── ...
#
#   Load:  Qwen3OmniMoeForConditionalGeneration.from_pretrained("output/exported")
#   Serve: vllm serve output/exported --omni --stage-configs-path ...
#
# ━━━ Generated output (--format speechlmm) ━━━
#
#   output/exported_speechlmm/
#   ├── config.json              ← model_type speechlmm when base was a SpeechLMM dir
#   ├── model*.safetensors
#   ├── configuration_speechlmm.py, modeling_speechlmm.py  (if present on base)
#   └── tokenizer / processor files

BASE_MODEL="/path/to/Qwen3-Omni-30B-A3B-Instruct"
# Optional: use a saved SpeechLMM directory as base when resuming native SpeechLMM training
# SPEECHLMM_BASE="/path/to/speechlmm_checkpoint_or_debug_model"

# ── 1. Adapter-only → qwen3-omni (inference) ─────────────────────────────────
# Training: freeze_audio_encoder, trainable_module_paths on proj1/proj2/ln_post.
# No LoRA files in checkpoint. Wrapper remap not involved.
python scripts/export_checkpoint.py \
    --base-model "${BASE_MODEL}" \
    --checkpoint output/adapter_run/checkpoint-500 \
    --format qwen3-omni \
    --output-dir output/exported

# ── 2. LoRA encoder + FT adapters → qwen3-omni (inference, main ASR/ST path) ─
# Training: lora_audio_encoder + freeze_audio_adapters: false, use_speechlmm_wrapper.
# Script auto-remaps LoRA audio_tower.* → thinker.audio_tower.*, then merges adapters.
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_run/checkpoint-500 \
#     --format qwen3-omni \
#     --output-dir output/exported

# ── 3. LoRA only → qwen3-omni (inference) ───────────────────────────────────
# Training: lora on encoder (or LLM), adapters frozen. Checkpoint has adapter_* only.
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_encoder_only/checkpoint-500 \
#     --format qwen3-omni \
#     --output-dir output/exported

# ── 4. Intermediate checkpoint (same flags as above) ─────────────────────────
# Export any checkpoint-N/ while training is still running or before deleting old steps.
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_run/checkpoint-200 \
#     --format qwen3-omni \
#     --output-dir output/exported_step200

# ── 5. Export latest weights from run root (optional layout) ─────────────────
# Some runs also copy adapter + trainable_modules to output_dir/ (not only checkpoint-*).
# Use that path if you did not keep checkpoint subdirs.
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_run \
#     --format qwen3-omni \
#     --output-dir output/exported

# ── 6. Resume training / lipread → speechlmm layout ──────────────────────────
# Same checkpoint as (2); flat layout for LlamaFactory. On a raw Qwen3-Omni base,
# --use-speechlmm-wrapper grafts the SpeechLMM modules before merge; for lipread
# also pass --lipread-encoder-weights <autoavsr.pth>. For a native SpeechLMM base
# dir, set --base-model to SPEECHLMM_BASE and drop --use-speechlmm-wrapper.
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_run/checkpoint-500 \
#     --format speechlmm \
#     --use-speechlmm-wrapper \
#     --output-dir output/exported_speechlmm

# ── 7. Legacy checkpoint (no trainable_modules.safetensors) ──────────────────
# Requires global_step*/ and zero_to_fp32.py in the checkpoint dir. Lists which
# modules to pull from the ZeRO shard (must match trainable_module_paths from training).
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_run/checkpoint-500 \
#     --trainable-module-paths audio_tower.proj1,audio_tower.proj2,audio_tower.ln_post \
#     --format qwen3-omni \
#     --output-dir output/exported

# ── 8. Optional: force output dtype ──────────────────────────────────────────
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_run/checkpoint-500 \
#     --format qwen3-omni \
#     --dtype bfloat16 \
#     --output-dir output/exported

# ── 9. Optional: smaller shards for slow filesystems ─────────────────────────
# python scripts/export_checkpoint.py \
#     --base-model "${BASE_MODEL}" \
#     --checkpoint output/lora_run/checkpoint-500 \
#     --format qwen3-omni \
#     --max-shard-size 2GB \
#     --output-dir output/exported
