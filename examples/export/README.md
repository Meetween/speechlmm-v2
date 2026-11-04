# Export configurations

Use `scripts/export_checkpoint.py` to turn a training checkpoint into a
standalone Hugging Face model directory. Run commands from the speechlmm-v2
repo root. See `export_checkpoint_example.sh` for copy-paste recipes.

## Output formats

| `--format` | Layout | Use for |
|---|---|---|
| `qwen3-omni` | Nested `thinker.*` weights, Qwen3OmniMoe config | Native Qwen / vLLM-Omni audio-text inference |
| `speechlmm` | Flat SpeechLMM layout | Resume training or use custom modules such as lipread |

`qwen3-omni` does not represent SpeechLMM-only modules such as the lipread
encoder and adapter. Use `speechlmm` when the exported model must retain
lipread capability.

`--base-model` must be the same full-weight path used as `model_name_or_path`
during training (local path, not a Hub ID alone).

## Typical command

```bash
python scripts/export_checkpoint.py \
    --base-model /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --checkpoint output/my_run/checkpoint-500 \
    --format qwen3-omni \
    --output-dir output/exported
```

Intermediate `checkpoint-N/` directories work the same as the final run root.

## Expected checkpoint contents

LoRA + trainable modules (common ASR/ST setup with the SpeechLMM wrapper):

```text
checkpoint-500/
├── adapter_config.json
├── adapter_model.safetensors
├── trainable_modules.safetensors
└── ...
```

LoRA only:

```text
checkpoint-500/
├── adapter_config.json
└── adapter_model.safetensors
```

Full-weight `trainable_module_paths` only (no LoRA):

```text
checkpoint-500/
└── trainable_modules.safetensors
```

If `trainable_modules.safetensors` is missing, pass
`--trainable-module-paths` and keep the DeepSpeed `global_step*/` shards so
the script can consolidate them.

## SpeechLMM wrapper remapping

Training with `use_speechlmm_wrapper: true` stores LoRA targets in the flat
namespace (`audio_tower.*`, `model.*`, `lm_head.*`, …). Export detects that
and remaps keys to `thinker.*` before loading onto a raw Qwen3-Omni base. No
manual remapping is required.

## Thinker expert LoRA

Checkpoints trained with
`lora_language_model_scope: attention_and_experts` include custom expert
adapter tensors (`lora_A_gate`, `lora_B_gate`, `lora_A_up`, `lora_B_up`,
`lora_A_down`, `lora_B_down`). Export re-registers the expert LoRA module,
merges adapters with bounded per-expert memory, and asserts that no `lora_*`
keys remain in the exported weights.

For `--format qwen3-omni`, Thinker MoE experts are written in the unfused
on-disk layout expected by Transformers / vLLM-Omni
(`thinker.model.layers.*.mlp.experts.{i}.{gate,up,down}_proj.weight`).

## Other flags

```bash
# Force output dtype
--dtype bfloat16

# Smaller shards on slow filesystems
--max-shard-size 2GB

# Legacy checkpoints without trainable_modules.safetensors
--trainable-module-paths audio_tower.proj1,audio_tower.proj2,audio_tower.ln_post
```

## merge_lora_checkpoint.py

`scripts/merge_lora_checkpoint.py` is a thin wrapper around the same export
path (`--format speechlmm`). Prefer `export_checkpoint.py` for new workflows
so remapping, expert LoRA merge, and both output formats stay single-sourced.
