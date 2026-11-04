# Training configurations

The YAML files in this directory are minimal examples for SpeechLMM
fine-tuning. Replace placeholder model and dataset paths with local paths
available in the training environment.

## Component LoRA

Set `finetuning_type: lora`, then enable only the model components needed by
the run. Component-specific rank and alpha values fall back to the global
`lora_rank` and `lora_alpha` when omitted.

### Thinker scope

Use `lora_language_model_scope` to select the Thinker targets:

- `none`: no Thinker LoRA.
- `attention`: LoRA on the attention projections and `lm_head`. On dense
  SpeechLMM (Qwen2.5) this is `q_proj`/`k_proj`/`v_proj`/`o_proj` only;
  `gate_proj`/`up_proj`/`down_proj` are not included.
- `attention_and_ffn`: attention and `lm_head` plus the dense FFN Linears.
  Use this for SpeechLMM-v2 L when you want Thinker MLP LoRA.
- `attention_and_experts`: attention and `lm_head` LoRA plus sparse,
  per-expert LoRA on every fused Thinker MoE FFN (Qwen3-Omni).

Do not use `lora_target: all` to mean Thinker attention. That flag
selects every unfrozen Linear. Prefer `lora_language_model_scope`.

The validated Qwen3-Omni-30B configuration is:

```yaml
### method
finetuning_type: lora
lora_language_model_scope: attention_and_experts

# Thinker attention and lm_head
lora_language_model_rank: 8
lora_language_model_alpha: 16

# Fused Thinker MoE experts
lora_language_model_experts_rank: 4
lora_language_model_experts_alpha: 8
```

Expert rank 4 adds 207,618,048 trainable parameters. Combined with attention
rank 8, the configuration has 215,535,616 trainable parameters. Alpha controls
adapter scaling and does not change the parameter count.

Expert LoRA requires Transformers 5.x fused Thinker experts, eager expert
execution, a non-quantized base model, and `finetuning_type: lora`. The
validated 30B training path uses DeepSpeed ZeRO-3.

Existing attention-only configurations using `lora_language_model: true`
remain supported and are equivalent to
`lora_language_model_scope: attention`. New configurations should use the
scope argument.

See `lora_all_components.yaml` for a complete mixed-rank example.

### Other components

The remaining component scopes are independent:

```yaml
# Whisper audio encoder
lora_audio_encoder: true
lora_audio_encoder_rank: 8
lora_audio_encoder_alpha: 16

# SpeechLMM audio projectors; ln_post is trained as a full-weight module
lora_audio_adapters: true
lora_audio_adapters_rank: 4
lora_audio_adapters_alpha: 8

# Visual speech encoder
lora_lipread_encoder: true
lora_lipread_encoder_rank: 8
lora_lipread_encoder_alpha: 16

# Visual speech projector
lora_lipread_adapter: true
lora_lipread_adapter_rank: 4
lora_lipread_adapter_alpha: 8
```

Relevant examples:

- `lora_thinker_attention_and_ffn.yaml`
- `lora_audio_adapters_only.yaml`
- `lora_audio_encoder_and_adapters.yaml`
- `lora_lipread.yaml`
- `lora_all_components.yaml`

## Training and evaluation dataset mixing

Training and evaluation have separate interleave probabilities. This allows
them to use different dataset counts.

Use `interleave_probs` for training:

```yaml
dataset: train_a,train_b,train_c
mix_strategy: interleave_under
interleave_probs: 0.5,0.3,0.2
```

Use `eval_interleave_probs` when evaluation datasets are merged and
interleaved:

```yaml
eval_dataset: dev_a,dev_b
eval_interleave_probs: 0.7,0.3
```

Do not set `eval_interleave_probs` when each evaluation dataset is evaluated
separately:

```yaml
eval_dataset: dev_a,dev_b
eval_on_each_dataset: true
```

Existing training configurations using only `interleave_probs` remain valid.
For merged evaluation, training probabilities are reused when
`eval_interleave_probs` is omitted and the training and evaluation dataset
counts match.

## Checkpoint resume

A checkpoint restores the model, optimizer, scheduler, RNG, and `global_step`.
It does not restore the position of the training dataloader.

With the default `ignore_data_skip: false`, resume rebuilds the dataloader and
discards the batches already consumed in the current epoch (`skip_first_batches`).
Those batches are still collated. For SpeechLMM that includes audio decoding and
feature extraction, so resuming at a high step can sit for a long time before the
first optimizer update and is easy to mistake for a hang.

`ignore_data_skip: true` on its own skips that replay and retrains the same
prefix. The weights continue from the checkpoint while the data starts again at
the first sample. Across repeated restarts the head of the dataset is
oversampled and the tail may never be reached. Nothing raises; the loss curve
still looks plausible.

Automatic resume only happens when `overwrite_output_dir` is false.
`src/llamafactory/hparams/parser.py` picks up the last checkpoint in `output_dir`
only in that case. With `overwrite_output_dir: true`, a restarted job trains
from step 0 unless `resume_from_checkpoint` is set explicitly.

Set `use_stateful_dataloader: true` to checkpoint the training dataloader
instead of replaying it. Each rank writes `dataloader_state_rank_{rank}.pt` in
the checkpoint directory, and resume loads that file. The flag forces
`ignore_data_skip: true`. Evaluation keeps the ordinary dataloader. The flag is
only supported for `stage: sft` without KTransformers, and it needs torchdata.

Resume must use the same world size, dataloader worker count, per-device batch
size, gradient accumulation, seed, `dataloader_drop_last`, and data mix
(`dataset`, `streaming`, `mix_strategy`, `interleave_probs`, `buffer_size`,
`cutoff_len`, `packing`, `neat_packing`). A missing per-rank file or any
mismatch raises, rather than silently retraining the prefix.

With `dataloader_num_workers > 0` the restored position is exact. With
`dataloader_num_workers: 0`, accelerate rewinds the saved position by
`world_size - 1` batches, so those batches are trained again after resume.

`streaming: true` takes the untested dispatcher path and logs a warning.

```yaml
### method
use_stateful_dataloader: true

### output
overwrite_output_dir: false
resume_from_checkpoint: /path/to/output_dir/checkpoint-1000
```

## Export

After training, see [Export configurations](../export/README.md).
