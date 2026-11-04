# SpeechLMM-v2

A multimodal and multilingual foundation model supporting audio, video and text.

## Overview

SpeechLMM-v2 is a foundation model designed to handle multiple modalities including audio, video, and text. Built on top of a fork of [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), it provides a comprehensive framework for training and fine-tuning multimodal language models.

> **PLGrid Athena:** see [README_plgrid-athena.md](README_plgrid-athena.md).

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git (for submodule management)

## Installation

### Clone

```bash
git clone --recurse-submodules https://github.com/Meetween/speechlmm-v2.git
cd speechlmm-v2
```

If submodules are missing:

```bash
git submodule update --init --recursive
```

### Using uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[deepspeed,logging]"
```

`logging` installs `wandb` and `tensorboard` for cluster training.

`llamafactory/` is installed automatically when you clone with `--recurse-submodules`.

## Project Structure

```
speechlmm-v2/
├── src/speechlmm/       # Model, data, evaluation
├── llamafactory/        # LLaMA Factory (git submodule)
├── configs/             # Training configs
├── docker/              # base.Dockerfile, train.Dockerfile
├── scripts/
│   ├── build/           # Docker / SIF build
│   ├── install/         # Cluster-specific installers (e.g. plgrid_athena.sh)
│   ├── pruning/         # Thinker layer pruning: subsets, block scoring, prune
│   ├── training/        # Local / container launchers
│   └── export_checkpoint.py
└── pyproject.toml
```

## Building Docker Images and SIF

From the repo root, set build variables (edit `examples/build/template_build.sh` or export them in your shell):

- `BASE_IMAGE_BASENAME`, `BASE_IMAGE_VERSION`
- `TRAIN_IMAGE_BASENAME`, `TRAIN_IMAGE_VERSION`
- `TRAIN_SIF_DIR` — output directory for the `.sif` file (`TRAIN_SIF_PATH` is derived automatically)

```bash
export TRAIN_SIF_DIR=/path/to/output
source scripts/build/env.sh
bash scripts/build/build_base.sh
bash scripts/build/build_train.sh
bash scripts/build/build_sif.sh
```

See `examples/build/template_build.sh` for a full example.

> **Note**: `build_sif.sh` pushes the training image to a local Docker registry (`localhost:5000`) before converting it to a Singularity image. Make sure the registry is running before executing this step.

## Training

Training arguments and example configurations are documented in
[Training configurations](examples/training/README.md).

### Training on local machine
For local training, see `scripts/training/run_llamafactory.sh`.

Set `WANDB_API_KEY` (and `WANDB_ENTITY` for team projects) before launching.
`run_llamafactory.sh` expects environment variables such as `SHARED_DIR` and `HF_HOME` to be set in your shell.

Run the following:
```bash
bash scripts/training/run_llamafactory.sh $CONFIG
```

### Training on local machine through singularity container
For container training, see `scripts/training/run_sif.sh`.

Bind-mounts and env vars must be set before launch.

```bash
bash scripts/training/run_sif.sh
```

## Exporting Checkpoints

Export flags, checkpoint layouts, SpeechLMM remapping, and expert LoRA notes
are documented in [Export configurations](examples/export/README.md).

```bash
python scripts/export_checkpoint.py \
    --base-model /path/to/Qwen3-Omni-30B-A3B-Instruct \
    --checkpoint output/my_training_run/checkpoint-500 \
    --format qwen3-omni \
    --output-dir output/exported
```

## Development

### Testing

The unit tests need no GPU and no datasets:

```bash
python -m unittest discover -s tests
```

Tests that require checkpoints or datasets skip unless their env var is set:
`SPEECHLMM_DEBUG_QWEN3`, `SPEECHLMM_QWEN25_SOURCE`, `SPEECHLMM_TEST_MIXED_DATA`,
plus `MODELS_DIR` and `TRAINING_CHECKPOINTS_DIR`.

### Code Formatting

This project uses `black` for code formatting. Install with `uv pip install -e ".[dev]"`, then:

```bash
black .
```

## License

See [LICENSE](LICENSE).

## Links

- [Repository](https://github.com/Meetween/speechlmm-v2)
- [Issues](https://github.com/Meetween/speechlmm-v2/issues)

## Acknowledgements

This project builds upon [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), a unified framework for efficient fine-tuning of large language models.

This work is part of [Meetween](https://meetween.eu) — *My Personal AI Mediator
for Virtual Meetings* — funded by the European Union under the Horizon Europe
programme, grant agreement No. 101135798 (HORIZON-CL4-2023-HUMAN-01-CNECT).

Views and opinions expressed are those of the author(s) only and do not
necessarily reflect those of the European Union or the European Commission.
Neither the European Union nor the granting authority can be held responsible
for them.
