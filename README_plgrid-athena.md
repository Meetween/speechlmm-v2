# PLGrid Athena setup

> For PLGrid Athena only. Other machines: [README.md](README.md).

Home quota is too small for PyTorch/uv caches. **Everything goes on `$SCRATCH`.**

## Preliminary setup

1. **GitHub SSH** — `ssh -T git@github.com` must succeed.
2. **Hugging Face** — request access to [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) if needed.

## System requirements

- Linux, NVIDIA A100 (CUDA 12.8)
- ≥40 GB GPU VRAM for full model (4× A100 for 30B with ZeRO-3)
- Repos and venv on `$SCRATCH`, not `$HOME`

## Installation

```bash
cd "$SCRATCH"
git clone --recurse-submodules git@github.com:Meetween/speechlmm-v2.git
cd speechlmm-v2
bash scripts/install/plgrid_athena.sh
source "${SPEECHLMM_VENV}/bin/activate"
```

Installs `.[deepspeed,logging]` (DeepSpeed, wandb, tensorboard) with cu128 PyTorch wheels.

Default paths (override before install if needed):

```bash
export SPEECHLMM_DIR="${SPEECHLMM_DIR:-$SCRATCH/speechlmm-v2}"
export SPEECHLMM_VENV="${SPEECHLMM_VENV:-$SPEECHLMM_DIR/.venv}"
```

Optional `~/.bashrc` snippet:

```bash
export SPEECHLMM_DIR="${SPEECHLMM_DIR:-$SCRATCH/speechlmm-v2}"
export SPEECHLMM_VENV="${SPEECHLMM_VENV:-$SPEECHLMM_DIR/.venv}"
export HF_HOME="${SCRATCH}/.cache/huggingface"
```

After `git pull`: `git submodule update --init --recursive`

**Do not** `pip install torch` from default PyPI — it pulls CUDA 13 wheels and fails on Athena.

## Check install

On a login node (with the venv activated):

```bash
python -c "import torch, torchcodec; print(torch.__version__, torchcodec.__version__)"
```

CUDA is not available on login nodes; run GPU checks inside a SLURM job.

## W&B (optional, for online logging)

Add to `~/.wandb_env` (entity slug from the wandb.ai URL — often a team name, not your username):

```bash
export WANDB_API_KEY=<your-key>
export WANDB_ENTITY=<team-slug>
export WANDB_PROJECT=speechlmm-v2
```

## Runtime notes

- **uv, not conda** — install and venv live in this repo under `$SCRATCH`.
- **llamafactory** — git submodule in `speechlmm-v2/llamafactory`; install only via `plgrid_athena.sh`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Storage quota of $HOME exceeded` | Keep caches on `$SCRATCH` (`HF_HOME`, `UV_CACHE_DIR`); remove `~/.cache/uv` |
| `ModuleNotFoundError` after `git pull` | `git submodule update --init --recursive` |
| `driver too old (found version 12080)` | Re-run `bash scripts/install/plgrid_athena.sh` (cu128) |
