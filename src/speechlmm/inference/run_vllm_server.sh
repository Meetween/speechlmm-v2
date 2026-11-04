
# enter a compute node;
# srun -p plgrid-gpu-a100 -A plgmeetween2026-gpu-a100 --gres=gpu:4 --cpus-per-task=16 --mem=128G --time=04:00:00 --pty bash


UV_VENV_PATH=/path/to/venv # Change it with your env

MODEL_PATH=/path/to/Qwen/Qwen3-Omni-30B-A3B-Instruct
CONFIG_PATH=/path/to/vllm_omni/qwen3_omni_moe.yaml

source $UV_VENV_PATH/bin/activate

export TORCHINDUCTOR_CACHE_DIR=${SCRATCH}/torch_cache
export VLLM_CACHE_ROOT=${SCRATCH}/vllm_cache

module load CUDA/12.8.0
vllm serve ${MODEL_PATH} --omni --port 8091 --stage-configs-path ${CONFIG_PATH}


