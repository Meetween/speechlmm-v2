#!/bin/bash
# Score every contiguous BLOCK_SIZE-layer Thinker drop that keeps the first and
# last layer: prune, evaluate JSD to the unpruned teacher on CALIB, then delete
# the pruned copy. Process RANK of WORLD_SIZE takes every WORLD_SIZE-th block.
set -euo pipefail

SRC=${SRC:?set SRC to the teacher checkpoint}
CALIB=${CALIB:?set CALIB to the calibration DatasetDict}
MEDIA_DIR=${MEDIA_DIR:?set MEDIA_DIR to the media root}
SCORE_ROOT=${SCORE_ROOT:?set SCORE_ROOT to the output directory}
BLOCK_SIZE=${BLOCK_SIZE:-8}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)

num_layers=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["thinker_config"]["text_config"]["num_hidden_layers"])' "${SRC}/config.json")
starts=($(seq 1 $((num_layers - 1 - BLOCK_SIZE))))

for index in "${!starts[@]}"; do
    (( index % WORLD_SIZE == RANK )) || continue
    start=${starts[$index]}
    block="${SCORE_ROOT}/block_${start}"
    [[ -f "${block}/eval_results.json" ]] && continue

    rm -rf "${block}"
    python "${PROJECT_DIR}/scripts/pruning/prune_layers.py" \
        --src "${SRC}" --dst "${block}/model" --drop "${start}:$((start + BLOCK_SIZE - 1))"
    cat > "${block}/score.yaml" <<EOF
model_name_or_path: ${block}/model
ref_model: ${SRC}
trust_remote_code: true
stage: kd
do_train: false
do_eval: true
finetuning_type: full
kd_ce_weight: 0
kd_jsd_weight: 1
eval_dataset: calib
tokenized_path: ${CALIB}
media_dir: ${MEDIA_DIR}
template: speechlmm_qwen2_omni
output_dir: ${block}
bf16: true
flash_attn: fa2
per_device_eval_batch_size: 1
report_to: none
EOF
    # One single-GPU process per rank, outside the srun process group.
    (
        cd "${PROJECT_DIR}/llamafactory"
        unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
        FORCE_TORCHRUN=1 NPROC_PER_NODE=1 MASTER_PORT=$((29500 + ${SLURM_PROCID:-0})) \
            llamafactory-cli train "${block}/score.yaml"
    )
    rm -rf "${block}/model"
done
