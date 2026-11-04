#!/usr/bin/env python3
"""Tokenize a dataset and save to tokenized_path without loading the full model.

    python -m speechlmm.data.preprocess_dataset <config.yaml> [run_name=...]
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def _ensure_distributed() -> None:
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def main() -> None:
    _ensure_distributed()
    # get_train_args() with no args reads sys.argv directly so read_args() loads the
    # YAML config and merges key=value overrides (same contract as llamafactory-cli).
    model_args, data_args, training_args, finetuning_args, _generating_args = get_train_args()
    if data_args.tokenized_path is None:
        raise SystemExit("tokenized_path must be set in the config")

    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage=finetuning_args.stage,
        **tokenizer_module,
    )
    print(f"Tokenized dataset saved to {data_args.tokenized_path}")


if __name__ == "__main__":
    main()
