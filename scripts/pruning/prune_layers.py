#!/usr/bin/env python3
"""Drop a contiguous block of Thinker decoder layers from a SpeechLMM checkpoint.

All other modules are copied unchanged, so the hidden size seen by the Talker stays the same.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

LAYER_PREFIX = "model.layers."


def rename_key(key: str, new_index: dict[int, int]) -> str | None:
    if not key.startswith(LAYER_PREFIX):
        return key
    index, _, tail = key[len(LAYER_PREFIX) :].partition(".")
    if int(index) not in new_index:
        return None
    return f"{LAYER_PREFIX}{new_index[int(index)]}.{tail}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--drop", required=True, help="Inclusive layer range START:END, e.g. 16:23")
    args = parser.parse_args()

    start, end = (int(value) for value in args.drop.split(":"))
    config = json.loads((args.src / "config.json").read_text())
    text_config = config["thinker_config"]["text_config"]
    num_layers = text_config["num_hidden_layers"]
    if not 0 < start <= end < num_layers - 1:
        raise SystemExit(f"--drop must keep the first and last layer, got {args.drop} for {num_layers} layers")

    kept = [index for index in range(num_layers) if not start <= index <= end]
    new_index = {old: new for new, old in enumerate(kept)}
    text_config["num_hidden_layers"] = len(kept)
    if "layer_types" in text_config:
        text_config["layer_types"] = [text_config["layer_types"][index] for index in kept]

    args.dst.mkdir(parents=True)
    for path in args.src.iterdir():
        if path.suffix != ".safetensors" and path.name not in ("config.json", "model.safetensors.index.json"):
            (shutil.copytree if path.is_dir() else shutil.copy2)(path, args.dst / path.name)
    (args.dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    index_path = args.src / "model.safetensors.index.json"
    if index_path.is_file():
        shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    elif (args.src / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    else:
        raise SystemExit(f"{args.src} has no model.safetensors or model.safetensors.index.json")
    weight_map = {}
    total_size = 0
    for shard in shards:
        tensors = {}
        with safe_open(args.src / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                renamed = rename_key(key, new_index)
                if renamed is not None:
                    tensors[renamed] = handle.get_tensor(key)
        save_file(tensors, args.dst / shard, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, shard))
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (args.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Kept layers {kept} in {args.dst}")


if __name__ == "__main__":
    main()
