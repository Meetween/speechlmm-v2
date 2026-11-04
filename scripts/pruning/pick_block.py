#!/usr/bin/env python3
"""Rank the scored blocks by JSD to the teacher, averaged per task family, and pick the lowest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Directory holding block_<start>/eval_results.json")
    parser.add_argument("--manifest", type=Path, required=True, help="subset_manifest.json of the calibration set")
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    family_of = {row["key"]: row["family"] for row in json.loads(args.manifest.read_text())["rows"]}
    ranking = []
    for path in args.scores.glob("block_*/eval_results.json"):
        metrics = json.loads(path.read_text())
        losses = defaultdict(list)
        for source, family in family_of.items():
            losses[family].append(metrics[f"eval_{source}_loss"])
        families = {family: sum(values) / len(values) for family, values in sorted(losses.items())}
        start = int(path.parent.name.removeprefix("block_"))
        drop = f"{start}:{start + args.block_size - 1}"
        ranking.append({"drop": drop, "score": sum(families.values()) / len(families), "families": families})

    ranking.sort(key=lambda block: block["score"])
    args.output.write_text(json.dumps({"drop": ranking[0]["drop"], "ranking": ranking}, indent=2) + "\n")
    for block in ranking:
        print(f"{block['drop']:>6}  {block['score']:.4f}")


if __name__ == "__main__":
    main()
