#!/usr/bin/env python3
"""Sample a tokenized training tree into the calibration or healing DatasetDict.

``--root`` is a prepared tree with ``assembly_manifest.json``, one
``linked_backing/<source>/`` per source and ``validation_<source>/`` splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

LANGS = ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv")
ST_PAIRS = ("en_cs", "en_hu", "en_sv", "cs_en", "hu_en", "sv_en", "en_de", "de_en")
EUROPARL_ST_PAIRS = ("en_de", "en_es", "en_fr", "en_it", "en_nl", "en_pt", "de_en", "es_en", "fr_en", "it_en", "nl_en", "pt_en")
EUROPARL_MT_PAIRS = ("en_de", "en_es", "en_fr", "en_it", "de_en", "es_en", "fr_en", "it_en")

# (source, rows, task family). Families are weighted equally when scoring.
CALIB = [
    *[(f"fleurs_asr_train_{lang}", 2, "asr") for lang in LANGS],
    *[(f"voxpopuli_eurospeech_asr_train_{lang}", 1, "asr") for lang in LANGS],
    *[(f"common_voice_asr_train_{lang}", 1, "asr") for lang in LANGS],
    *[(f"europarl_st_st_train_{pair}", 1, "st") for pair in EUROPARL_ST_PAIRS],
    *[(f"dgt_tm_st_train_{pair}", 1, "st") for pair in ST_PAIRS],
    *[(f"topical_chat_st_train_{pair}", 1, "st") for pair in ST_PAIRS],
    *[(f"europarl_st_mt_train_{pair}", 1, "mt") for pair in EUROPARL_MT_PAIRS],
    *[(f"spokensquad_sqa_train_{lang}", 2, "sqa") for lang in LANGS],
    *[(f"iwslt26_sqa_train_{name}", 4, "sqa") for name in ("en", "en_de", "en_it")],
    *[(f"librisqa_sqa_train_{lang}", 1, "tqa") for lang in LANGS],
    ("icsi_segmented_ssum_train_en", 8, "sum"),
    ("icsi_segmented_tsum_train_en", 8, "sum"),
    ("slurp_slu_train_en", 16, "slu"),
    ("speechmassive_slu_train_de", 8, "slu"),
    ("speechmassive_slu_train_fr", 8, "slu"),
    ("ami_avspeakd_train_en", 8, "meeting"),
    ("ami_achap_train_en", 4, "meeting"),
    ("ytseg_achap_train_en", 4, "meeting"),
    ("ytseg_asr_train_en", 4, "meeting"),
]

# Validation splits only. 16 rows each in the heal set, logged as one loss per split.
HEAL_EVAL = (
    *[f"fleurs_asr_train_{lang}" for lang in LANGS],
    *[f"europarl_st_st_train_en_{lang}" for lang in ("de", "es", "fr", "it", "nl", "pt")],
    *[f"dgt_tm_st_train_{pair}" for pair in ("en_cs", "cs_en", "en_hu", "hu_en", "en_sv", "sv_en")],
    *[
        f"europarl_st_mt_train_{pair}"
        for pair in ("en_de", "de_en", "en_fr", "fr_en", "es_en", "it_en", "nl_en", "pt_en")
    ],
    *[f"librisqa_sqa_train_{lang}" for lang in LANGS],
    "slurp_slu_train_en",
    "icsi_segmented_ssum_train_en",
    "icsi_segmented_tsum_train_en",
    "ami_achap_train_en",
    "ytseg_achap_train_en",
    "ami_avspeakd_train_en",
)


def load_source(root: Path, source: str) -> Dataset:
    files = sorted((root / "linked_backing" / source).glob("*.arrow"))
    return concatenate_datasets([Dataset.from_file(str(path)) for path in files])


def sample(dataset: Dataset, count: int, seed: str) -> Dataset:
    if count >= len(dataset):
        return dataset
    return dataset.select(sorted(random.Random(seed).sample(range(len(dataset)), count)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("calib", "heal"), required=True)
    parser.add_argument("--per-source", type=int, default=64, help="Heal train rows per source.")
    parser.add_argument("--eval-rows", type=int, default=32, help="Heal rows per validation split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = json.loads((args.root / "assembly_manifest.json").read_text())["source_counts"]
    dataset = DatasetDict()
    rows = []
    if args.mode == "calib":
        missing = [source for source, _, _ in CALIB if source not in sources]
        if missing:
            raise SystemExit(f"Sources not in {args.root}: {missing}")
        for source, count, family in CALIB:
            dataset[f"validation_{source}"] = sample(load_source(args.root, source), count, f"{args.seed}:{source}")
            rows.append({"key": source, "family": family, "rows": len(dataset[f"validation_{source}"])})
    else:
        train = [sample(load_source(args.root, source), args.per_source, f"{args.seed}:{source}") for source in sorted(sources)]
        dataset["train"] = concatenate_datasets(train)
        for source in HEAL_EVAL:
            split = load_from_disk(str(args.root / f"validation_{source}"))
            dataset[f"validation_{source}"] = sample(split, args.eval_rows, f"{args.seed}:eval:{source}")

    dataset.save_to_disk(str(args.output))
    manifest = {"mode": args.mode, "root": str(args.root), "seed": args.seed, "rows": rows}
    (args.output / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print({name: len(split) for name, split in dataset.items()})


if __name__ == "__main__":
    main()
