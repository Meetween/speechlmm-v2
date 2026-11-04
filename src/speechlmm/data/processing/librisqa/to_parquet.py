"""Convert normalized LibriSQA Part II JSONL to SpeechLMM Parquet.

Each output row contains a multiple-choice question, its candidate answers,
the selected answer, and the source language for text-only SQA training.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.librisqa.to_parquet --help
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import re

from speechlmm.data.utils import datasets_dir
from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row

logger = logging.getLogger(__name__)


TASKS = ["sqa"]
SOURCE = "librisqa"


def get_from_original(original_dir: Path) -> list[dict]:
    subsets = ["LibriSQA-PartI", "LibriSQA-PartII"]

    full_df: list[dict] = []
    for subset in subsets:
        original_json = original_dir / f"{subset}.jsonl"
        with original_json.open() as f:
            subset_df = [json.loads(line.strip()) for line in f]
            full_df.extend(subset_df)
            logger.info(f"Loaded {len(subset_df)} samples from {subset}")
    logger.info(f"Loaded {len(full_df)} samples.")

    return full_df


def _word_count(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.split())


def _base_row(
    entry: dict,
    source: str,
    task: str,
) -> dict:
    row = empty_row()
    source_dataset = entry.get("source_dataset")
    text_input = entry.get("text_input")
    if source_dataset == "LibriSQA-PartI":
        text_output = entry["text_output"]
        metadata = None
    elif source_dataset == "LibriSQA-PartII":
        text_input = re.sub(r"\n*Options:.+$", "", text_input, flags=re.DOTALL).strip()
        text_output = entry["answer_label"]
        metadata = {"options": list(entry["answer_choices"].values())}
    else:
        raise ValueError(f"Unknown source_dataset: {entry['source_dataset']}")

    row.update(
        {
            "sample_id": entry["sample_id"],
            "task": [task],
            "split": entry["split"],
            "source_dataset": source_dataset or source,
            "license": entry.get("license"),
            "text_input": text_input,
            "text_input_normalized": text_input,
            "text_prompt": entry["text_prompt"],
            "text_output": text_output,
            "text_output_normalized": text_output,
            "metadata": metadata,
            "num_words_input": _word_count(text_input),
            "num_words_output": _word_count(text_output),
            "language_text_input": entry["language_text_input"],
            "language_text_output": entry["language_text_output"],
            "language_text_prompt": entry["language_text_prompt"],
        }
    )

    return row


def build_task_rows(
    df: list,
    source: str,
    task: str,
) -> list[dict]:
    return [
        _base_row(
            entry,
            source=source,
            task=task,
        )
        for entry in df
    ]


def build_samples(
    df: list[dict],
    source: str,
    tasks: Sequence[str],
    splits: Sequence[str],
    languages: Sequence[str],
) -> list[dict]:
    samples = []

    selected = [
        entry
        for entry in df
        if entry["split"] in splits
        and (not languages or (entry.get("language_text_input") or entry.get("language_audio_input")) in languages)
    ]
    for task in tasks:
        samples.extend(
            build_task_rows(
                source=source,
                task=task,
                df=selected,
            )
        )

    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check LibriSQA annotations and filters.")
    for split in splits:
        part = df[df["split"] == split]
        if part.empty:
            logger.warning("No samples for split=%s", split)
            continue

        out = output_dir / f"{split}.parquet"
        # Cast to the shared schema so every corpus emits identical column types.
        table = pa.Table.from_pandas(part.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False)
        pq.write_table(table, out)
        logger.info("Wrote %s (%d rows)", out, len(part))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LibriSQA annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["sqa"], help="Tasks to extract")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to build")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv"],
        help="Languages to build",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or datasets_dir() / SOURCE
    parquet_dir = args.parquet_dir or output_dir / "parquet"

    df = get_from_original(output_dir / "original")

    samples = build_samples(
        df=df,
        source=SOURCE,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages,
    )

    write_parquets(samples, args.splits, parquet_dir)


if __name__ == "__main__":
    main()
