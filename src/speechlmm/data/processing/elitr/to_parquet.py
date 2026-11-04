"""Convert ELITR meeting transcripts and minutes to SpeechLMM Parquet.

Pair each meeting transcript with its minutes and emit text-summarization
rows for the requested splits and languages. Incomplete meeting directories
are skipped with a warning.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.elitr.to_parquet --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from speechlmm.data.utils import datasets_dir
from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row

logger = logging.getLogger(__name__)


TASKS = ["tsum"]
SOURCE = "elitr"
LICENSE = "cc-by-nc-4.0"


def list_minute_files(root: Path) -> list[Path]:
    entries = root.glob("minutes_GENER*")
    return sorted(entries)


def list_transcript_files(root: Path) -> list[Path]:
    entries = root.glob("transcript_MAN*")
    return sorted(entries)


def list_meeting_ids(root: Path) -> list[Path]:
    entries = root.glob("meeting_*")
    return sorted(entries)


def get_from_original(original_dir: Path, splits: Sequence[str], languages: Sequence[str]) -> dict[str, dict[str, list[dict]]]:
    full_df = {}

    for split in splits:
        full_df[split] = {}
        for language in languages:
            full_df[split][language] = []
            meeting_dir = original_dir / f"elitr-minuting-corpus-{language}" / split
            meetings = list_meeting_ids(meeting_dir)
            for meeting in meetings:
                transcript_files = list_transcript_files(meeting)
                minute_files = list_minute_files(meeting)
                if not transcript_files or not minute_files:
                    logger.warning("Skipping incomplete ELITR meeting: %s", meeting)
                    continue
                with transcript_files[0].open() as f:
                    transcripts = f.readlines()

                with minute_files[0].open() as f:
                    minutes = f.readlines()

                entry = {
                    "split": split,
                    "meeting": meeting.name,
                    "language": language,
                    "transcripts": "".join(transcripts),
                    "minutes": "".join(minutes),
                }
                full_df[split][language].append(entry)

    return full_df


def _word_count(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.split())


def _base_row(
    entry: dict,
    source: str,
    task: str,
    split: str,
    language: str,
) -> dict:
    row = empty_row()
    row.update(
        {
            "sample_id": f"{language}_{entry['meeting']}",
            "task": [task],
            "split": split,
            "source_dataset": source,
            "license": LICENSE,
            "text_input": entry["transcripts"],
            "text_input_normalized": entry["transcripts"],
            "text_output": entry["minutes"],
            "text_output_normalized": entry["minutes"],
            "num_words_output": _word_count(entry["minutes"]),
            "language_text_input": language,
            "language_text_output": language,
        }
    )

    return row


def build_task_rows(
    df: list,
    source: str,
    task: str,
    split: str,
    language: str,
) -> list[dict]:
    return [
        _base_row(
            entry,
            source=source,
            task=task,
            split=split,
            language=language,
        )
        for entry in df
    ]


def build_samples(
    df: dict[str, dict[str, list[dict]]],
    source: str,
    tasks: Sequence[str],
    splits: Sequence[str],
) -> list[dict]:
    samples = []

    for task in tasks:
        for split in splits:
            for language in df[split].keys():
                samples.extend(
                    build_task_rows(
                        source=source,
                        task=task,
                        split=split,
                        language=language,
                        df=df[split][language],
                    )
                )

    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check ELITR annotations path.")
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
    parser = argparse.ArgumentParser(description="Convert ELITR annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["tsum"], help="Tasks to extract")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test", "test2"], help="Splits to build")
    parser.add_argument("--languages", nargs="+", default=["cs", "en"], help="Languages to build")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or datasets_dir() / SOURCE
    parquet_dir = args.parquet_dir or output_dir / "parquet"

    df = get_from_original(output_dir / "original", args.splits, args.languages)

    samples = build_samples(
        df=df,
        source=SOURCE,
        tasks=args.tasks,
        splits=args.splits,
    )

    write_parquets(samples, args.splits, parquet_dir)


if __name__ == "__main__":
    main()
