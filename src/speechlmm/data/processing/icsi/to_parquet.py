"""Convert ICSI meetings to SpeechLMM summarization Parquet.

Load segmented data from source Parquet or unsegmented data from Arrow,
build speech- and text-summarization rows, and materialize audio under the
dataset root.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.icsi.to_parquet --help
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, load_from_disk

from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row
from speechlmm.data.utils import probe_duration_seconds, datasets_dir, AUDIO_TASKS

logger = logging.getLogger(__name__)

# audio_input_path is stored relative to the dataset dir; resolved at load via media_dir.
DEFAULT_SIGNALS_ROOT = "signals"

TASKS = ["tsum", "ssum"]
SOURCE = "icsi_unsegmented"
LICENSE = "cc-by-4.0"
AUDIO_CODEC = "wav"


def get_from_arrow(arrow_dir, splits) -> dict:
    full_df = {}

    for split in splits:
        obj = load_from_disk(arrow_dir / split)
        full_df[split] = obj.to_pandas()

    return full_df


def get_from_parquet(parquet_dir, splits) -> dict:
    full_df = {}

    for split in splits:
        ori_parquet_file = parquet_dir / f"{split}.parquet"
        dataset = load_dataset("parquet", data_files=ori_parquet_file.as_posix(), split="train")
        full_df[split] = dataset.to_pandas()

    return full_df


def _save_audio(audio_file: Path, audio_bytes: bytes | None, source_path: str) -> None:
    if audio_file.exists():
        return
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    if audio_bytes is not None:
        audio_file.write_bytes(audio_bytes)
    elif Path(source_path).is_file():
        shutil.copyfile(source_path, audio_file)
    else:
        raise FileNotFoundError(f"ICSI audio is unavailable: {source_path}")


def _word_count(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.split())


def audio_input_path(split: str, language: str, audio_id: str, signals_root: str) -> Path:
    return Path(signals_root) / "audio" / language / split / f"{audio_id}"


def _base_row(
    entry,
    source: str,
    task: str,
    split: str,
    language: str,
    signals_root: str,
    output_dir: Path,
) -> dict:
    use_audio = task in AUDIO_TASKS
    audio_path = entry["audio"]["path"]
    audio_name = Path(audio_path).name
    sample_id = Path(audio_name).stem
    row = empty_row()
    row.update(
        {
            "sample_id": sample_id,
            "task": [task],
            "split": split,
            "source_dataset": source,
            "license": LICENSE,
            "text_output": entry["summary"],
            "text_output_normalized": entry["summary"],
            "num_words_output": _word_count(entry["summary"]),
            "language_text_input": language,
            "language_text_output": language,
        }
    )
    if use_audio:
        rel_audio_path = audio_input_path(split, language, audio_name, signals_root)
        abs_audio_path = output_dir / rel_audio_path
        _save_audio(abs_audio_path, entry["audio"].get("bytes"), audio_path)
        duration, _, sample_rate = probe_duration_seconds(abs_audio_path)
        row.update(
            {
                "audio_input_path": rel_audio_path.as_posix(),
                "language_audio_input": language,
                "audio_input_duration": duration,
                "audio_input_codec": AUDIO_CODEC,
                "audio_input_sample_rate": sample_rate,
            }
        )
    else:
        row.update({"text_input": entry["transcript_without_names"]})

    return row


def build_task_rows(
    df: pd.DataFrame,
    source: str,
    task: str,
    split: str,
    language: str,
    signals_root: str,
    output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    rows = [
        _base_row(
            entry, source=source, task=task, split=split, language=language, signals_root=signals_root, output_dir=output_dir
        )
        for _, entry in df.iterrows()
    ]

    return rows


def build_samples(
    df: dict[str, pd.DataFrame],
    source: str,
    tasks: Sequence[str],
    splits: Sequence[str],
    languages: Sequence[str],
    signals_root: str,
    output_dir: Path,
) -> list[dict]:

    samples = []
    for split in splits:
        for task in tasks:
            for language in languages:
                samples.extend(
                    build_task_rows(
                        df=df[split],
                        source=source,
                        task=task,
                        split=split,
                        language=language,
                        signals_root=signals_root,
                        output_dir=output_dir,
                    )
                )

    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check ICSI annotations path.")

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
    parser = argparse.ArgumentParser(description="Convert ICSI annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--source", choices=["icsi_segmented", "icsi_unsegmented"], default="icsi_segmented")
    parser.add_argument("--tasks", nargs="+", default=["tsum", "ssum"], help="Tasks to extract")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Splits to build")
    parser.add_argument("--languages", nargs="+", default=["en"], help="Languages to build")
    parser.add_argument(
        "--signals-root",
        type=str,
        default=DEFAULT_SIGNALS_ROOT,
        help="Prefix for the stored audio_input_path; relative ('signals') by default, resolved via LLaMA-Factory media_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or datasets_dir() / args.source
    parquet_dir = args.parquet_dir or output_dir / "parquet"

    if args.source == "icsi_unsegmented":
        df = get_from_arrow(output_dir / "original" / "arrow" , args.splits)
    else:
        df = get_from_parquet(output_dir / "original" / "parquet", args.splits)

    samples = build_samples(
        df,
        source=args.source,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages,
        signals_root=args.signals_root,
        output_dir=output_dir,
    )

    write_parquets(samples, args.splits, parquet_dir)


if __name__ == "__main__":
    main()
