"""Normalize prebuilt LipCrops metadata into SpeechLMM Parquet.

Read source Parquet and revised transcripts from ``original/``, select
lipread samples for the requested languages, and emit per-split metadata.
Video paths remain relative to the dataset root.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.lipcrops.to_parquet --help
"""

from __future__ import annotations
import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from speechlmm.data.utils import datasets_dir, DEFAULT_SIGNALS_ROOT
from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row

logger = logging.getLogger(__name__)

TASKS = ("lipread",)
SOURCE = "lipcrops"
LICENSE = "unknown"


def get_parquet_from_original(
    original_dir: Path, splits: Sequence[str]
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in splits:
        parquet_files = sorted(original_dir.rglob(f"{split}*.parquet"))
        if not parquet_files:
            logger.warning(
                "No LipCrops parquet found for split=%s under %s; skipping",
                split,
                original_dir,
            )
            continue
        frames[split] = pd.read_parquet(parquet_files)

    return frames


def get_transcripts_from_original(original_dir: Path) -> dict[str, str]:
    transcript_files = sorted(original_dir.rglob("*.txt"))
    if not transcript_files:
        raise FileNotFoundError(
            f"No revised LipCrops transcripts found under {original_dir}"
        )
    return {
        path.stem: path.read_text(encoding="utf-8").strip() for path in transcript_files
    }


def _word_count(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.split())


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, Sequence) else []


def _base_row(
    entry: dict,
    transcripts: dict[str, str],
    source: str,
    task: str,
    split: str,
    languages: Sequence[str],
    signals_root: str,
    dataset_root: Path
) -> dict | None:
    sample_id = entry.get("sample_id")
    if sample_id not in transcripts:
        return None

    row_tasks = _as_list(entry.get("task"))
    if task not in row_tasks:
        return None

    language = entry.get("language")
    if languages and language not in languages:
        return None

    transcript = " ".join(transcripts[sample_id].split())
    if not transcript:
        return None

    lip_video_path = entry.get("lip_video_path")
    if not lip_video_path or entry.get("has_lip_crop") is False:
        return None

    video_path = dataset_root / DEFAULT_SIGNALS_ROOT / str(lip_video_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        logger.warning(
            "Missing/empty lip_video for sample_id=%s, path=%s",
            sample_id,
            video_path,
        )
        return None

    row = empty_row()
    row.update(
        {
            "sample_id": sample_id,
            "task": [task],
            "split": split,
            "source_dataset": source,
            "license": entry.get("license") or LICENSE,
            "video_input_path": f"{signals_root}/{lip_video_path}",
            "video_input_duration": entry.get("face_video_duration"),
            "video_input_width": entry.get("lip_video_width"),
            "video_input_height": entry.get("lip_video_height"),
            "video_input_fps": entry.get("video_fps"),
            "language_video_input": language,
            "text_output": transcript,
            "text_output_normalized": _normalize_text(transcript),
            "num_words_output": _word_count(transcript),
            "language_text_output": language,
            "has_face": entry.get("has_face"),
            "num_visible_speakers": entry.get("num_speakers"),
            "speaker_id": _as_list(entry.get("speaker_id")),
            "num_speakers": entry.get("num_speakers"),
            "has_overlapping_speech": entry.get("has_overlapping_speech"),
            "repetition_score": entry.get("repetition_score"),
            "alignment_score": entry.get("alignment_score"),
            "wer_estimate": entry.get("wer_estimate"),
            "metadata": {},
        }
    )
    return row


def build_task_rows(
    df: pd.DataFrame,
    transcripts: dict[str, str],
    source: str,
    task: str,
    split: str,
    languages: Sequence[str],
    signals_root: str,
    dataset_root: Path
) -> list[dict]:
    rows: list[dict] = []
    for entry in df.to_dict(orient="records"):
        row = _base_row(
            entry,
            transcripts=transcripts,
            source=source,
            task=task,
            split=split,
            languages=languages,
            signals_root=signals_root,
            dataset_root=dataset_root
        )
        if row is not None:
            rows.append(row)

    skipped = len(df) - len(rows)
    if skipped:
        logger.info("Split %s: kept %d, skipped %d", split, len(rows), skipped)

    return rows


def build_samples(
    df: dict[str, pd.DataFrame],
    transcripts: dict[str, str],
    source: str,
    tasks: Sequence[str],
    splits: Sequence[str],
    languages: Sequence[str],
    signals_root: str,
    dataset_root: Path
) -> list[dict]:
    samples: list[dict] = []
    for task in tasks:
        for split in splits:
            if split not in df:
                continue
            samples.extend(
                build_task_rows(
                    source=source,
                    task=task,
                    split=split,
                    languages=languages,
                    df=df[split],
                    transcripts=transcripts,
                    signals_root=signals_root,
                    dataset_root=dataset_root
                )
            )

    return samples


def write_parquets(
    samples: list[dict], splits: Sequence[str], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError(
            "No samples produced; check LipCrops annotations and filters."
        )
    for split in splits:
        part = df[df["split"] == split]
        if part.empty:
            logger.warning("No samples for split=%s", split)
            continue

        out = output_dir / f"{split}.parquet"
        # Cast to the shared schema so every corpus emits identical column types.
        table = pa.Table.from_pandas(
            part.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False
        )
        pq.write_table(table, out)
        logger.info("Wrote %s (%d rows)", out, len(part))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert LipCrops annotations to SpeechLMM metadata Parquet."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument(
        "--tasks", nargs="+", default=["lipread"], help="Tasks to extract"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train"], help="Splits to build"
    )
    parser.add_argument(
        "--languages", nargs="+", default=["en"], help="Languages to build"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or datasets_dir() / SOURCE
    parquet_dir = args.parquet_dir or output_dir / "parquet"

    df = get_parquet_from_original(output_dir / "original" / "parquet", args.splits)
    transcripts = get_transcripts_from_original(
        output_dir / "original" / "lipcrops_transcripts"
    )
    samples = build_samples(
        df=df,
        transcripts=transcripts,
        source=SOURCE,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages,
        signals_root=DEFAULT_SIGNALS_ROOT,
        dataset_root=output_dir,
    )

    write_parquets(samples, args.splits, parquet_dir)


if __name__ == "__main__":
    main()
