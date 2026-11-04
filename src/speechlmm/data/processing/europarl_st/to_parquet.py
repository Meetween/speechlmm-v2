"""Build Europarl-ST metadata Parquet for the SpeechLMM schema.

One row per segment per task, written to per-(direction, split) Parquet:

  st   audio (source-language speech segment) -> text_output = target translation
  mt   text_input = source transcript          -> text_output = target translation

Directions are ``{src}/{tgt}`` over the languages in ``LANGUAGES``. Task
instructions are NOT stored in ``text_prompt`` (added later by ``parquet_to_sharegpt``).

Europarl-ST segments are time spans inside speech-level ``.m4a`` files
(``{src}/audios/{audio_id}.m4a``). The per-segment ST clips are cut separately by
``cut_audio`` into ``signals/`` at the exact path returned by :func:`clip_relpath`.

Usage:
    PYTHONPATH=src python3 -m speechlmm.data.processing.europarl_st.to_parquet
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

SOURCE = "Europarl-ST"
LICENSE = "CC-BY-NC-4.0"

# Cut ST clips: 16 kHz mono, stored relative to the dataset dir (resolved via media_dir).
CLIP_CODEC = "flac"
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


def _word_count(text: str | None) -> int | None:
    return len(text.split()) if text else None


def _audio_date(audio_id: str) -> str:
    """Session date field of an Europarl audio id (e.g. 'en.20080924.31.3-242' -> '20080924')."""
    parts = audio_id.split(".")
    return parts[1] if len(parts) > 1 else "misc"


def clip_subpath(src: str, audio_id: str, start: float, end: float) -> str:
    """Path of the per-segment ST clip *relative to the signals root*.

    Sharded by source language and session date to keep directories small; the
    source language is part of the path because the same audio id can occur under
    different source languages with different audio content.
    """
    return f"{src}/{_audio_date(audio_id)}/{audio_id}_{start:.2f}-{end:.2f}.{CLIP_CODEC}"


def clip_relpath(signals_root: str, src: str, audio_id: str, start: float, end: float) -> str:
    """Clip path relative to the dataset dir. Shared by to_parquet and cut_audio."""
    return f"{signals_root}/{clip_subpath(src, audio_id, start, end)}"


def source_audio_path(root: Path, src: str, audio_id: str) -> Path:
    return root / src / "audios" / f"{audio_id}.m4a"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_segments(seg_lst_path: Path) -> list[tuple[str, float, float]]:
    """Parse ``segments.lst`` lines '<audio_id> <start> <end>' into typed tuples."""
    segments: list[tuple[str, float, float]] = []
    for line in _read_lines(seg_lst_path):
        parts = line.split()
        if len(parts) != 3:
            continue
        audio_id, start, end = parts
        segments.append((audio_id, float(start), float(end)))
    return segments


def build_direction_rows(
    src: str,
    tgt: str,
    split: str,
    direction_dir: Path,
    signals_root: str,
    tasks: Sequence[str],
) -> list[dict]:
    """Build ST/MT rows for one ``{src}/{tgt}/{split}`` directory."""
    seg_lst = direction_dir / "segments.lst"
    src_txt_path = direction_dir / f"segments.{src}"
    tgt_txt_path = direction_dir / f"segments.{tgt}"
    if not (seg_lst.exists() and src_txt_path.exists() and tgt_txt_path.exists()):
        logger.warning("Missing segment files in %s, skipping", direction_dir)
        return []

    segments = parse_segments(seg_lst)
    src_lines = _read_lines(src_txt_path)
    tgt_lines = _read_lines(tgt_txt_path)
    if not (len(segments) == len(src_lines) == len(tgt_lines)):
        logger.warning(
            "Length mismatch in %s (lst=%d src=%d tgt=%d), truncating to min",
            direction_dir,
            len(segments),
            len(src_lines),
            len(tgt_lines),
        )
    n = min(len(segments), len(src_lines), len(tgt_lines))

    rows: list[dict] = []
    for i in range(n):
        audio_id, start, end = segments[i]
        transcript = src_lines[i].strip()
        translation = tgt_lines[i].strip()
        if not translation:
            continue
        base_id = f"{SOURCE}_{src}_{tgt}_{split}_{i}"

        if "st" in tasks:
            row = empty_row()
            row.update(
                {
                    "sample_id": f"{base_id}_st",
                    "task": ["st"],
                    "split": split,
                    "source_dataset": SOURCE,
                    "license": LICENSE,
                    "audio_input_path": clip_relpath(signals_root, src, audio_id, start, end),
                    "audio_input_duration": round(end - start, 3),
                    "audio_input_sample_rate": SAMPLE_RATE,
                    "audio_input_num_channels": NUM_CHANNELS,
                    "audio_input_codec": CLIP_CODEC,
                    "language_audio_input": src,
                    "text_output": translation,
                    "text_output_normalized": translation,
                    "num_words_output": _word_count(translation),
                    "language_text_output": tgt,
                }
            )
            rows.append(row)

        if "mt" in tasks and transcript:
            row = empty_row()
            row.update(
                {
                    "sample_id": f"{base_id}_mt",
                    "task": ["mt"],
                    "split": split,
                    "source_dataset": SOURCE,
                    "license": LICENSE,
                    "text_input": transcript,
                    "text_input_normalized": transcript,
                    "num_words_input": _word_count(transcript),
                    "language_text_input": src,
                    "text_output": translation,
                    "text_output_normalized": translation,
                    "num_words_output": _word_count(translation),
                    "language_text_output": tgt,
                }
            )
            rows.append(row)

    return rows


def write_direction_parquet(rows: list[dict], split: str, parquet_dir: Path) -> int:
    if not rows:
        return 0
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = parquet_dir / f"{split}.parquet"
    table = pa.Table.from_pandas(df.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False)
    pq.write_table(table, out)
    logger.info("Wrote %s (%d rows)", out, len(df))
    return len(df)


def discover_directions(root: Path, languages: Sequence[str]) -> list[tuple[str, str]]:
    """All ordered ``(src, tgt)`` pairs among ``languages`` that exist on disk."""
    directions: list[tuple[str, str]] = []
    for src in languages:
        for tgt in languages:
            if src == tgt:
                continue
            if (Path(root) / src / tgt).is_dir():
                directions.append((src, tgt))
    return directions


def build_samples(
    root: Path,
    signals_root: str,
    directions: Sequence[tuple[str, str]],
    splits: Sequence[str],
    tasks: Sequence[str],
) -> list:
    rows: list[dict] = []
    for split in splits:
        for src, tgt in directions:
            direction_dir = root / src / tgt / split
            if not direction_dir.is_dir():
                continue
            direction_rows = build_direction_rows(src, tgt, split, direction_dir, signals_root, tasks)
            rows.extend(direction_rows)

    return rows


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check Europarl-ST annotations path.")

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
    parser = argparse.ArgumentParser(description="Convert Europarl-ST annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--europarl-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--languages", nargs="+", default=["de", "en", "es", "fr", "it", "nl", "pt"])
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--tasks", nargs="+", default=["st", "mt"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    default_dataset_root = args.dataset_root or datasets_dir() / "europarl_st"
    europarl_root = args.europarl_root or default_dataset_root / "original" / "v1.1"
    output_dir = args.output_dir or default_dataset_root / "parquet"
    directions = discover_directions(europarl_root, args.languages)
    samples = build_samples(
        root=europarl_root,
        signals_root=DEFAULT_SIGNALS_ROOT,
        directions=directions,
        splits=args.splits,
        tasks=args.tasks,
    )
    write_parquets(samples, args.splits, output_dir)


if __name__ == "__main__":
    main()
