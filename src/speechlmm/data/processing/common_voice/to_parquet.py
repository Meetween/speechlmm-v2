"""Build Common Voice metadata Parquet for the SpeechLMM schema.

One row per validated clip per task, written to per-(language, split) Parquet:

  asr   audio (clip) -> text_output = sentence

Common Voice ships one directory per locale
(``<cv_root>/<lang>/{train,dev,test,validated,...}.tsv`` + ``clips/``); this reads
the official ``train``/``dev``/``test`` TSVs. Columns are read by *header name*
(``path``, ``sentence``, and the optional ``client_id``/``gender``), so the parser
tolerates the column-order differences across Common Voice releases.

Task instructions are NOT stored in ``text_prompt`` (added later by
``parquet_to_sharegpt``).

Usage:
    PYTHONPATH=src python3 -m speechlmm.data.processing.common_voice.to_parquet \\
        --cv-root <cv-corpus-26.0-.../> --languages en de it
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from speechlmm.data.utils import datasets_dir, DEFAULT_SIGNALS_ROOT
from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row

logger = logging.getLogger(__name__)

SOURCE = "CommonVoice"
LICENSE = "CC0-1.0"
AUDIO_CODEC = "mp3"

# SpeechLMM lang -> Common Voice on-disk locale directory name.
LANGUAGES = ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv")
SPLITS = ("train", "dev", "test")
CV_LOCALE: dict[str, str] = {"sv": "sv-SE"}


def cv_locale(language: str) -> str:
    return CV_LOCALE.get(language, language)


# Clips are symlinked to <dataset>/signals/<lang>/ by prepare.download(); the stored
# audio_input_path is relative and resolved at load time via LLaMA-Factory media_dir.


def _word_count(text: str | None) -> int | None:
    return len(text.split()) if text else None


def audio_input_path(signals_root: str, language: str, clip_path: str) -> str:
    """Relative clip path. ``clip_path`` is the Common Voice ``path`` column (a filename)."""
    return f"{signals_root}/{language}/{clip_path}"


def load_clip_durations(lang_dir: Path) -> dict[str, float]:
    """Map clip filename -> duration (seconds) from ``clip_durations.tsv`` if present."""
    path = lang_dir / "clip_durations.tsv"
    if not path.exists():
        return {}
    durations: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        clip_col = cols.get("clip") or cols.get("path")
        dur_col = cols.get("duration[ms]") or cols.get("duration")
        if not (clip_col and dur_col):
            return {}
        for row in reader:
            try:
                durations[row[clip_col]] = float(row[dur_col]) / 1000.0
            except (TypeError, ValueError):
                continue
    return durations


def build_split_rows(
    tsv_path: Path,
    language: str,
    split: str,
    signals_root: str,
    durations: dict[str, float],
) -> list[dict]:
    rows: list[dict] = []
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        fields = {c.lower(): c for c in (reader.fieldnames or [])}
        path_col = fields.get("path")
        sent_col = fields.get("sentence")
        client_col = fields.get("client_id")
        gender_col = fields.get("gender")
        if not (path_col and sent_col):
            raise ValueError(f"{tsv_path} missing required 'path'/'sentence' columns: {reader.fieldnames}")

        for i, entry in enumerate(reader):
            clip_path = (entry.get(path_col) or "").strip()
            sentence = (entry.get(sent_col) or "").strip()
            if not (clip_path and sentence):
                continue

            row = empty_row()
            row.update(
                {
                    "sample_id": f"{SOURCE}_{language}_{split}_{Path(clip_path).stem}",
                    "task": ["asr"],
                    "split": split,
                    "source_dataset": SOURCE,
                    "license": LICENSE,
                    "audio_input_path": audio_input_path(signals_root, language, clip_path),
                    "audio_input_codec": AUDIO_CODEC,
                    "language_audio_input": language,
                    "text_output": sentence,
                    "text_output_normalized": sentence,
                    "num_words_output": _word_count(sentence),
                    "language_text_output": language,
                }
            )
            if clip_path in durations:
                row["audio_input_duration"] = round(durations[clip_path], 3)
            if client_col and entry.get(client_col):
                row["speaker_id"] = [entry[client_col]]
            if gender_col and (entry.get(gender_col) or "").strip():
                row["speaker_gender"] = [entry[gender_col].strip()]
            rows.append(row)
    return rows


def write_split_parquet(rows: list[dict], language: str, split: str, parquet_dir: Path) -> int:
    if not rows:
        return 0
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = parquet_dir / f"{language}_{split}.parquet"
    table = pa.Table.from_pandas(df.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False)
    pq.write_table(table, out)
    logger.info("Wrote %s (%d rows)", out, len(df))
    return len(df)


def to_parquet_one(
    cv_root: Path,
    parquet_dir: Path,
    language: str,
    split: str,
    signals_root: str,
) -> int:
    lang_dir = Path(cv_root) / cv_locale(language)
    tsv_path = lang_dir / f"{split}.tsv"
    if not tsv_path.exists():
        logger.warning("No TSV for %s/%s (%s), skipping", language, split, tsv_path)
        return 0
    durations = load_clip_durations(lang_dir)
    rows = build_split_rows(tsv_path, language, split, signals_root, durations)
    return write_split_parquet(rows, language, split, parquet_dir)


def build_samples(
    root: Path,
    signals_root: str,
    languages: list[str],
    splits: list[str]
) -> list[dict]:
    rows = []
    for split in splits:
        for language in languages:
            language_dir = root / cv_locale(language)
            tsv_path = language_dir / f"{split}.tsv"
            if not tsv_path.exists():
                logger.warning("No TSV for %s/%s (%s), skipping", language, split, tsv_path)
                continue
            durations = load_clip_durations(language_dir)
            language_rows = build_split_rows(tsv_path, language, split, signals_root, durations)
            rows.extend(language_rows)

    return rows


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check Common Voice annotations path.")

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
    parser = argparse.ArgumentParser(description="Convert Common Voice TSVs to SpeechLMM metadata Parquet.")
    parser.add_argument("--cv-root", type=Path, default=None, help="Extracted cv-corpus dir with <lang>/ subdirs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parquet output dir (default: $DATASETS_DIR/common_voice/parquet)",
    )
    parser.add_argument("--languages", nargs="+", default=["de", "en", "es", "fr", "it", "nl", "pt"], help="Languages to include")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to build")
    parser.add_argument("--tasks", nargs="+", default=["asr"], help="Tasks to build (asr)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    default_dataset_root = datasets_dir() / "common_voice"
    cv_root = args.cv_root or default_dataset_root / "original" / "cv-corpus-26.0-2026-06-12"
    output_dir = args.output_dir or default_dataset_root / "parquet"

    samples = build_samples(
        root=cv_root,
        languages=args.languages,
        splits=args.splits,
        signals_root=DEFAULT_SIGNALS_ROOT,
    )
    write_parquets(samples, args.splits, output_dir)


if __name__ == "__main__":
    main()
