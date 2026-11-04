"""Build SpeechLMM ASR metadata from FLEURS TSV annotations.

Each output row maps one utterance to its transcript and relative audio path.
The preparation pipeline downloads the referenced audio and enriches codec,
sample-rate, and duration metadata afterward.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.fleurs.to_parquet --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from speechlmm.data.corpus_preparer import datasets_dir
from speechlmm.data.processing.fleurs.download import FLEURS_MIRROR, download_tsv
from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row
from speechlmm.data.utils import probe_duration_seconds, AUDIO_TASKS, DEFAULT_SIGNALS_ROOT

logger = logging.getLogger(__name__)


# audio_input_path is stored relative to the dataset dir; resolved at load via media_dir.
# Audio unpacks under <dataset_root>/signals/audio/<language>/<split>/, so the prefix is "signals".

LICENSE = "CC-BY-4.0"
SOURCE = "FLEURS"
AUDIO_CODEC = "wav"


def _word_count(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.split())


def audio_input_path(split: str, language: str, audio_id: str, signals_root: str) -> Path:
    return Path(signals_root) / "audio" / language / split / f"{audio_id}.{AUDIO_CODEC}"


def _base_row(id: str, task: str, split: str, language: str, audio_id: str, signals_root: str, dataset_root: Path) -> dict:
    use_audio = task in AUDIO_TASKS

    row = empty_row()
    row.update(
        {"sample_id": f"fleurs_{split}_{id}_{task}", "task": [task], "split": split, "source_dataset": SOURCE, "license": LICENSE}
    )

    if use_audio:
        path = audio_input_path(split, language, audio_id, signals_root)
        duration, codec, sample_rate = probe_duration_seconds(dataset_root / path)
        row.update(
            {
                "audio_input_path": str(path),
                "language_audio_input": language,
                "audio_input_duration": duration,
                "audio_input_codec": codec,
                "audio_input_sample_rate": sample_rate,
            }
        )

    return row


def build_task_rows(task: str, split: str, language: str, df: pd.DataFrame, signals_root: str, dataset_root: Path) -> list[dict]:
    rows: list[dict] = []

    id_col = df.iloc[:, 0]
    audio_file_col = df.iloc[:, 1]
    txt_col = df.iloc[:, 2]
    speaker_col = df.iloc[:, 6]

    for id, audio_path, transcript_plain, speaker in zip(id_col, audio_file_col, txt_col, speaker_col):
        audio_id = audio_path.removesuffix(f".{AUDIO_CODEC}")
        row = _base_row(f"{id}_{audio_id}", task, split, language, audio_id, signals_root, dataset_root)
        row.update(
            {
                "text_output": transcript_plain,
                "text_output_normalized": transcript_plain,
                "num_words_output": _word_count(transcript_plain),
                "language_text_output": language,
            }
        )

        if speaker:
            row.update({"speaker_id": [speaker]})
        rows.append(row)

    return rows


def build_samples(
    tsv_dir: Path,
    tasks: Sequence[str],
    splits: Sequence[str],
    languages: Sequence[str],
    dataset_root: Path,
    signals_root: str
) -> list[dict]:
    samples: list[dict] = []
    for split in splits:
        for language in languages:
            tsv_path = tsv_dir / language / f"{split}.tsv"
            if not tsv_path.exists():
                logger.warning("FLEURS TSV not found: %s", tsv_path)
                continue

            df = pd.read_csv(tsv_path, sep="\t", dtype=str, header=None)

            for task in tasks:
                samples.extend(
                    build_task_rows(
                        task=task,
                        split=split,
                        language=language,
                        df=df,
                        signals_root=signals_root,
                        dataset_root=dataset_root,
                    )
                )

    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check FLEURS annotations path.")

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


def enrich_audio_metadata(
    parquet_dir: Path,
    dataset_root: Path,
    splits: Sequence[str],
    workers: int,
) -> None:
    """Fill audio metadata after FLEURS archives have been downloaded."""
    for split in splits:
        parquet_path = parquet_dir / f"{split}.parquet"
        if not parquet_path.is_file():
            continue

        frame = pq.read_table(parquet_path).to_pandas()
        relative_paths = sorted({path for path in frame["audio_input_path"] if isinstance(path, str) and path})
        with ThreadPoolExecutor(max_workers=workers) as executor:
            probed = dict(
                zip(
                    relative_paths,
                    executor.map(
                        probe_duration_seconds,
                        (dataset_root / path for path in relative_paths),
                    ),
                )
            )

        for index, relative_path in frame["audio_input_path"].items():
            if relative_path not in probed:
                continue
            duration, codec, sample_rate = probed[relative_path]
            frame.at[index, "audio_input_duration"] = duration
            frame.at[index, "audio_input_codec"] = codec
            frame.at[index, "audio_input_sample_rate"] = sample_rate

        table = pa.Table.from_pandas(frame.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False)
        temporary_path = parquet_path.with_suffix(".tmp.parquet")
        pq.write_table(table, temporary_path)
        temporary_path.replace(parquet_path)
        logger.info("Updated audio metadata in %s", parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FLEURS annotations to SpeechLMM metadata Parquet.")
    parser.add_argument(
        "--fleurs-root",
        type=Path,
        default=None,
        help="TSV cache dir (default: $DATASETS_DIR/speechlmm-v2/fleurs/tsv)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to build")
    parser.add_argument("--languages", nargs="+", default=["en"], help="Languages to build")
    parser.add_argument("--tasks", nargs="+", default=["asr"], help="Tasks to build")
    parser.add_argument(
        "--signals-root",
        default=DEFAULT_SIGNALS_ROOT,
        help="Prefix for the stored audio_input_path; relative ('signals') by default, resolved via LLaMA-Factory media_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset_root = datasets_dir() / "fleurs"
    tsv_dir = args.fleurs_root or dataset_root / "tsv"
    output_dir = args.output_dir or dataset_root / "parquet"

    download_tsv(tsv_dir, args.splits, args.languages, base_url=FLEURS_MIRROR)

    samples = build_samples(
        tsv_dir,
        args.tasks,
        args.splits,
        args.languages,
        dataset_root,
        signals_root=args.signals_root,
    )
    write_parquets(samples, args.splits, output_dir)


if __name__ == "__main__":
    main()
