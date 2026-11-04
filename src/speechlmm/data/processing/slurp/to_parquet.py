"""Convert Hugging Face SLURP utterances to SpeechLMM Parquet.

Map each utterance to its intent and transcript, write the decoded audio
under the dataset root, and store relative media paths plus audio metadata
for spoken-language-understanding training.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.slurp.to_parquet --help
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from collections.abc import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, get_dataset_split_names, load_dataset

from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row
from speechlmm.data.utils import probe_duration_seconds, datasets_dir, AUDIO_TASKS

logger = logging.getLogger(__name__)


# audio_input_path is stored relative to the dataset dir; resolved at load via media_dir.
DEFAULT_SIGNALS_ROOT = "signals"

TASKS = ["slu"]
HF_NAME = "qmeeus/slurp"
SOURCE = "slurp"
LICENSE = "cc-by-nc-4.0"
AUDIO_CODEC = "flac"


def get_from_HF(splits: Sequence[str], languages: Sequence[str]) -> dict:
    available_splits = get_dataset_split_names(HF_NAME)
    full_ds = {}
    for split in splits:
        full_ds[split] = {}
        for language in languages:
            if split not in available_splits:
                continue
            full_ds[split][language] = load_dataset(HF_NAME, split=split)
            # do not decode audio, just store the path to the audio file
            full_ds[split][language] = full_ds[split][language].cast_column("audio", Audio(decode=False))

    return full_ds


def _save_audio(audio_file: Path, audio_bytes: bytes | None, source_path: str) -> None:
    if audio_file.exists():
        return
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    if audio_bytes is not None:
        audio_file.write_bytes(audio_bytes)
    elif Path(source_path).is_file():
        shutil.copyfile(source_path, audio_file)
    else:
        raise FileNotFoundError(f"SLURP audio is unavailable: {source_path}")


def _word_count(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.split())


def audio_input_path(split: str, language: str, audio_id: str, signals_root: str) -> Path:
    return Path(signals_root) / "audio" / language / split / f"{audio_id}"


def _intent_name(value: object, intents: list[str]) -> str:
    if isinstance(value, str) and value in intents:
        return value
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SLURP intent label: {value!r}") from exc
    if index < 0 or index >= len(intents):
        raise ValueError(f"SLURP intent index out of range: {index}")
    return intents[index]


def _base_row(
    entry,
    task: str,
    split: str,
    language: str,
    intents: list[str],
    signals_root: str,
    output_dir: Path,
) -> dict:
    use_audio = task in AUDIO_TASKS
    audio_path = entry["audio"]["path"]
    slurp_id = entry["slurp_id"]
    audio_name = Path(audio_path).name
    sample_id = f"{slurp_id}_{Path(audio_name).stem.removeprefix('audio-')}"
    intent = _intent_name(entry["intent"], intents)
    row = empty_row()
    metadata = {"intents": intents}
    row.update(
        {
            "sample_id": sample_id,
            "task": [task],
            "split": split,
            "source_dataset": SOURCE,
            "license": LICENSE,
            "text_output": intent,
            "text_output_normalized": intent,
            "num_words_output": _word_count(intent),
            "language_text_input": language,
            "language_text_output": language,
            "metadata": metadata,
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

    return row


def build_task_rows(
    df: pd.DataFrame,
    task: str,
    split: str,
    language: str,
    intents: list[str],
    signals_root: str,
    output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    rows = [
        _base_row(entry, task, split, language, intents=intents, signals_root=signals_root, output_dir=output_dir) for entry in df
    ]

    return rows


def build_samples(
    df: dict,
    tasks: Sequence[str],
    splits: Sequence[str],
    signals_root: str,
    output_dir: Path,
) -> list[dict]:
    samples = []
    for task in tasks:
        metadata = None

        for split in splits:
            for language in df[split].keys():
                if metadata is None:
                    metadata = get_metadata(df[split][language])

                samples.extend(
                    build_task_rows(
                        task=task,
                        split=split,
                        language=language,
                        intents=metadata["intents"],
                        df=df[split][language],
                        signals_root=signals_root,
                        output_dir=output_dir,
                    )
                )

    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:

    for split in splits:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(samples)
        if df.empty:
            raise RuntimeError("No samples produced; check SLURP annotations path.")

        part = df[df["split"] == split]
        if part.empty:
            logger.warning("No samples for split=%s", split)
            continue

        out = output_dir / f"{split}.parquet"
        # Cast to the shared schema so every corpus emits identical column types.
        table = pa.Table.from_pandas(part.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False)
        pq.write_table(table, out)
        logger.info("Wrote %s (%d rows)", out, len(part))


def get_metadata(df) -> dict[str, list[str]]:
    names = getattr(df.features["intent"], "names", None)
    if not names:
        raise ValueError("SLURP intent labels are missing from the dataset features")
    return {"intents": list(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SLURP annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["slu"], help="Tasks to extract")
    parser.add_argument("--splits", nargs="+", default=["train", "devel", "test", "train_synthetic"], help="Splits to build")
    parser.add_argument("--languages", nargs="+", default=["en"], help="Languages to build")
    parser.add_argument(
        "--signals-root",
        type=str,
        default=DEFAULT_SIGNALS_ROOT,
        help="Prefix for the stored audio_input_path; relative ('signals') by default, resolved via LLaMA-Factory media_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or datasets_dir() / SOURCE
    parquet_dir = args.parquet_dir or output_dir / "parquet"

    df = get_from_HF(args.splits, args.languages)

    samples = build_samples(
        df,
        tasks=args.tasks,
        splits=args.splits,
        signals_root=args.signals_root,
        output_dir=output_dir,
    )

    write_parquets(samples, args.splits, parquet_dir)


if __name__ == "__main__":
    main()
