"""Convert Hugging Face IWSLT26 utterances to SpeechLMM Parquet.

Map each utterance to its intent and transcript, write the decoded audio
under the dataset root, and store relative media paths plus audio metadata
for spoken-language-understanding training.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.iwslt26.to_parquet --help
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
from datasets import Audio, load_dataset

from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row
from speechlmm.data.utils import datasets_dir, AUDIO_TASKS, DEFAULT_SIGNALS_ROOT

logger = logging.getLogger(__name__)


# audio_input_path is stored relative to the dataset dir; resolved at load via media_dir.

TASKS = ["asr", "st", "sqa", "ssum", "achap", "instruct"]
HF_NAME = "YapayNet/iwslt2026-if-augmented"
SOURCE = "iwslt26"
LICENSE = "upstream"
# Hugging Face uses "validation"; the SpeechLMM pipeline uses "dev".
HF_SPLIT_ALIASES = {
    "train": "train",
    "dev": "validation",
    "devel": "validation",
    "validation": "validation",
    "test": "test",
}


def resolve_hf_split(split: str) -> str:
    """Map a SpeechLMM split name onto the Hugging Face IWSLT26 split key."""
    if split not in HF_SPLIT_ALIASES:
        raise ValueError(
            f"Unsupported IWSLT26 split {split!r}; expected one of {sorted(HF_SPLIT_ALIASES)}"
        )
    return HF_SPLIT_ALIASES[split]

import re
SQA_PROMPT_QUESTION_PATTERN = re.compile(r"^(?P<prompt>.*?(?:[:?]))\s*(?P<question>.+)$")


def get_from_HF(tasks: Sequence[str]) -> dict:
    full_ds = {}
    for task in tasks:
        full_ds[task] = load_dataset(HF_NAME, task, trust_remote_code=True)
        # do not decode audio, just store the path to the audio file
        full_ds[task] = full_ds[task].cast_column("audio", Audio(decode=False))
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
        raise FileNotFoundError(f"IWSLT26 audio is unavailable: {source_path}")


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
        raise ValueError(f"Invalid IWSLT26 intent label: {value!r}") from exc
    if index < 0 or index >= len(intents):
        raise ValueError(f"IWSLT26 intent index out of range: {index}")
    return intents[index]


def _get_prompt_and_question(text: str) -> tuple[str, str]:
    m = SQA_PROMPT_QUESTION_PATTERN.match(text)
    if m:
        return m.group("prompt").strip(), m.group("question").strip()
    return "", text

def _base_row(
    entry,
    task: str,
    split: str,
    signals_root: str,
    output_dir: Path,
) -> dict:
    use_audio = task in AUDIO_TASKS
    audio_path = entry["audio"]["path"]
    audio_name = Path(audio_path).name

    messages = entry["messages"]
    
    prompt, question, answer = "", "", ""
    for message in messages:
        if "role" in message:
            if message["role"] == "user":
                prompt, question = _get_prompt_and_question(message["content"])
            elif message["role"] == "assistant":
                answer = message["content"]

    language_audio_input = entry["audio_lang"]
    row = empty_row()
    row.update(
        {
            "sample_id": entry["id"],
            "task": [entry["task"]],
            "split": split,
            "audio_id": entry["audio_id"],
            "source_dataset": entry["source_dataset"],
            "license": LICENSE,
            "text_output": answer,
            "text_output_normalized": answer,
            "text_prompt": question,
            "language_text_prompt": entry["prompt_lang"],
            "num_words_output": _word_count(answer),
            "language_text_output": entry["answer_lang"]
        }
    )

    if use_audio:
        rel_audio_path = audio_input_path(split, language_audio_input, audio_name, signals_root)
        abs_audio_path = output_dir / rel_audio_path
        _save_audio(abs_audio_path, entry["audio"].get("bytes"), audio_path)
        row.update(
            {
                "audio_input_path": rel_audio_path.as_posix(),
                "audio_input_duration": entry["audio_duration_s"],
                "audio_input_codec": entry["audio_format"],
                "audio_input_sample_rate": entry["audio_sample_rate"],
                "audio_input_num_channels": entry["audio_channels"],
                "language_audio_input": language_audio_input,
            }
        )

    return row


def build_task_rows(
    df: pd.DataFrame,
    task: str,
    split: str,
    signals_root: str,
    output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []

    rows = [
        _base_row(entry, task, split, signals_root=signals_root, output_dir=output_dir) for entry in df
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
        for split in splits:
            samples.extend(
                build_task_rows(
                    task=task,
                    split=split,
                    df=df[task][resolve_hf_split(split)],
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
            raise RuntimeError("No samples produced; check IWSLT26 annotations path.")

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
        raise ValueError("IWSLT26 intent labels are missing from the dataset features")
    return {"intents": list(names)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IWSLT26 annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["asr", "sqa"], help="Tasks to extract")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"], help="Splits to build")
    parser.add_argument(
        "--signals-root",
        type=str,
        default=DEFAULT_SIGNALS_ROOT,
        help="Prefix for the stored audio_input_path; relative ('signals') by default, resolved via LLaMA-Factory media_dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or datasets_dir() / "iwslt26"
    parquet_dir = args.parquet_dir or output_dir / "parquet"

    df = get_from_HF(tasks=args.tasks)

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
