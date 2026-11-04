"""Normalize prebuilt DGT-TM metadata into SpeechLMM Parquet.

Read source Parquet from ``original/`` and emit per-split rows for ASR,
speech translation, or machine translation. Audio paths remain relative to
the dataset root and are probed when an audio task is requested.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.dgt_tm.to_parquet --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS
from speechlmm.data.utils import AUDIO_TASKS, DEFAULT_SIGNALS_ROOT, probe_duration_seconds, datasets_dir

logger = logging.getLogger(__name__)

# audio_input_path is stored relative to the dataset dir; resolved at load via media_dir.

TASKS = ["asr", "mt", "st"]
SOURCE = "dgt_tm"
SAMPLE_PREFIX="EU10_mt"

def create_mt_parquet(original_dir: Path):
    audio_to_row: dict[str, str] = {}
    asr_parquet_dir = original_dir / "asr"

    asr_ori_parquets = list(asr_parquet_dir.rglob(f"*.parquet"))
    for asr_ori_parquet in asr_ori_parquets:
        txt_to_audio: dict[str, str] = {}
        asr_df = pd.read_parquet(asr_ori_parquet)
        for d in asr_df.itertuples(index=True):
            asr_row = d._asdict()
            audio_to_row[asr_row["audio_input_path"]] = asr_row

    logger.info(f"asr entries:{len(audio_to_row.keys())}")

    st_parquet_dir = original_dir / "st"
    mt_parquet_dir = original_dir / "mt"
    dedup_map: dict[str, set[str]] = {}
    # populate parquet starting from dev and test, so that they contain a reasonable amount of entries
    # due to deduplication the latter split could be significantly smaller
    for split in ["dev", "test", "selected_train", "train"]:
        st_ori_parquets = list(st_parquet_dir.rglob(f"{split}*.parquet")) 
        for st_ori_parquet in st_ori_parquets:
            idx = 0
            mt_rows = []
            st_df = pd.read_parquet(st_ori_parquet)
            for d in st_df.itertuples(index=True):
                st_row = d._asdict()
                key = st_row["audio_input_path"]
                if key not in audio_to_row:
                    continue
                asr_row = audio_to_row[key]
                if asr_row["language_text_output"] != st_row["language_audio_input"]:
                    continue

                text_input = asr_row["text_output"]
                text_output = st_row["text_output"]
                if text_input in dedup_map and text_output in dedup_map[text_input]:
                    continue

                mt_row = {   
                    "sample_id": f"{SAMPLE_PREFIX}_{idx}",
                    "text_input": text_input,
                    "language_text_input": asr_row["language_text_output"],
                    "text_output": text_output,
                    "language_text_output": st_row["language_text_output"],
                    "source_dataset": st_row["source_dataset"], 
                    "license": st_row["license"],
                    "task": ['mt'],
                    "split": st_row["split"],
                    "num_words_output": st_row["num_words_output"],
                }
                if text_input not in dedup_map:
                    dedup_map[text_input] = set()
                dedup_map[text_input].add(text_output)

                mt_rows.append(mt_row)
                idx += 1

            if not mt_rows:
                logger.info("No MT entries for %s", st_ori_parquet)
                continue
            mt_parquet_dir.mkdir(parents=True, exist_ok=True)
            mt_df = pd.DataFrame(mt_rows)
            logger.info("mt entries: %s written into %s", len(mt_rows), mt_parquet_dir / st_ori_parquet.name)
            mt_df.to_parquet(mt_parquet_dir / st_ori_parquet.name, index=False)  


def get_from_original(original_dir: Path, splits: Sequence[str], tasks: Sequence[str]) -> dict[str, pd.DataFrame]:
    full_df = {}
    for split in splits:
        df = {}
        for task in tasks:
            actual_split = f"selected_{split}" if split == "train" else split

            original_task_dir = original_dir / task
            ori_parquets = list(original_task_dir.rglob(f"{actual_split}*.parquet"))
            if not ori_parquets:
                raise FileNotFoundError(f"No DGT-TM parquet found for split={actual_split} under {original_dir}")
            df[task] = pd.read_parquet(ori_parquets)
            if task == "mt" and split == "train":
                extra_parquets = list(original_task_dir.rglob(f"{split}*.parquet"))
                extra_parquets = [path for path in extra_parquets if path not in ori_parquets]
                if extra_parquets:
                    df[task] = pd.concat([df[task], pd.read_parquet(extra_parquets)], ignore_index=True)
        full_df[split] = pd.concat([df[x] for x in tasks], ignore_index=True)

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
    output_dir: Path,
    signals_root: str
) -> dict:

    use_audio = task in AUDIO_TASKS
    row = entry._asdict()

    path = row.get("audio_input_path")
    if use_audio and path:
        sl = row.get("language_audio_input")
        if not sl:
            raise ValueError(f"Missing audio input language for sample_id={row.get('sample_id')}")
        row.update({
            "audio_input_path": f'{signals_root}/audio/{sl}/{path}'
        })

    if use_audio and not row.get("audio_input_duration"):
        duration, codec, sample_rate = probe_duration_seconds(output_dir / row["audio_input_path"])
        row.update({
            "audio_input_duration": duration,
            "audio_input_codec": codec,
            "audio_input_sample_rate": sample_rate
        })

    row.update({"source_dataset": source, "split": split, "task": [task]})
    if task == "asr" and row.get("asr_text"):
        row.update({
                "text_output": row["asr_text"],
                "text_output_normalized": row["asr_text"],
                "language_text_output": row.get("language_audio_input"),
        })
    elif task == "mt":
        text_input = row.get("text_input") or row.get("asr_text")
        row.update({
            "audio_input_path": None,
            "audio_input_duration": None,
            "audio_input_codec": None,
            "audio_input_sample_rate": None,
            "text_input": text_input,
            "text_input_normalized": row.get("text_input_normalized") or text_input,
            "language_text_input": row.get("language_text_input") or row.get("language_audio_input"),
        })
    return row


def _entry_tasks(entry) -> list[str]:
    value = getattr(entry, "task", None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    found: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if item is None or isinstance(item, (bytes, bytearray)):
            continue
        if isinstance(item, str):
            found.append(item)
            continue
        try:
            stack.extend(list(item))
        except TypeError:
            found.append(str(item))
    return found


def build_task_rows(
        df: pd.DataFrame,
        source: str,
        task: str,
        split: str,
        output_dir: Path,
        signals_root: str
    ) -> list[dict]:
    rows: list[dict] = []

    for entry in df.itertuples(index=False):
        entry_tasks = _entry_tasks(entry)
        if entry_tasks and task not in entry_tasks:
            continue
        rows.append(
            _base_row(
                entry,
                source=source,
                task=task,
                split=split,
                output_dir=output_dir,
                signals_root=signals_root
            )
        )

    return rows


def build_samples(
    df: dict[str, pd.DataFrame],
    source: str,
    tasks: Sequence[str],
    splits: Sequence[str],
    output_dir: Path,
    signals_root: str
) -> list[dict]:
    samples = []

    for task in tasks:
        for split in splits:
            a = build_task_rows(
                source=source,
                task=task,
                split=split,
                df=df[split],
                output_dir=output_dir,
                signals_root=signals_root
            )

            samples.extend(a)

    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check DGT-TM annotations path.")
    defaults = {
        "speaker_id": [],
        "speaker_gender": [],
        "speaker_age_group": [],
        "metadata": {},
    }
    for column, default in defaults.items():
        if column not in df:
            df[column] = [default.copy() for _ in range(len(df))]
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
    parser = argparse.ArgumentParser(description="Convert DGT-TM annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["asr", "st"], help="Tasks to extract")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to build")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or datasets_dir() / SOURCE
    parquet_dir = args.parquet_dir or output_dir / "parquet"

    df = get_from_original(output_dir / "original", args.splits, args.tasks)

    samples = build_samples(
        df=df,
        source=SOURCE,
        tasks=args.tasks,
        splits=args.splits,
        output_dir=output_dir,
        signals_root=DEFAULT_SIGNALS_ROOT
    )

    write_parquets(samples, args.splits, parquet_dir)


if __name__ == "__main__":
    main()
