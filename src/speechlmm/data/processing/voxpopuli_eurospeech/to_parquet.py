"""Normalize prebuilt VoxPopuli+EuroSpeech metadata into SpeechLMM Parquet.

Read source Parquet from ``original/`` and emit per-split ASR rows.
Audio paths remain relative to the dataset root.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.voxpopuli_eurospeech.to_parquet --help
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
from speechlmm.data.utils import AUDIO_TASKS, probe_duration_seconds, datasets_dir

logger = logging.getLogger(__name__)

# audio_input_path is stored relative to the dataset dir; resolved at load via media_dir.
DEFAULT_SIGNALS_ROOT = "signals"

TASKS = ["asr"]
SOURCE = "voxpopuli_eurospeech"

def get_from_original(original_dir: Path, splits: Sequence[str]) -> dict[str, pd.DataFrame]:
    full_df = {}
    for split in splits:
        full_df[split] = pd.read_parquet(original_dir / f'{split}.parquet')

    return full_df

def build_samples(
    df: dict[str, pd.DataFrame],
    tasks: Sequence[str],
    splits: Sequence[str],
    signals_root: str
) -> list[dict]:
    samples = []

    for split in splits:
        logger.info(df[split].iloc[0])
        df[split]["audio_input_path"] = df[split]["audio_input_path"].apply(lambda x: f"{signals_root}/{str(x)}" if pd.notna(x) else x)
        df[split]["task"] = df[split]["task"].apply(lambda x: [x] if isinstance(x, str) else x)
        df[split]["language_text_output"] = df[split]["language_audio_input"]
        df[split]["sample_id"] = df[split]["sample_id"].apply(lambda x: str(x))

        samples.extend(df[split].to_dict(orient="records"))
        logger.info(samples[-1])
    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check VoxPopuli+EuroSpeech annotations path.")
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


