"""Build YTSeg metadata Parquet for the SpeechLMM schema.

Loads the YTSeg ``text`` config from the Hugging Face Hub and emits one row per
video per task into per-split Parquet:

  asr   audio -> text_output = reference transcript
  achap audio -> text_output = timestamped chapter markers ("MM:SS - title")

The audio itself is fetched separately by ``download`` into ``signals/<video_id>.mp3``.
Official YTSeg splits are used as-is; only the split *names* are normalised
(HF ``validation`` -> our ``val``).

Usage:
    PYTHONPATH=src python3 -m speechlmm.data.processing.ytseg.to_parquet
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from speechlmm.data.utils import datasets_dir
from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AUDIO_CODEC = "mp3"

# Our split name -> Hugging Face split name.
HF_SPLITS = {"train": "train", "val": "validation", "test": "test"}

CHAPTER_HEADER_RE = re.compile(r"\[CSTART\]\s*(\d{1,2}:\d{2}:\d{2})\s*-\s*(.+?)\s*\[CEND\]")


def _word_count(text: str | None) -> int | None:
    return len(text.split()) if text else None


def _ts_to_mmss(ts: str) -> str:
    parts = ts.split(":")
    if len(parts) != 3:
        return ts
    h, m, s = (int(p) for p in parts)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_achap_output(target_text_ts_ref: str | None) -> str | None:
    """Extract 'MM:SS - title' lines from YTSeg timestamped chapter targets."""
    if not target_text_ts_ref:
        return None
    lines: list[str] = []
    for match in CHAPTER_HEADER_RE.finditer(target_text_ts_ref):
        title = match.group(2).strip()
        if title:
            lines.append(f"{_ts_to_mmss(match.group(1))} - {title}")
    return "\n".join(lines) if lines else None


def join_transcript(text_ref) -> str | None:
    if not text_ref:
        return None
    if isinstance(text_ref, list):
        parts = [str(x).strip() for x in text_ref if str(x).strip()]
        return " ".join(parts) if parts else None
    return str(text_ref).strip() or None


def signals_path(video_id: str, signals_root: str) -> str:
    # Stored relative to the dataset dir; resolved at load time via media_dir.
    return f"{signals_root}/{video_id}.mp3"


def _base_row(video_id: str, task: str, split: str, record: dict, source: str, license: str, signals_root: str) -> dict:
    duration = record.get("duration")
    transcript = join_transcript(record.get("text_ref"))
    row = empty_row()
    row.update(
        {
            "sample_id": f"ytseg_{split}_{video_id}_{task}",
            "task": [task],
            "split": split,
            "source_dataset": source,
            "license": license,
            "audio_input_path": signals_path(video_id, signals_root),
            "audio_input_duration": float(duration) if duration is not None else None,
            "audio_input_sample_rate": SAMPLE_RATE,
            "audio_input_num_channels": 1,
            "audio_input_codec": AUDIO_CODEC,
            "language_audio_input": "en",
            "num_speakers": record.get("num_speakers"),
        }
    )
    if duration and transcript and duration > 0:
        row["words_per_second"] = _word_count(transcript) / duration
        row["chars_per_second"] = len(transcript) / duration
    return row


def build_task_rows(video_id: str, split: str, record: dict, source: str, license: str, signals_root: str) -> list[dict]:
    rows: list[dict] = []
    transcript = join_transcript(record.get("text_ref"))
    achap_text = parse_achap_output(record.get("target_text_ts_ref"))

    if transcript:
        row = _base_row(video_id, "asr", split, record, source, license, signals_root)
        row.update(
            {
                "text_output": transcript,
                "text_output_normalized": transcript,
                "num_words_output": _word_count(transcript),
                "language_text_output": "en",
            }
        )
        rows.append(row)

    if achap_text:
        row = _base_row(video_id, "achap", split, record, source, license, signals_root)
        row.update(
            {
                "text_output": achap_text,
                "text_output_normalized": achap_text,
                "num_words_output": _word_count(achap_text),
                "language_text_output": "en",
            }
        )
        rows.append(row)

    return rows


def build_samples(
    dataset_id: str,
    splits: list,
    source: str,
    license: str,
    signals_root: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Load the HF text config for each split and build schema rows (asr, achap)."""
    from datasets import load_dataset

    samples: list[dict] = []
    for split in splits:
        hf_split = HF_SPLITS[split]
        logger.info("Loading %s (text, %s)", dataset_id, hf_split)
        ds = load_dataset(dataset_id, "text", split=hf_split)
        n = len(ds) if max_samples is None else min(max_samples, len(ds))
        for i in range(n):
            record = ds[i]
            samples.extend(build_task_rows(record["video_id"], split, record, source, license, signals_root))
        logger.info("split=%s: %d videos", split, n)
    return samples


def write_parquets(samples: list[dict], splits: list, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check the dataset id / splits.")

    for split in splits:
        part = df[df["split"] == split]
        if part.empty:
            continue
        out = output_dir / f"{split}.parquet"
        table = pa.Table.from_pandas(part.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False)
        pq.write_table(table, out)
        logger.info("Wrote %s (%d rows)", out, len(part))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YTSeg metadata Parquet from the Hugging Face Hub.")
    parser.add_argument("--dataset-id", default="retkowski/ytseg")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--source", default="YTSeg")
    parser.add_argument("--license", default="CC-BY-NC-SA-4.0")
    parser.add_argument("--tasks", nargs="+", default=["asr", "achap"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit videos per split (debug)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset_root = args.dataset_root or datasets_dir() / "ytseg"
    output_dir = dataset_root / "parquet"

    samples = build_samples(args.dataset_id, args.splits, args.source, args.license, args.max_samples)

    write_parquets(samples, args.splits, output_dir)


if __name__ == "__main__":
    main()
