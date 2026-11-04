"""Convert cleaned metadata Parquet files to ShareGPT JSON."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from speechlmm.data.prompt_renderer import normalized_locale, render_user_content, NO_ANSWER
from speechlmm.data.utils import TRANSLATION_TASKS, MONOLINGUAL_TASKS

logger = logging.getLogger(__name__)

def _build_assistant_content(row: dict, task: str, no_answer: bool | None = None) -> str:
    """Build the assistant turn from text and optional generated speech."""
    parts: list[str] = []

    text_output = row.get("text_output") if no_answer is None else no_answer
    if text_output is not None and str(text_output).strip():
        parts.append(str(text_output))

    if row.get("audio_output_path"):
        parts.append("<audio>")

    if not parts:
        logger.warning("Sample has no output for task=%s, sample_id=%s", task, row.get("sample_id"))
        return ""
    return "\n".join(parts)


def convert_row(row: dict,
    task: str,
    data_root: str | None = None,
    random_thr: float | None = 0.1
    ) -> dict | None:
    """Convert one metadata row to a ShareGPT conversation."""
    audios: list[str] = []
    videos: list[str] = []
    lipread: list[str] = []

    user_content, no_answer = render_user_content(row, task, audios, videos, lipread, random_thr=random_thr)
    assistant_content = _build_assistant_content(row, task, no_answer=no_answer)
    if not user_content or not assistant_content:
        return None

    if data_root:
        audios = [str(Path(data_root) / path) for path in audios]
        videos = [str(Path(data_root) / path) for path in videos]
        lipread = [str(Path(data_root) / path) for path in lipread]

    entry: dict = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    if audios:
        entry["audios"] = audios
    if videos:
        entry["videos"] = videos
    if lipread:
        entry["lipread"] = lipread

    codec_tokens = row.get("audio_output_codec_tokens")
    if codec_tokens is not None:
        entry["codec_tokens"] = codec_tokens
    if row.get("sample_id") is not None:
        entry["sample_id"] = row["sample_id"]
    if row.get("speaker"):
        entry["speaker"] = row["speaker"]
    elif isinstance(row.get("speaker_id"), list) and row["speaker_id"]:
        entry["speaker"] = row["speaker_id"][0]
    return entry


def _input_language(row: dict, task: str) -> str:
    if task == "lipread" and row.get("video_input_path"):
        language = row.get("language_video_input") or row.get("language_text_input")
    elif row.get("audio_input_path"):
        language = (
            row.get("language_audio_input")
            or row.get("language_video_input")
            or row.get("language_text_input")
        )
    elif row.get("video_input_path"):
        language = row.get("language_video_input") or row.get("language_audio_input") or row.get("language_text_input")
    else:
        language = row.get("language_text_input")
    return normalized_locale(str(language)) if language else ""


def _output_language(row: dict) -> str:
    language = row.get("language_text_output") or row.get("language_audio_output")

    return normalized_locale(str(language)) if language else ""


def _get_direction(src_language: str, tgt_language: str) -> str:
    return f"{src_language}_{tgt_language}"


def convert_parquet(
    parquet_path: Path,
    tasks: Sequence[str],
    languages: Sequence[str],
    data_root: str | None = None,
    random_thr: float | None = 0.1
) -> dict[str, dict[str, list[dict]]]:
    """Read one Parquet file and group ShareGPT entries by task, input/output language."""
    normalized_languages = {normalized_locale(language): language for language in languages}
    results = {task: {} for task in tasks}
    for task in tasks:
        for src_language in languages: 
            for tgt_language in languages:
                direction = _get_direction(src_language, tgt_language)
                results[task][direction]= []

    df = pd.read_parquet(parquet_path)
    for d in df.itertuples(index=True):
        row = d._asdict()
        row_tasks = row.get("task") or []
        if isinstance(row_tasks, str):
            row_tasks = [row_tasks]

        for task in tasks:
            if task not in row_tasks:
                continue
            normalized_src_language = _input_language(row, task)
            configured_src_language = normalized_languages.get(normalized_src_language)
            if configured_src_language is None:
                continue
            normalized_tgt_language = _output_language(row)
            if not normalized_tgt_language and task in MONOLINGUAL_TASKS:
                normalized_tgt_language = normalized_src_language
            configured_tgt_language = normalized_languages.get(normalized_tgt_language)
            if configured_tgt_language is None:
                continue
            configured_direction = _get_direction(configured_src_language, configured_tgt_language)
            entry = convert_row(row, task, data_root, random_thr=random_thr)
            if entry is not None:
                results[task][configured_direction].append(entry)
    return results



def convert_dataset(
    parquet_files: Sequence[Path],
    output_dir: Path,
    dataset_name: str,
    tasks: Sequence[str],
    languages: Sequence[str],
    split: str = "train",
    data_root: str | None = None,
    random_thr: float | None = 0.1
) -> dict[str, str]:
    """Convert one logical dataset split and return its registry mapping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = {task: {} for task in tasks}
    for task in tasks:
        for src_language in languages: 
            for tgt_language in languages:
                direction = _get_direction(src_language, tgt_language)
                entries[task][direction]= []

    for parquet_file in parquet_files:
        converted = convert_parquet(Path(parquet_file), tasks, languages, data_root, random_thr=random_thr)
        logger.info(f"Converted {parquet_file} for tasks: {converted.keys()} entries.")
        for task in tasks:
            logger.info(f"Converted {parquet_file} for tasks task: {task} for languages: {converted[task].keys()}")
            for src_language in languages: 
                for tgt_language in languages:
                    direction = _get_direction(src_language, tgt_language)
                    entries[task][direction].extend(converted[task][direction])

    written: dict[str, str] = {}
    for task in tasks:
        for src_language in languages: 
            for tgt_language in languages:

                # in bilingual task, do not save sharegpt is input and output languages are equal
                if task in TRANSLATION_TASKS and src_language == tgt_language:
                    continue

                # in monolingual task (neither translation nor mixed tasks), do not save sharegpt is input and output languages differ
                if task in MONOLINGUAL_TASKS and src_language != tgt_language:
                    continue

                # flag to identify if the entry is monolingual or bilingual, for naming the output file
                bilingual_entry = src_language != tgt_language
                direction = _get_direction(src_language, tgt_language)
                normalized_direction = _get_direction(normalized_locale(src_language), normalized_locale(tgt_language))
                task_entries = entries[task][direction]
                if not task_entries:
                    logger.info(
                        "No %s samples for dataset=%s task=%s direction=%s",
                        split,
                        dataset_name,
                        task,
                        direction,
                    )
                    continue

                suffix = normalized_direction if bilingual_entry else normalized_locale(src_language)
                name = f"{dataset_name}_{task}_{split}_{suffix}"
                out_name = f"{name}.json"
                with open(output_dir / out_name, "w", encoding="utf-8") as file:
                    json.dump(task_entries, file, ensure_ascii=False, indent=2)
                written[name] = out_name
                logger.info("Wrote %d samples to %s", len(task_entries), output_dir / out_name)
    return written