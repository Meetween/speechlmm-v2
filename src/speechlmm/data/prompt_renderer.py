"""Render task prompts for ShareGPT conversion."""

from __future__ import annotations

import json
import logging
import re
import string
from functools import lru_cache
from importlib.resources import files
from string import Formatter
import numpy as np
import random

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_VERSION = "v0_1"
_ALLOWED_FIELDS = {
    "format_in",
    "format_out",
    "format_qst",
    "input",
    "intents",
    "no_answer",
    "lang_in",
    "lang_out",
    "lang_qst",
    "options",
    "question",
}
_LANGUAGE_MAP = {
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hu": "Hungarian",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "sv": "Swedish",
}
_SUBTASK_BY_SOURCE = {
    "librisqaparti": "sqa-abstractive",
    "librisqapartii": "sqa-multiple",
    "mmsu": "sqa-multiple",
    "slurp": "slu-list",
    "speechmassive": "slu-list",
    "synthspokensquad": "sqa-extractive",
}
_OUTPUT_FORMAT_BY_TASK = {"s2st": "speech", "tts": "speech"}

NO_ANSWER = 'unanswerable'

def normalized_locale(locale: str) -> str:
    """Return the base, lowercase locale code (for example ``en-US`` -> ``en``)."""
    return locale.split("-", 1)[0].lower()


@lru_cache(maxsize=None)
def load_prompt_templates(version: str = DEFAULT_PROMPT_VERSION) -> dict[str, tuple[str, ...]]:
    """Load and validate one packaged prompt set."""
    resource = files("speechlmm.data").joinpath("prompts", f"{version}.json")
    try:
        data = json.loads(resource.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown prompt version: {version}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Prompt set {version} must be a JSON object")

    templates: dict[str, tuple[str, ...]] = {}
    for task, parts in data.items():
        if not isinstance(task, str) or not isinstance(parts, list) or not parts:
            raise ValueError(f"Invalid prompt entry for {task!r} in {version}")
        if not all(isinstance(part, str) and part for part in parts):
            raise ValueError(f"Prompt parts for {task!r} in {version} must be non-empty strings")
        fields = {field_name for part in parts for _, field_name, _, _ in Formatter().parse(part) if field_name is not None}
        unknown_fields = fields - _ALLOWED_FIELDS
        if unknown_fields:
            raise ValueError(f"Prompt {task!r} in {version} has unknown placeholders: {sorted(unknown_fields)}")
        templates[task] = tuple(parts)
    return templates


def _language(row: dict, *keys: str, default: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            locale = normalized_locale(str(value))
            return _LANGUAGE_MAP.get(locale, locale)
    return default


def _source_variant(row: dict, task: str) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("prompt_variant"):
        return str(metadata["prompt_variant"])
    source = row.get("source_dataset")
    if not source:
        return task
    normalized_source = re.sub(r"[-_]", "", str(source)).lower()
    return _SUBTASK_BY_SOURCE.get(normalized_source, task)


def _primary_input(
    row: dict,
    task: str,
    audios: list[str],
    videos: list[str],
    lipread: list[str],
) -> tuple[str, str, str]:
    video_path = row.get("video_input_path")
    if task == "lipread" and video_path:
        lipread.append(video_path)
        language = _language(
            row,
            "language_video_input",
            "language_text_input",
            default="the source language",
        )
        return "video", language, "<lipread>"

    input_formats: list[str] = []
    input_values: list[str] = []
    input_language: str | None = None

    audio_path = row.get("audio_input_path")
    if audio_path:
        audios.append(audio_path)
        input_formats.append("speech")
        input_values.append("<audio>")
        input_language = _language(row, "language_audio_input", default="")

    if video_path:
        videos.append(video_path)
        input_formats.append("video")
        input_values.append("<video>")
        language = _language(
            row,
            "language_video_input",
            "language_audio_input",
            "language_text_input",
            default="",
        )
        input_language = input_language or language

    text_input = row.get("text_input")
    if text_input:
        input_formats.append("text")
        input_values.append(str(text_input).strip())
        language = _language(row, "language_text_input", default="")
        input_language = input_language or language

    return (
        " and ".join(input_formats) or "input",
        input_language or "the source language",
        " ".join(input_values),
    )


def _question(row: dict, audios: list[str], default_language: str) -> tuple[str, str, str]:
    text_prompt = row.get("text_prompt")
    if text_prompt:
        language = _language(row, "language_text_prompt", default=default_language)
        return "text", language, str(text_prompt).strip()

    audio_prompt = row.get("audio_prompt_path")
    if audio_prompt:
        audios.append(audio_prompt)
        language = _language(row, "language_audio_prompt", default=default_language)
        return "speech", language, "<audio>"

    return "question", default_language, ""


def _sqa_options(metadata: object, answer: object) -> str:
    if not isinstance(metadata, dict):
        return ""
    options = metadata.get("options")
    if options is None:
        return ""
    if isinstance(options, np.ndarray):
        options = options.tolist()
    if not isinstance(options, (list, tuple)) or not options:
        return ""

    numeric = isinstance(answer, str) and answer.isdigit()
    labels = [str(index + 1) for index in range(len(options))]
    if not numeric:
        labels = list(string.ascii_uppercase[: len(options)])
        if len(labels) < len(options):
            labels.extend(str(index + 1) for index in range(len(labels), len(options)))
    return "\n".join(f"{label}: {option}" for label, option in zip(labels, options))


def _slu_intents(metadata: object, random_thr: float | None = 0.1, answer: str | None = None) -> tuple[str, str | None]:
    no_answer: str = None

    if not isinstance(metadata, dict):
        return "", no_answer
    intents = metadata.get("intents")
    if intents is None:
        return "", no_answer
    if isinstance(intents, np.ndarray):
        intents = intents.tolist()
    if not isinstance(intents, (list, tuple)):
        return "", no_answer

    if random_thr is not None and random_thr < 1.0:
        if answer is not None and random.random() < random_thr:
            intents = [intent for intent in intents if intent != answer]
            no_answer = NO_ANSWER
    # select half of the intents randomly if there are more than 10
    intents_len = len(intents)
    if intents_len > 10:
        intents = random.sample(intents, intents_len // 2)
    
    return ", ".join(str(intent) for intent in intents if intent), no_answer


def render_user_content(
    row: dict,
    task: str,
    audios: list[str],
    videos: list[str],
    lipread: list[str] | None = None,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    random_thr: float | None = 0.1
) -> tuple[str, str | None]:
    """Render the user turn and collect referenced input media paths."""
    if lipread is None:
        lipread = []
    format_in, lang_in, input_value = _primary_input(row, task, audios, videos, lipread)
    if task == "sqa":
        format_qst, lang_qst, question = _question(row, audios, lang_in)
    else:
        format_qst, lang_qst, question = "question", lang_in, ""
    lang_out = _language(
        row,
        "language_audio_output" if row.get("audio_output_path") else "language_text_output",
        default=lang_in,
    )
    format_out = _OUTPUT_FORMAT_BY_TASK.get(task, "text")

    text_prompt = row.get("text_prompt")
    if text_prompt and task != "sqa":
        rendered = [str(text_prompt).strip()]
        if input_value:
            rendered.append(input_value)
        audio_prompt = row.get("audio_prompt_path")
        if audio_prompt:
            audios.append(audio_prompt)
            rendered.append("<audio>")
        return "\n".join(rendered), None

    variant = _source_variant(row, task)
    templates = load_prompt_templates(prompt_version)
    parts = templates.get(variant) or templates.get(task)
    if not parts:
        parts = ("Process the following {format_in} input.", "Input: {input}")
        logger.warning("No prompt template for task=%s variant=%s", task, variant)

    metadata = row.get("metadata")
    intents, no_answer = _slu_intents(metadata, random_thr=random_thr, answer=row.get("text_output"))
    options = _sqa_options(metadata, row.get("text_output"))

    context = {
        "format_in": format_in,
        "format_out": format_out,
        "format_qst": format_qst,
        "input": input_value,
        "intents": intents,
        "no_answer": NO_ANSWER,
        "lang_in": lang_in,
        "lang_out": lang_out,
        "lang_qst": lang_qst,
        "options": options,
        "question": question,
    }
    rendered = [part.format(**context) for part in parts]

    joined_templates = "\n".join(parts)
    if input_value and "{input}" not in joined_templates:
        rendered.append(input_value)
    if task == "sqa" and question and "{question}" not in joined_templates:
        rendered.append(question)
    if task != "sqa" and row.get("audio_prompt_path"):
        audios.append(row["audio_prompt_path"])
        rendered.append("<audio>")
    return "\n".join(rendered), no_answer
