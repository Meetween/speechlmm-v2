"""Build synthetic spoken SQuAD metadata Parquet files for SpeechLMM.

The real speechlmm-tts-synth metadata is one CSV row per synthesized audio part:
context rows contain the spoken passage, and question rows contain one spoken
question plus the extractive answer in ``answers_json``. This module joins those
rows into one SpeechLMM sample per answerable QA pair.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from speechlmm.data.schema import SCHEMA, empty_row

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress bars are optional.
    tqdm = None

logger = logging.getLogger(__name__)
csv.field_size_limit(2**31 - 1)

SOURCE = "SynthSpokenSQuAD"
DEFAULT_TASK = "sqa"
DEFAULT_LICENSE = "unknown"
DEFAULT_IMPOSSIBLE_ANSWER_TEXT = "unanswerable"
DEFAULT_TEXT_MODALITY_PROBABILITY = 0.3
SUPPORTED_SUFFIXES = {".csv"}
MANIFEST_NAME = "audio_manifest.csv"
MANIFEST_PREFIX = "audio_manifest"


@dataclass(frozen=True)
class AudioRef:
    archive_path: Path
    member_name: str
    audio_path: str
    duration: float | None
    sample_rate: int | None
    num_channels: int | None
    codec: str | None


@dataclass(frozen=True)
class ContextRow:
    context_id: str
    source_part_id: str
    language: str
    model_id: str
    context_text: str | None
    audio: AudioRef


@dataclass(frozen=True)
class QuestionRow:
    sample_id: str
    context_id: str
    language: str
    model_id: str
    context_text: str | None
    context_audio: AudioRef
    split: str
    question_text: str | None
    question_audio: AudioRef
    answer_text: str


class _NullProgress:
    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, n: int) -> None:
        return None


def _progress_bar(total: int, desc: str, unit: str) -> Any:
    if tqdm is None:
        return _NullProgress()
    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _drain_progress_queue(progress_queue: SimpleQueue[int], progress: Any) -> None:
    updated = 0
    while not progress_queue.empty():
        updated += progress_queue.get()
    if updated:
        progress.update(updated)


def _safe_id(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")


def manifest_name_for_language(language: str) -> str:
    return f"{MANIFEST_PREFIX}_{_safe_id(language)}.csv"


def manifest_name() -> str:
    return f"{MANIFEST_PREFIX}.csv"


def discover_audio_manifests(parquet_dir: Path) -> list[Path]:
    manifests = sorted(Path(parquet_dir).glob(f"{MANIFEST_PREFIX}_*.csv"))
    if manifests:
        return manifests

    legacy_manifest = Path(parquet_dir) / MANIFEST_NAME
    return [legacy_manifest] if legacy_manifest.exists() else []


def _metadata_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Metadata path not found: {path}")
    if not files:
        raise FileNotFoundError(f"No CSV metadata files found under: {[str(p) for p in paths]}")
    return files


def _derive_archive_root(metadata_files: Sequence[Path], archive_root: Path | None) -> Path:
    if archive_root:
        return Path(archive_root)

    first = Path(metadata_files[0]).resolve()
    if first.parent.name == "metadata":
        return first.parent.parent / "output"
    return first.parent.parent / "output"


def _normalise_language(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower().replace("_", "-")
    match = re.search(r"(?:^|[-/])([a-z]{2})(?:[-/]|$)", text)
    return match.group(1) if match else text


def _wanted(value: str | None, wanted_values: set[str] | None) -> bool:
    if not wanted_values:
        return True
    return value in wanted_values


def _optional_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(value: str | None) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _word_count(text: str | None) -> int | None:
    return len(text.split()) if text else None


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _split_from_metadata(value: str | None, splits: set[str] | None) -> str | None:
    split = str(value or "").strip().lower()
    split = {"validation": "val"}.get(split, split)
    return split if splits is None or split in splits else None


def _answer_text(answers_json: str | None) -> str | None:
    if not answers_json:
        return None
    try:
        answers = json.loads(answers_json)
    except json.JSONDecodeError:
        return None

    texts = answers.get("text")
    if isinstance(texts, list):
        for text in texts:
            if text:
                return str(text).strip()
    if isinstance(texts, str) and texts.strip():
        return texts.strip()
    return None


def _audio_path(row: dict[str, str], signals_root: str) -> str:
    model_id = _safe_id(row.get("model_id") or "unknown_model")
    language = _normalise_language(row.get("audio_language")) or "unknown"
    audio_name = row.get("audio_name")
    if not audio_name:
        raise ValueError("Missing audio_name in generated metadata row")
    return str(Path(signals_root) / "audio" / f"{model_id}-{language}" / audio_name)


def _archive_path(row: dict[str, str], archive_root: Path) -> Path:
    relpath = row.get("archive_relpath")
    if relpath:
        return archive_root / relpath

    archive_path = row.get("archive_path")
    if archive_path:
        return Path(archive_path)
    raise ValueError("Missing both archive_relpath and archive_path in generated metadata row")


def _duration(row: dict[str, str]) -> float | None:
    return _optional_float(row.get("audio_duration_seconds") or row.get("estimated_duration_seconds"))


def _audio_ref(row: dict[str, str], archive_root: Path, signals_root: str) -> AudioRef:
    audio_name = row.get("audio_name")
    if not audio_name:
        raise ValueError("Missing audio_name in generated metadata row")
    return AudioRef(
        archive_path=_archive_path(row, archive_root),
        member_name=audio_name,
        audio_path=_audio_path(row, signals_root),
        duration=_duration(row),
        sample_rate=_optional_int(row.get("sample_rate")),
        num_channels=_optional_int(row.get("channels")),
        codec=(row.get("audio_format") or None),
    )


def _is_generated(row: dict[str, str]) -> bool:
    return row.get("status") == "generated"


def _row_language(row: dict[str, str]) -> str | None:
    return _normalise_language(row.get("audio_language") or row.get("input_text_language"))


def _context_key(row: dict[str, str], source_part_id: str) -> tuple[str, str]:
    return (row.get("source_instance_id") or "", source_part_id)


def _read_contexts(
    metadata_files: Sequence[Path],
    archive_root: Path,
    signals_root: str,
    languages: set[str] | None,
    models: set[str] | None,
) -> dict[int, dict[tuple[str, str], ContextRow]]:
    contexts_by_file: dict[int, dict[tuple[str, str], ContextRow]] = {}

    for file_idx, path in enumerate(metadata_files):
        contexts: dict[tuple[str, str], ContextRow] = {}
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                if not _is_generated(row) or row.get("part_level") != "context":
                    continue
                language = _row_language(row)
                model_id = row.get("model_id") or ""
                source_part_id = row.get("source_part_id") or ""
                context_id = row.get("context_id") or ""
                if not language or not source_part_id or not context_id:
                    continue
                if not _wanted(language, languages) or not _wanted(model_id, models):
                    continue
                contexts.setdefault(
                    _context_key(row, source_part_id),
                    ContextRow(
                        context_id=context_id,
                        source_part_id=source_part_id,
                        language=language,
                        model_id=model_id,
                        context_text=row.get("input_text") or "",
                        audio=_audio_ref(row, archive_root, signals_root),
                    ),
                )
        contexts_by_file[file_idx] = contexts
        logger.info("Indexed %d context rows from %s", len(contexts), path)

    return contexts_by_file


def _iter_questions(
    metadata_files: Sequence[Path],
    contexts_by_file: dict[int, dict[tuple[str, str], ContextRow]],
    archive_root: Path,
    signals_root: str,
    splits: set[str] | None,
    languages: set[str] | None,
    models: set[str] | None,
    include_question_text: bool,
    impossible_answer_text: str,
) -> Iterator[tuple[QuestionRow, str | None]]:
    for file_idx, path in enumerate(metadata_files):
        contexts = contexts_by_file[file_idx]
        with Path(path).open(newline="") as f:
            for row in csv.DictReader(f):
                if not _is_generated(row) or row.get("part_level") != "question":
                    continue

                language = _row_language(row)
                model_id = row.get("model_id") or ""
                if not language or not _wanted(language, languages) or not _wanted(model_id, models):
                    continue

                context = contexts.get(_context_key(row, row.get("parent_part_id") or ""))
                if context is None or context.context_id != (row.get("context_id") or ""):
                    continue
                answer = _answer_text(row.get("answers_json"))
                if _is_true(row.get("is_impossible")) and not answer:
                    answer = impossible_answer_text
                if not answer:
                    continue

                qa_id = row.get("qa_id") or row.get("question_id") or row.get("source_part_id") or row.get("audio_name")
                sample_id = f"{SOURCE.lower()}_{model_id}_{language}_{qa_id}"
                split = _split_from_metadata(row.get("split"), splits) or "train"
                question_text = row.get("input_text") or ""
                question_audio = _audio_ref(row, archive_root, signals_root)

                yield (
                    QuestionRow(
                        sample_id=sample_id,
                        context_id=context.context_id,
                        language=language,
                        model_id=model_id,
                        split=split,
                        context_text=context.context_text,
                        question_text=question_text,
                        answer_text=answer,
                        context_audio=context.audio,
                        question_audio=question_audio,
                    ),
                    split,
                )


def _count_contexts(
    metadata_files: Sequence[Path],
    contexts_by_file: dict[int, dict[tuple[str, str], ContextRow]],
    archive_root: Path,
    signals_root: str,
    splits: set[str] | None,
    languages: set[str] | None,
    models: set[str] | None,
    include_question_text: bool,
    impossible_answer_text: str,
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    counts: Counter[tuple[str, str]] = Counter()
    existing_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    for question, existing_split in _iter_questions(
        metadata_files,
        contexts_by_file,
        archive_root,
        signals_root,
        splits,
        languages,
        models,
        include_question_text,
        impossible_answer_text,
    ):
        key = (question.language, question.context_id)
        counts[key] += 1
        if existing_split:
            existing_splits[key].add(existing_split)

    logger.info("Counted %d QA rows across %d language/context groups", sum(counts.values()), len(counts))
    return counts, existing_splits


def assign_context_splits(
    counts: Counter[tuple[str, str]],
    existing_splits: dict[tuple[str, str], set[str]],
    *,
    val_size: int,
    test_size: int,
    seed: int,
    respect_existing_splits: bool,
) -> dict[tuple[str, str], str]:
    assignments: dict[tuple[str, str], str] = {}

    if respect_existing_splits:
        mixed_contexts = 0
        for key in counts:
            splits = existing_splits.get(key) or {"train"}
            if len(splits) == 1:
                assignments[key] = sorted(splits)[0]
            else:
                mixed_contexts += 1
                assignments[key] = "train"
        if mixed_contexts:
            logger.warning("Assigned %d mixed-split context groups to train to avoid leakage", mixed_contexts)
        return assignments

    by_language: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (language, context_id), count in counts.items():
        by_language[language].append((context_id, count))

    rng = random.Random(seed)
    for language, groups in sorted(by_language.items()):
        rng.shuffle(groups)
        split_counts = {"val": 0, "test": 0, "train": 0}
        for context_id, count in groups:
            if split_counts["val"] < val_size:
                split = "val"
            elif split_counts["test"] < test_size:
                split = "test"
            else:
                split = "train"
            assignments[(language, context_id)] = split
            split_counts[split] += count
        logger.info("Assigned language=%s split counts=%s", language, split_counts)

    return assignments


def _stable_probability(seed: int, sample_id: str, field: str) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{field}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _format_input(value: str, as_text: bool) -> str:
    return value if as_text else "<audio>"


def _random_question_formats(
    question: QuestionRow,
    *,
    text_modality_probability: float,
    seed: int,
) -> tuple[str, str, str | None, str | None]:
    context_as_text = _stable_probability(seed, question.sample_id, "context") < text_modality_probability
    question_as_text = _stable_probability(seed, question.sample_id, "question") < text_modality_probability

    if context_as_text:
        context_audio_path = None
        context_text = question.context_text
    else:
        context_audio_path = question.context_audio.audio_path
        context_text = "<audio>"
    if question_as_text:
        question_audio_path = None
        question_text = question.question_text
    else:
        question_audio_path = question.question_audio.audio_path
        question_text = "<audio>"

    return context_audio_path, question_audio_path, context_text, question_text


def _schema_row(
    question: QuestionRow,
    task: str,
    split: str,
    *,
    text_modality_probability: float,
    seed: int,
) -> dict[str, Any]:

    context_audio_path, question_audio_path, context_text, question_text = _random_question_formats(
        question,
        text_modality_probability=text_modality_probability,
        seed=seed,
    )

    row = empty_row()
    row.update(
        {
            "sample_id": question.sample_id,
            "task": [task],
            "split": split,
            "source_dataset": SOURCE,
            "license": DEFAULT_LICENSE,
        }
    )

    row.update(
        {
            "text_output": question.answer_text,
            "text_output_normalized": question.answer_text,
            "num_words_output": _word_count(question.answer_text),
            "language_text_output": question.language,
        }
    )

    if context_audio_path:
        row.update(
            {
                "audio_input_path": context_audio_path,
                "audio_input_duration": question.context_audio.duration,
                "audio_input_sample_rate": question.context_audio.sample_rate,
                "audio_input_num_channels": question.context_audio.num_channels,
                "audio_input_codec": question.context_audio.codec,
                "language_audio_input": question.language,
            }
        )
    else:
        row.update(
            {
                "text_input": context_text,
                "language_text_input": question.language,
            }
        )

    if question_audio_path:
        row.update(
            {
                "audio_prompt_path": question_audio_path,
                "audio_prompt_duration": question.question_audio.duration,
                "audio_prompt_sample_rate": question.question_audio.sample_rate,
                "audio_prompt_num_channels": question.question_audio.num_channels,
                "audio_prompt_codec": question.question_audio.codec,
                "language_audio_prompt": question.language,
            }
        )
    else:
        row.update(
            {
                "text_prompt": question_text,
                "language_text_prompt": question.language,
            }
        )

    return row


def _manifest_row(audio: AudioRef) -> dict[str, str]:
    return {
        "archive_path": str(audio.archive_path),
        "member_name": audio.member_name,
        "audio_path": audio.audio_path,
    }


def _write_table(
    rows: list[dict[str, Any]],
    key: str,
    output_dir: Path,
    writers: dict[str, pq.ParquetWriter],
) -> None:
    if not rows:
        return
    split = key
    out = output_dir / f"{split}.parquet"
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    if key not in writers:
        writers[key] = pq.ParquetWriter(out, SCHEMA)
    writers[key].write_table(table)
    rows.clear()


def open_manifest_writer(output_dir: Path) -> csv.DictWriter:
    path = output_dir / manifest_name()
    handle = path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=("archive_path", "member_name", "audio_path"))
    writer.writeheader()
    return writer, handle


def write_parquets_and_manifest(
    metadata_files: Sequence[Path],
    contexts_by_file: dict[int, dict[tuple[str, str], ContextRow]],
    assignments: dict[tuple[str, str], str],
    output_dir: Path,
    *,
    archive_root: Path,
    splits: Sequence[str],
    languages: set[str] | None,
    signals_root: str,
    models: set[str] | None,
    include_question_text: bool,
    impossible_answer_text: str,
    text_modality_probability: float,
    seed: int,
    chunk_size: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted_splits = set(splits)
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    writers: dict[str, pq.ParquetWriter] = {}
    counts: Counter[str] = Counter()

    manifest_seen: set[tuple[str, str, str]] = set()
    manifest_counts: int = 0
    manifest_writer, manifest_handle = open_manifest_writer(output_dir)

    try:
        for question, _ in _iter_questions(
            metadata_files,
            contexts_by_file,
            archive_root,
            signals_root,
            splits,
            languages,
            models,
            include_question_text,
            impossible_answer_text,
        ):
            split = assignments.get((question.language, question.context_id), "train")
            if split not in wanted_splits:
                continue

            key = split
            row = _schema_row(
                question,
                DEFAULT_TASK,
                split,
                text_modality_probability=text_modality_probability,
                seed=seed,
            )
            buffers[key].append(row)
            counts[key] += 1

            for path, audio in (
                (row.get("audio_prompt_path"), question.question_audio),
                (row.get("audio_input_path"), question.context_audio),
            ):
                if not path:
                    continue

                manifest_key = (str(audio.archive_path), audio.member_name, audio.audio_path)
                if manifest_key not in manifest_seen:
                    manifest_seen.add(manifest_key)
                    manifest_writer.writerow(_manifest_row(audio))
                    manifest_counts += 1

            if len(buffers[key]) >= chunk_size:
                _write_table(buffers[key], key, output_dir, writers)
                logger.info("Wrote chunk for split=%s", key)
    finally:
        manifest_handle.close()

    for key, rows in buffers.items():
        _write_table(rows, key, output_dir, writers)

    for writer in writers.values():
        writer.close()

    written = {split: output_dir / f"{split}.parquet" for split in counts}
    for key, count in sorted(counts.items()):
        logger.info("Wrote %s (%d rows)", written[key], count)
    logger.info(
        "Wrote audio extraction manifest %s (%d unique audio files)",
        output_dir / manifest_name(),
        manifest_counts,
    )
    return written


def build_and_write_parquets(
    metadata_paths: Sequence[Path],
    output_dir: Path,
    *,
    archive_root: Path | None = None,
    signals_root: str = "signals",
    splits: list[str] | None = None,
    languages: list[str] | None = None,
    include_question_text: bool = False,
    models: list[str] | None = None,
    val_size: int = 250,
    test_size: int = 250,
    seed: int = 13,
    respect_existing_splits: bool = True,
    impossible_answer_text: str = DEFAULT_IMPOSSIBLE_ANSWER_TEXT,
    text_modality_probability: float = DEFAULT_TEXT_MODALITY_PROBABILITY,
    chunk_size: int = 10000,
) -> dict[str, Path]:
    if not 0.0 <= text_modality_probability <= 1.0:
        raise ValueError("text_modality_probability must be between 0 and 1")
    metadata_paths = [metadata_paths] if isinstance(metadata_paths, Path) else list(metadata_paths)
    metadata_files = _metadata_files(metadata_paths)
    archive_root = _derive_archive_root(metadata_files, archive_root)
    wanted_languages = {_normalise_language(lang) for lang in languages} if languages else None
    wanted_languages = {lang for lang in wanted_languages if lang} if wanted_languages else None
    wanted_models = set(models) if models else None

    logger.info("Using archive root: %s", archive_root)
    contexts_by_file = _read_contexts(metadata_files, archive_root, signals_root, wanted_languages, wanted_models)
    counts, existing_splits = _count_contexts(
        metadata_files,
        contexts_by_file,
        archive_root,
        signals_root,
        set(splits),
        wanted_languages,
        wanted_models,
        include_question_text,
        impossible_answer_text,
    )
    if not counts:
        raise RuntimeError("No synthetic spoken SQuAD samples found.")

    assignments = assign_context_splits(
        counts,
        existing_splits,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        respect_existing_splits=respect_existing_splits,
    )
    return write_parquets_and_manifest(
        metadata_files,
        contexts_by_file,
        assignments,
        output_dir,
        archive_root=archive_root,
        signals_root=signals_root,
        splits=splits,
        languages=wanted_languages,
        models=wanted_models,
        include_question_text=include_question_text,
        impossible_answer_text=impossible_answer_text,
        text_modality_probability=text_modality_probability,
        seed=seed,
        chunk_size=chunk_size,
    )


def _extract_from_archive(
    archive_path: Path,
    items: list[tuple[str, Path]],
    *,
    overwrite: bool,
    progress_queue: SimpleQueue[int] | None = None,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    if not archive_path.exists():
        stats["missing_archive"] += len(items)
        logger.error("Archive not found: %s", archive_path)
        if progress_queue is not None:
            progress_queue.put(len(items))
        return stats

    with tarfile.open(archive_path) as tar:
        names = set(tar.getnames())
        basename_to_name: dict[str, str] | None = None
        for member_name, output_path in items:
            if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
                stats["skipped_existing"] += 1
                if progress_queue is not None:
                    progress_queue.put(1)
                continue

            actual_member = member_name
            if actual_member not in names:
                if basename_to_name is None:
                    basename_to_name = {Path(name).name: name for name in names}
                actual_member = basename_to_name.get(Path(member_name).name, "")
            if not actual_member:
                stats["missing_member"] += 1
                if progress_queue is not None:
                    progress_queue.put(1)
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(actual_member)
            if src is None:
                stats["missing_member"] += 1
                if progress_queue is not None:
                    progress_queue.put(1)
                continue
            with src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            stats["extracted"] += 1
            if progress_queue is not None:
                progress_queue.put(1)
    return stats


def extract_audio_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    workers: int = 4,
    overwrite: bool = False,
) -> Counter[str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Audio manifest not found: {manifest_path}")

    by_archive: dict[Path, list[tuple[str, Path]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["archive_path"], row["member_name"], row["audio_path"])
            if key in seen:
                continue
            seen.add(key)
            by_archive[Path(row["archive_path"])].append((row["member_name"], Path(dataset_root) / row["audio_path"]))

    logger.info("Extracting %d audio files from %d archives", len(seen), len(by_archive))
    total: Counter[str] = Counter()
    progress_queue: SimpleQueue[int] = SimpleQueue()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _extract_from_archive,
                archive_path,
                items,
                overwrite=overwrite,
                progress_queue=progress_queue,
            )
            for archive_path, items in by_archive.items()
        }
        with _progress_bar(total=len(seen), desc="Extracting audio", unit="file") as progress:
            while futures:
                done, futures = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                _drain_progress_queue(progress_queue, progress)
                for future in done:
                    total.update(future.result())
            _drain_progress_queue(progress_queue, progress)
    logger.info("Audio extraction stats: %s", dict(total))
    return total
