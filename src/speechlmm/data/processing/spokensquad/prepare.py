"""Synthetic spoken SQuAD corpus preparation pipeline."""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args, add_video_integrity_args
from speechlmm.data.processing.spokensquad.to_parquet import (
    DEFAULT_IMPOSSIBLE_ANSWER_TEXT,
    DEFAULT_TEXT_MODALITY_PROBABILITY,
    build_and_write_parquets,
    discover_audio_manifests,
    extract_audio_manifest,
)

logger = logging.getLogger(__name__)


class SpokenSquadCorpusPreparer(CorpusPreparer):
    name = "spokensquad"
    default_tasks: tuple[str, ...] = ("sqa",)
    default_splits: tuple[str, ...] = ("train", "val")
    default_languages: tuple[str, ...] = ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv")

    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None
    ) -> None:
        super().__init__(dataset_root, tasks, splits, languages)

        self.metadata_paths = self.dataset_root / "original" / "metadata"
        self.archive_root = None
        self.signals_root = "signals"
        self.include_question_text = False
        self.models = None
        self.val_size = 250
        self.test_size = 250
        self.seed = 13
        self.respect_existing_splits = True
        self.impossible_answer_text = DEFAULT_IMPOSSIBLE_ANSWER_TEXT
        self.text_modality_probability = DEFAULT_TEXT_MODALITY_PROBABILITY
        self.chunk_size = 10000
        self.workers = 4
        self.overwrite_audio = False


    def additional_setting(
        self,
        argv: list[str] | None = None,
    ) -> None:
        parser = argparse.ArgumentParser()
        add_common_extra_args(parser)
        args, _ = parser.parse_known_args(argv)
        if args.metadata:
            self.metadata_paths = args.metadata
        if args.archive_root:
            self.archive_root = args.archive_root
        if args.signals_root:
            self.signals_root = args.signals_root
        if args.include_question_text:
            self.include_question_text = args.include_question_text
        if args.models:
            self.models = args.models
        if args.val_size:
            self.val_size = args.val_size
        if args.test_size:
            self.test_size = args.test_size
        if args.seed:
            self.seed = args.seed
        if args.random_splits:
            self.respect_existing_splits = not args.random_splits
        if args.impossible_answer_text:
            self.impossible_answer_text = args.impossible_answer_text
        if args.text_modality_probability:
            self.text_modality_probability = args.text_modality_probability
        if args.chunk_size:
            self.chunk_size = args.chunk_size
        if args.workers:
            self.workers = args.workers
        if args.overwrite_audio:
            self.overwrite_audio = args.overwrite_audio

    def to_parquet(self) -> None:
        build_and_write_parquets(
            self.metadata_paths,
            self.parquet_dir,
            archive_root=self.archive_root,
            signals_root=self.get_signals_prefix(),
            splits=self.splits,
            languages=self.languages,
            include_question_text=self.include_question_text,
            models=self.models,
            val_size=self.val_size,
            test_size=self.test_size,
            seed=self.seed,
            respect_existing_splits=self.respect_existing_splits,
            impossible_answer_text=self.impossible_answer_text,
            text_modality_probability=self.text_modality_probability,
            chunk_size=self.chunk_size,
        )

    def download(self) -> None:
        manifest_paths = discover_audio_manifests(self.parquet_dir)
        if not manifest_paths:
            raise FileNotFoundError(f"No audio manifests found under: {self.parquet_dir}")

        total: Counter[str] = Counter()
        for manifest_path in manifest_paths:
            logger.info("Extracting audio listed in %s", manifest_path)
            total.update(
                extract_audio_manifest(
                    manifest_path,
                    self.dataset_root,
                    workers=self.workers,
                    overwrite=self.overwrite_audio,
                )
            )
        logger.info("Combined audio extraction stats: %s", dict(total))



def add_common_args(
    parser: argparse.ArgumentParser,
    default_tasks: tuple[str, ...],
    default_splits: tuple[str, ...],
    default_languages: tuple[str, ...],
) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset dir (default: $DATASETS_DIR/speechlmm-v2/ami)",
    )
    parser.add_argument("--tasks", nargs="+", default=list(default_tasks), help="Tasks to export to ShareGPT")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(default_languages),
        help="Languages to export to ShareGPT",
    )
    parser.add_argument("--splits", nargs="+", default=list(default_splits), help="Splits to build")
    

def add_common_extra_args(
    parser: argparse.ArgumentParser
) -> None:
    parser.add_argument("--skip-parquet", action="store_true", help="Reuse already-downloaded parquet")
    parser.add_argument("--skip-download", action="store_true", help="Reuse already-downloaded media")
    parser.add_argument("--skip-filter", action="store_true", help="Skip duration, task, and corrupted-video filtering")
    add_video_integrity_args(parser)

    parser.add_argument("--metadata", nargs="+", type=Path, required=False, help="Metadata file(s) or directories")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Root containing archive_relpath entries (default: <metadata-parent>/../output)",
    )
    parser.add_argument(
        "--signals-root", default="signals", help="Relative output prefix for extracted audio paths stored in Parquet"
    )
    parser.add_argument("--models", nargs="+", default=None, help="TTS model ids to keep, e.g. omnivoice xtts_v2")
    parser.add_argument("--val-size", type=int, default=250, help="Target number of validation QA samples per language")
    parser.add_argument("--test-size", type=int, default=250, help="Target number of test QA samples per language")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for grouped split assignment")
    parser.add_argument(
        "--random-splits",
        action="store_true",
        help="Ignore metadata split labels and randomly create train/val/test by context group",
    )
    parser.add_argument(
        "--include-question-text", action="store_true", help="Also include question text in the user turn when available"
    )
    parser.add_argument(
        "--impossible-answer-text",
        default=DEFAULT_IMPOSSIBLE_ANSWER_TEXT,
        help="Assistant text target for SQuAD v2 impossible questions",
    )
    parser.add_argument(
        "--text-modality-probability",
        type=float,
        default=DEFAULT_TEXT_MODALITY_PROBABILITY,
        help="Independent probability that passage/question is provided as text instead of speech",
    )
    parser.add_argument("--chunk-size", type=int, default=10000, help="Rows per Parquet write batch")
    parser.add_argument("--workers", type=int, default=4, help="Parallel archive extraction workers")
    parser.add_argument("--overwrite-audio", action="store_true", help="Overwrite already extracted audio files")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the SpokenSQuAD corpus.")
    add_common_args(
        parser,
        default_tasks=SpokenSquadCorpusPreparer.default_tasks,
        default_languages=SpokenSquadCorpusPreparer.default_languages,
        default_splits=SpokenSquadCorpusPreparer.default_splits,
    )
    args, extra_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    preparer = SpokenSquadCorpusPreparer(
        dataset_root=args.dataset_root,
        splits=args.splits,
        tasks=args.tasks,
        languages=args.languages,
    )

    preparer.additional_setting(args=extra_args)

    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
