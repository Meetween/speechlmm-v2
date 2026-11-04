"""Prepare IWSLT26 spoken-language-understanding data for SpeechLMM.

Load the requested Hugging Face splits, build intent-classification Parquet,
materialize each utterance's audio under ``signals/``, and convert the result
to ShareGPT.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.iwslt26.prepare --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_video_integrity_args
from speechlmm.data.processing.iwslt26.to_parquet import get_from_HF, build_samples, write_parquets

logger = logging.getLogger(__name__)

class IWSLT26CorpusPreparer(CorpusPreparer):
    name = "iwslt26"
    default_tasks: tuple[str, ...] = ("sqa",)
    default_splits: tuple[str, ...] = ("train", "dev")
    default_languages: tuple[str, ...] = ("de", "en", "it",)

    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
    ) -> None:
        super().__init__(dataset_root, tasks=tasks, splits=splits, languages=languages)
        self.signals_root = self.signals_dir.name

    def to_parquet(self) -> None:
        df = get_from_HF(self.tasks)
        samples = build_samples(
            df,
            tasks=self.tasks,
            splits=self.splits,
            signals_root=self.get_signals_prefix(),
            output_dir=self.dataset_root,
        )

        write_parquets(samples, self.splits, self.parquet_dir)


def add_common_args(
    parser: argparse.ArgumentParser,
    default_tasks: tuple[str, ...],
    default_splits: tuple[str, ...],
    default_languages: tuple[str, ...],
) -> None:
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset dir (default: $DATASETS_DIR/speechlmm-v2/iwslt26)")
    parser.add_argument("--tasks", nargs="+", default=list(default_tasks), help="Tasks to extract")
    parser.add_argument("--splits", nargs="+", default=list(default_splits), help="Splits to build")
    parser.add_argument("--languages", nargs="+", default=list(default_languages), help="Languages to build")
    parser.add_argument("--skip-parquet", action="store_true", help="Reuse already-downloaded parquet")
    parser.add_argument("--skip-download", action="store_true", help="Reuse already-downloaded media")
    parser.add_argument("--skip-filter", action="store_true", help="Skip duration, task, and corrupted-video filtering")
    add_video_integrity_args(parser)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the IWSLT26 corpus.")
    add_common_args(
        parser,
        default_tasks=IWSLT26CorpusPreparer.default_tasks,
        default_languages=IWSLT26CorpusPreparer.default_languages,
        default_splits=IWSLT26CorpusPreparer.default_splits,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = IWSLT26CorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages
    )
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
