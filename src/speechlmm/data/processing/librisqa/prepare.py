"""Prepare LibriSQA multiple-choice data for SpeechLMM.

Read normalized LibriSQA Part II JSONL, build text-only SQA Parquet for the
requested splits and languages, then convert it to ShareGPT. No media
download is required.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.librisqa.prepare --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.librisqa.to_parquet import get_from_original, build_samples, write_parquets

logger = logging.getLogger(__name__)


class LIBRISQACorpusPreparer(CorpusPreparer):
    name = "librisqa"
    default_tasks: tuple[str, ...] = ("sqa",)
    default_splits: tuple[str, ...] = ("train", "test")
    default_languages: tuple[str, ...] = ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv")

    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
    ) -> None:
        super().__init__(dataset_root, tasks=tasks, splits=splits, languages=languages)

    def to_parquet(self) -> None:
        df = get_from_original(self.dataset_root / "original")

        samples = build_samples(
            df=df,
            source=self.name,
            tasks=self.tasks,
            splits=self.splits,
            languages=self.languages
        )
        write_parquets(samples, self.splits, self.parquet_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the LibriSQA corpus.")
    add_common_args(
        parser,
        default_tasks=LIBRISQACorpusPreparer.default_tasks,
        default_languages=LIBRISQACorpusPreparer.default_languages,
        default_splits=LIBRISQACorpusPreparer.default_splits,
    )
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = LIBRISQACorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages
    )
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
