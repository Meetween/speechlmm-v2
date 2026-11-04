"""Prepare TopicalChat data for SpeechLMM.

Normalize locally available source Parquet into per-split ASR, speech
translation, and machine translation metadata, then convert it to ShareGPT.
Referenced media must already exist under the dataset root.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.topical_chat.prepare --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args

from speechlmm.data.processing.topical_chat.to_parquet import create_mt_parquet, get_from_original, build_samples, write_parquets

logger = logging.getLogger(__name__)


class TopicalChatCorpusPreparer(CorpusPreparer):
    name = "topical_chat"
    default_tasks: tuple[str, ...] = ("asr", "mt", "st")
    default_splits: tuple[str, ...] = ("train", "dev", "test")
    default_languages: tuple[str, ...] = ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv")

    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        skip_mt_create: bool = False,
    ) -> None:
        super().__init__(dataset_root, tasks=tasks, splits=splits, languages=languages)
        self.skip_mt_create = skip_mt_create

    def additional_setting(self, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--skip-mt-create", action="store_true")
        args, _ = parser.parse_known_args(argv)
        if args.skip_mt_create:
            self.skip_mt_create = True

    def to_parquet(self) -> None:
        if "mt" in self.tasks and not self.skip_mt_create:
            create_mt_parquet(self.dataset_root / "original")

        df = get_from_original(self.dataset_root / "original", self.splits, self.tasks)

        samples = build_samples(
            df=df,
            source=self.name,
            tasks=self.tasks,
            splits=self.splits,
            output_dir=self.dataset_root,
            signals_root=self.get_signals_prefix(),
        )

        write_parquets(samples, self.splits, self.parquet_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the TopicalChat corpus.")
    add_common_args(
        parser,
        default_tasks=TopicalChatCorpusPreparer.default_tasks,
        default_splits=TopicalChatCorpusPreparer.default_splits,
        default_languages=TopicalChatCorpusPreparer.default_languages,
    )
    parser.add_argument("--skip-mt-create", action="store_true", help="Reuse already-generated MT original parquet")
    args, extra_args = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = TopicalChatCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages,
        skip_mt_create=args.skip_mt_create,
    )
    preparer.additional_setting(argv=extra_args)
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
