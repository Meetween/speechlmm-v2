"""Prepare Common Voice ASR data for SpeechLMM.

Read locale TSVs from an extracted Common Voice release, write metadata
Parquet, link each locale's ``clips/`` directory under ``signals/``, and
convert the result to ShareGPT. The gated Common Voice release must already
exist locally.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.common_voice.prepare \\
        --cv-root <cv-corpus-root> --languages en
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.common_voice.to_parquet import (
    DEFAULT_SIGNALS_ROOT,
    LANGUAGES,
    SPLITS,
    build_samples,
    cv_locale,
    write_parquets,
)

logger = logging.getLogger(__name__)


class CommonVoiceCorpusPreparer(CorpusPreparer):
    name = "common_voice"
    cv_version = "cv-corpus-26.0-2026-06-12"
    default_tasks: tuple[str, ...] = ("asr",)
    default_splits: tuple[str, ...] = tuple(SPLITS)
    default_languages: tuple[str, ...] = tuple(LANGUAGES)

    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
    ) -> None:
        super().__init__(dataset_root, tasks, splits, languages)
        self.cv_root = self.dataset_root / "original" / self.cv_version

    def additional_setting(
        self,
        argv: list[str] | None = None,
    ) -> None:
        parser = argparse.ArgumentParser()
        add_common_extra_args(parser)
        args, _ = parser.parse_known_args(argv)
        if args.cv_root:
            self.cv_root = args.cv_root

    def to_parquet(self) -> None:
        samples = build_samples(
            root=self.cv_root,
            languages=self.languages,
            splits=self.splits,
            signals_root=self.get_signals_prefix(),
        )
        write_parquets(samples, self.splits, self.parquet_dir)

    def download(self) -> None:
        """Link each locale's clips directory under ``signals/<language>``."""
        for language in self.languages:
            clips_dir = self.cv_root / cv_locale(language) / "clips"
            if not clips_dir.is_dir():
                logger.warning("No clips dir for %s (%s)", language, clips_dir)
                continue
            link = self.signals_dir / language
            self.signals_dir.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                continue
            link.symlink_to(os.path.relpath(clips_dir, self.signals_dir))


def add_common_extra_args(
    parser: argparse.ArgumentParser
) -> None:
    parser.add_argument("--cv-root", type=Path, default=None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the Common Voice corpus (ASR).")
    add_common_args(
        parser,
        default_tasks=AmiCorpusPreparer.default_tasks,
        default_languages=AmiCorpusPreparer.default_languages,
        default_splits=AmiCorpusPreparer.default_splits,
    )
    args, extra_args = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = CommonVoiceCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        languages=args.languages,
        splits=args.splits,
    )
    preparer.additional_setting(args=extra_args)
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
