"""FLEURS corpus preparation pipeline.

FLEURS-specific steps (the rest is provided by ``CorpusPreparer``):
  1. parse FLEURS XML annotations -> metadata Parquet (per split)
  2. download Mix-Headset WAVs referenced by the Parquet

The generic Parquet -> ShareGPT JSON (+ dataset_info.json) step lives in the base class.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer

from speechlmm.data.processing.fleurs.download import (
    FLEURS_MIRROR,
    download_tsv,
    download_audios,
)
from speechlmm.data.processing.fleurs.to_parquet import (
    build_samples,
    enrich_audio_metadata,
    write_parquets,
)

logger = logging.getLogger(__name__)


class FleursCorpusPreparer(CorpusPreparer):
    name = "fleurs"
    default_tasks: tuple[str, ...] = ("asr",)
    default_splits: tuple[str, ...] = ("train", "dev", "test")
    default_languages: tuple[str, ...] = ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv",)
    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
    ) -> None:
        super().__init__(dataset_root, tasks, splits, languages)
        self.tsv_dir = self.dataset_root / "tsv"
        self.workers = 4

    def additional_setting(
        self,
        argv: list[str] | None = None,
    ) -> None:
        parser = argparse.ArgumentParser()
        add_common_extra_args(parser)
        args, _ = parser.parse_known_args(argv)
        if args.tsv_dir:
            self.tsv_dir = args.tsv_dir
        if args.workers:
            self.workers = args.workers

    def to_parquet(self) -> None:
        download_tsv(self.tsv_dir, self.splits, self.languages, base_url=FLEURS_MIRROR)

        samples = build_samples(
            self.tsv_dir,
            self.tasks,
            self.splits,
            self.languages,
            self.dataset_root,
            signals_root=self.get_signals_prefix()
        )
        write_parquets(samples, self.splits, self.parquet_dir)

    def download(self) -> None:
        download_audios(self.signals_dir, self.splits, self.languages)
        enrich_audio_metadata(self.parquet_dir, self.dataset_root, self.splits, self.workers)


def add_common_extra_args(
    parser: argparse.ArgumentParser
) -> None:
    parser.add_argument(
        "--tsv-dir",
        type=Path,
        default=None,
        help="Extracted FLEURS tsv dir (default: $DATASETS_DIR/speechlmm-v2/fleurs/original/tsv)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the FLEURS corpus.")
    add_common_args(
        parser,
        default_tasks=FleursCorpusPreparer.default_tasks,
        default_languages=FleursCorpusPreparer.default_languages,
        default_splits=FleursCorpusPreparer.default_splits,
    )
    args, extra_args = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = FleursCorpusPreparer(
        dataset_root=args.dataset_root,
        splits=args.splits,
        tasks=args.tasks,
        languages=args.languages,
        workers=args.workers,
    )
    preparer.additional_setting(args=extra_args)
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
