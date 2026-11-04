"""Europarl-ST corpus preparation pipeline.

Europarl-ST-specific steps (the rest is provided by ``CorpusPreparer``):
  1. parse the official ``v1.1/{src}/{tgt}/{split}`` annotations -> metadata Parquet
     (one file per direction+split, tasks ``st`` and ``mt``)
  2. cut per-segment 16 kHz mono ST clips from the speech-level ``.m4a`` audio

The generic Parquet -> ShareGPT JSON (+ dataset_info.json) step lives in the base class.
Europarl-ST ships parallel text for many language pairs. Per-direction Parquet files
are aggregated into language-specific ShareGPT datasets by the base class.

Usage:
    python -m speechlmm.data.prepare_corpus --corpus europarl_st
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path


from speechlmm.data.utils import DEFAULT_SIGNALS_ROOT
from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.europarl_st.cut_audio import cut_all
from speechlmm.data.processing.europarl_st.to_parquet import (
    discover_directions,
    build_samples,
    write_parquets,
)

logger = logging.getLogger(__name__)


class EuroparlSTCorpusPreparer(CorpusPreparer):
    name = "europarl_st"
    europarl_version = "v1.1"
    default_tasks: tuple[str, ...] = ("st", "mt")
    default_splits: tuple[str, ...] = ("train", "dev", "test")
    default_languages: tuple[str, ...] = ("de", "en", "es", "fr", "it", "nl", "pt")

    def __init__(
        self,
        dataset_root: Path | None = None,
        europarl_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        workers: int = 8,
    ) -> None:
        super().__init__(dataset_root, tasks, splits, languages)
        self.europarl_root = self.dataset_root / "original" / self.europarl_version
        self.workers = 8
        self.directions = self._directions()

    def additional_setting(
        self,
        argv: list[str] | None = None,
    ) -> None:
        parser = argparse.ArgumentParser()
        add_common_extra_args(parser)
        args, _ = parser.parse_known_args(argv)
        if args.europarl_root:
            self.europarl_root = args.europarl_root
        if args.workers:
            self.workers = args.workers

    def _directions(self) -> list[tuple[str, str]]:
        return discover_directions(self.europarl_root, self.languages)

    def to_parquet(self) -> None:
        samples = build_samples(
            root=self.europarl_root,
            signals_root=self.get_signals_prefix(),
            directions=self.directions,
            splits=self.splits, 
            tasks=self.tasks
        )

        write_parquets(samples, self.splits, self.parquet_dir)

    def download(self) -> None:
        """Cut per-segment ST clips (only needed for the ST task)."""
        if "st" not in self.tasks:
            logger.info("ST not requested; skipping audio cutting")
            return
        cut_all(
            self.europarl_root,
            self.signals_dir,
            self.directions,
            self.splits,
            workers=self.workers,
        )


def add_common_extra_args(
    parser: argparse.ArgumentParser
) -> None:
    parser.add_argument(
        "--europarl-root",
        type=Path,
        default=None,
        help="Extracted Europarl-ST dir with <lang>/ subdirs (default: $DATASETS_DIR/speechlmm-v2/europarl_st/original/v1.1)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the Europarl-ST corpus (ST + MT).")
    add_common_args(
        parser,
        default_tasks=EuroparlSTCorpusPreparer.default_tasks,
        default_languages=EuroparlSTCorpusPreparer.default_languages,
        default_splits=EuroparlSTCorpusPreparer.default_splits,
    )
    args, extra_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = EuroparlSTCorpusPreparer(
        dataset_root=args.dataset_root,
        languages=args.languages,
        splits=args.splits,
        tasks=args.tasks
    )
    preparer.additional_setting(args=extra_args)
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
