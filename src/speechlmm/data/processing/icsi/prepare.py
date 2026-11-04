"""Prepare segmented and unsegmented ICSI data for SpeechLMM.

Build speech- and text-summarization Parquet from local ICSI Arrow or
Parquet sources, materialize referenced audio, and register the segmented
and unsegmented variants as separate ShareGPT datasets.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.icsi.prepare --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.icsi.to_parquet import get_from_arrow, get_from_parquet, build_samples, write_parquets
from speechlmm.data.utils import datasets_dir, intersection, DEFAULT_SIGNALS_ROOT

logger = logging.getLogger(__name__)

class ICSICorpusPreparer(CorpusPreparer):
    name = "icsi"
    default_tasks: tuple[str, ...] = ("tsum", "ssum")
    default_splits: tuple[str, ...] = ("train", "validation", "test")
    default_languages: tuple[str, ...] = ("en",)

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

    def get_signals_prefix(self) -> str:
        return f"{ICSICorpusPreparer.name}/{self.version}/{DEFAULT_SIGNALS_ROOT}"


class ICSIUnsegmentedCorpusPreparer(ICSICorpusPreparer):
    name = "icsi_unsegmented"

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
        df = get_from_arrow(self.dataset_root / "original" / "arrow", self.splits)

        samples = build_samples(
            df,
            source=self.name,
            tasks=self.tasks,
            splits=self.splits,
            languages=self.languages,
            signals_root=self.get_signals_prefix(),
            output_dir=self.dataset_root,
        )

        write_parquets(samples, self.splits, self.parquet_dir)


class ICSISegmentedCorpusPreparer(ICSICorpusPreparer):
    name = "icsi_segmented"

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
        df = get_from_parquet(self.dataset_root / "original" / "parquet", self.splits)

        samples = build_samples(
            df,
            source=self.name,
            tasks=self.tasks,
            splits=self.splits,
            languages=self.languages,
            signals_root=self.get_signals_prefix(),
            output_dir=self.dataset_root,
        )

        write_parquets(samples, self.splits, self.parquet_dir)

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the ICSI corpus.")
    add_common_args(
        parser,
        default_tasks=ICSICorpusPreparer.default_tasks,
        default_languages=ICSICorpusPreparer.default_languages,
        default_splits=ICSICorpusPreparer.default_splits,
    )
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    preparer_specs: list[tuple[type[ICSICorpusPreparer], tuple[str, ...], bool]] = [
        (ICSISegmentedCorpusPreparer, ICSISegmentedCorpusPreparer.default_tasks, False),
        (ICSIUnsegmentedCorpusPreparer, ICSIUnsegmentedCorpusPreparer.default_tasks, False)
    ]
    
    for index, (preparer_cls, supported_tasks, skip_data_steps) in enumerate(preparer_specs):
        tasks = intersection(args.tasks, supported_tasks)
        if not tasks:
            continue

        preparer = preparer_cls(
            dataset_root=args.dataset_root,
            splits=args.splits,
            tasks=tasks,
            languages=args.languages,
        )
        preparer.prepare(
            skip_parquet=args.skip_parquet or skip_data_steps,
            skip_download=args.skip_download or skip_data_steps,
            skip_filter=args.skip_filter,
            merge_registry=index > 0,
            args=args,
        )


if __name__ == "__main__":
    main()
