"""Register prebuilt VoxPopuli+EuroSpeech data for SpeechLMM.

This preparer does not download or generate the source metadata. It validates
that the requested Parquet files already exist under the dataset root, then
converts them to ShareGPT and updates the dataset registry.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.voxpopuli_eurospeech.prepare --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.voxpopuli_eurospeech.to_parquet import get_from_original, build_samples, write_parquets

logger = logging.getLogger(__name__)


class VPESCorpusPreparer(CorpusPreparer):
    name = "voxpopuli_eurospeech"
    default_tasks: tuple[str, ...] = ("asr",)
    default_splits: tuple[str, ...] = ("train", "dev", "test")
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

        df = get_from_original(self.dataset_root / "original", self.splits)

        samples = build_samples(
            df=df,
            tasks=self.tasks,
            splits=self.splits,
            signals_root=self.get_signals_prefix()
        )

        write_parquets(samples, self.splits, self.parquet_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the VoxPopuli+EuroSpeech corpus.")
    add_common_args(
        parser,
        default_tasks=VPESCorpusPreparer.default_tasks,
        default_languages=VPESCorpusPreparer.default_languages,
        default_splits=VPESCorpusPreparer.default_splits,
    )
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = VPESCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages
    )
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
