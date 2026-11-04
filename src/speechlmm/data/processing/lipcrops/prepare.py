"""Prepare LipCrops visual-speech-recognition data for SpeechLMM.

LipCrops-specific steps (the rest is provided by ``CorpusPreparer``):
  1. filter the source metadata to requested splits, languages, and lipread samples
  2. replace source transcripts with the revised transcripts
  3. write metadata Parquet and convert it to ShareGPT

The lip-crop videos must already be available below ``signals/``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.lipcrops.to_parquet import (
    build_samples,
    get_parquet_from_original,
    get_transcripts_from_original,
    write_parquets,
)

logger = logging.getLogger(__name__)


class LipCropsCorpusPreparer(CorpusPreparer):
    name = "lipcrops"
    default_tasks: tuple[str, ...] = ("lipread",)
    default_splits: tuple[str, ...] = ("train", "dev", "test",)
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
        self.set_duration_limit(120)

    def to_parquet(self) -> None:
        frames = get_parquet_from_original(
            self.dataset_root / "original" / "parquet", self.splits
        )
        transcripts = get_transcripts_from_original(
            self.dataset_root / "original" / "lipcrops_transcripts"
        )

        samples = build_samples(
            df=frames,
            transcripts=transcripts,
            source=self.name,
            tasks=self.tasks,
            splits=self.splits,
            languages=self.languages,
            signals_root=self.get_signals_prefix(),
            dataset_root=self.dataset_root
        )
        write_parquets(samples, self.splits, self.parquet_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the LipCrops corpus.")
    add_common_args(
        parser,
        default_tasks=LipCropsCorpusPreparer.default_tasks,
        default_splits=LipCropsCorpusPreparer.default_splits,
        default_languages=LipCropsCorpusPreparer.default_languages,
    )
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = LipCropsCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages
    )
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
