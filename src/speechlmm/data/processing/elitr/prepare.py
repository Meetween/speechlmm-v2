"""Prepare ELITR meeting summaries for SpeechLMM.

Read local meeting transcript and minutes files, build text-summarization
Parquet for the requested splits and languages, then convert it to ShareGPT.
No media download is required.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.elitr.prepare --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.elitr.to_parquet import build_samples, get_from_original, write_parquets

logger = logging.getLogger(__name__)


class ELITRCorpusPreparer(CorpusPreparer):
    name = "elitr"
    default_tasks: tuple[str, ...] = ("tsum",)
    default_splits: tuple[str, ...] = ("train", "dev", "test", "test2")
    default_languages: tuple[str, ...] = ("cs", "en")

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
        df = get_from_original(self.dataset_root / "original", self.splits, self.languages)
        samples = build_samples(
            df=df,
            source=self.name,
            tasks=self.tasks,
            splits=self.splits,
        )
        write_parquets(samples, self.splits, self.parquet_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the ELITR corpus.")
    add_common_args(
        parser,
        default_tasks=DGTTMCorpusPreparer.default_tasks,
        default_splits=DGTTMCorpusPreparer.default_splits,
        default_languages=DGTTMCorpusPreparer.default_languages,
    )
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = ELITRCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages,
    )
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
