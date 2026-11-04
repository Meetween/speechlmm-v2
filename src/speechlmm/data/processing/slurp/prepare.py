"""Prepare SLURP spoken-language-understanding data for SpeechLMM.

Load the requested Hugging Face splits, build intent-classification Parquet,
materialize each utterance's audio under ``signals/``, and convert the result
to ShareGPT.

Usage:
    PYTHONPATH=src python -m speechlmm.data.processing.slurp.prepare --help
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.slurp.to_parquet import get_from_HF, build_samples, write_parquets

logger = logging.getLogger(__name__)


class SlurpCorpusPreparer(CorpusPreparer):
    name = "slurp"
    default_tasks: tuple[str, ...] = ("slu",)
    default_splits: tuple[str, ...] = ("train", "devel", "test", "train_synthetic")
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

    def to_parquet(self) -> None:
        df = get_from_HF(self.splits, self.languages)

        samples = build_samples(
            df,
            tasks=self.tasks,
            splits=self.splits,
            signals_root=self.get_signals_prefix(),
            output_dir=self.dataset_root,
        )

        write_parquets(samples, self.splits, self.parquet_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the SLURP corpus.")
    add_common_args(
        parser,
        default_tasks=SlurpCorpusPreparer.default_tasks,
        default_languages=SlurpCorpusPreparer.default_languages,
        default_splits=SlurpCorpusPreparer.default_splits,
    )
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = SlurpCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages
    )
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
