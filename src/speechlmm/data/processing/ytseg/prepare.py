"""YTSeg corpus preparation pipeline.

YTSeg-specific steps (the rest is provided by ``CorpusPreparer``):
  1. build metadata Parquet from the Hugging Face ``text`` config (per split)
  2. download the 16 kHz mono MP3 audio from the Hugging Face ``audio`` config

Both steps read from the Hugging Face Hub, so run this where there is internet.
``ytseg_cc_by`` reuses this pipeline via :class:`YTSegCorpusPreparer`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.ytseg.download import download_audio, video_ids_from_parquet
from speechlmm.data.processing.ytseg.to_parquet import build_samples, write_parquets

logger = logging.getLogger(__name__)


class YTSegCorpusPreparer(CorpusPreparer):
    name = "ytseg"
    dataset_id = "retkowski/ytseg"
    license = "CC-BY-NC-SA-4.0"
    source = "YTSeg"
    default_tasks: tuple[str, ...] = ("asr", "achap")
    default_splits: tuple[str, ...] = ("train", "val", "test")
    default_languages: tuple[str, ...] = ("en",)

    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None
    ) -> None:
        super().__init__(dataset_root, tasks, splits, languages)
        self.max_samples = None

    def additional_setting(
            self,
            argv: list[str] | None = None,
        ) -> None:
        parser = argparse.ArgumentParser()
        add_common_extra_args(parser)
        args, _ = parser.parse_known_args(argv)
        if args.max_samples:
            self.max_samples = args.max_samples

    def to_parquet(self) -> None:
        samples = build_samples(
            self.dataset_id,
            splits=self.splits,
            source=self.source,
            license=self.license,
            signals_root=self.get_signals_prefix(),
            max_samples=self.max_samples,
        )
        write_parquets(samples, self.splits, self.parquet_dir)

    def download(self) -> None:
        wanted = video_ids_from_parquet(self.parquet_dir)
        download_audio(self.dataset_id, self.splits, self.signals_dir, wanted_ids=wanted or None)


def add_common_extra_args(
    parser: argparse.ArgumentParser
) -> None:
    parser.add_argument("--max-samples", type=int, default=None, help="Limit videos per split (debug)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the YTSeg corpus.")
    add_common_args(
        parser,
        default_tasks=YTSegCorpusPreparer.default_tasks,
        default_languages=YTSegCorpusPreparer.default_languages,
        default_splits=YTSegCorpusPreparer.default_splits,
    )
    args, extra_args = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer = YTSegCorpusPreparer(
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        splits=args.splits,
        languages=args.languages
    )
    preparer.additional_setting(args=extra_args)
    preparer.prepare(skip_parquet=args.skip_parquet, skip_download=args.skip_download, skip_filter=args.skip_filter, args=args)


if __name__ == "__main__":
    main()
