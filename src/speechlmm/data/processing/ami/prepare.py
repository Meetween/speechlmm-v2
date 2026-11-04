"""AMI corpus preparation pipeline.

AMI-specific steps (the rest is provided by ``CorpusPreparer``):
  1. parse AMI XML annotations -> metadata Parquet (per split)
  2. download audio/video signals referenced by the Parquet

The generic Parquet -> ShareGPT JSON (+ dataset_info.json) step lives in the base class.
Three preparers share one dataset root. asr/ssum/tsum are one row per meeting.
achap and avspeakd are chunked from the same XML (12 minute chapter windows,
2 minute speaker windows) and written under ``parquet_achap`` / ``parquet_avspeakd``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

from speechlmm.data.corpus_preparer import CorpusPreparer, add_common_args
from speechlmm.data.processing.ami.chunks import (
    AVSPEAKD_MAX_SECONDS,
    AmiChunk,
    achap_chunks,
    avspeakd_chunks,
    chunks_to_rows,
)
from speechlmm.data.processing.ami.download import (
    download_signals,
    materialize_chunks,
    media_paths_from_parquet,
)
from speechlmm.data.processing.ami.to_parquet import (
    build_samples_ami,
    enrich_video_metadata,
    write_parquets,
)
from speechlmm.data.utils import intersection

logger = logging.getLogger(__name__)


class AmiCommonCorpusPreparer(CorpusPreparer):
    name = "ami"
    license = "CC-BY-4.0"
    source = "AMI"
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
        super().__init__(dataset_root, tasks=tasks, splits=splits, languages=languages)
        logger.info(f"self.name: {self.name}")

    def additional_setting(
        self,
        argv: list[str] | None = None,
    ) -> None:
        parser = argparse.ArgumentParser()
        add_common_extra_args(parser)
        args, _ = parser.parse_known_args(argv)
        self.ami_annotations = args.ami_annotations if args.ami_annotations else self.dataset_root / "original" / self.name / "manual"
        self.partition = args.partition if args.partition else "scenario"
        self.workers = args.workers if args.workers else 4
        self.include_unsplit = args.include_unsplit if args.include_unsplit else False

    def _signals_root_for(self, suffix: str | None = None) -> str:
        prefix = self.get_signals_prefix()
        return f"{prefix}_{suffix}" if suffix else prefix

    def _source_signals(self) -> Path:
        return self.dataset_root / "signals"


class AmiCorpusPreparer(AmiCommonCorpusPreparer):
    default_tasks: tuple[str, ...] = ("asr", "ssum", "tsum")

    def to_parquet(self) -> None:
        samples = build_samples_ami(
            ami_root=self.ami_annotations,
            signals_root=self.get_signals_prefix(),
            partition=self.partition,
            include_unsplit=self.include_unsplit,
        )
        write_parquets(samples, self.splits, self.parquet_dir)
        enrich_video_metadata(
            self.parquet_dir,
            self.dataset_root,
            self.splits,
            self.workers,
        )

    def download(self) -> None:
        wanted = media_paths_from_parquet(self.parquet_dir)
        if not wanted:
            logger.warning("No media paths in %s; skipping download", self.parquet_dir)
            return
        download_signals(
            self.parquet_dir,
            self.signals_dir,
            workers=self.workers,
        )


class AmiChunkCorpusPreparer(AmiCommonCorpusPreparer):
    task_suffix: str

    def _build_chunks(self) -> list[AmiChunk]:
        raise NotImplementedError

    @cached_property
    def chunks(self) -> list[AmiChunk]:
        return self._build_chunks()

    def to_parquet(self) -> None:
        samples = chunks_to_rows(self.chunks, self._signals_root_for(self.task_suffix))
        write_parquets(samples, self.splits, self.parquet_dir)

    def download(self) -> None:
        if self.datasets_root is None:
            raise OSError("Set DATASETS_DIR so chunk media can be written")
        materialize_chunks(
            self.chunks,
            datasets_root=self.datasets_root,
            source_signals=self._source_signals(),
            signals_root=self._signals_root_for(self.task_suffix),
            workers=self.workers,
        )


class AmiAVSpeakDCorpusPreparer(AmiChunkCorpusPreparer):
    default_tasks: tuple[str, ...] = ("avspeakd",)
    task_suffix = "avspeakd"

    def additional_setting(
        self,
        argv: list[str] | None = None,
    ) -> None:
        super().additional_setting(argv)
        self.set_duration_limit(AVSPEAKD_MAX_SECONDS)

    def _build_chunks(self) -> list[AmiChunk]:
        return avspeakd_chunks(
            self.ami_annotations,
            partition=self.partition,
            include_unsplit=self.include_unsplit,
            splits=self.splits,
        )


class AmiAChapCorpusPreparer(AmiChunkCorpusPreparer):
    default_tasks: tuple[str, ...] = ("achap",)
    task_suffix = "achap"

    def _build_chunks(self) -> list[AmiChunk]:
        return achap_chunks(
            self.ami_annotations,
            partition=self.partition,
            include_unsplit=self.include_unsplit,
            splits=self.splits,
        )


def add_common_extra_args(
    parser: argparse.ArgumentParser
) -> None:

    # parameters specific to AmiCorpusPreparer
    parser.add_argument("--partition", choices=("scenario", "full", "full_asr"), default="scenario",
        help="AMI official split partition")
    parser.add_argument("--ami-annotations", type=Path, default=None,
        help="AMI XML root (default: <dataset-root>/original/ami/manual)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--include-unsplit", action="store_true", help="Include meetings with no official AMI split")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the AMI corpus.")
    add_common_args(
        parser,
        default_tasks=AmiCorpusPreparer.default_tasks,
        default_languages=AmiCorpusPreparer.default_languages,
        default_splits=AmiCorpusPreparer.default_splits,
    )
    args, extra_args = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    preparer_specs: list[tuple[type[AmiCorpusPreparer], tuple[str, ...], bool]] = [
        (AmiCorpusPreparer, AmiCorpusPreparer.default_tasks, False),
        (AmiAVSpeakDCorpusPreparer, AmiAVSpeakDCorpusPreparer.default_tasks, False),
        (AmiAChapCorpusPreparer, AmiAChapCorpusPreparer.default_tasks, False),
    ]

    for index, (preparer_cls, supported_tasks, skip_data_steps) in enumerate(preparer_specs):
        tasks = intersection(args.tasks, supported_tasks)
        if not tasks:
            continue

        preparer = preparer_cls(
            dataset_root=args.dataset_root,
            splits=args.splits,
            tasks=tasks,
            languages=args.languages
        )
        preparer.additional_setting(argv=extra_args)
                
        preparer.prepare(
            skip_parquet=args.skip_parquet or skip_data_steps,
            skip_download=args.skip_download or skip_data_steps,
            skip_filter=args.skip_filter,
            merge_registry=index > 0,
            args=args,
        )


if __name__ == "__main__":
    main()
