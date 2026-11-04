"""Shared skeleton for corpus preparation.

Each corpus subclasses :class:`CorpusPreparer` and supplies its own steps; the base
owns the common pipeline (parse -> optional download -> ShareGPT -> dataset_info) and
the standard dataset layout (``parquet/``, ``signals/``, ``sharegpt/``). Registration
of the ShareGPT files into ``dataset_info.json`` is delegated to
:mod:`speechlmm.data.registry_dataset`.
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import random

from speechlmm.data.filter_parquet import filter
from speechlmm.data.parquet_to_sharegpt import convert_dataset
from speechlmm.data.registry_dataset import write_registry_dataset
from speechlmm.data.utils import datasets_dir, intersection, DEFAULT_SIGNALS_ROOT
from speechlmm.data.versioning import get_versions_or_defaults, DATASETS_VERSION, DEFAULT_CORPUS_VERSION

logger = logging.getLogger(__name__)


class CorpusPreparer(ABC):
    """Base corpus-preparation pipeline.

    Subclasses set ``name`` and implement :meth:`to_parquet` (and override
    :meth:`download` if the corpus has media to fetch). Everything else is shared.
    """

    name: str
    version: str
    default_tasks: tuple[str, ...] = ("achap", "asr", "avspeakd", "lipread", "mt", "slu", "sqa", "ssum", "st", "s2st", "tsum", "tts",)
    default_splits: tuple[str, ...] =  ("train", "dev", "test",)
    default_languages: tuple[str, ...] = ("cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "sv", "de-CZ", "de-DE", "en-US", "en-UK", "es-ES", "fr-FR", "hu-HU", "it-IT", "nl-NL", "pt-PT", "sv-SE",)
    filter_suffix: str = ""
    task_suffix = None
    random_thr = None

    def __init__(
        self,
        dataset_root: Path | None = None,
        tasks: Sequence[str] | None = None,
        splits: Sequence[str] | None = None,
        languages: Sequence[str] | None = None
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root else datasets_dir() / self.name
        task_suffix_str = f"_{self.task_suffix}" if self.task_suffix else ""
        self.parquet_dir = self.dataset_root / f"parquet{task_suffix_str}"
        self.signals_dir = self.dataset_root / f"signals{task_suffix_str}"
        self.sharegpt_dir = self.dataset_root / f"sharegpt{task_suffix_str}"
        self.tasks = tuple(tasks) if tasks is not None else self.default_tasks
        self.splits = tuple(splits) if splits is not None else self.default_splits
        self.languages = tuple(languages) if languages is not None else self.default_languages
#        self.name = "Generic"
        self.version = DEFAULT_CORPUS_VERSION

        self.duration_limit: float | None = 720
        self.task_excluded: list[str] | None = None
        self.check_video_integrity: bool = True
        self.deep_video_check: bool = False
        try:
            self.datasets_root: Path | None = datasets_dir()
        except EnvironmentError:
            # ``dataset_root`` is the corpus dir, so the datasets root cannot be
            # derived from it; leave it unset rather than guess a wrong root.
            self.datasets_root = None
        if self.random_thr is not None and self.random_thr < 1.0:
            random.seed(314)  # for reproducibility of random filtering

    def set_duration_limit(self, value: float | None) -> None:
        self.duration_limit = value

    def set_task_excluded(self, value: list[str] | None) -> None:
        self.task_excluded = value

    def set_check_video_integrity(self, value: bool, deep: bool = False) -> None:
        self.check_video_integrity = value
        self.deep_video_check = deep

    def _media_root(self) -> Path | None:
        """Root the media paths stored in the Parquet are relative to.

        ``video_input_path`` is ``<corpus>/<version>/signals/...`` (see
        :meth:`get_signals_prefix`), i.e. relative to the datasets tree
        (``$DATASETS_DIR`` / ``--dataset-root``), not to ``dataset_root``.
        """
        return self.datasets_root

    def _filter_parquet_dir(self) -> Path:
        return self.dataset_root / f"filtered_{self.parquet_dir.name}"

    def _rejected_parquet_dir(self) -> Path:
        return self.dataset_root / f"rejected_{self.parquet_dir.name}"

    def _filtered_sharegpt_dir(self) -> Path:
        return self.dataset_root / f"filtered_{self.sharegpt_dir.name}"

    @abstractmethod
    def to_parquet(self) -> None:
        """Parse raw annotations into per-split Parquet under ``self.parquet_dir``."""

    def download(self) -> None:
        """Fetch media referenced by the Parquet. No-op unless the corpus overrides it."""

    def parquet_files_for_split(self, split: str) -> Sequence[Path]:
        """Return the Parquet files that contribute to one logical dataset split."""
        return (self.parquet_dir / f"{split}.parquet",)

    def to_sharegpt(self) -> dict[str, str]:
        """Convert each split's Parquet into ShareGPT JSON; returns {key: sharegpt/<file>}."""
        for stale_file in self.sharegpt_dir.glob(f"{self.name}_*.json"):
            stale_file.unlink()

        written: dict[str, str] = {}
        for split in self.splits:
            parquet_files = list(self.parquet_files_for_split(split))
            existing_files = [path for path in parquet_files if path.exists()]
            for missing_file in set(parquet_files) - set(existing_files):
                logger.warning("No parquet for split=%s (%s), skipping", split, missing_file)
            if not existing_files:
                continue
            split_written = convert_dataset(
                existing_files,
                self.sharegpt_dir,
                self.name,
                tasks=self.tasks,
                languages=self.languages,
                split=split,
                random_thr=self.random_thr,
            )
            if split_written:
                written.update(
                    {key: f"{self.sharegpt_dir.name}/{name}" for key, name in split_written.items()}
                )
        return written

    def do_filter(self) -> None:
        if (
            self.duration_limit is None
            and self.task_excluded is None
            and not self.check_video_integrity
        ):
            logger.info("No filter criteria set; skipping filter")
            return

        filtered_parquet_dir = self._filter_parquet_dir()
        rejected_parquet_dir = self._rejected_parquet_dir()
        logger.info(
            "Filtering with duration_limit=%s seconds, task_excluded=%s, check_video_integrity=%s",
            self.duration_limit,
            self.task_excluded,
            self.check_video_integrity,
        )
        filter(
            self.parquet_dir,
            filtered_parquet_dir,
            rejected_parquet_dir,
            self.duration_limit,
            self.task_excluded,
            check_video_integrity=self.check_video_integrity,
            media_root=self._media_root(),
            deep_video_check=self.deep_video_check,
        )
        self.parquet_dir = filtered_parquet_dir
        self.sharegpt_dir = self._filtered_sharegpt_dir()

    def write_info(self, written: dict[str, str], *, replace: bool = True) -> None:
        """Register the written ShareGPT files into this corpus's ``dataset_info.json``."""
        write_registry_dataset(written, self.dataset_root, replace=replace)

    def validate_tasks(self) -> None:
        unsupported_tasks = set(self.tasks) - set(self.default_tasks)
        if unsupported_tasks:
            raise ValueError(
                f"{self.name} only supports {set(self.default_tasks)}, got: {sorted(unsupported_tasks)}"
            )
        
    def prepare(
        self,
        skip_parquet: bool = False,
        skip_download: bool = False,
        skip_filter: bool = False,
        merge_registry: bool = False,
        args: argparse.Namespace | None = None,
    ) -> None:
        if args is not None:
            apply_video_integrity_args(self, args)

        self.validate_tasks()
    
        if skip_parquet:
            logger.info("Skipping parquet generation")
        else:
            self.to_parquet()

        if skip_download:
            logger.info("Skipping media download")
        else:
            self.download()

        if skip_filter:
            logger.info("Skipping filter")
        else:
            self.do_filter()

        written = self.to_sharegpt()
        self.write_info(written, replace=not merge_registry)
        logger.info("%s dataset ready: dataset_dir=%s", self.name, self.dataset_root)


    def additional_setting(
        self,
        argv: list[str] | None = None,
    ) -> None:
        pass

    def set_version(self, version: str) -> None:
        self.version = version

    def get_signals_prefix(self) -> str:
        return f"{self.name}/{self.version}/{DEFAULT_SIGNALS_ROOT}"


def add_video_integrity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-check-video-integrity",
        dest="check_video_integrity",
        action="store_false",
        help="Keep samples whose video file is missing or corrupted (the check is on by default)",
    )
    parser.add_argument(
        "--deep-video-check",
        action="store_true",
        help="Decode every frame when checking videos (slow)",
    )


def apply_video_integrity_args(preparer: "CorpusPreparer", args: argparse.Namespace) -> None:
    if getattr(args, "check_video_integrity", None) is None:
        return
    preparer.set_check_video_integrity(
        args.check_video_integrity, deep=getattr(args, "deep_video_check", False)
    )


def add_common_args(
    parser: argparse.ArgumentParser,
    default_tasks: tuple[str, ...],
    default_splits: tuple[str, ...],
    default_languages: tuple[str, ...],
) -> None:
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset dir (default: $DATASETS_DIR).")
    parser.add_argument("--dataset-version", type=str, default=DATASETS_VERSION, help="Specific version of the dataset described into versions.json' (default is 'v1.0')")
    parser.add_argument("--version-file", type=Path, default=None, help="Json file containing the versions of the corpora (default: None). If not present the default version file 'versions.json' is used.")
    parser.add_argument("--corpora", type=str, nargs="+",  default=None, help="Specific corpora to process")
    parser.add_argument("--tasks", nargs="+", default=list(default_tasks), help="Tasks to export to ShareGPT")
    parser.add_argument("--languages", nargs="+", default=list(default_languages), help="Languages to export to ShareGPT")
    parser.add_argument("--splits", nargs="+", default=list(default_splits), help="Splits to build")
    parser.add_argument("--skip-parquet", action="store_true", help="Reuse already-downloaded parquet")
    parser.add_argument("--skip-download", action="store_true", help="Reuse already-downloaded media")
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip duration, task, and corrupted-video filtering",
    )
    add_video_integrity_args(parser)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare all corpora.")
    add_common_args(
        parser,
        CorpusPreparer.default_tasks,
        CorpusPreparer.default_splits,
        CorpusPreparer.default_languages,
    )

    args, extra_args = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from speechlmm.data.preparers import CORPUS_PREPARERS, preparers_for

    if args.corpora is not None:
        actual_corpus_names = []
        for corpus in args.corpora:
            if corpus in CORPUS_PREPARERS:
                actual_corpus_names.append(corpus)
            else:
                logger.error("The corpus %s cannot be processed because not implemented", corpus)
    else:
        actual_corpus_names = list(CORPUS_PREPARERS)

    commit, corpus_versions = get_versions_or_defaults(
        actual_corpus_names,
        version=args.dataset_version,
        path=args.version_file
    )

    for corpus_name in actual_corpus_names:
        preparer_classes = preparers_for(corpus_name)

        for preparer_cls in preparer_classes:

            splits = intersection(args.splits, preparer_cls.default_splits)
            tasks = intersection(args.tasks, preparer_cls.default_tasks)
            languages = intersection(args.languages, preparer_cls.default_languages)
            corpus_version = corpus_versions[corpus_name]

            if not tasks:
                continue

            logger.info(
                "Preparing %s corpus with tasks=%s, splits=%s, languages=%s, version=%s, commit=%s",
                preparer_cls.name,
                tasks,
                splits,
                languages,
                corpus_version,
                commit
            )

            datasets_root = Path(args.dataset_root) if args.dataset_root else datasets_dir()
            preparer = preparer_cls(
                dataset_root=datasets_root / corpus_name / corpus_version,
                splits=splits,
                tasks=tasks,
                languages=languages
            )
            preparer.datasets_root = datasets_root
            preparer.set_version(corpus_version)
            preparer.additional_setting(argv=extra_args)

            preparer.prepare(
                skip_parquet=args.skip_parquet,
                skip_download=args.skip_download,
                skip_filter=args.skip_filter,
                args=args,
            )


if __name__ == "__main__":
    main()
