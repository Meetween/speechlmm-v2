"""Filter metadata Parquet rows by duration and/or task."""

from __future__ import annotations

import argparse
import logging
from functools import reduce
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from speechlmm.data.utils import datasets_dir, resolve_parquet_dir
from speechlmm.data.video_integrity import (
    DEFAULT_WORKERS,
    MISSING_FILE_REASON,
    check_videos,
)

logger = logging.getLogger(__name__)

VIDEO_PATH_COLUMN = "video_input_path"


def audio_video_condition(table: pa.Table, limit: float) -> list:
    audio_condition = pc.less_equal(
        pc.fill_null(table["audio_input_duration"], 0), limit * 1.001
    )
    video_condition = pc.less_equal(
        pc.fill_null(table["video_input_duration"], 0), limit * 1.001
    )
    return [audio_condition, video_condition]


def task_condition(table: pa.Table, values: list[str]) -> list[pa.Array]:
    conditions: list[pa.Array] = []
    for value in values:
        mask = pa.array(
            [lst is not None and value not in lst for lst in table["task"].to_pylist()]
        )
        conditions.append(mask)
    return conditions


def video_integrity_condition(
    table: pa.Table,
    media_root: Path | None,
    deep: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> list[pa.Array]:
    """Reject rows whose ``video_input_path`` points at a corrupted video.

    Rows whose video path is null or empty (audio-only samples) are kept. Each
    distinct file is decoded once, however many rows reference it.
    """
    if VIDEO_PATH_COLUMN not in table.column_names:
        return []

    paths = table[VIDEO_PATH_COLUMN].to_pylist()
    checks = check_videos(paths, media_root=media_root, deep=deep, workers=workers)
    if not checks:
        return []

    corrupted = {path for path, check in checks.items() if not check.ok}
    for path in sorted(corrupted):
        check = checks[path]
        logger.warning(
            "corrupted video: %s (%s, backend=%s)",
            path,
            check.reason,
            check.backend or "path",  # no backend ran: rejected on the path itself
        )
    # Genuinely corrupted files still resolve on disk, so only a run where every
    # single path is missing points at a wrong root rather than at bad media.
    if corrupted and all(
        check.reason == MISSING_FILE_REASON for check in checks.values()
    ):
        raise RuntimeError(
            f"none of the {len(checks)} videos exist under media_root={media_root}: "
            "is it correct? Pass --media-root or --no-check-video-integrity."
        )

    # ``video_input_path`` is null for audio-only samples: keep those rows.
    mask = pa.array(
        [not isinstance(path, str) or path not in corrupted for path in paths],
        type=pa.bool_(),
    )
    return [mask]


def filter_table(
    file: Path,
    filtered_file: Path,
    rejected_file: Path,
    duration_limit: float | None,
    task_excluded: list[str] | None,
    check_video_integrity: bool = True,
    media_root: Path | None = None,
    deep_video_check: bool = False,
    video_check_workers: int = DEFAULT_WORKERS,
) -> None:
    table = pq.read_table(file)

    conditions: list = []
    if duration_limit is not None:
        conditions.extend(audio_video_condition(table, duration_limit))
    if task_excluded is not None:
        conditions.extend(task_condition(table, task_excluded))
    if check_video_integrity:
        conditions.extend(
            video_integrity_condition(
                table,
                media_root,
                deep=deep_video_check,
                workers=video_check_workers,
            )
        )

    if not conditions:
        filtered_table = table
        rejected_table = table.slice(0, 0)
    else:
        mask = pc.fill_null(reduce(pc.and_kleene, conditions), False)
        filtered_table = table.filter(mask)
        rejected_table = table.filter(pc.invert(mask))

    logger.info(
        "size: %d filtered: %d rejected: %d file: %s",
        table.num_rows,
        filtered_table.num_rows,
        rejected_table.num_rows,
        file,
    )
    pq.write_table(filtered_table, filtered_file)
    pq.write_table(rejected_table, rejected_file)


def filter(
    parquet_dir: Path,
    filtered_parquet_dir: Path,
    rejected_parquet_dir: Path,
    duration_limit: float | None,
    task_excluded: list[str] | None,
    check_video_integrity: bool = True,
    media_root: Path | None = None,
    deep_video_check: bool = False,
    video_check_workers: int = DEFAULT_WORKERS,
) -> None:
    filtered_parquet_dir.mkdir(parents=True, exist_ok=True)
    rejected_parquet_dir.mkdir(parents=True, exist_ok=True)
    for file in sorted(parquet_dir.glob("*.parquet")):
        filter_table(
            file,
            filtered_parquet_dir / file.name,
            rejected_parquet_dir / file.name,
            duration_limit,
            task_excluded,
            check_video_integrity=check_video_integrity,
            media_root=media_root,
            deep_video_check=deep_video_check,
            video_check_workers=video_check_workers,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Filter metadata Parquet by audio/video duration, task, and/or corrupted video."
    )
    parser.add_argument(
        "--corpus",
        help="Corpus name under $DATASETS_DIR/speechlmm-v2 (e.g. europarl_st)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset dir (or a parquet dir) to read instead",
    )
    parser.add_argument(
        "--input-duration-limit",
        type=int,
        help="Audio/video duration limit in seconds (default: 720)",
    )
    parser.add_argument(
        "--task-excluded",
        nargs="+",
        help="Tasks to exclude from the filtered output",
    )
    parser.add_argument(
        "--no-check-video-integrity",
        dest="check_video_integrity",
        action="store_false",
        help="Keep samples whose video file is missing or corrupted (the check is on by default)",
    )
    parser.add_argument(
        "--deep-video-check",
        action="store_true",
        help="Decode every frame instead of just the first and the last one (slow)",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        help="Root the video paths are relative to (default: $DATASETS_DIR)",
    )
    parser.add_argument(
        "--video-check-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Threads used to check videos (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _, parquet_dir = resolve_parquet_dir(args.corpus, args.dataset_root)
    _, filtered_parquet_dir = resolve_parquet_dir(args.corpus, args.dataset_root, "filtered_")
    _, rejected_parquet_dir = resolve_parquet_dir(args.corpus, args.dataset_root, "rejected_")
    if not parquet_dir.is_dir():
        raise SystemExit(f"No parquet dir at {parquet_dir}")

    media_root = args.media_root
    if args.check_video_integrity and media_root is None:
        try:
            media_root = datasets_dir()
        except EnvironmentError as exception:
            raise SystemExit(
                f"{exception}, or pass --media-root, or disable the video check "
                "with --no-check-video-integrity"
            )

    filter(
        parquet_dir,
        filtered_parquet_dir,
        rejected_parquet_dir,
        args.input_duration_limit,
        args.task_excluded,
        check_video_integrity=args.check_video_integrity,
        media_root=media_root,
        deep_video_check=args.deep_video_check,
        video_check_workers=args.video_check_workers,
    )


if __name__ == "__main__":
    main()
