"""Utilities to detect corrupted / unreadable video files.

Two backends are used, in this order:

1. ``torchcodec`` (already a dependency of this project), which decodes the
   file through the FFmpeg libraries linked into the wheel.
2. the ``ffmpeg`` command line tool, if it happens to be on ``PATH``.

A file is reported as corrupted when it is missing or empty, when the container
cannot be opened, when it carries no video stream, or when the frames that are
actually decoded raise an error.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 8

# Every failure carrying this reason means the path itself never resolved, which
# is how a wrong media root looks from here.
MISSING_FILE_REASON = "file does not exist"


@dataclass
class VideoCheck:
    """Outcome of a single video inspection."""

    path: Path
    ok: bool
    reason: str | None = None  # why it was rejected, ``None`` when ok
    backend: str | None = None


def check_video(
    path: str | Path,
    deep: bool = False,
    timeout: float | None = 300.0,
) -> VideoCheck:
    """Check whether ``path`` is a readable, non-corrupted video file.

    Args:
        path: file to inspect.
        deep: when ``False`` (default) only the container metadata plus the
            first and last frames are decoded, which is enough to catch
            truncated or zero-length files and costs a few milliseconds. When
            ``True`` every frame is decoded, which catches corruption in the
            middle of the file but is as slow as playing the whole video.
        timeout: seconds allowed to the ``ffmpeg`` subprocess, when that
            backend is used. ``None`` disables the timeout.

    Returns:
        A :class:`VideoCheck` whose ``ok`` is False when the file is corrupted.
    """
    path = Path(path)

    if not path.exists():
        return VideoCheck(path, ok=False, reason=MISSING_FILE_REASON)
    if not path.is_file():
        return VideoCheck(path, ok=False, reason="not a regular file")
    if path.stat().st_size == 0:
        return VideoCheck(path, ok=False, reason="file is empty")

    if deep and shutil.which("ffmpeg") is not None:
        return _check_with_ffmpeg(path, timeout=timeout, deep=True)

    try:
        return _check_with_torchcodec(path, deep=deep)
    except ImportError:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "neither torchcodec nor the ffmpeg executable is available"
            )
        return _check_with_ffmpeg(path, timeout=timeout, deep=deep)


def check_videos(
    paths: list[str | None],
    media_root: Path | None = None,
    deep: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, VideoCheck]:
    """Check many videos concurrently, keyed by the path as it was given.

    Duplicate paths are inspected once, and null or empty paths are skipped, so
    the returned mapping may be smaller than ``paths``. Relative paths are
    resolved against ``media_root`` when it is given (metadata Parquet stores
    them relative to the datasets root).
    """
    unique_paths = sorted(
        {path for path in paths if isinstance(path, str) and path}
    )
    if not unique_paths:
        return {}

    def resolve(path: str) -> Path:
        candidate = Path(path)
        if media_root is None or candidate.is_absolute():
            return candidate
        return Path(media_root) / candidate

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = executor.map(
            lambda path: check_video(resolve(path), deep=deep), unique_paths
        )
        return dict(zip(unique_paths, results))


def _check_with_torchcodec(path: Path, deep: bool) -> VideoCheck:
    from torchcodec.decoders import VideoDecoder

    try:
        decoder = VideoDecoder(str(path))
        num_frames = len(decoder)

        if num_frames < 1:
            return VideoCheck(
                path,
                ok=False,
                reason="no decodable video frame",
                backend="torchcodec",
            )

        indices = range(num_frames) if deep else {0, num_frames - 1}
        for index in indices:
            decoder[index]
    except Exception as exception:  # any decoding failure means "corrupted"
        return VideoCheck(
            path,
            ok=False,
            reason=f"{type(exception).__name__}: {exception}",
            backend="torchcodec",
        )

    return VideoCheck(path, ok=True, backend="torchcodec")


def _check_with_ffmpeg(path: Path, timeout: float | None, deep: bool = True) -> VideoCheck:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
    ]
    if not deep:
        command.extend(["-frames:v", "1"])
    command.extend(["-f", "null", "-"])
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return VideoCheck(
            path, ok=False, reason="ffmpeg timed out", backend="ffmpeg"
        )

    errors = process.stderr.strip()
    if process.returncode != 0 or errors:
        return VideoCheck(
            path,
            ok=False,
            reason=errors or f"ffmpeg exited with code {process.returncode}",
            backend="ffmpeg",
        )
    return VideoCheck(path, ok=True, backend="ffmpeg")
