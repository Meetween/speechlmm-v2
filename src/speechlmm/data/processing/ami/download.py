"""Download AMI media referenced by metadata Parquet into ``signals/``.

Reads ``audio_input_path`` and ``video_input_path`` from the Parquet files and
fetches each file from the official Edinburgh AMI mirror (same source as Kaldi,
Lhotse, ESPnet). Skips files that already exist locally unless ``force`` is set.

Usage:
    PYTHONPATH=src python3 -m speechlmm.data.processing.ami.download
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from speechlmm.data.processing.ami.chunks import (
    AVSPEAKD_VIDEO_FPS,
    AVSPEAKD_VIDEO_HEIGHT,
    AVSPEAKD_VIDEO_WIDTH,
    AmiChunk,
    chunk_audio_path,
    chunk_video_path,
)
from speechlmm.data.utils import datasets_dir

logger = logging.getLogger(__name__)

AMI_MIRROR = "http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"


def media_paths_from_parquet(parquet_dir: Path) -> list[str]:
    """Collect unique relative media paths under ``signals/`` from Parquet."""
    paths: set[str] = set()
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path)
        for col in ("audio_input_path", "video_input_path"):
            if col not in table.column_names:
                continue
            for path in table.column(col).to_pylist():
                if path:
                    paths.add(path.replace("\\", "/"))
    return sorted(paths)


def _signals_relative_path(rel_path: str) -> Path:
    parts = Path(rel_path).parts
    if parts and parts[0] == "signals":
        return Path(*parts[1:])
    return Path(rel_path)


def _mirror_url(rel_path: str, base_url: str) -> str:
    """Map ``signals/{meeting}/audio|video/...`` to the AMI mirror URL."""
    return f"{base_url.rstrip('/')}/{_signals_relative_path(rel_path).as_posix()}"


def _download_file(
    rel_path: str,
    signals_dir: Path,
    base_url: str,
    force: bool,
) -> tuple[str, str]:
    dest = signals_dir / _signals_relative_path(rel_path)
    if dest.exists() and not force and dest.stat().st_size > 0:
        return rel_path, "skipped"

    url = _mirror_url(rel_path, base_url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "speechlmm-ami-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
        return rel_path, "ok"
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return rel_path, f"error HTTP {exc.code}"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return rel_path, f"error {exc}"


def download_license(output_dir: Path) -> None:
    dest = output_dir / "LICENCE-CC-BY-4.0.txt"
    if dest.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(LICENSE_URL, headers={"User-Agent": "speechlmm-ami-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    logger.info("Wrote %s", dest)


def download_signals(
    parquet_dir: Path,
    signals_dir: Path,
    *,
    base_url: str = AMI_MIRROR,
    workers: int = 4,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Download every audio/video file referenced in ``parquet_dir``."""
    parquet_dir = Path(parquet_dir)
    signals_dir = Path(signals_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet dir not found: {parquet_dir}")

    media_paths = media_paths_from_parquet(parquet_dir)
    if not media_paths:
        raise RuntimeError(f"No media paths found in {parquet_dir}")

    logger.info("Found %d media file(s) in %s", len(media_paths), parquet_dir)

    if dry_run:
        for rel_path in media_paths:
            print(f"{_mirror_url(rel_path, base_url)} -> {signals_dir / _signals_relative_path(rel_path)}")
        return

    download_license(signals_dir)
    _download_media(media_paths, signals_dir, base_url=base_url, workers=workers, force=force)


def _download_media(
    media_paths: Sequence[str],
    signals_dir: Path,
    *,
    base_url: str = AMI_MIRROR,
    workers: int = 4,
    force: bool = False,
) -> None:
    counts = {"ok": 0, "skipped": 0, "error": 0}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_file, rel_path, signals_dir, base_url, force): rel_path
            for rel_path in media_paths
        }
        for i, future in enumerate(as_completed(futures), 1):
            rel_path, status = future.result()
            if status == "ok":
                counts["ok"] += 1
                logger.info("[%d/%d] %s downloaded", i, len(media_paths), rel_path)
            elif status == "skipped":
                counts["skipped"] += 1
            else:
                counts["error"] += 1
                errors.append(f"{rel_path}: {status}")
                logger.error("[%d/%d] %s %s", i, len(media_paths), rel_path, status)

    logger.info(
        "Media done. downloaded=%d skipped=%d errors=%d -> %s",
        counts["ok"], counts["skipped"], counts["error"], signals_dir,
    )
    if errors:
        logger.error("Failed downloads:\n%s", "\n".join(errors))
        raise RuntimeError(f"{len(errors)} AMI downloads failed")


def _meeting_audio(meeting_id: str) -> str:
    return f"{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"


def _meeting_video(meeting_id: str, view: str) -> str:
    return f"{meeting_id}/video/{meeting_id}.{view}.avi"


AUDIO_CUT_ARGS = ("-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le")
VIDEO_CUT_ARGS = (
    "-an",
    "-vf", f"scale={AVSPEAKD_VIDEO_WIDTH}:{AVSPEAKD_VIDEO_HEIGHT},fps={AVSPEAKD_VIDEO_FPS}",
    "-c:v", "libx264",
    "-profile:v", "baseline",
    "-pix_fmt", "yuv420p",
    "-r", str(AVSPEAKD_VIDEO_FPS),
)


def _cut(source: Path, dest: Path, start: float, duration: float, codec_args: Sequence[str]) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg picks the container from the extension, so keep it last.
    tmp = dest.with_name(f"{dest.stem}.part{dest.suffix}")
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        *codec_args, str(tmp),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail}")
    tmp.replace(dest)


def materialize_chunks(
    chunks: Sequence[AmiChunk],
    *,
    datasets_root: Path,
    source_signals: Path,
    signals_root: str,
    workers: int = 4,
) -> None:
    """Download each source meeting file that is missing, then ffmpeg-cut the chunks."""
    sources: set[str] = set()
    for chunk in chunks:
        sources.add(_meeting_audio(chunk.meeting_id))
        if chunk.view:
            sources.add(_meeting_video(chunk.meeting_id, chunk.view))
    _download_media(sorted(sources), source_signals, workers=workers)

    def cut(chunk: AmiChunk) -> None:
        audio_src = source_signals / _meeting_audio(chunk.meeting_id)
        audio_dest = datasets_root / chunk_audio_path(signals_root, chunk.chunk_id)
        _cut(audio_src, audio_dest, chunk.start, chunk.duration, AUDIO_CUT_ARGS)
        if chunk.view:
            video_src = source_signals / _meeting_video(chunk.meeting_id, chunk.view)
            video_dest = datasets_root / chunk_video_path(signals_root, chunk.chunk_id)
            _cut(video_src, video_dest, chunk.start, chunk.duration, VIDEO_CUT_ARGS)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(cut, chunk) for chunk in chunks]
        for future in as_completed(futures):
            future.result()
    logger.info("Cut %d chunk(s) under %s", len(chunks), signals_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AMI media for Parquet meetings.")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Directory with train/val/test.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Signals root; writes paths from Parquet relative to this dir",
    )
    parser.add_argument("--base-url", default=AMI_MIRROR, help="AMI corpus mirror base URL")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs only, do not download")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = datasets_dir()  / "ami"
    parquet_dir = args.parquet_dir or root / "parquet"
    signals_dir = args.output_dir or root / "signals"
    download_signals(
        parquet_dir,
        signals_dir,
        base_url=args.base_url,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
