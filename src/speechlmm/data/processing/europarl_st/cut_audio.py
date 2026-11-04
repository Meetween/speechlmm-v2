"""Cut Europarl-ST per-segment ST clips from speech-level ``.m4a`` with ffmpeg.

Europarl-ST provides speech-level audio plus ``segments.lst`` time spans; the schema
stores one clip per sample, so ST needs per-segment clips. This reads the raw
``{src}/{tgt}/{split}/segments.lst`` files, dedups by ``(src, audio_id, start, end)``
(the same span is referenced by several target directions), and writes 16 kHz mono
clips to the exact paths produced by :func:`to_parquet.clip_subpath`. Idempotent:
existing non-empty clips are skipped.

Usage:
    PYTHONPATH=src python3 -m speechlmm.data.processing.europarl_st.cut_audio \\
        --workers 8
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from speechlmm.data.utils import datasets_dir
from speechlmm.data.processing.europarl_st.to_parquet import (
    CLIP_CODEC,
    SAMPLE_RATE,
    clip_subpath,
    discover_directions,
    parse_segments,
    source_audio_path,
)

logger = logging.getLogger(__name__)


def iter_cut_specs(
    root: Path,
    signals_dir: Path,
    directions: list[tuple[str, str]],
    splits: list[str],
):
    """Yield unique ``(src_audio, start, end, out_clip)`` cut specs across directions."""
    seen: set[tuple[str, str, float, float]] = set()
    for src, tgt in directions:
        for split in splits:
            seg_lst = root / src / tgt / split / "segments.lst"
            if not seg_lst.exists():
                continue
            for audio_id, start, end in parse_segments(seg_lst):
                key = (src, audio_id, start, end)
                if key in seen:
                    continue
                seen.add(key)
                out_clip = signals_dir / clip_subpath(src, audio_id, start, end)
                yield source_audio_path(root, src, audio_id), start, end, out_clip


def cut_one(src_audio: Path, start: float, end: float, out_clip: Path, force: bool) -> str:
    if out_clip.exists() and not force and out_clip.stat().st_size > 0:
        return "skipped"
    if not src_audio.exists():
        logger.error("Missing source audio: %s", src_audio)
        return "missing_source"

    out_clip.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_clip.with_suffix(out_clip.suffix + ".part")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src_audio),
        "-t",
        f"{max(end - start, 0.0):.3f}",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        CLIP_CODEC,
        str(tmp),  # force muxer: the .part temp suffix hides the format
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tmp.replace(out_clip)
        return "ok"
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        logger.error(
            "ffmpeg failed for %s [%.2f-%.2f]: %s", src_audio, start, end, exc.stderr.decode(errors="ignore").strip()[:200]
        )
        return "error"


def cut_all(
    root: Path,
    signals_dir: Path,
    directions: list[tuple[str, str]],
    splits: list[str],
    workers: int = 8,
    force: bool = False,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required to cut Europarl-ST clips")

    specs = list(iter_cut_specs(root, signals_dir, directions, splits))
    logger.info("Cutting %d unique clips with %d workers", len(specs), workers)

    counts = {"ok": 0, "skipped": 0, "error": 0, "missing_source": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(cut_one, src_audio, start, end, out_clip, force): out_clip for src_audio, start, end, out_clip in specs
        }
        for i, future in enumerate(as_completed(futures), 1):
            counts[future.result()] += 1
            if i % 5000 == 0:
                logger.info(
                    "[%d/%d] ok=%d skipped=%d error=%d missing=%d",
                    i,
                    len(specs),
                    counts["ok"],
                    counts["skipped"],
                    counts["error"],
                    counts["missing_source"],
                )

    logger.info(
        "Cut done. ok=%d skipped=%d error=%d missing_source=%d",
        counts["ok"],
        counts["skipped"],
        counts["error"],
        counts["missing_source"],
    )
    if counts["error"] or counts["missing_source"]:
        raise RuntimeError(f"{counts['error']} ffmpeg errors, {counts['missing_source']} missing sources")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut Europarl-ST per-segment ST clips (16 kHz mono).")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Extracted v1.1 dir (default: $DATASETS_DIR/speechlmm-v2/europarl_st_v1_1)",
    )
    parser.add_argument("--signals-dir", type=Path, default=None, help="Output signals dir (default: <dataset>/signals)")
    parser.add_argument("--languages", nargs="+", default=["de", "en", "es", "fr", "it", "nl", "pt"], help="Languages to include")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], help="Splits to build")
    parser.add_argument("--workers", type=int, default=8, help="Parallel ffmpeg workers")
    parser.add_argument("--force", action="store_true", help="Re-cut even if clip exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    default_dataset_root = datasets_dir() / "europarl_st"
    europarl_root = args.dataset_root or default_dataset_root / "v1.1"
    signals_dir = args.signals_dir or default_dataset_root / "signals"

    directions = discover_directions(europarl_root, args.languages)
    cut_all(europarl_root, signals_dir, directions, args.splits, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
