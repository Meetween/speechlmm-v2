"""Download YTSeg audio from the Hugging Face Hub into ``signals/<video_id>.mp3``.

The HF ``audio`` config embeds each video's audio as 16 kHz mono MP3. When the Hub
snapshot parquet shards are present locally (``HF_HOME/hub/datasets--...``), audio is
streamed shard-by-shard to avoid materialising the full split in memory. Otherwise
falls back to ``datasets.load_dataset`` (needs network + much more RAM).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pyarrow.parquet as pq

from speechlmm.data.utils import datasets_dir
from speechlmm.data.processing.ytseg.to_parquet import HF_SPLITS

logger = logging.getLogger(__name__)


def video_ids_from_parquet(parquet_dir: Path) -> set[str]:
    ids: set[str] = set()
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path, columns=["audio_input_path"])
        for path in table.column("audio_input_path").to_pylist():
            if path:
                ids.add(Path(path).stem)
    return ids


def _write_mp3(dest: Path, audio: dict, force: bool) -> str:
    if dest.exists() and not force and dest.stat().st_size > 0:
        return "skipped"
    raw = audio.get("bytes")
    if raw is None and audio.get("path"):
        raw = Path(audio["path"]).read_bytes()
    if not raw:
        return "empty"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(raw)
    tmp.replace(dest)
    return "ok"


def _hub_audio_dir(dataset_id: str) -> Path | None:
    """Return ``.../snapshots/<rev>/audio`` with the most parquet shards, if any."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    org, repo = dataset_id.split("/", 1)
    snapshots = hf_home / "hub" / f"datasets--{org}--{repo}" / "snapshots"
    if not snapshots.is_dir():
        return None
    best: Path | None = None
    best_n = 0
    for snap in snapshots.iterdir():
        audio = snap / "audio"
        if not audio.is_dir():
            continue
        n = sum(1 for _ in audio.glob("*.parquet"))
        if n > best_n:
            best_n = n
            best = audio
    return best


def _download_split_from_hub_shards(
    audio_dir: Path,
    hf_split: str,
    signals_dir: Path,
    *,
    wanted_ids: set[str] | None,
    force: bool,
    counts: dict[str, int],
) -> None:
    shards = sorted(audio_dir.glob(f"{hf_split}-*.parquet"))
    if not shards:
        logger.warning("No hub audio shards for split=%s under %s", hf_split, audio_dir)
        return

    logger.info("Streaming %d hub shard(s) for split=%s", len(shards), hf_split)
    processed = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=1, columns=["video_id", "audio"]):
            video_id = batch.column("video_id")[0].as_py()
            if wanted_ids is not None and video_id not in wanted_ids:
                continue
            audio = batch.column("audio")[0].as_py()
            status = _write_mp3(signals_dir / f"{video_id}.mp3", audio, force)
            counts[status] = counts.get(status, 0) + 1
            if status == "empty":
                logger.warning("No audio bytes for %s", video_id)
            processed += 1
            if processed % 500 == 0:
                logger.info("[%s] %d rows from shards", hf_split, processed)


def _download_split_via_load_dataset(
    dataset_id: str,
    hf_split: str,
    signals_dir: Path,
    *,
    wanted_ids: set[str] | None,
    force: bool,
    counts: dict[str, int],
) -> None:
    from datasets import Audio, load_dataset

    logger.info("Loading %s (audio, %s) via datasets", dataset_id, hf_split)
    ds = load_dataset(dataset_id, "audio", split=hf_split)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i in range(len(ds)):
        video_id = ds[i]["video_id"]
        if wanted_ids is not None and video_id not in wanted_ids:
            continue
        status = _write_mp3(signals_dir / f"{video_id}.mp3", ds[i]["audio"], force)
        counts[status] = counts.get(status, 0) + 1
        if status == "empty":
            logger.warning("No audio bytes for %s", video_id)
        if (i + 1) % 500 == 0:
            logger.info("[%s] %d/%d processed", hf_split, i + 1, len(ds))


def download_audio(
    dataset_id: str,
    splits,
    signals_dir: Path,
    *,
    wanted_ids: set[str] | None = None,
    force: bool = False,
) -> None:
    """Write ``signals/<video_id>.mp3`` for every wanted video in the given splits."""
    signals_dir = Path(signals_dir)
    counts = {"ok": 0, "skipped": 0, "empty": 0}
    hub_audio = _hub_audio_dir(dataset_id)

    for split in splits:
        hf_split = HF_SPLITS[split]
        if hub_audio is not None:
            _download_split_from_hub_shards(
                hub_audio, hf_split, signals_dir,
                wanted_ids=wanted_ids, force=force, counts=counts,
            )
        else:
            _download_split_via_load_dataset(
                dataset_id, hf_split, signals_dir,
                wanted_ids=wanted_ids, force=force, counts=counts,
            )

    logger.info(
        "Audio done. written=%d skipped=%d empty=%d -> %s",
        counts["ok"], counts["skipped"], counts["empty"], signals_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YTSeg audio (MP3) from the Hugging Face Hub.")
    parser.add_argument("--dataset-id", default="retkowski/ytseg")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--force", action="store_true", help="Re-download existing MP3 files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.dataset_root or datasets_dir() / "ytseg"
    wanted = video_ids_from_parquet(root / "parquet")
    download_audio(args.dataset_id, args.splits, root / "signals", wanted_ids=wanted or None, force=args.force)


if __name__ == "__main__":
    main()
