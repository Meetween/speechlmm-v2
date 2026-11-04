"""Utils for data preparation and preprocessing scripts for SpeechLMM-v2."""

from collections.abc import Sequence
import os 
import json
from pathlib import Path

from collections.abc import Sequence

DEFAULT_SIGNALS_ROOT = "signals"
AUDIO_TASKS = ["asr", "slu", "sqa", "st", "s2st", "achap", "ssum", "avspeakd"]
VIDEO_TASKS = ["lipread", "avspeakd"]

# list of tasks, which must be bilingual or monolingual
# the other task could be either bilingual or monolingual, according to the dataset
TRANSLATION_TASKS = {"mt", "st", "s2st"}
MONOLINGUAL_TASKS = {"asr"}

DATASETS_SUBDIR = "speechlmm-v2/v1.0"

def intersection(a: Sequence[str], b: Sequence[str]) -> list[str]:
    allowed = set(b)
    return [item for item in a if item in allowed]


def datasets_dir() -> Path:
    """Root of the datasets tree, from ``DATASETS_DIR`` or ``SCRATCH_DIR/data``."""
    if os.environ.get("DATASETS_DIR"):
        return Path(os.environ["DATASETS_DIR"])
    if os.environ.get("SCRATCH_DIR"):
        return Path(os.environ["SCRATCH_DIR"]) / "data" / DATASETS_SUBDIR
    raise EnvironmentError("Set DATASETS_DIR or SCRATCH_DIR to locate the datasets directory")


def resolve_parquet_dir(corpus: str | None, dataset_root: Path | None, prefix: str = '') -> tuple[str, Path]:
    if dataset_root:
        root = Path(dataset_root)
        return root.name, (root if root.name == f"{prefix}parquet" else root / f"{prefix}parquet")
    if corpus:
        return corpus, datasets_dir() / corpus / f"{prefix}parquet"
    raise SystemExit("Pass --corpus <name> or --dataset-root <path>")


def probe_duration_seconds(path: str | Path) -> tuple[float | None, str | None, int | None]:
    try:
        import av

        with av.open(path) as container:
            audio_stream = container.streams.audio[0]

            codec = audio_stream.codec_context.name

            sample_rate = audio_stream.codec_context.sample_rate

            if container.duration is not None:
                duration = container.duration / av.time_base
            elif audio_stream.duration is not None:
                duration = float(audio_stream.duration * audio_stream.time_base)
            else:
                duration = None

        return duration, codec, sample_rate
    except Exception:
        return None, None, None


def probe_video_duration_seconds(path: str | Path) -> float | None:
    try:
        import av

        with av.open(path) as container:
            video_stream = next(
                (stream for stream in container.streams if stream.type == "video"),
                None,
            )
            if video_stream is None:
                return None
            if video_stream.duration is not None:
                return float(video_stream.duration * video_stream.time_base)
            if container.duration is not None:
                return container.duration / av.time_base
            if video_stream.frames and video_stream.average_rate:
                return video_stream.frames / float(video_stream.average_rate)
    except Exception:
        return None

    return None

def intersection(a: Sequence[str], b: Sequence[str]) -> list[str]:
    allowed = set(b)
    return [item for item in a if item in allowed]
