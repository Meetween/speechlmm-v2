"""Build AMI metadata Parquet files for the SpeechLMM schema.

One row per meeting per task (asr, achap, ssum, tsum, avspeakd), with fields
filled according to the official metadata schema. Task instructions are NOT
stored in ``text_prompt`` (those are added later by ``parquet_to_sharegpt``).

Per-task text semantics:
  asr      audio -> text_output = plain transcript
  achap    audio -> text_output = chapter timestamps + titles
  ssum     audio -> text_output = abstractive summary
  tsum     text_input = plain transcript -> text_output = abstractive summary
  avspeakd audio + video -> text_output = plain-text diarization (Speaker N: M:SS - M:SS)

Official splits from ``corpusResources/meetings.xml`` (scenario partition).
https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml

Usage:
    PYTHONPATH=src python3 -m speechlmm.data.processing.ami.to_parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xml.etree.ElementTree as ET

from speechlmm.data.schema import SCHEMA, SCHEMA_COLUMNS, empty_row
from speechlmm.data.utils import AUDIO_TASKS, VIDEO_TASKS, DEFAULT_SIGNALS_ROOT, probe_video_duration_seconds, datasets_dir

logger = logging.getLogger(__name__)

NITE = "http://nite.sourceforge.net/"
HREF_RE = re.compile(r"(?P<file>[^#]+)#id\((?P<start>[^)]+)\)(?:\.\.id\((?P<end>[^)]+)\))?")
SERIES_RE = re.compile(r"^([A-Z]+\d+)")
LICENSE = "CC-BY-4.0"
SOURCE = "AMI"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CODEC = "wav"

# AMI "full" partition: official series groupings used when a meeting has no
# split in meetings.xml. A *series* is the recording-group prefix (e.g. "EN2001"),
# shared by all of its sessions ("EN2001a", "EN2001b", ...).
# See https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
FULL_TRAIN_SERIES = {
    "EN2001",
    "EN2003",
    "EN2004",
    "EN2005",
    "EN2006",
    "EN2009",
    "IN1001",
    "IN1002",
    "IN1005",
    "IN1007",
    "IN1008",
    "IN1009",
    "IN1012",
    "IN1013",
    "IN1014",
    "IN1016",
}
FULL_DEV_SERIES = {"IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"}
FULL_TEST_SERIES = {"EN2002"}
# "full_asr" partition adjustments (Kaldi/ESPnet AMI ASR recipe): these series
# move into train, and are removed from the dev / test sets respectively.
FULL_ASR_EXTRA_TRAIN = {"ES2003", "TS3006", "ES2014", "TS3007"}
FULL_ASR_DEV_REMOVE = {"ES2003", "TS3006", "ES2014", "TS3007"}
FULL_ASR_TEST_REMOVE = {"ES2014", "TS3007"}


# Media paths are stored relative to the dataset dir; resolved at load via media_dir.
DEFAULT_VIDEO_VIEW = "Closeup1"
DIARIZATION_TIMESTAMP_RESOLUTION = 0.1


def _tag(elem) -> str:
    return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag


def _nid(elem) -> str | None:
    return elem.get(f"{{{NITE}}}id")


def _series(meeting_id: str) -> str:
    # A meeting_id is one session (e.g. "ES2003a"); the series is its recording-group
    # prefix (e.g. "ES2003"), shared by every session of the same participants.
    m = SERIES_RE.match(meeting_id)
    return m.group(1) if m else meeting_id


def _word_count(text: str | None) -> int | None:
    if not text:
        return None
    return len(text.split())


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_meeting_index(ami_root: Path) -> dict[str, dict]:
    meetings: dict[str, dict] = {}
    root = ET.parse(ami_root / "corpusResources/meetings.xml").getroot()
    for meeting in root:
        if _tag(meeting) != "meeting":
            continue
        meeting_id = meeting.get("observation")
        if not meeting_id:
            continue
        speakers = []
        for child in meeting:
            if _tag(child) == "speaker":
                speakers.append(
                    {
                        "agent": child.get("nxt_agent"),
                        "speaker_id": child.get("global_name"),
                        "role": child.get("role"),
                        "camera": child.get("camera"),
                    }
                )
        meetings[meeting_id] = {
            "type": meeting.get("type"),
            "duration": float(meeting.get("duration")) if meeting.get("duration") else None,
            "visibility": meeting.get("visibility"),
            "seen_type": meeting.get("seen_type"),
            "speakers": speakers,
        }
    return meetings


def split_from_meetings_xml(info: dict) -> str | None:
    if info.get("visibility") == "unseen":
        return "test"
    seen = info.get("seen_type")
    if seen == "training":
        return "train"
    if seen == "development":
        return "val"
    return None


def split_for_partition(meeting_id: str, info: dict, partition: str) -> str | None:
    split = split_from_meetings_xml(info)
    series = _series(meeting_id)

    if partition == "scenario":
        return split

    if partition == "full":
        if split:
            return split
        if series in FULL_TEST_SERIES:
            return "test"
        if series in FULL_DEV_SERIES:
            return "val"
        if series in FULL_TRAIN_SERIES:
            return "train"
        if series == "IB4005":
            return None
        return split

    if partition == "full_asr":
        base = split_for_partition(meeting_id, info, "full")
        if meeting_id in FULL_ASR_EXTRA_TRAIN or series in FULL_ASR_EXTRA_TRAIN:
            return "train"
        if base == "val" and (meeting_id in FULL_ASR_DEV_REMOVE or series in FULL_ASR_DEV_REMOVE):
            return "train"
        if base == "test" and (meeting_id in FULL_ASR_TEST_REMOVE or series in FULL_ASR_TEST_REMOVE):
            return "train"
        return base

    raise ValueError(f"Unknown partition: {partition}")


def load_topic_ontology(ami_root: Path) -> dict[str, str]:
    tree = ET.parse(ami_root / "ontologies/default-topics.xml")
    return {_nid(node): node.get("name", "") for node in tree.iter() if _nid(node)}


def load_word_times(words_path: Path) -> dict[str, tuple[float, float]]:
    times: dict[str, tuple[float, float]] = {}
    for elem in ET.parse(words_path).getroot():
        wid = _nid(elem)
        start, end = elem.get("starttime"), elem.get("endtime")
        if not wid or start is None or end is None:
            continue
        times[wid] = (float(start), float(end))
    return times


def _word_ids(start_id: str, end_id: str | None) -> list[str]:
    if end_id is None:
        return [start_id]
    m0 = re.search(r"(\d+)$", start_id)
    m1 = re.search(r"(\d+)$", end_id)
    if not m0 or not m1:
        return [start_id]
    base = start_id[: m0.start()]
    return [f"{base}{i}" for i in range(int(m0.group(1)), int(m1.group(1)) + 1)]


def _topic_label(topic, ontology: dict[str, str]) -> str:
    label = (topic.get("other_description") or "").strip()
    if label:
        return label
    for child in topic:
        if _tag(child) != "pointer":
            continue
        href = child.get("href", "")
        if "#id(" not in href:
            continue
        topic_id = href.split("#id(", 1)[1].rstrip(")")
        return ontology.get(topic_id, topic_id)
    return ""


def load_chapters(meeting_id: str, ami_root: Path, ontology: dict[str, str], words_cache: dict) -> list[dict]:
    topic_path = ami_root / "topics" / f"{meeting_id}.topic.xml"
    if not topic_path.exists():
        return []

    chapters = []
    for topic in ET.parse(topic_path).getroot():
        if _tag(topic) != "topic":
            continue

        starts, ends = [], []
        for child in topic:
            if _tag(child) != "child":
                continue
            m = HREF_RE.match(child.get("href", ""))
            if not m:
                continue
            words_path = ami_root / "words" / Path(m.group("file")).name
            if words_path not in words_cache:
                words_cache[words_path] = load_word_times(words_path)
            word_times = words_cache[words_path]
            for wid in _word_ids(m.group("start"), m.group("end")):
                if wid in word_times:
                    s, e = word_times[wid]
                    starts.append(s)
                    ends.append(e)

        if not starts:
            continue
        chapters.append(
            {
                "start_sec": min(starts),
                "end_sec": max(ends),
                "title": _topic_label(topic, ontology),
            }
        )

    chapters.sort(key=lambda x: x["start_sec"])
    return chapters


def format_chapters(chapters: list[dict]) -> str:
    return "\n".join(f"{_format_timestamp(ch['start_sec'])} - {ch['title']}" for ch in chapters)


def _append_word(parts: list[str], text: str, is_punc: bool) -> None:
    if not text:
        return
    if is_punc and parts:
        parts[-1] += text
    else:
        parts.append(text)


def build_transcript(meeting_id: str, ami_root: Path, agent_to_speaker: dict[str, str]) -> tuple[str, str, float | None]:
    utterances: list[tuple[float, str, str]] = []
    max_end = 0.0

    for words_path in sorted((ami_root / "words").glob(f"{meeting_id}.*.words.xml")):
        agent = words_path.name.split(".")[1]
        speaker = agent_to_speaker.get(agent, agent)
        parts: list[str] = []
        start_time = None
        for elem in ET.parse(words_path).getroot():
            if _tag(elem) != "w":
                continue
            text = (elem.text or "").strip()
            if not text:
                continue
            if start_time is None and elem.get("starttime") is not None:
                start_time = float(elem.get("starttime"))
            if elem.get("endtime") is not None:
                max_end = max(max_end, float(elem.get("endtime")))
            _append_word(parts, text, elem.get("punc") == "true")
        if parts and start_time is not None:
            utterances.append((start_time, speaker, " ".join(parts)))

    if not utterances:
        return "", "", None

    utterances.sort(key=lambda x: x[0])
    with_names = [f"{speaker}: {text}" for _, speaker, text in utterances]
    without_names = [text for _, _, text in utterances]
    return "\n".join(with_names), "\n".join(without_names), max_end


def _section_text(elem) -> str:
    parts: list[str] = []
    if elem.text and elem.text.strip():
        parts.append(elem.text.strip())
    for child in elem:
        if _tag(child) == "sentence":
            if child.text and child.text.strip():
                parts.append(child.text.strip())
        elif child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return " ".join(parts).strip()


def load_abstractive_summary(meeting_id: str, ami_root: Path) -> str | None:
    summ_path = ami_root / "abstractive" / f"{meeting_id}.abssumm.xml"
    if not summ_path.exists():
        return None

    sections = []
    labels = {"abstract": "Abstract", "actions": "Actions", "decisions": "Decisions", "problems": "Problems"}
    root = ET.parse(summ_path).getroot()
    for key, title in labels.items():
        for elem in root.iter():
            if _tag(elem) != key:
                continue
            text = _section_text(elem)
            if text and text != "NA":
                sections.append(f"{title}:\n{text}")

    return "\n\n".join(sections) if sections else None


def audio_input_path_ami(id: str, signals_root: str) -> str:
    return f"{signals_root}/{id}/audio/{id}.Mix-Headset.wav"


def video_input_path_ami(id: str, signals_root: str, video_view: str = DEFAULT_VIDEO_VIEW) -> str:
    return f"{signals_root}/{id}/video/{id}.{video_view}.avi"


def _format_diarization_timestamp(seconds: float) -> str:
    snapped = round(seconds / DIARIZATION_TIMESTAMP_RESOLUTION) * DIARIZATION_TIMESTAMP_RESOLUTION
    total_secs = int(round(snapped))
    hours, rem = divmod(total_secs, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _parse_segment_file(
    path: Path,
    meeting_id: str,
    speaker_map: dict[tuple[str, str], str],
) -> list[tuple[float, float, str]]:
    parts = path.name.split(".")
    if len(parts) < 3 or parts[-1] != "xml" or parts[-2] != "segments":
        return []
    agent = parts[1]
    speaker_id = speaker_map.get((meeting_id, agent))
    if not speaker_id:
        return []

    segments: list[tuple[float, float, str]] = []
    for seg in ET.parse(path).getroot():
        if _tag(seg) != "segment":
            continue
        if "transcriber_start" not in seg.attrib or "transcriber_end" not in seg.attrib:
            continue
        start = float(seg.attrib["transcriber_start"])
        end = float(seg.attrib["transcriber_end"])
        if end - start <= 0:
            continue
        segments.append((start, end, speaker_id))
    return segments


def load_meeting_segments(
    meeting_id: str,
    ami_root: Path,
    agent_to_speaker: dict[str, str],
) -> list[tuple[float, float, str]]:
    """Return sorted (start, end, speaker_id) segments for one meeting."""
    speaker_map = {(meeting_id, agent): sid for agent, sid in agent_to_speaker.items()}
    seg_dir = ami_root / "segments"
    if not seg_dir.is_dir():
        return []

    segments: list[tuple[float, float, str]] = []
    prefix = f"{meeting_id}."
    for name in sorted(os.listdir(seg_dir)):
        if not name.startswith(prefix) or not name.endswith(".segments.xml"):
            continue
        segments.extend(_parse_segment_file(seg_dir / name, meeting_id, speaker_map))
    segments.sort(key=lambda x: x[0])
    return segments


def format_meeting_diarization(segments: list[tuple[float, float, str]]) -> str:
    speaker_idx: dict[str, int] = {}
    for _, _, spk in segments:
        if spk not in speaker_idx:
            speaker_idx[spk] = len(speaker_idx)

    lines = []
    for start, end, spk in segments:
        idx = speaker_idx[spk]
        lines.append(f"Speaker {idx}: {_format_diarization_timestamp(start)} - {_format_diarization_timestamp(end)}")
    return "\n".join(lines)


def has_overlapping_speech(segments: list[tuple[float, float, str]]) -> bool:
    for i, (s0, e0, _) in enumerate(segments):
        for s1, e1, _ in segments[i + 1 :]:
            if s0 < e1 and s1 < e0:
                return True
    return False


def build_meeting_diarization(
    meeting_id: str,
    ami_root: Path,
    agent_to_speaker: dict[str, str],
) -> tuple[str, bool] | None:
    segments = load_meeting_segments(meeting_id, ami_root, agent_to_speaker)
    if not segments:
        return None
    return format_meeting_diarization(segments), has_overlapping_speech(segments)



def _base_row_ami(
    meeting_id: str,
    task: str,
    split: str,
    info: dict,
    signals_root: str,
    transcript_plain: str,
) -> dict:
    speaker_ids = [s["speaker_id"] for s in info.get("speakers", []) if s.get("speaker_id")]
    num_speakers = len(speaker_ids) if speaker_ids else None
    duration = info.get("duration")
    use_audio = task in AUDIO_TASKS
    use_video = task in VIDEO_TASKS

    if use_audio and duration and transcript_plain:
        words_per_second = _word_count(transcript_plain) / duration if duration > 0 else None
        chars_per_second = len(transcript_plain) / duration if duration > 0 else None
    else:
        words_per_second = None
        chars_per_second = None

    row = empty_row()
    row.update(
        {
            "sample_id": f"ami_{split}_{meeting_id}_{task}",
            "task": [task],
            "split": split,
            "source_dataset": SOURCE,
            "license": LICENSE,
            "speaker_id": speaker_ids or None,
            "num_speakers": num_speakers,
            "words_per_second": words_per_second,
            "chars_per_second": chars_per_second,
        }
    )

    if use_audio:
        row.update(
            {
                "audio_input_path": audio_input_path_ami(meeting_id, signals_root),
                "audio_input_duration": duration,
                "audio_input_sample_rate": AUDIO_SAMPLE_RATE,
                "audio_input_num_channels": 1,
                "audio_input_codec": AUDIO_CODEC,
                "language_audio_input": "en",
            }
        )

    if use_video:
        row.update(
            {
                "video_input_path": video_input_path_ami(meeting_id, signals_root),
                "language_video_input": "en",
            }
        )

    if task == "tsum":
        row.update(
            {
                "text_input": transcript_plain,
                "text_input_normalized": transcript_plain,
                "num_words_input": _word_count(transcript_plain),
                "language_text_input": "en",
            }
        )

    return row



def build_task_rows_ami(
    meeting_id: str,
    split: str,
    info: dict,
    signals_root: str,
    transcript_plain: str,
    chapters: list[dict],
    abstractive_summary: str | None,
    diarization: tuple[str, bool] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    chapters_text = format_chapters(chapters) if chapters else None

    if transcript_plain:
        row = _base_row_ami(meeting_id, "asr", split, info, signals_root, transcript_plain)
        row.update(
            {
                "text_output": transcript_plain,
                "text_output_normalized": transcript_plain,
                "num_words_output": _word_count(transcript_plain),
                "language_text_output": "en",
            }
        )
        rows.append(row)

    if chapters_text:
        row = _base_row_ami(meeting_id, "achap", split, info, signals_root, transcript_plain)
        row.update(
            {
                "text_output": chapters_text,
                "text_output_normalized": chapters_text,
                "num_words_output": _word_count(chapters_text),
                "language_text_output": "en",
            }
        )
        rows.append(row)

    if abstractive_summary:
        row = _base_row_ami(meeting_id, "ssum", split, info, signals_root, transcript_plain)
        row.update(
            {
                "text_output": abstractive_summary,
                "text_output_normalized": abstractive_summary,
                "num_words_output": _word_count(abstractive_summary),
                "language_text_output": "en",
            }
        )
        rows.append(row)

        if transcript_plain:
            row = _base_row_ami(meeting_id, "tsum", split, info, signals_root, transcript_plain)
            row.update(
                {
                    "text_output": abstractive_summary,
                    "text_output_normalized": abstractive_summary,
                    "num_words_output": _word_count(abstractive_summary),
                    "language_text_output": "en",
                }
            )
            rows.append(row)

    if diarization:
        diar_text, overlap = diarization
        row = _base_row_ami(meeting_id, "avspeakd", split, info, signals_root, transcript_plain)
        row.update(
            {
                "text_output": diar_text,
                "text_output_normalized": diar_text,
                "num_words_output": _word_count(diar_text),
                "has_overlapping_speech": overlap,
                "language_text_output": "en",
            }
        )
        rows.append(row)

    return rows


def list_meeting_ids(ami_root: Path) -> list[str]:
    meeting_ids = {p.name.split(".")[0] for p in (ami_root / "words").glob("*.words.xml")}
    return sorted(meeting_ids)



def build_samples_ami(
    ami_root: Path,
    signals_root: str,
    partition: str,
    include_unsplit: bool,
) -> list[dict]:
    meetings = load_meeting_index(ami_root)
    ontology = load_topic_ontology(ami_root)
    words_cache: dict[Path, dict[str, tuple[float, float]]] = {}
    samples: list[dict] = []

    for meeting_id in list_meeting_ids(ami_root):
        info = meetings.get(meeting_id, {"speakers": [], "duration": None})
        split = split_for_partition(meeting_id, info, partition)
        if split is None and not include_unsplit:
            continue
        if split is None:
            split = "train"

        agent_to_speaker = {s["agent"]: s["speaker_id"] for s in info["speakers"] if s.get("agent")}
        _, transcript_plain, transcript_end = build_transcript(meeting_id, ami_root, agent_to_speaker)
        if not transcript_plain:
            continue

        chapters = load_chapters(meeting_id, ami_root, ontology, words_cache)
        abstractive_summary = load_abstractive_summary(meeting_id, ami_root)
        diarization = build_meeting_diarization(meeting_id, ami_root, agent_to_speaker)

        duration = info.get("duration")
        if transcript_end is not None:
            duration = max(duration or 0.0, transcript_end)
        if chapters:
            duration = max(duration or 0.0, max(ch["end_sec"] for ch in chapters))
        info = {**info, "duration": duration}

        samples.extend(
            build_task_rows_ami(
                meeting_id,
                split,
                info,
                signals_root,
                transcript_plain,
                chapters,
                abstractive_summary,
                diarization,
            )
        )

    return samples


def write_parquets(samples: list[dict], splits: Sequence[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)
    if df.empty:
        raise RuntimeError("No samples produced; check AMI annotations path.")

    for split in splits:
        part = df[df["split"] == split]
        if part.empty:
            logger.warning("No samples for split=%s", split)
            continue
        out = output_dir / f"{split}.parquet"
        # Cast to the shared schema so every corpus emits identical column types.
        table = pa.Table.from_pandas(part.reindex(columns=SCHEMA_COLUMNS), schema=SCHEMA, preserve_index=False)
        pq.write_table(table, out)
        logger.info("Wrote %s (%d rows)", out, len(part))


def enrich_video_metadata(
    parquet_dir: Path,
    dataset_root: Path,
    splits: Sequence[str],
    workers: int,
) -> None:
    """Fill video durations from downloaded media."""
    for split in splits:
        parquet_path = parquet_dir / f"{split}.parquet"
        if not parquet_path.is_file():
            continue

        frame = pq.read_table(parquet_path).to_pandas()
        relative_paths = sorted(
            {
                path
                for path in frame["video_input_path"]
                if isinstance(path, str) and path
            }
        )
        if not relative_paths:
            continue
        with ThreadPoolExecutor(max_workers=workers) as executor:
            durations = dict(
                zip(
                    relative_paths,
                    executor.map(
                        probe_video_duration_seconds,
                        (dataset_root / path for path in relative_paths),
                    ),
                )
            )
        missing = sum(duration is None for duration in durations.values())
        if missing:
            logger.warning(
                "Could not probe %d/%d videos for split=%s",
                missing,
                len(relative_paths),
                split,
            )

        for index, relative_path in frame["video_input_path"].items():
            if relative_path in durations:
                frame.at[index, "video_input_duration"] = durations[relative_path]

        table = pa.Table.from_pandas(
            frame.reindex(columns=SCHEMA_COLUMNS),
            schema=SCHEMA,
            preserve_index=False,
        )
        temporary_path = parquet_path.with_suffix(".tmp.parquet")
        pq.write_table(table, temporary_path)
        temporary_path.replace(parquet_path)
        logger.info("Updated video metadata in %s", parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AMI annotations to SpeechLMM metadata Parquet.")
    parser.add_argument("--ami-annotations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--signals-root",
        default=DEFAULT_SIGNALS_ROOT,
        help="Prefix for the stored audio_input_path; relative ('signals') by default, resolved via LLaMA-Factory media_dir.",
    )
    parser.add_argument(
        "--partition",
        choices=("scenario", "full", "full_asr"),
        default="scenario",
    )
    parser.add_argument(
        "--include-unsplit",
        action="store_true",
        help="Include meetings without an official AMI split (assigned to train)",
    )
    parser.add_argument("--tasks", nargs="+", default=["asr", "tsum", "ssum", "achap", "avspeakd"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel video metadata probe workers",
    )
    args = parser.parse_args()

    # this main referes to AMI dataset for "asr", "ssum", "tsum", only
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    default_dataset_root = datasets_dir() / "ami"
    ami_root = args.ami_annotations or default_dataset_root / "manual"
    parquet_dir = args.output_dir or default_dataset_root / "parquet"
    dataset_root = parquet_dir.parent if args.output_dir else default_dataset_root
    if not ami_root.exists():
        raise FileNotFoundError(f"AMI annotations not found: {ami_root}")

    samples = build_samples_ami(
        ami_root=ami_root,
        signals_root=args.signals_root,
        partition=args.partition,
        include_unsplit=args.include_unsplit,
    )
    splits = sorted({sample["split"] for sample in samples})
    write_parquets(samples, splits, parquet_dir)
    enrich_video_metadata(parquet_dir, dataset_root, splits, args.workers)

    task_counts = pd.Series([t for s in samples for t in s["task"]]).value_counts().to_dict()
    counts = pd.Series([s["split"] for s in samples]).value_counts().to_dict()
    logger.info("Done. %d rows. Split counts: %s. Task counts: %s", len(samples), counts, task_counts)


if __name__ == "__main__":
    main()
