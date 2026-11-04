"""Chapter and speaker-window chunks for AMI achap and avspeakd.

Whole meetings are too long for those tasks. achap packs topic chapters into
windows of at most 12 minutes (stride one chapter). avspeakd packs speaker
segments into windows of at most 2 minutes (stride three segments). A turn
longer than 2 minutes is cut into 2 minute pieces first. Times in the text
are rebased to the start of each window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from speechlmm.data.processing.ami.to_parquet import (
    LICENSE,
    SOURCE,
    _format_diarization_timestamp,
    _format_timestamp,
    _word_count,
    has_overlapping_speech,
    list_meeting_ids,
    load_chapters,
    load_meeting_index,
    load_meeting_segments,
    load_topic_ontology,
    split_for_partition,
)
from speechlmm.data.schema import empty_row

ACHAP_MAX_SECONDS = 12 * 60
AVSPEAKD_MAX_SECONDS = 2 * 60
AVSPEAKD_STRIDE = 3
# start + 120 is not exact in binary floats. A turn that the annotations
# place on the cap must still count as inside it.
_CAP_SLACK = 1e-4
# Training clips are 1 fps CIF. The released avspeakd mp4s use this encode.
AVSPEAKD_VIDEO_FPS = 1
AVSPEAKD_VIDEO_WIDTH = 352
AVSPEAKD_VIDEO_HEIGHT = 288

# Room camera published by the AMI mirror, by site.
# TS3003–TS3007 have Overview1 and Overview2. From TS3008a onward only
# Overview2 is on the mirror, and TS3008b/c/d have no overview video at all.
_TS_WITHOUT_ROOM_VIDEO = frozenset({"TS3008b", "TS3008c", "TS3008d"})


def room_views(meeting_id: str) -> tuple[str, ...]:
    site = meeting_id[:2]
    if site == "ES":
        return ("Corner",)
    if site == "IS":
        return ("C",)
    if site != "TS" or meeting_id in _TS_WITHOUT_ROOM_VIDEO:
        return ()
    series = int(meeting_id[2:6])
    if series >= 3008:
        return ("Overview2",)
    return ("Overview1", "Overview2")


AUDIO_SAMPLE_RATE = 16000
AUDIO_CODEC = "wav"


@dataclass(frozen=True)
class AmiChunk:
    chunk_id: str
    meeting_id: str
    split: str
    task: str
    start: float
    end: float
    text: str
    view: str | None
    overlap: bool | None
    speaker_ids: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


def chapter_windows(chapters: Sequence[dict]) -> list[tuple[float, float, list[dict]]]:
    """Overlapping chapter packs. A chapter longer than 12 minutes is cut at the cap."""
    windows: list[tuple[float, float, list[dict]]] = []
    for index, first in enumerate(chapters):
        start = first["start_sec"]
        chosen = [first]
        for chapter in chapters[index + 1 :]:
            if chapter["end_sec"] - start <= ACHAP_MAX_SECONDS:
                chosen.append(chapter)
            else:
                break
        end = min(chosen[-1]["end_sec"], start + ACHAP_MAX_SECONDS)
        if end <= start:
            continue
        windows.append((start, end, list(chosen)))
    return windows


def _format_rebased_timestamp(seconds: float) -> str:
    return _format_timestamp(round(seconds))


def format_chapter_window(start: float, chapters: Sequence[dict]) -> str:
    lines = [
        f"{_format_rebased_timestamp(chapter['start_sec'] - start)} - {chapter['title']}"
        for chapter in chapters
    ]
    return "\n".join(lines)


def split_long_turns(
    segments: Sequence[tuple[float, float, str]],
    limit: float = AVSPEAKD_MAX_SECONDS,
) -> list[tuple[float, float, str]]:
    """Cut a turn longer than the cap into cap-sized pieces, in place.

    The stride counts those pieces. IS1009c has one 128s turn; the released
    windows keep a 120s piece and a short remainder, and every later window
    moves up by one segment. Pieces stay where the original turn was, so a
    backchannel that starts during the turn is still a later segment.
    """
    pieces: list[tuple[float, float, str]] = []
    for start, end, speaker in segments:
        cursor = start
        while end - cursor > limit + _CAP_SLACK:
            pieces.append((cursor, cursor + limit, speaker))
            cursor += limit
        if end > cursor:
            pieces.append((cursor, end, speaker))
    return pieces


def speaker_index(segments: Sequence[tuple[float, float, str]]) -> dict[str, int]:
    """Number speakers by first turn. A shared start yields to the shorter turn.

    ES2004a opens with two speakers at t=0. The released numbering gives the
    lower index to the turn that ends first, not to the earlier file name.
    """
    index: dict[str, int] = {}
    for _start, _end, speaker in sorted(segments, key=lambda seg: (seg[0], seg[1])):
        if speaker not in index:
            index[speaker] = len(index)
    return index


def diarization_windows(
    segments: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float, list[tuple[float, float, str]]]]:
    """Stride-3 windows of at most two minutes, ending on the last turn that fits.

    Same-start turns are ordered by end time. A later turn that crosses the cap
    ends the scan, and an anchor turn longer than the cap is hard-cut.
    """
    windows: list[tuple[float, float, list[tuple[float, float, str]]]] = []
    ordered = sorted(enumerate(segments), key=lambda item: (item[1][0], item[1][1], item[0]))
    for anchor in range(0, len(segments), AVSPEAKD_STRIDE):
        start = segments[anchor][0]
        cap = start + AVSPEAKD_MAX_SECONDS
        if segments[anchor][1] > cap + _CAP_SLACK:
            end = cap
        else:
            end = None
            for index, (seg_start, seg_end, _speaker) in ordered:
                if seg_end <= start:
                    continue
                if seg_start >= cap + _CAP_SLACK:
                    break
                if seg_end > cap + _CAP_SLACK and index != anchor:
                    if seg_start < start:
                        continue
                    break
                end = seg_end
            if end is None:
                continue
        pending: list[tuple[float, float, str, float, float]] = []
        for seg_start, seg_end, speaker in segments:
            if seg_end <= start or seg_start >= end:
                continue
            clipped_start = max(seg_start, start)
            clipped_end = min(seg_end, end)
            if clipped_end <= clipped_start:
                continue
            pending.append((seg_start, seg_end, speaker, clipped_start, clipped_end))
        if not pending:
            continue
        # Equal spans keep the segment-file order, which is also the speaker numbering.
        pending.sort(key=lambda seg: (seg[0], seg[1]))
        chosen = [(clipped_start, clipped_end, speaker) for _start, _end, speaker, clipped_start, clipped_end in pending]
        windows.append((start, end, chosen))
    return windows


def format_diarization_window(
    start: float,
    segments: Sequence[tuple[float, float, str]],
    speakers: dict[str, int],
) -> str:
    lines = []
    for seg_start, seg_end, speaker in segments:
        lines.append(
            f"Speaker {speakers[speaker]}: {_format_diarization_timestamp(seg_start - start)}"
            f" - {_format_diarization_timestamp(seg_end - start)}"
        )
    return "\n".join(lines)


def _speaker_ids(info: dict) -> tuple[str, ...]:
    return tuple(speaker["speaker_id"] for speaker in info.get("speakers", []) if speaker.get("speaker_id"))


def _iter_meeting_context(ami_root: Path, partition: str, include_unsplit: bool, splits: Sequence[str]):
    meetings = load_meeting_index(ami_root)
    allowed = set(splits)
    for meeting_id in list_meeting_ids(ami_root):
        info = meetings.get(meeting_id, {"speakers": [], "duration": None})
        split = split_for_partition(meeting_id, info, partition)
        if split is None and not include_unsplit:
            continue
        if split is None:
            split = "train"
        if split not in allowed:
            continue
        yield meeting_id, split, info


def achap_chunks(
    ami_root: Path,
    *,
    partition: str,
    include_unsplit: bool,
    splits: Sequence[str],
) -> list[AmiChunk]:
    ontology = load_topic_ontology(ami_root)
    words_cache: dict = {}
    chunks: list[AmiChunk] = []
    for meeting_id, split, info in _iter_meeting_context(ami_root, partition, include_unsplit, splits):
        chapters = load_chapters(meeting_id, ami_root, ontology, words_cache)
        speakers = _speaker_ids(info)
        for index, (start, end, chosen) in enumerate(chapter_windows(chapters)):
            chunk_id = f"{meeting_id}_chunk{index:04d}"
            chunks.append(
                AmiChunk(
                    chunk_id=chunk_id,
                    meeting_id=meeting_id,
                    split=split,
                    task="achap",
                    start=start,
                    end=end,
                    text=format_chapter_window(start, chosen),
                    view=None,
                    overlap=None,
                    speaker_ids=speakers,
                )
            )
    return chunks


def avspeakd_chunks(
    ami_root: Path,
    *,
    partition: str,
    include_unsplit: bool,
    splits: Sequence[str],
) -> list[AmiChunk]:
    chunks: list[AmiChunk] = []
    for meeting_id, split, info in _iter_meeting_context(ami_root, partition, include_unsplit, splits):
        views = room_views(meeting_id)
        if not views:
            continue
        agent_to_speaker = {
            speaker["agent"]: speaker["speaker_id"]
            for speaker in info.get("speakers", [])
            if speaker.get("agent")
        }
        segments = split_long_turns(load_meeting_segments(meeting_id, ami_root, agent_to_speaker))
        if not segments:
            continue
        speakers = speaker_index(segments)
        speaker_ids = _speaker_ids(info) or tuple(speakers)
        windows = diarization_windows(segments)
        for view in views:
            for index, (start, end, chosen) in enumerate(windows):
                chunk_id = f"{meeting_id}_{view}_chunk{index:04d}"
                chunks.append(
                    AmiChunk(
                        chunk_id=chunk_id,
                        meeting_id=meeting_id,
                        split=split,
                        task="avspeakd",
                        start=start,
                        end=end,
                        text=format_diarization_window(start, chosen, speakers),
                        view=view,
                        overlap=has_overlapping_speech(chosen),
                        speaker_ids=speaker_ids,
                    )
                )
    return chunks


def chunk_audio_path(signals_root: str, chunk_id: str) -> str:
    return f"{signals_root}/{chunk_id}/audio/{chunk_id}.wav"


def chunk_video_path(signals_root: str, chunk_id: str) -> str:
    return f"{signals_root}/{chunk_id}/video/{chunk_id}.mp4"


def chunk_to_row(chunk: AmiChunk, signals_root: str) -> dict:
    text = chunk.text
    duration = chunk.duration
    words = _word_count(text)
    row = empty_row()
    row.update(
        {
            "sample_id": f"ami_{chunk.split}_{chunk.chunk_id}_{chunk.task}",
            "task": [chunk.task],
            "split": chunk.split,
            "source_dataset": SOURCE,
            "license": LICENSE,
            "speaker_id": list(chunk.speaker_ids) or None,
            "num_speakers": len(chunk.speaker_ids) or None,
            "text_output": text,
            "text_output_normalized": text,
            "num_words_output": words,
            "language_text_output": "en",
            "audio_input_path": chunk_audio_path(signals_root, chunk.chunk_id),
            "audio_input_duration": duration,
            "audio_input_sample_rate": AUDIO_SAMPLE_RATE,
            "audio_input_num_channels": 1,
            "audio_input_codec": AUDIO_CODEC,
            "language_audio_input": "en",
            "words_per_second": (words / duration) if words and duration > 0 else None,
            "chars_per_second": (len(text) / duration) if text and duration > 0 else None,
        }
    )
    if chunk.task == "avspeakd":
        row.update(
            {
                "video_input_path": chunk_video_path(signals_root, chunk.chunk_id),
                "video_input_duration": duration,
                "video_input_fps": float(AVSPEAKD_VIDEO_FPS),
                "video_input_width": AVSPEAKD_VIDEO_WIDTH,
                "video_input_height": AVSPEAKD_VIDEO_HEIGHT,
                "language_video_input": "en",
                "has_overlapping_speech": chunk.overlap,
            }
        )
    return row


def chunks_to_rows(chunks: Sequence[AmiChunk], signals_root: str) -> list[dict]:
    return [chunk_to_row(chunk, signals_root) for chunk in chunks]
