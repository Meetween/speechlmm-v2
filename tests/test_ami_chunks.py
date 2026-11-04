"""Window packing for AMI achap and avspeakd chunks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speechlmm.data.processing.ami.chunks import (  # noqa: E402
    chapter_windows,
    diarization_windows,
    format_chapter_window,
    format_diarization_window,
    speaker_index,
    split_long_turns,
)


class ChapterWindowTests(unittest.TestCase):
    def test_packs_while_span_fits_and_strides_by_one(self):
        # ES2002a topic spans. The released achap clips use these durations.
        chapters = [
            {"start_sec": 50.42, "end_sec": 88.71, "title": "introduction of participants and their roles"},
            {"start_sec": 89.30, "end_sec": 151.79, "title": "project goals and design process"},
            {"start_sec": 151.79, "end_sec": 453.61, "title": "drawing animals on the whiteboard"},
            {"start_sec": 455.51, "end_sec": 531.81, "title": "project budget"},
            {"start_sec": 531.83, "end_sec": 735.70, "title": "possible issues with project goals"},
            {"start_sec": 753.67, "end_sec": 976.83, "title": "initial ideas about RC design"},
            {"start_sec": 977.17, "end_sec": 1109.45, "title": "closing"},
        ]
        windows = chapter_windows(chapters)
        durations = [round(end - start, 2) for start, end, _chosen in windows]
        self.assertEqual(durations, [685.28, 646.40, 583.91, 653.94, 577.62, 355.78, 132.28])
        text = format_chapter_window(windows[0][0], windows[0][2])
        self.assertTrue(text.startswith("00:00 - introduction of participants and their roles\n00:39 - "))

    def test_long_chapter_is_cut_at_twelve_minutes(self):
        chapters = [{"start_sec": 10.0, "end_sec": 2000.0, "title": "long"}]
        start, end, chosen = chapter_windows(chapters)[0]
        self.assertEqual((start, end, len(chosen)), (10.0, 730.0, 1))
        self.assertEqual(format_chapter_window(start, chosen), "00:00 - long")


class DiarizationWindowTests(unittest.TestCase):
    def test_stride_drops_segments_that_cross_the_cap(self):
        segments = [
            (0.0, 10.0, "A"),
            (5.0, 20.0, "B"),
            (30.0, 40.0, "A"),
            (100.0, 130.0, "C"),  # runs past the first window's 120s cap
            (110.0, 115.0, "A"),
        ]
        windows = diarization_windows(segments)
        self.assertEqual([start for start, _end, _chosen in windows], [0.0, 100.0])
        _start, end, chosen = windows[0]
        self.assertEqual(end, 40.0)
        self.assertEqual([speaker for _s, _e, speaker in chosen], ["A", "B", "A"])

    def test_overlapping_turn_is_clipped_to_the_last_turn_that_fits(self):
        # The later turn ends first. The clip ends there, and the earlier
        # turn is cut back to it. A turn that starts inside and runs past the
        # cap is kept, clipped, and does not extend the clip.
        segments = [
            (0.0, 10.0, "A"),
            (50.0, 115.0, "B"),
            (60.0, 110.0, "C"),
            (100.0, 140.0, "A"),
        ]
        start, end, chosen = diarization_windows(segments)[0]
        self.assertEqual((start, end), (0.0, 110.0))
        self.assertEqual(chosen, [(0.0, 10.0, "A"), (50.0, 110.0, "B"), (60.0, 110.0, "C"), (100.0, 110.0, "A")])

    def test_anchor_turn_longer_than_two_minutes_is_hard_cut(self):
        segments = [(0.0, 300.0, "A"), (10.0, 20.0, "B")]
        start, end, chosen = diarization_windows(segments)[0]
        self.assertEqual((start, end), (0.0, 120.0))
        self.assertEqual(chosen[0], (0.0, 120.0, "A"))
        self.assertIn((10.0, 20.0, "B"), chosen)

    def test_same_start_turns_follow_end_time(self):
        segments = [(0.0, 10.0, "A"), (1.0, 8.0, "C"), (1.0, 3.0, "B")]
        _start, _end, chosen = diarization_windows(segments)[0]
        self.assertEqual([speaker for _s, _e, speaker in chosen], ["A", "B", "C"])

    def test_equal_spans_keep_segment_file_order(self):
        segments = [(0.0, 5.0, "B"), (0.0, 5.0, "A")]
        _start, _end, chosen = diarization_windows(segments)[0]
        self.assertEqual([speaker for _s, _e, speaker in chosen], ["B", "A"])

    def test_same_start_turn_that_fits_is_not_hidden_by_a_crossing_turn(self):
        segments = [
            (0.0, 10.0, "A"),
            (50.0, 130.0, "B"),
            (50.0, 80.0, "C"),
        ]
        start, end, chosen = diarization_windows(segments)[0]
        self.assertEqual((start, end), (0.0, 80.0))
        self.assertEqual(chosen, [(0.0, 10.0, "A"), (50.0, 80.0, "C"), (50.0, 80.0, "B")])

    def test_shared_start_numbers_the_shorter_turn_first(self):
        segments = [(0.0, 5.0, "A"), (0.0, 2.0, "B"), (3.0, 4.0, "C")]
        self.assertEqual(list(speaker_index(segments)), ["B", "A", "C"])

    def test_long_turn_is_split_in_place_before_the_stride(self):
        # The long turn sits on an index that makes its remainder the next
        # anchor. A backchannel during the turn stays after that remainder.
        segments = [(0.0, 1.0, "A"), (2.0, 3.0, "A"), (10.0, 138.4, "B"), (50.0, 51.0, "C")]
        split = split_long_turns(segments)
        self.assertEqual(
            split,
            [(0.0, 1.0, "A"), (2.0, 3.0, "A"), (10.0, 130.0, "B"), (130.0, 138.4, "B"), (50.0, 51.0, "C")],
        )
        starts = [start for start, _end, _chosen in diarization_windows(split)]
        self.assertEqual(starts, [0.0, 130.0])

    def test_turn_that_lands_on_the_cap_is_kept(self):
        # 0.936 + 120 is not exact in binary floating point. The turn that
        # the annotations put on the two-minute mark still belongs in the clip.
        start = 0.936
        segments = [(start, start + 120, "A"), (start + 120, start + 121, "B")]
        window_start, end, chosen = diarization_windows(segments)[0]
        self.assertAlmostEqual(end - window_start, 120.0, places=4)
        self.assertEqual([speaker for _s, _e, speaker in chosen], ["A"])

    def test_speaker_numbers_follow_the_whole_meeting(self):
        segments = [(0.0, 1.0, "A"), (1.0, 2.0, "B"), (2.0, 3.0, "A")]
        speakers = speaker_index(segments)
        text = format_diarization_window(1.0, [(1.0, 2.0, "B"), (2.0, 3.0, "A")], speakers)
        self.assertEqual(text, "Speaker 1: 0:00 - 0:01\nSpeaker 0: 0:01 - 0:02")


class RoomViewTests(unittest.TestCase):
    def test_ts_overview_follows_the_mirror(self):
        from speechlmm.data.processing.ami.chunks import room_views

        self.assertEqual(room_views("ES2002a"), ("Corner",))
        self.assertEqual(room_views("IS1000a"), ("C",))
        self.assertEqual(room_views("TS3007d"), ("Overview1", "Overview2"))
        self.assertEqual(room_views("TS3008a"), ("Overview2",))
        self.assertEqual(room_views("TS3008b"), ())
        self.assertEqual(room_views("TS3012d"), ("Overview2",))


if __name__ == "__main__":
    unittest.main()
