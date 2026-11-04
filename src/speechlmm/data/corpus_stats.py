"""Corpus statistics from the SpeechLMM-v2 metadata Parquet.

Corpus-agnostic: reads every ``parquet/*.parquet`` under a dataset root and reports the
numbers our dataset READMEs need, broken down by task, by split and by source language:

  * sample counts and audio/video hours;
  * audio/video input-clip duration distributions (mean / median / min / max), in seconds;
  * output word-count distribution (mean / median / min / max).

The source-language breakdown is used for every task (ASR, ST, MT, ...): for ST/MT a
source clip is counted once for its source language regardless of how many target
languages it is translated into, so per-source-language hours stay additive.

Audio and video statistics are computed independently over *unique* input clips,
deduplicated by their respective media paths within each group. Media shared across
tasks/directions is counted once in corpus, split, and language totals. Sample and word
counts are over rows (one row = one training sample).

Works for any corpus following the schema in ``schema.py`` (ASR, ST, MT, chaptering,
summarization, ...).

Usage:
    # by corpus name under $DATASETS_DIR/speechlmm-v2 (or SCRATCH_DIR/data/...)
    PYTHONPATH=src python3 -m speechlmm.data.corpus_stats --corpus europarl_st

    # or point straight at a dataset dir / a parquet dir
    PYTHONPATH=src python3 -m speechlmm.data.corpus_stats --dataset-root /path/to/common_voice

    # emit GitHub-Markdown tables (for pasting into a README)
    PYTHONPATH=src python3 -m speechlmm.data.corpus_stats --corpus ami --markdown
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from speechlmm.data.utils import datasets_dir, resolve_parquet_dir

logger = logging.getLogger(__name__)

# Columns read from the Parquet (a subset of the full schema); missing ones are tolerated.
_COLUMNS = [
    "task",
    "split",
    "audio_input_duration",
    "audio_input_path",
    "video_input_duration",
    "video_input_path",
    "num_words_output",
    "language_audio_input",
    "language_text_input",
    "language_video_input",
    "language_text_output",
]


def _first_task(value) -> str:
    """The schema stores ``task`` as a list; report its first (only) element."""
    if isinstance(value, str):
        return value
    if value is None:
        return "?"
    try:
        return str(value[0]) if len(value) else "?"
    except TypeError:
        return "?"


def load_frame(parquet_dir: Path) -> tuple[pd.DataFrame, int]:
    """Read every Parquet under ``parquet_dir`` into one normalized DataFrame."""
    files = sorted(glob.glob(str(parquet_dir / "*.parquet")))
    frames = []
    for f in files:
        names = set(pq.read_schema(f).names)
        cols = [c for c in _COLUMNS if c in names]
        frames.append(pq.read_table(f, columns=cols).to_pandas())
    df = pd.concat(frames, ignore_index=True)  # aligns columns, missing -> NaN
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["task"] = df["task"].map(_first_task)
    df["src"] = (
        df["language_audio_input"]
        .fillna(df["language_video_input"])
        .fillna(df["language_text_input"])
    )
    df["tgt"] = df["language_text_output"]
    return df, len(files)


def _audio_summary(group: pd.DataFrame) -> tuple[int, float, pd.Series]:
    """(#unique clips, hours, unique-clip durations) deduplicated by ``audio_input_path``."""
    clips = group.loc[
        group["audio_input_path"].notna() & group["audio_input_path"].ne(""),
        ["audio_input_path", "audio_input_duration"],
    ].drop_duplicates("audio_input_path")
    durations = clips["audio_input_duration"]
    hours = float(durations.sum(skipna=True)) / 3600.0
    return len(clips), hours, durations


def _video_summary(group: pd.DataFrame) -> tuple[int, float, pd.Series]:
    """(#unique clips, hours, unique-clip durations) deduplicated by ``video_input_path``."""
    clips = group.loc[
        group["video_input_path"].notna() & group["video_input_path"].ne(""),
        ["video_input_path", "video_input_duration"],
    ].drop_duplicates("video_input_path")
    durations = clips["video_input_duration"]
    hours = float(durations.sum(skipna=True)) / 3600.0
    return len(clips), hours, durations


def _dist(
    series: pd.Series,
) -> tuple[float, float, float, float, float] | None:
    """(sum, mean, median, min, max) over non-null values, or None if empty."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return (
        float(s.sum()),
        float(s.mean()),
        float(s.median()),
        float(s.min()),
        float(s.max()),
    )


def _fmt_dist(d) -> str:
    if d is None:
        return "-"
    sum, mean, med, lo, hi = d
    return f"{sum:.1f} / {mean:.1f} / {med:.1f} / {lo:.1f} / {hi:.1f}"


class Table:
    """Minimal aligned / Markdown table printer."""

    def __init__(self, headers: list[str], markdown: bool):
        self.headers = headers
        self.rows: list[list[str]] = []
        self.markdown = markdown

    def add(self, *cells) -> None:
        self.rows.append([str(c) for c in cells])

    def render(self) -> str:
        cols = (
            list(zip(*([self.headers] + self.rows)))
            if self.rows
            else [[h] for h in self.headers]
        )
        widths = [max(len(c) for c in col) for col in cols]
        if self.markdown:
            head = (
                "| "
                + " | ".join(h.ljust(w) for h, w in zip(self.headers, widths))
                + " |"
            )
            sep = "| " + " | ".join("-" * w for w in widths) + " |"
            body = [
                "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
                for row in self.rows
            ]
            return "\n".join([head, sep, *body])
        head = "  ".join(h.ljust(w) for h, w in zip(self.headers, widths))
        body = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in self.rows]
        return "\n".join([head, *body])


def report(corpus: str, df: pd.DataFrame, n_files: int, markdown: bool) -> list[str]:
    splits_order = ["train", "dev", "val", "test"]
    present = [s for s in df["split"].dropna().unique()]
    splits = [s for s in splits_order if s in present]
    splits += sorted(s for s in present if s not in splits)
    tasks = sorted(df["task"].unique())

    audio_clips, audio_hours, _ = _audio_summary(df)
    video_clips, video_hours, _ = _video_summary(df)
    lines = [
        f"# {corpus}",
        f"parquet files: {n_files:,}   rows: {len(df):,}   tasks: {', '.join(tasks)}",
        f"unique input audio: {audio_clips:,} clips / {audio_hours:,.1f} h",
        f"unique input video: {video_clips:,} clips / {video_hours:,.1f} h",
        "",
        "(samples/words are over rows; media hours + duration are over unique clips per group)",
    ]
    t = Table(
        [
            "task",
            "samples",
            "audio clips",
            "audio h",
            "video clips",
            "video h",
            "audio dur s: sum/mean/med/min/max",
            "video dur s: sum/mean/med/min/max",
            "words: sum/mean/med/min/max",
        ],
        markdown,
    )
    for task in tasks:
        g = df[df["task"] == task]
        n_audio, audio_h, audio_durations = _audio_summary(g)
        n_video, video_h, video_durations = _video_summary(g)
        t.add(
            task,
            f"{len(g):,}",
            f"{n_audio:,}",
            f"{audio_h:,.1f}",
            f"{n_video:,}",
            f"{video_h:,.1f}",
            _fmt_dist(_dist(audio_durations)),
            _fmt_dist(_dist(video_durations)),
            _fmt_dist(_dist(g["num_words_output"])),
        )
    lines += ["", "## Per task"]
    lines.append(t.render())

    t = Table(
        ["split", "samples", "audio clips", "audio h", "video clips", "video h"],
        markdown,
    )
    for s in splits:
        g = df[df["split"] == s]
        n_audio, audio_h, _ = _audio_summary(g)
        n_video, video_h, _ = _video_summary(g)
        t.add(
            s,
            f"{len(g):,}",
            f"{n_audio:,}",
            f"{audio_h:,.1f}",
            f"{n_video:,}",
            f"{video_h:,.1f}",
        )
    t.add(
        "total",
        f"{len(df):,}",
        f"{audio_clips:,}",
        f"{audio_hours:,.1f}",
        f"{video_clips:,}",
        f"{video_hours:,.1f}",
    )
    lines += ["", "## Per split (media deduplicated by path)"]
    lines.append(t.render())

    # Per source language only when the corpus is multilingual (otherwise == the total).
    # This is the language breakdown for every task, including ST/MT (a source clip is
    # counted once for its source language, regardless of how many targets it translates to).
    if df["src"].nunique(dropna=True) > 1:
        t = Table(
            [
                "language",
                "samples",
                "audio clips",
                "audio h",
                "video clips",
                "video h",
                "audio dur s: sum/mean/med/min/max",
                "video dur s: sum/mean/med/min/max",
                "words: sum/mean/med/min/max",
            ],
            markdown,
        )
        for src, g in sorted(
            df.groupby("src", dropna=False), key=lambda kv: -len(kv[1])
        ):
            n_audio, audio_h, audio_durations = _audio_summary(g)
            n_video, video_h, video_durations = _video_summary(g)
            t.add(
                f"{src if pd.notna(src) else '?'}",
                f"{len(g):,}",
                f"{n_audio:,}",
                f"{audio_h:,.1f}",
                f"{n_video:,}",
                f"{video_h:,.1f}",
                _fmt_dist(_dist(audio_durations)),
                _fmt_dist(_dist(video_durations)),
                _fmt_dist(_dist(g["num_words_output"])),
            )
        lines += ["", "## Per source language (media deduplicated by path)"]
        lines.append(t.render())

    return lines


def render_md(lines: list[str]) -> None:
    print("\n".join(lines))


def report_csv(corpus: str, df: pd.DataFrame) -> list[dict]:
    splits = sorted(df["split"].dropna().unique())
    tasks = sorted(df["task"].unique())

    stats: list[dict] = []
    for task in tasks:
        for split in splits:
            df_ts = df[(df["task"] == task) & (df["split"] == split)]
            for (src, tgt), g in sorted(
                df_ts.groupby(["src", "tgt"], dropna=False), key=lambda kv: -len(kv[1])
            ):
                audio_clips, audio_hours, audio_durations = _audio_summary(g)
                video_clips, video_hours, video_durations = _video_summary(g)
                audio_dist = _dist(audio_durations) or ("", "", "", "", "")
                video_dist = _dist(video_durations) or ("", "", "", "", "")
                word_dist = _dist(g["num_words_output"]) or ("", "", "", "", "")
                stats.append(
                    {
                        "corpus": corpus,
                        "task": task,
                        "split": split,
                        "src": src,
                        "tgt": tgt,
                        "samples": len(g),
                        "audio_clips": audio_clips,
                        "audio_hours": audio_hours,
                        "audio_total_duration": audio_dist[0],
                        "audio_mean_duration": audio_dist[1],
                        "audio_median_duration": audio_dist[2],
                        "audio_min_duration": audio_dist[3],
                        "audio_max_duration": audio_dist[4],
                        "video_clips": video_clips,
                        "video_hours": video_hours,
                        "video_total_duration": video_dist[0],
                        "video_mean_duration": video_dist[1],
                        "video_median_duration": video_dist[2],
                        "video_min_duration": video_dist[3],
                        "video_max_duration": video_dist[4],
                        "total_num_words": word_dist[0],
                        "mean_num_words": word_dist[1],
                        "median_num_words": word_dist[2],
                        "min_num_words": word_dist[3],
                        "max_num_words": word_dist[4],
                    }
                )
    return stats


def render_csv(stats: list[dict]) -> None:
    import csv
    import sys

    writer = csv.DictWriter(sys.stdout, fieldnames=stats[0].keys())
    writer.writeheader()
    writer.writerows(stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Report corpus statistics from the metadata Parquet."
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
        "--markdown", action="store_true", help="Emit GitHub-Markdown tables"
    )
    parser.add_argument("--csv", action="store_true", help="Emit statistics in csv")
    parser.add_argument("--filter", action="store_true", help="Compute stats on filtered data")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    name, parquet_dir = resolve_parquet_dir(args.corpus, args.dataset_root, 'filtered_' if args.filter else '')
    if not parquet_dir.is_dir():
        raise SystemExit(f"No parquet dir at {parquet_dir}")
    df, n_files = load_frame(parquet_dir)
    if df.empty:
        raise SystemExit(f"No parquet rows under {parquet_dir}")
    if args.csv:
        report_array = report_csv(name, df)
        render_csv(report_array)
    else:
        report_array = report(name, df, n_files, args.markdown)
        render_md(report_array)


if __name__ == "__main__":
    main()
