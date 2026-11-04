"""Distill Talker codec tokens into an offline cache, filtered by Whisper WER.

1. ``generate`` asks the official Omni model to repeat each row's target text, once with greedy
   and ``--samples`` times with sampled Talker decoding, and writes each candidate's audio and
   codes to ``<work-dir>/candidates.jsonl``. Candidates whose Thinker reply differs from the
   target, or whose Talker output reaches ``--talker-max-new-tokens``, are not written::

      PYTHONPATH=src python -m speechlmm.data.extract_codec_tokens generate \\
        --backbone qwen2_5_omni --model /path/to/Qwen2.5-Omni-7B \\
        --from-json data/talker.json --work-dir /path/to/work

2. ``speechlmm-score-audio`` (speechlmm-evaluation) transcribes every candidate::

      speechlmm-score-audio --audio /path/to/work/candidates.jsonl --result /path/to/work/scores.jsonl

3. ``select`` caches, per row, the lowest-WER candidate at or below ``--threshold``::

      PYTHONPATH=src python -m speechlmm.data.extract_codec_tokens select \\
        --backbone qwen2_5_omni --encoder-tag qwen25-distill-v1 \\
        --work-dir /path/to/work --cache-root /path/to/talker-codec-cache

Training reads the cache through ``codec_cache`` in ``dataset_info.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import wave
from pathlib import Path

import numpy as np

from speechlmm.data.codec_cache import N_CODEBOOKS, CacheRecord, CodecCache
from speechlmm.data.distill_talker import (
    OmniDistillEncoder,
    assistant_text_from_row,
    distill_prompt,
    normalize_text,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


def load_rows(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def write_wav(path: Path, samples) -> None:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def generate_candidates(
    rows: list[dict], *, encoder, work_dir: Path, samples: int = 3, language: str = "en"
) -> dict[str, int]:
    stats = {"rows": 0, "done": 0, "missing": 0, "candidates": 0, "mismatch": 0, "truncated": 0, "failed": 0}
    (work_dir / "audio").mkdir(parents=True, exist_ok=True)
    candidates_path = work_dir / "candidates.jsonl"
    done = set()
    if candidates_path.exists():
        done = {json.loads(line)["row_id"] for line in candidates_path.read_text(encoding="utf-8").splitlines() if line}
    with candidates_path.open("a", encoding="utf-8") as out:
        for index, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or "")
            target = assistant_text_from_row(row)
            if not (sample_id and target):
                logger.warning("Skipping row without sample_id or target text: %s", sample_id or row)
                stats["missing"] += 1
                continue
            if sample_id in done:
                stats["done"] += 1
                continue
            stats["rows"] += 1
            speaker = row.get("speaker") or "ethan"
            for attempt in range(samples + 1):
                try:
                    codes, reply, wav = encoder.encode_text(distill_prompt(target), speaker, sample=attempt > 0)
                except Exception:
                    logger.exception("Failed to distill %s", sample_id)
                    stats["failed"] += 1
                    continue
                if normalize_text(reply) != normalize_text(target):
                    logger.info("Skipping a %s candidate: the Thinker replied %r", sample_id, reply)
                    stats["mismatch"] += 1
                    continue
                if len(codes) >= encoder.talker_max_new_tokens - 1:
                    logger.info("Skipping a %s candidate: the Talker stopped at talker_max_new_tokens", sample_id)
                    stats["truncated"] += 1
                    continue
                audio_path = (work_dir / "audio" / f"{index:08d}_{attempt}.wav").resolve()
                write_wav(audio_path, wav)
                candidate = {
                    "sample_id": f"{sample_id}#{attempt}",
                    "row_id": sample_id,
                    "sampled": attempt > 0,
                    "audio_path": str(audio_path),
                    "text": target,
                    "language": row.get("language") or language,
                    "speaker": speaker,
                    "codes": codes,
                }
                out.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                out.flush()
                stats["candidates"] += 1
    return stats


def select_candidates(work_dir: Path, *, cache: CodecCache, threshold: float = 0.0) -> dict[str, int]:
    scores = {row["sample_id"]: row for row in load_rows(work_dir / "scores.jsonl")}
    best: dict[str, tuple[float, dict]] = {}
    unscored = 0
    for candidate in load_rows(work_dir / "candidates.jsonl"):
        score = scores.get(candidate["sample_id"], {}).get("score")
        if score is None:
            unscored += 1
            continue
        current = best.get(candidate["row_id"])
        if current is None or score < current[0]:
            best[candidate["row_id"]] = (score, candidate)
    stats = {"kept": 0, "rejected": 0, "unscored": unscored}
    try:
        for row_id, (score, candidate) in best.items():
            if score > threshold:
                stats["rejected"] += 1
                continue
            cache.put(CacheRecord(row_id, candidate["codes"], candidate["speaker"]))
            stats["kept"] += 1
    finally:
        cache.flush()
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Distill Talker codec tokens into an offline cache.")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Write greedy and sampled candidates for every row")
    generate.add_argument("--backbone", required=True, choices=sorted(N_CODEBOOKS))
    generate.add_argument("--model", type=Path, required=True, help="Official Omni checkpoint")
    generate.add_argument("--from-json", type=Path, required=True)
    generate.add_argument("--work-dir", type=Path, required=True)
    generate.add_argument("--samples", type=int, default=3, help="Sampled candidates per row, besides greedy")
    generate.add_argument("--language", default="en", help="Language of rows without a language field")
    generate.add_argument("--device", default="cuda")
    generate.add_argument("--thinker-max-new-tokens", type=int, default=256)
    generate.add_argument("--talker-max-new-tokens", type=int, default=512)
    select = commands.add_parser("select", help="Cache the best scored candidate of every row")
    select.add_argument("--backbone", required=True, choices=sorted(N_CODEBOOKS))
    select.add_argument("--encoder-tag", required=True)
    select.add_argument("--work-dir", type=Path, required=True)
    select.add_argument("--cache-root", type=Path, required=True)
    select.add_argument("--threshold", type=float, default=0.0, help="Highest Whisper WER (percent) to keep")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "generate":
        encoder = OmniDistillEncoder(
            args.backbone,
            str(args.model),
            device=args.device,
            thinker_max_new_tokens=args.thinker_max_new_tokens,
            talker_max_new_tokens=args.talker_max_new_tokens,
        )
        stats = generate_candidates(
            load_rows(args.from_json),
            encoder=encoder,
            work_dir=args.work_dir,
            samples=args.samples,
            language=args.language,
        )
    else:
        cache = CodecCache(args.cache_root, args.backbone, args.encoder_tag)
        stats = select_candidates(args.work_dir, cache=cache, threshold=args.threshold)
    logger.info("%s done: %s", args.command, stats)


if __name__ == "__main__":
    main()
