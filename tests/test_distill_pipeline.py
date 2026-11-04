"""CPU tests for Omni self-distill → Whisper-scored candidates → codec cache → Talker training lookup."""

from __future__ import annotations

import json
import sys
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LF_SRC = ROOT / "llamafactory" / "src"
for path in (SRC, LF_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llamafactory.data.converter import get_dataset_converter  # noqa: E402
from llamafactory.data.parser import DatasetAttr  # noqa: E402
from llamafactory.hparams import DataArguments  # noqa: E402

from speechlmm.data.codec_cache import CodecCache  # noqa: E402
from speechlmm.data.distill_talker import (  # noqa: E402
    OmniDistillEncoder,
    assistant_text_from_row,
    distill_prompt,
    normalize_text,
)
from speechlmm.data.extract_codec_tokens import (  # noqa: E402
    generate_candidates,
    main,
    select_candidates,
)
from speechlmm.models.talker_loss import pad_codec_batch  # noqa: E402


def _row(sample_id: str, text: str, speaker: str = "ethan") -> dict:
    return {
        "sample_id": sample_id,
        "speaker": speaker,
        "messages": [
            {"role": "user", "content": f"Read this aloud: {text}"},
            {"role": "assistant", "content": text},
        ],
    }


ROWS = [_row("smoke_000", "Hello world"), _row("smoke_001", "Good morning", "chelsie")]


class FakeEncoder:
    """Voices the text after the distill instruction, or a fixed reply when one is set."""

    def __init__(self, n_codebooks: int = 1, reply: str | None = None, talker_max_new_tokens: int = 512):
        self.n_codebooks = n_codebooks
        self.reply = reply
        self.talker_max_new_tokens = talker_max_new_tokens
        self.calls = []

    def encode_text(self, prompt: str, speaker: str, *, sample: bool = True):
        self.calls.append(sample)
        reply = self.reply if self.reply is not None else prompt.split("\n", 1)[1]
        frames = range(len(reply.split()) * 2 + len(self.calls))
        codes = list(frames) if self.n_codebooks == 1 else [[t] * self.n_codebooks for t in frames]
        return codes, reply, [0.0, 0.5, -0.5]


def _score(work: Path, wer_by_attempt: dict[int, float | None]) -> None:
    rows = []
    for line in (work / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        sample_id = json.loads(line)["sample_id"]
        rows.append({"sample_id": sample_id, "score": wer_by_attempt[int(sample_id.rsplit("#", 1)[1])]})
    (work / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class DistillPipelineTests(unittest.TestCase):
    def test_prompt_helpers(self):
        self.assertEqual(assistant_text_from_row(ROWS[0]), "Hello world")
        self.assertEqual(assistant_text_from_row({"text": " Hi "}), "Hi")
        self.assertTrue(distill_prompt("Hello world").endswith("\nHello world"))
        self.assertEqual(normalize_text("Hello,  World!"), "hello world")

    def test_generate_writes_greedy_then_sampled_candidates_and_resumes(self):
        with TemporaryDirectory() as raw:
            work = Path(raw)
            encoder = FakeEncoder()
            stats = generate_candidates(ROWS, encoder=encoder, work_dir=work, samples=2)
            self.assertEqual(stats["candidates"], 6)
            self.assertEqual(encoder.calls, [False, True, True] * 2)
            candidate = json.loads((work / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(candidate["sample_id"], "smoke_000#0")
            self.assertEqual(candidate["language"], "en")
            with wave.open(candidate["audio_path"]) as audio:
                self.assertEqual((audio.getframerate(), audio.getnframes()), (24000, 3))
            again = generate_candidates(ROWS, encoder=encoder, work_dir=work, samples=2)
            self.assertEqual((again["done"], again["candidates"]), (2, 0))

    def test_generate_drops_wrong_replies_and_cut_off_talker_output(self):
        with TemporaryDirectory() as raw:
            stats = generate_candidates(ROWS[:1], encoder=FakeEncoder(reply="Sure! Hello world."), work_dir=Path(raw))
            self.assertEqual((stats["mismatch"], stats["candidates"]), (4, 0))
        with TemporaryDirectory() as raw:
            stats = generate_candidates(ROWS[:1], encoder=FakeEncoder(talker_max_new_tokens=5), work_dir=Path(raw))
            self.assertEqual((stats["truncated"], stats["candidates"]), (4, 0))
        with TemporaryDirectory() as raw:
            encoder = FakeEncoder()
            stats = generate_candidates([{"text": "hi"}], encoder=encoder, work_dir=Path(raw))
            self.assertEqual((stats["missing"], encoder.calls), (1, []))

    def test_select_keeps_the_lowest_wer_candidate_under_the_threshold(self):
        with TemporaryDirectory() as raw:
            work = Path(raw)
            generate_candidates(ROWS, encoder=FakeEncoder(), work_dir=work, samples=3)
            _score(work, {0: 20.0, 1: 0.0, 2: None, 3: 10.0})
            cache = CodecCache(work / "cache", "qwen2_5_omni", "qwen25-distill-v1")
            stats = select_candidates(work, cache=cache)
            self.assertEqual(stats, {"kept": 2, "rejected": 0, "unscored": 2})
            candidates = [json.loads(line) for line in (work / "candidates.jsonl").read_text().splitlines()]
            chosen = next(c for c in candidates if c["sample_id"] == "smoke_001#1")
            record = CodecCache.from_dir(cache.dir).lookup("smoke_001")
            self.assertEqual((record.codes, record.speaker), (chosen["codes"], "chelsie"))

            _score(work, {0: 20.0, 1: 5.0, 2: None, 3: 10.0})
            strict = CodecCache(work / "cache", "qwen2_5_omni", "strict")
            self.assertEqual(select_candidates(work, cache=strict)["rejected"], 2)
            self.assertEqual(select_candidates(work, cache=strict, threshold=5.0)["kept"], 2)

    def test_qwen3_codes_pad_to_sixteen_codebooks(self):
        with TemporaryDirectory() as raw:
            work = Path(raw)
            generate_candidates(ROWS[:1], encoder=FakeEncoder(n_codebooks=16), work_dir=work, samples=0)
            _score(work, {0: 0.0})
            cache = CodecCache(work / "cache", "qwen3_omni", "qwen3-distill-v1")
            select_candidates(work, cache=cache)
            record = CodecCache.from_dir(cache.dir).lookup("smoke_000")
            labels, mask = pad_codec_batch([record.codes], num_codebooks=16, ignore_index=-100)
            self.assertEqual(tuple(labels.shape)[-1], 16)
            self.assertTrue(bool(mask[0]))

    def test_cli_generate_then_select(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            rows_path = root / "talker.json"
            rows_path.write_text(json.dumps(ROWS), encoding="utf-8")
            work = root / "work"
            with patch("speechlmm.data.extract_codec_tokens.OmniDistillEncoder", lambda *a, **k: FakeEncoder()):
                main(["generate", "--backbone", "qwen2_5_omni", "--model", "/unused", "--from-json", str(rows_path), "--work-dir", str(work)])
            _score(work, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
            main(["select", "--backbone", "qwen2_5_omni", "--encoder-tag", "t", "--work-dir", str(work), "--cache-root", str(root / "cache")])
            self.assertIsNotNone(CodecCache(root / "cache", "qwen2_5_omni", "t").lookup("smoke_001"))

    def test_converter_reads_codes_from_the_cache(self):
        if not hasattr(DatasetAttr("file", "talker.json"), "codec_cache"):
            self.skipTest("needs LlamaFactory #38 codec_cache columns")
        with TemporaryDirectory() as raw:
            work = Path(raw)
            generate_candidates(ROWS, encoder=FakeEncoder(), work_dir=work, samples=0)
            _score(work, {0: 0.0})
            cache = CodecCache(work / "cache", "qwen2_5_omni", "qwen25-distill-v1")
            select_candidates(work, cache=cache)
            dataset_attr = DatasetAttr("file", "talker.json")
            dataset_attr.formatting = "sharegpt"
            dataset_attr.messages = "messages"
            dataset_attr.role_tag = "role"
            dataset_attr.content_tag = "content"
            dataset_attr.user_tag = "user"
            dataset_attr.assistant_tag = "assistant"
            dataset_attr.speaker = "speaker"
            dataset_attr.sample_id = "sample_id"
            dataset_attr.codec_cache = str(cache.dir)
            converted = get_dataset_converter("sharegpt", dataset_attr, DataArguments())(ROWS[0])
            self.assertEqual(converted["_speaker"], "ethan")
            self.assertEqual(converted["_codec_tokens"], cache.lookup("smoke_000").codes)
            self.assertEqual(converted["_response"][0]["content"], "Hello world")

            inline = dict(ROWS[0], codec_tokens=[1, 2, 3])
            dataset_attr.codec_tokens = "codec_tokens"
            self.assertEqual(get_dataset_converter("sharegpt", dataset_attr, DataArguments())(inline)["_codec_tokens"], [1, 2, 3])


class _Processor:
    def apply_chat_template(self, conversation, **kwargs):
        return conversation[-1]["content"][0]["text"]

    def __call__(self, text, **kwargs):
        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        return SimpleNamespace(to=lambda device: batch)

    def batch_decode(self, ids, **kwargs):
        self.decoded = ids.tolist()
        return ["Hello world"]


class OmniEncoderTests(unittest.TestCase):
    def _encode(self, backbone: str, model, *, sample: bool = True):
        processor = _Processor()
        encoder = OmniDistillEncoder(backbone, "/unused", device="cpu")
        with patch.object(encoder, "_load", return_value=(model, processor)):
            codes, reply, wav = encoder.encode_text("Read this aloud: Hello world", "ethan", sample=sample)
        return codes, reply, wav, processor

    def test_qwen25_keeps_the_codes_sent_to_token2wav(self):
        class Model:
            device = "cpu"
            speaker_map = ("Chelsie", "Ethan")
            token2wav = torch.nn.Identity()

            def generate(self, input_ids, generation_mode, speaker, talker_do_sample, **kwargs):
                assert generation_mode == "audio" and speaker == "Ethan" and talker_do_sample is False
                self.token2wav(torch.tensor([[10, 11, 12]]))
                return torch.tensor([[1, 2, 3, 7, 8]]), torch.ones(1, 4)

        codes, reply, wav, processor = self._encode("qwen2_5_omni", Model(), sample=False)
        self.assertEqual(codes, [10, 11, 12])
        self.assertEqual(reply, "Hello world")
        self.assertEqual(wav.tolist(), [1.0] * 4)
        self.assertEqual(processor.decoded, [[7, 8]])
        self.assertEqual(len(Model.token2wav._forward_pre_hooks), 0)

    def test_qwen3_keeps_the_codes_sent_to_the_vocoder(self):
        class Vocoder:
            def chunked_decode(self, codes, **kwargs):
                return "wav"

        class Model:
            device = "cpu"
            code2wav = Vocoder()

            def generate(self, input_ids, return_audio, speaker, talker_do_sample, **kwargs):
                assert return_audio and speaker == "ethan" and talker_do_sample is True
                self.code2wav.chunked_decode(torch.arange(32).reshape(1, 16, 2), chunk_size=300)
                return torch.tensor([[1, 2, 3, 7]]), torch.zeros(1, 1, 4)

        model = Model()
        codes, _, _, _ = self._encode("qwen3_omni", model)
        self.assertEqual(len(codes), 2)
        self.assertEqual(codes[0], list(range(0, 32, 2)))
        self.assertNotIn("chunked_decode", vars(model.code2wav))


if __name__ == "__main__":
    unittest.main()
