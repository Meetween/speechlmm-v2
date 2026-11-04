"""Talker generate() helpers (no GPU, no checkpoints)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speechlmm.models.talker_generate import (  # noqa: E402
    _remap_speaker,
    _wants_audio,
    attach_omni_container,
    maybe_load_spk_dict,
    speechlmm_generate,
)


class TalkerGenerateTests(unittest.TestCase):
    def test_qwen25_speaker_case_remap(self):
        wrap = SimpleNamespace(
            config=SimpleNamespace(talker_speaker="ethan"),
            speaker_map={"Ethan": {}, "Chelsie": {}},
        )
        kwargs = {"speaker": "ethan"}
        _remap_speaker(wrap, kwargs)
        self.assertEqual(kwargs["speaker"], "Ethan")

    def test_qwen3_empty_speaker_map_keeps_name(self):
        wrap = SimpleNamespace(config=SimpleNamespace(talker_speaker="ethan"), speaker_map={})
        kwargs = {"speaker": "ethan"}
        _remap_speaker(wrap, kwargs)
        self.assertEqual(kwargs["speaker"], "ethan")

    def test_unknown_speaker(self):
        wrap = SimpleNamespace(config=SimpleNamespace(talker_speaker="ethan"), speaker_map={"Ethan": {}})
        with self.assertRaises(ValueError):
            _remap_speaker(wrap, {"speaker": "aiden"})

    def test_wants_audio_flags(self):
        wrap = SimpleNamespace(has_talker=True)
        self.assertTrue(_wants_audio(wrap, {"return_audio": True}))
        self.assertFalse(_wants_audio(wrap, {"return_audio": False}))
        self.assertFalse(_wants_audio(wrap, {"generation_mode": "text"}))
        self.assertTrue(_wants_audio(wrap, {}))
        self.assertFalse(_wants_audio(SimpleNamespace(has_talker=False), {}))

    def test_talker_without_vocoder_returns_text(self):
        captured = {}

        def fake_generate(model, *args, **kwargs):
            captured["kwargs"] = kwargs
            return "ids"

        wrap = SimpleNamespace(
            has_talker=True,
            code2wav=None,
            config=SimpleNamespace(),
            native_generate_cls=None,
        )
        with patch("speechlmm.models.talker_generate.GenerationMixin.generate", fake_generate):
            with patch("speechlmm.models.talker_generate.logger") as mock_logger:
                mock_logger.warning_once = mock_logger.warning
                out = speechlmm_generate(wrap, speaker="ethan")
        self.assertEqual(out, "ids")
        mock_logger.warning.assert_called()
        self.assertIn("returns text only", mock_logger.warning.call_args[0][0])
        self.assertNotIn("speaker", captured["kwargs"])
        self.assertNotIn("return_audio", captured["kwargs"])

    def test_talker_without_vocoder_raises_when_audio_required(self):
        wrap = SimpleNamespace(has_talker=True, code2wav=None, config=SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "talker and code2wav"):
            speechlmm_generate(wrap, return_audio=True)

    def test_text_path_drops_audio_kwargs(self):
        captured = {}

        def fake_generate(model, *args, **kwargs):
            captured["kwargs"] = kwargs
            return "ids"

        wrap = SimpleNamespace(has_talker=False, config=SimpleNamespace(), native_generate_cls=None)
        with patch("speechlmm.models.talker_generate.GenerationMixin.generate", fake_generate):
            out = speechlmm_generate(wrap, speaker="ethan", generation_mode="text")
        self.assertEqual(out, "ids")
        self.assertNotIn("speaker", captured["kwargs"])
        self.assertNotIn("return_audio", captured["kwargs"])
        self.assertNotIn("generation_mode", captured["kwargs"])

    def test_omni_path_uses_wrap_talker_and_vocoder(self):
        class Omni:
            def __init__(self):
                self.talker = "old_talker"
                self.token2wav = "old_t2w"
                self.code2wav = "old_c2w"
                self.has_talker = False
                self.kwargs = None

            def generate(self, *args, **kwargs):
                self.kwargs = kwargs
                return ("text", "wav")

        omni = Omni()
        wrap = SimpleNamespace(
            has_talker=True,
            talker="wrap_talker",
            code2wav="wrap_vocoder",
            thinker="wrap_thinker",
            omni=omni,
            speaker_map={"Ethan": {}},
            config=SimpleNamespace(talker_speaker="ethan", backbone_type="qwen3_omni"),
        )
        text, wav = speechlmm_generate(wrap, speaker="ethan", return_audio=True)
        self.assertEqual((text, wav), ("text", "wav"))
        self.assertEqual(omni.talker, "wrap_talker")
        self.assertEqual(omni.token2wav, "wrap_vocoder")
        self.assertEqual(omni.code2wav, "wrap_vocoder")
        self.assertTrue(omni.has_talker)
        self.assertEqual(omni.kwargs["speaker"], "Ethan")
        self.assertNotIn("generation_mode", omni.kwargs)

    def test_missing_vocoder(self):
        wrap = SimpleNamespace(
            has_talker=True,
            code2wav=None,
            config=SimpleNamespace(backbone_type="qwen3_omni"),
            speaker_map={},
        )
        with self.assertRaises(ValueError):
            speechlmm_generate(wrap, return_audio=True)

    def test_return_audio_without_talker_raises(self):
        wrap = SimpleNamespace(has_talker=False, code2wav=None, config=SimpleNamespace())
        with self.assertRaises(ValueError):
            speechlmm_generate(wrap, return_audio=True)

    def test_text_fallback_without_vocoder(self):
        captured = {}

        def fake_generate(model, *args, **kwargs):
            captured["kwargs"] = kwargs
            return "ids"

        wrap = SimpleNamespace(
            has_talker=True,
            talker=object(),
            code2wav=None,
            config=SimpleNamespace(),
        )
        with patch("speechlmm.models.talker_generate.GenerationMixin.generate", fake_generate):
            out = speechlmm_generate(wrap, speaker="ethan")
        self.assertEqual(out, "ids")
        self.assertNotIn("speaker", captured["kwargs"])
        self.assertNotIn("return_audio", captured["kwargs"])

    def test_generate_requires_omni_container(self):
        wrap = SimpleNamespace(
            has_talker=True,
            code2wav=object(),
            omni=None,
            native_thinker_cls=None,
            native_generate_cls=None,
            native_omni_config=None,
            speaker_map={},
            config=SimpleNamespace(backbone_type="qwen3_omni", talker_speaker="ethan"),
        )
        with self.assertRaises(RuntimeError):
            speechlmm_generate(wrap, return_audio=True)

    def test_qwen25_dit_rope_fix_applies_on_tf52(self):
        try:
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as q25
        except Exception as exc:
            self.skipTest(str(exc))
        import speechlmm.models.talker_generate as tg

        nested = "rotate_half_codec" in q25.apply_rotary_pos_emb.__code__.co_varnames
        before = q25.apply_rotary_pos_emb
        tg._Q25_DIT_ROPE_FIXED = False
        tg._fix_qwen25_token2wav_rope()
        if nested:
            self.assertIs(q25.apply_rotary_pos_emb, tg._llama_apply_rotary_pos_emb)
        else:
            self.assertIs(q25.apply_rotary_pos_emb, before)

    def test_attach_omni_container_rebuilds_thinker(self):
        class Thinker:
            generation_config = SimpleNamespace(eos_token_id=None)

        class Omni:
            def generate(self, *args, **kwargs):
                return ("ids", "wav")

        class Vocoder:
            def __init__(self):
                self.n_float = 0

            def float(self):
                self.n_float += 1
                return self

        audio, visual, text, lm = object(), object(), object(), object()
        talker = object()
        vocoder = Vocoder()
        wrap = SimpleNamespace(
            omni=None,
            thinker=None,
            has_talker=True,
            audio_tower=audio,
            visual=visual,
            model=text,
            lm_head=lm,
            talker=talker,
            code2wav=vocoder,
            vocab_size=16,
            spatial_merge_size=2,
            rope_deltas=None,
            speaker_map={"Ethan": {"bos_token": 1}},
            generation_config="gen",
            native_thinker_cls=Thinker,
            native_generate_cls=Omni,
            native_omni_config=lambda cfg: SimpleNamespace(enable_audio_output=True),
            config=SimpleNamespace(
                backbone_type="qwen2_5_omni",
                thinker_config="thinker_cfg",
                im_end_token_id=151645,
            ),
        )
        with patch("transformers.PreTrainedModel.__init__", lambda self, config=None, *a, **k: None):
            attach_omni_container(wrap)
        self.assertIsInstance(wrap.omni, Omni)
        self.assertIsInstance(wrap.thinker, Thinker)
        self.assertEqual(wrap.thinker.generation_config.eos_token_id, 151645)
        self.assertIs(wrap.omni.thinker, wrap.thinker)
        self.assertIs(wrap.thinker.model, text)
        self.assertIs(wrap.thinker.audio_tower, audio)
        self.assertIs(wrap.omni.talker, talker)
        self.assertIs(wrap.omni.token2wav, vocoder)
        self.assertEqual(wrap.omni.speaker_map["Ethan"]["bos_token"], 1)
        self.assertGreaterEqual(vocoder.n_float, 1)

        text_ids, wav = speechlmm_generate(wrap, speaker="ethan", return_audio=True)
        self.assertEqual((text_ids, wav), ("ids", "wav"))

    def test_attach_keeps_live_omni(self):
        existing = SimpleNamespace()
        wrap = SimpleNamespace(
            omni=existing,
            has_talker=True,
            talker="t",
            code2wav=None,
            speaker_map={},
            config=SimpleNamespace(backbone_type="qwen3_omni"),
        )
        attach_omni_container(wrap)
        self.assertIs(wrap.omni, existing)
        self.assertEqual(wrap.omni.talker, "t")

    def test_maybe_load_spk_dict(self):
        import tempfile

        import torch

        wrap = SimpleNamespace(speaker_map={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spk_dict.pt"
            torch.save({"Ethan": {"bos_token": 7}}, path)
            maybe_load_spk_dict(wrap, tmp)
        self.assertEqual(wrap.speaker_map["Ethan"]["bos_token"], 7)


if __name__ == "__main__":
    unittest.main()
