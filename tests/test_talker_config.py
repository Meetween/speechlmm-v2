"""Talker config / token-id tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TalkerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from speechlmm.models.configuration_speechlmm import SpeechLMMConfig
        except Exception as exc:
            raise unittest.SkipTest(f"transformers/SpeechLMMConfig unavailable: {exc}") from exc
        cls.SpeechLMMConfig = SpeechLMMConfig

    def test_qwen25_tts_ids_from_talker_config(self):
        talker = SimpleNamespace(
            tts_text_pad_token_id=151859,
            tts_text_start_token_id=151860,
            tts_text_end_token_id=151861,
        )
        qwen2 = SimpleNamespace(
            thinker_config=SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=8, vocab_size=32),
                vision_config=SimpleNamespace(spatial_merge_size=2),
                initializer_range=0.02,
                tie_word_embeddings=False,
                user_token_id=872,
            ),
            talker_config=talker,
            token2wav_config=SimpleNamespace(),
        )
        cfg = self.SpeechLMMConfig.from_qwen2_5_omni_config(qwen2)
        self.assertEqual(cfg.tts_pad_token_id, 151859)
        self.assertEqual(cfg.tts_bos_token_id, 151860)
        self.assertEqual(cfg.tts_eos_token_id, 151861)
        self.assertNotEqual(cfg.tts_pad_token_id, 151671)
        self.assertEqual(cfg.talker_num_codebooks, 1)

    def test_qwen3_tts_ids_from_top_level(self):
        qwen3 = SimpleNamespace(
            thinker_config=SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=8, vocab_size=32),
                vision_config=SimpleNamespace(spatial_merge_size=2),
                initializer_range=0.02,
                tie_word_embeddings=False,
            ),
            talker_config=SimpleNamespace(
                num_code_groups=16,
                accept_hidden_layer=24,
                speaker_id={"ethan": 2302, "chelsie": 2301},
            ),
            code2wav_config=SimpleNamespace(),
            im_start_token_id=151644,
            im_end_token_id=151645,
            tts_pad_token_id=151671,
            tts_bos_token_id=151672,
            tts_eos_token_id=151673,
            system_token_id=8948,
            user_token_id=872,
            assistant_token_id=77091,
        )
        cfg = self.SpeechLMMConfig.from_qwen3_omni_config(qwen3)
        self.assertEqual(cfg.tts_pad_token_id, 151671)
        self.assertEqual(cfg.tts_bos_token_id, 151672)
        self.assertEqual(cfg.tts_eos_token_id, 151673)
        self.assertEqual(cfg.talker_num_codebooks, 16)
        self.assertEqual(cfg.resolve_talker_speaker_id("ethan"), 2302)
        with self.assertRaises(ValueError):
            cfg.resolve_talker_speaker_id("not-a-speaker")

    def test_accept_hidden_layer_hook_values(self):
        from speechlmm.models.modeling_speechlmm_qwen2_5_omni import (
            SpeechLMMQwen2_5OmniForConditionalGeneration,
        )
        from speechlmm.models.modeling_speechlmm_qwen3_omni import (
            SpeechLMMQwen3OmniForConditionalGeneration,
        )

        self.assertEqual(SpeechLMMQwen2_5OmniForConditionalGeneration.talker_accept_hidden_layer(SimpleNamespace()), -1)
        xl = SimpleNamespace(config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24)))
        self.assertEqual(SpeechLMMQwen3OmniForConditionalGeneration.talker_accept_hidden_layer(xl), 24)


if __name__ == "__main__":
    unittest.main()
