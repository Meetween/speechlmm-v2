"""The attention backend chosen on a Qwen-Omni config survives SpeechLMM wrapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AttentionImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import Qwen2_5OmniConfig, Qwen3OmniMoeConfig

            from speechlmm.models.configuration_speechlmm import SpeechLMMConfig
        except Exception as exc:
            raise unittest.SkipTest(f"transformers/SpeechLMMConfig unavailable: {exc}") from exc
        cls.families = (
            (Qwen2_5OmniConfig, SpeechLMMConfig.from_qwen2_5_omni_config),
            (Qwen3OmniMoeConfig, SpeechLMMConfig.from_qwen3_omni_config),
        )
        cls.SpeechLMMConfig = SpeechLMMConfig

    def _qwen(self, config_cls, backend):
        qwen = config_cls()
        qwen._attn_implementation = backend
        return qwen

    def test_wrapping_keeps_the_requested_backend(self):
        for config_cls, wrap in self.families:
            for backend in ("flash_attention_2", "sdpa"):
                with self.subTest(family=config_cls.__name__, backend=backend):
                    qwen = self._qwen(config_cls, backend)
                    wrapped = wrap(qwen)
                    self.assertEqual(wrapped._attn_implementation, backend)
                    self.assertEqual(qwen.thinker_config._attn_implementation, backend)
                    self.assertEqual(qwen.thinker_config.text_config._attn_implementation, backend)
                    self.assertEqual(qwen.thinker_config.audio_config._attn_implementation, backend)
                    self.assertEqual(qwen.thinker_config.vision_config._attn_implementation, backend)

    def test_explicit_backend_wins(self):
        for config_cls, wrap in self.families:
            with self.subTest(family=config_cls.__name__):
                qwen = self._qwen(config_cls, "flash_attention_2")
                wrapped = wrap(qwen, attn_implementation="sdpa")
                self.assertEqual(wrapped._attn_implementation, "sdpa")


if __name__ == "__main__":
    unittest.main()
