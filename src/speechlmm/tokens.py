"""Single source of truth for all SpeechLMM-specific special tokens.

LlamaFactory files import from here instead of hard-coding token strings
in ``llamafactory.extras.constants``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llamafactory.hparams import ModelArguments

# ------------------------------------------------------------------
# Lipread tokens (static — tied to the lipread encoder framing)
# ------------------------------------------------------------------

LIPREAD_PLACEHOLDER = "<lipread>"
LIPREAD_BOS_TOKEN = "<|lipread_start|>"
LIPREAD_PAD_TOKEN = "<|lipread_pad|>"
LIPREAD_EOS_TOKEN = "<|lipread_end|>"
LIPREAD_FRAME_SIZE = 88
LIPREAD_FPS = 25
DUMMY_LIPREAD_FRAMES = 16

LIPREAD_SPECIAL_TOKENS = (LIPREAD_BOS_TOKEN, LIPREAD_PAD_TOKEN, LIPREAD_EOS_TOKEN)


def lipread_special_tokens() -> list[str]:
    """Lipread tokens to register on the tokenizer (training or export)."""
    return list(LIPREAD_SPECIAL_TOKENS)


def build_speechlmm_special_tokens(model_args: ModelArguments) -> list[str]:
    """Return the full list of special tokens to add to the tokenizer."""
    del model_args  # reserved for future conditional tokens
    return lipread_special_tokens()
