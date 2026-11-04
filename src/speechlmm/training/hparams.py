"""Dataclass mixins for SpeechLMM-specific training hyperparameters.

Defines per-component LoRA, freeze, and model flags that are mixed into
the training framework's argument classes via multiple inheritance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# ---------------------------------------------------------------
# Per-component LoRA flags
# ---------------------------------------------------------------

LORA_COMPONENTS = (
    "audio_encoder",
    "audio_adapters",
    "language_model",
    "lipread_encoder",
    "lipread_adapter",
    "talker",
)

FREEZE_MAP = {
    "audio_encoder": "freeze_audio_encoder",
    "audio_adapters": "freeze_audio_adapters",
    "language_model": "freeze_language_model",
    "lipread_encoder": "freeze_lipread_encoder",
    "lipread_adapter": "freeze_lipread_adapter",
    "talker": "freeze_talker",
}


@dataclass
class SpeechLMMComponentLoraArguments:
    """Per-component LoRA flags for SpeechLMM sub-modules."""

    lora_audio_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to the audio encoder (Whisper) layers."},
    )
    lora_audio_encoder_rank: int | None = field(
        default=None,
        metadata={"help": "LoRA rank for the audio encoder. Falls back to lora_rank if not set."},
    )
    lora_audio_encoder_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the audio encoder. Falls back to lora_alpha if not set."},
    )
    lora_audio_adapters: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to the audio adapter layers (proj1, proj2). ln_post is auto-added to trainable_module_paths."},
    )
    lora_audio_adapters_rank: int | None = field(
        default=None,
        metadata={"help": "LoRA rank for the audio adapters. Falls back to lora_rank if not set."},
    )
    lora_audio_adapters_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the audio adapters. Falls back to lora_alpha if not set."},
    )
    lora_language_model_scope: str | None = field(
        default=None,
        metadata={
            "help": (
                "Thinker LoRA scope: none, attention, attention_and_ffn, or "
                "attention_and_experts. Preferred over the legacy "
                "lora_language_model and lora_language_model_experts flags."
            )
        },
    )
    lora_language_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Legacy alias enabling Thinker attention and lm_head LoRA. "
                "Prefer lora_language_model_scope."
            )
        },
    )
    lora_language_model_rank: int | None = field(
        default=None,
        metadata={"help": "LoRA rank for the language model. Falls back to lora_rank if not set."},
    )
    lora_language_model_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the language model. Falls back to lora_alpha if not set."},
    )
    lora_language_model_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Legacy alias adding sparse per-expert LoRA to the Thinker MoE FFN "
                "experts. Prefer lora_language_model_scope=attention_and_experts."
            )
        },
    )
    lora_language_model_experts_rank: int | None = field(
        default=None,
        metadata={"help": "LoRA rank for Thinker MoE experts. Falls back to lora_rank if not set."},
    )
    lora_language_model_experts_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for Thinker MoE experts. Falls back to lora_alpha if not set."},
    )
    lora_lipread_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to the lipread encoder layers."},
    )
    lora_lipread_encoder_rank: int | None = field(
        default=None,
        metadata={"help": "LoRA rank for the lipread encoder. Falls back to lora_rank if not set."},
    )
    lora_lipread_encoder_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the lipread encoder. Falls back to lora_alpha if not set."},
    )
    lora_lipread_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to the lipread adapter layer."},
    )
    lora_lipread_adapter_rank: int | None = field(
        default=None,
        metadata={"help": "LoRA rank for the lipread adapter. Falls back to lora_rank if not set."},
    )
    lora_lipread_adapter_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the lipread adapter. Falls back to lora_alpha if not set."},
    )
    lora_talker: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to the Talker (speech generation) sub-model."},
    )
    lora_talker_rank: int | None = field(
        default=None,
        metadata={"help": "LoRA rank for the Talker. Falls back to lora_rank if not set."},
    )
    lora_talker_alpha: int | None = field(
        default=None,
        metadata={"help": "LoRA alpha for the Talker. Falls back to lora_alpha if not set."},
    )

    def _validate_component_lora(self) -> None:
        """Cross-validate component LoRA flags (called from FinetuningArguments.__post_init__)."""
        if self.lora_language_model_scope is not None:
            allowed_scopes = (
                "none",
                "attention",
                "attention_and_ffn",
                "attention_and_experts",
            )
            if self.lora_language_model_scope not in allowed_scopes:
                raise ValueError(
                    "lora_language_model_scope must be one of "
                    f"{allowed_scopes}, got {self.lora_language_model_scope!r}."
                )

            if self.lora_language_model and self.lora_language_model_scope == "none":
                raise ValueError(
                    "lora_language_model_scope='none' conflicts with "
                    "lora_language_model=True."
                )
            if (
                self.lora_language_model_experts
                and self.lora_language_model_scope != "attention_and_experts"
            ):
                raise ValueError(
                    "lora_language_model_experts=True conflicts with "
                    f"lora_language_model_scope={self.lora_language_model_scope!r}."
                )

            self.lora_language_model = self.lora_language_model_scope != "none"
            self.lora_language_model_experts = (
                self.lora_language_model_scope == "attention_and_experts"
            )

        has_component_lora = any(getattr(self, f"lora_{c}") for c in LORA_COMPONENTS)

        if has_component_lora and self.finetuning_type != "lora":  # type: ignore[attr-defined]
            raise ValueError("lora_* component flags require finetuning_type='lora'.")

        if has_component_lora and self.lora_target != ["all"]:  # type: ignore[attr-defined]
            warnings.warn(
                "Both lora_* component flags and lora_target are set. "
                "Component flags take precedence; lora_target will be ignored.",
                UserWarning,
                stacklevel=3,
            )

        for comp, freeze_attr in FREEZE_MAP.items():
            if getattr(self, f"lora_{comp}"):
                setattr(self, freeze_attr, False)

        for comp in LORA_COMPONENTS:
            if getattr(self, f"lora_{comp}"):
                if getattr(self, f"lora_{comp}_rank") is None:
                    setattr(self, f"lora_{comp}_rank", self.lora_rank)  # type: ignore[attr-defined]
                if getattr(self, f"lora_{comp}_alpha") is None:
                    setattr(self, f"lora_{comp}_alpha", self.lora_alpha)  # type: ignore[attr-defined]

        if self.lora_language_model_experts:
            if self.finetuning_type != "lora":  # type: ignore[attr-defined]
                raise ValueError("lora_language_model_experts requires finetuning_type='lora'.")
            if not self.lora_language_model:
                raise ValueError("lora_language_model_experts requires lora_language_model=True.")
            if self.lora_language_model_experts_rank is None:
                self.lora_language_model_experts_rank = self.lora_rank  # type: ignore[attr-defined]
            if self.lora_language_model_experts_alpha is None:
                self.lora_language_model_experts_alpha = self.lora_alpha  # type: ignore[attr-defined]
            if self.lora_language_model_experts_rank is None or self.lora_language_model_experts_rank <= 0:
                raise ValueError("lora_language_model_experts_rank must be a positive integer.")
            if self.lora_language_model_experts_alpha is None or self.lora_language_model_experts_alpha <= 0:
                raise ValueError("lora_language_model_experts_alpha must be a positive integer.")

        if (
            self.finetuning_type == "lora"  # type: ignore[attr-defined]
            and not has_component_lora
            and self.freeze_audio_encoder  # type: ignore[attr-defined]
            and not self.freeze_audio_adapters  # type: ignore[attr-defined]
        ):
            warnings.warn(
                "freeze_audio_encoder=True with freeze_audio_adapters=False has no effect "
                "under old-style lora_target — the 'audio_tower' prefix freezes adapters too. "
                "Use lora_audio_adapters=True for fine-grained LoRA control.",
                UserWarning,
                stacklevel=3,
            )

        if self.lora_audio_adapters and self.finetuning_type == "lora":  # type: ignore[attr-defined]
            paths = list(self.trainable_module_paths or [])  # type: ignore[attr-defined]
            if "audio_tower.ln_post" not in paths:
                paths.append("audio_tower.ln_post")
            self.trainable_module_paths = paths  # type: ignore[attr-defined]
        elif (
            not self.freeze_audio_adapters  # type: ignore[attr-defined]
            and not self.lora_audio_adapters  # type: ignore[attr-defined]
        ):
            paths = list(self.trainable_module_paths or [])  # type: ignore[attr-defined]
            # Qwen3 adapters: proj1/proj2/ln_post. Qwen2.5: proj/ln_post.
            # If the YAML already selected a backbone-specific set, only ensure ln_post.
            if "audio_tower.proj" in paths or "audio_tower.proj1" in paths:
                if "audio_tower.ln_post" not in paths:
                    paths.append("audio_tower.ln_post")
            else:
                for prefix in ("audio_tower.proj1", "audio_tower.proj2", "audio_tower.ln_post"):
                    if prefix not in paths:
                        paths.append(prefix)
            self.trainable_module_paths = paths  # type: ignore[attr-defined]

        if self.lora_lipread_adapter and self.finetuning_type == "lora":  # type: ignore[attr-defined]
            paths = list(self.trainable_module_paths or [])  # type: ignore[attr-defined]
            if "lipread_adapter.ln_post" not in paths:
                paths.append("lipread_adapter.ln_post")
            self.trainable_module_paths = paths  # type: ignore[attr-defined]
        elif (
            not self.freeze_lipread_adapter  # type: ignore[attr-defined]
            and not self.lora_lipread_adapter  # type: ignore[attr-defined]
        ):
            paths = list(self.trainable_module_paths or [])  # type: ignore[attr-defined]
            for prefix in ("lipread_adapter.ln_post", "lipread_adapter.linear_1", "lipread_adapter.linear_2"):
                if prefix not in paths:
                    paths.append(prefix)
            self.trainable_module_paths = paths  # type: ignore[attr-defined]


# ---------------------------------------------------------------
# SpeechLMM-specific freeze flags
# ---------------------------------------------------------------

@dataclass
class SpeechLMMFreezeArguments:
    """Freeze flags for SpeechLMM sub-modules."""

    freeze_audio_encoder: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Whether to freeze the audio encoder (Whisper) layers. "
                "Inherits from freeze_audio_tower if not set."
            )
        },
    )
    freeze_audio_adapters: bool | None = field(
        default=None,
        metadata={
            "help": (
                "Whether to freeze the audio adapter layers (proj1, proj2, ln_post). "
                "Inherits from freeze_audio_tower if not set."
            )
        },
    )
    freeze_talker: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the Talker (speech generation) sub-model in SpeechLMM training."},
    )
    freeze_code2wav: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the Code2Wav (waveform synthesis) sub-model in SpeechLMM training."},
    )
    freeze_code_predictor: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to freeze the Talker's code_predictor (residual codebook predictor) "
                "in SpeechLMM training. Only relevant when freeze_talker is False."
            )
        },
    )
    freeze_lipread_encoder: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the lipread encoder in SpeechLMM training."},
    )
    freeze_lipread_adapter: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the lipread adapter in SpeechLMM training."},
    )

    def _resolve_freeze_defaults(self) -> None:
        """Inherit audio sub-component freeze from freeze_audio_tower when unset."""
        if self.freeze_audio_encoder is None:
            self.freeze_audio_encoder = self.freeze_audio_tower  # type: ignore[attr-defined]
        if self.freeze_audio_adapters is None:
            self.freeze_audio_adapters = self.freeze_audio_tower  # type: ignore[attr-defined]


# ---------------------------------------------------------------
# SpeechLMM-specific model arguments
# ---------------------------------------------------------------

@dataclass
class SpeechLMMModelArguments:
    """Model arguments specific to SpeechLMM."""

    use_speechlmm_wrapper: bool = field(
        default=False,
        metadata={"help": "Wrap a Qwen3-Omni checkpoint in the SpeechLMM model (zero-copy)."},
    )
    lipread_encoder_weights: str | None = field(
        default=None,
        metadata={"help": "Path to pretrained AutoAVSR weights for the lipread encoder."},
    )
    periodic_trainable_save_steps: int | None = field(
        default=None,
        metadata={
            "help": (
                "Every N optimizer steps, save only the trainable_module_paths weights to "
                "checkpoint-N/trainable_modules.safetensors. Requires save_strategy: 'no'."
            )
        },
    )
