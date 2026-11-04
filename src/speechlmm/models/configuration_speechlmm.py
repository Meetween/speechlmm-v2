"""Configuration for SpeechLMM: a customizable wrapper around Qwen Omni components."""

from transformers import PreTrainedConfig
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _is_qwen2_5_backbone(backbone_type: str | None = None, audio_encoder_type: str | None = None) -> bool:
    return backbone_type == "qwen2_5_omni"


class SpeechLMMConfig(PreTrainedConfig):
    """Configuration for SpeechLMMForConditionalGeneration.

    Wraps Qwen3-Omni or Qwen2.5-Omni sub-configs (Thinker, Talker, Code2Wav /
    Token2Wav) and adds SpeechLMM-specific parameters for encoder swapping,
    loss weighting, and architecture selection.
    """

    model_type = "speechlmm"

    # Default nested types stay Qwen3; from_qwen2_5_omni_config passes live objects.
    sub_configs = {
        "thinker_config": Qwen3OmniMoeThinkerConfig,
        "talker_config": Qwen3OmniMoeTalkerConfig,
        "code2wav_config": Qwen3OmniMoeCode2WavConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        talker_config=None,
        code2wav_config=None,
        # --- SpeechLMM-specific ---
        backbone_type: str = "qwen3_omni",
        audio_encoder_type: str = "qwen3_omni",
        custom_audio_config: dict | None = None,
        vision_encoder_type: str = "qwen3_omni",
        custom_vision_config: dict | None = None,
        code2wav_type: str = "qwen3_omni",
        custom_code2wav_config: dict | None = None,
        enable_vision: bool = True,
        enable_talker: bool = True,
        enable_code2wav: bool = True,
        thinker_loss_weight: float = 1.0,
        talker_loss_weight: float = 1.0,
        # --- Lipread encoder / adapter ---
        lipread_encoder_dim: int = 768,
        lipread_adapter_hidden: int = 1024,
        lipread_encoder_weights: str | None = None,
        # --- Token IDs ---
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        tts_pad_token_id: int = 151671,
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        system_token_id: int = 8948,
        user_token_id: int = 872,
        assistant_token_id: int = 77091,
        lipread_bos_token_id: int = 151676,
        lipread_eos_token_id: int = 151678,
        lipread_pad_token_id: int = 151677,
        speaker_bos_token_id: int | None = None,
        talker_speaker: str = "ethan",
        talker_loss_reduction: str = "token",
        talker_num_codebooks: int | None = None,
        **kwargs,
    ):
        use_qwen2_5 = _is_qwen2_5_backbone(backbone_type, audio_encoder_type)

        # --- Build sub-configs from dicts if needed ---
        if thinker_config is None:
            thinker_config = {}
            logger.info("thinker_config is None, using defaults")
        if isinstance(thinker_config, dict):
            thinker_cls = Qwen2_5OmniThinkerConfig if use_qwen2_5 else Qwen3OmniMoeThinkerConfig
            self.thinker_config = thinker_cls(**thinker_config)
        else:
            self.thinker_config = thinker_config

        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None, using defaults")
        if isinstance(talker_config, dict):
            talker_cls = Qwen2_5OmniTalkerConfig if use_qwen2_5 else Qwen3OmniMoeTalkerConfig
            self.talker_config = talker_cls(**talker_config)
        else:
            self.talker_config = talker_config

        if code2wav_config is None:
            code2wav_config = {}
            logger.info("code2wav_config is None, using defaults")
        if isinstance(code2wav_config, dict):
            code2wav_cls = (
                Qwen2_5OmniToken2WavConfig if use_qwen2_5 else Qwen3OmniMoeCode2WavConfig
            )
            self.code2wav_config = code2wav_cls(**code2wav_config)
        else:
            self.code2wav_config = code2wav_config

        # --- SpeechLMM-specific params ---
        self.backbone_type = backbone_type
        self.audio_encoder_type = audio_encoder_type
        self.custom_audio_config = custom_audio_config or {}
        self.vision_encoder_type = vision_encoder_type
        self.custom_vision_config = custom_vision_config or {}
        self.code2wav_type = code2wav_type
        self.custom_code2wav_config = custom_code2wav_config or {}

        self.enable_vision = enable_vision
        self.enable_talker = enable_talker
        self.enable_code2wav = enable_code2wav
        self.thinker_loss_weight = thinker_loss_weight
        self.talker_loss_weight = talker_loss_weight
        self.lipread_encoder_dim = lipread_encoder_dim
        self.lipread_adapter_hidden = lipread_adapter_hidden
        self.lipread_encoder_weights = lipread_encoder_weights
        # TODO(P1): Add codec format config (codec_type, vocab_size,
        #   num_codebooks) for RVQ vs FSQ swapping.
        # TODO(P2): Add speaker conditioning config (speaker_encoder_type,
        #   speaker_embed_dim, num_speakers).

        # --- Token IDs ---
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        if use_qwen2_5:
            tts_pad_token_id = getattr(self.talker_config, "tts_text_pad_token_id", tts_pad_token_id)
            tts_bos_token_id = getattr(self.talker_config, "tts_text_start_token_id", tts_bos_token_id)
            tts_eos_token_id = getattr(self.talker_config, "tts_text_end_token_id", tts_eos_token_id)
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.system_token_id = system_token_id
        self.user_token_id = user_token_id
        self.assistant_token_id = assistant_token_id
        self.lipread_bos_token_id = lipread_bos_token_id
        self.lipread_eos_token_id = lipread_eos_token_id
        self.lipread_pad_token_id = lipread_pad_token_id
        self.speaker_bos_token_id = speaker_bos_token_id
        self.talker_speaker = talker_speaker
        if talker_loss_reduction not in ("token", "sample"):
            raise ValueError(f"talker_loss_reduction must be 'token' or 'sample', got {talker_loss_reduction!r}")
        self.talker_loss_reduction = talker_loss_reduction
        if talker_num_codebooks is None:
            talker_num_codebooks = 1 if use_qwen2_5 else int(getattr(self.talker_config, "num_code_groups", 16))
        self.talker_num_codebooks = talker_num_codebooks

        self.initializer_range = getattr(
            self.thinker_config,
            "initializer_range",
            getattr(self.thinker_config, "init_std", 0.02),
        )
        self.hidden_size = self.thinker_config.text_config.hidden_size
        self.tie_word_embeddings = getattr(self.thinker_config, "tie_word_embeddings", False)

        # PreTrainedConfig.__init__ overwrites `_attn_implementation` on every sub-config
        # (including the shared Thinker), so pass the Thinker's backend through.
        if kwargs.get("attn_implementation") is None:
            thinker = self.thinker_config
            inherited = getattr(thinker, "_attn_implementation", None) or getattr(
                thinker.text_config, "_attn_implementation", None
            )
            if inherited is not None:
                kwargs["attn_implementation"] = inherited

        super().__init__(**kwargs)

    # Proxy properties for token IDs — allows get_rope_index and other
    # methods that expect flat config access to work transparently.
    # Qwen2.5-Omni stores some of these as *_token_index.
    @property
    def image_token_id(self):
        return getattr(
            self.thinker_config,
            "image_token_id",
            getattr(self.thinker_config, "image_token_index", None),
        )

    @property
    def video_token_id(self):
        return getattr(
            self.thinker_config,
            "video_token_id",
            getattr(self.thinker_config, "video_token_index", None),
        )

    @property
    def audio_token_id(self):
        return getattr(
            self.thinker_config,
            "audio_token_id",
            getattr(self.thinker_config, "audio_token_index", None),
        )

    @property
    def vision_start_token_id(self):
        return self.thinker_config.vision_start_token_id

    @property
    def audio_start_token_id(self):
        return self.thinker_config.audio_start_token_id

    @property
    def audio_end_token_id(self):
        return getattr(self.thinker_config, "audio_end_token_id", None)

    @property
    def position_id_per_seconds(self):
        return self.thinker_config.position_id_per_seconds

    @property
    def seconds_per_chunk(self):
        return getattr(self.thinker_config, "seconds_per_chunk", 2)

    @property
    def is_qwen2_5_backbone(self) -> bool:
        return _is_qwen2_5_backbone(self.backbone_type, self.audio_encoder_type)

    def get_text_config(self, decoder=False) -> PreTrainedConfig:
        return self.thinker_config.get_text_config()

    @staticmethod
    def sync_lipread_token_ids_from_tokenizer(config: "SpeechLMMConfig", tokenizer) -> None:
        """Set ``lipread_*_token_id`` on *config* from *tokenizer* vocabulary."""
        if getattr(config, "model_type", None) != "speechlmm":
            return
        try:
            from speechlmm.tokens import LIPREAD_BOS_TOKEN, LIPREAD_EOS_TOKEN, LIPREAD_PAD_TOKEN
        except ImportError:
            LIPREAD_BOS_TOKEN = "<|lipread_start|>"
            LIPREAD_PAD_TOKEN = "<|lipread_pad|>"
            LIPREAD_EOS_TOKEN = "<|lipread_end|>"

        vocab = tokenizer.get_vocab()
        for attr, token in (
            ("lipread_bos_token_id", LIPREAD_BOS_TOKEN),
            ("lipread_pad_token_id", LIPREAD_PAD_TOKEN),
            ("lipread_eos_token_id", LIPREAD_EOS_TOKEN),
        ):
            if token in vocab:
                setattr(config, attr, vocab[token])

    @staticmethod
    def sync_talker_token_ids_from_tokenizer(config: "SpeechLMMConfig", tokenizer) -> None:
        if getattr(config, "model_type", None) != "speechlmm":
            return
        vocab = tokenizer.get_vocab()
        for attr, token in (
            ("im_start_token_id", "<|im_start|>"),
            ("im_end_token_id", "<|im_end|>"),
        ):
            if token in vocab:
                setattr(config, attr, vocab[token])

    def resolve_talker_speaker_id(self, name: str | None = None, speaker_map: dict | None = None) -> int:
        key = (name or self.talker_speaker or "ethan")
        key_l = str(key).lower()
        speaker_id = getattr(self.talker_config, "speaker_id", None)
        if isinstance(speaker_id, dict):
            if key_l in speaker_id:
                return int(speaker_id[key_l])
            if key in speaker_id:
                return int(speaker_id[key])
            raise ValueError(f"Unknown Talker speaker {key!r}; available: {list(speaker_id)}")
        if speaker_map:
            for map_key, value in speaker_map.items():
                if str(map_key).lower() == key_l:
                    bos = value["bos_token"] if isinstance(value, dict) else value
                    return int(bos)
        if self.speaker_bos_token_id is not None:
            return int(self.speaker_bos_token_id)
        raise ValueError(f"Unknown Talker speaker {key!r}")

    @classmethod
    def from_qwen3_omni_config(cls, qwen3_config, **overrides):
        """Create a SpeechLMMConfig from an existing Qwen3OmniMoeConfig."""
        overrides.setdefault("backbone_type", "qwen3_omni")
        overrides.setdefault("audio_encoder_type", "qwen3_omni")
        overrides.setdefault("vision_encoder_type", "qwen3_omni")
        overrides.setdefault("code2wav_type", "qwen3_omni")
        return cls(
            thinker_config=qwen3_config.thinker_config,
            talker_config=qwen3_config.talker_config,
            code2wav_config=qwen3_config.code2wav_config,
            im_start_token_id=qwen3_config.im_start_token_id,
            im_end_token_id=qwen3_config.im_end_token_id,
            tts_pad_token_id=qwen3_config.tts_pad_token_id,
            tts_bos_token_id=qwen3_config.tts_bos_token_id,
            tts_eos_token_id=qwen3_config.tts_eos_token_id,
            system_token_id=qwen3_config.system_token_id,
            user_token_id=qwen3_config.user_token_id,
            assistant_token_id=qwen3_config.assistant_token_id,
            **overrides,
        )

    @classmethod
    def from_qwen2_5_omni_config(cls, qwen2_config, **overrides):
        """Create a SpeechLMMConfig from an existing Qwen2_5OmniConfig.

        Qwen2.5 stores the waveform head as ``token2wav_config``; SpeechLMM keeps
        the flat ``code2wav`` name for freeze/LoRA compatibility with Qwen3.
        """
        overrides.setdefault("backbone_type", "qwen2_5_omni")
        overrides.setdefault("audio_encoder_type", "qwen2_5_omni")
        overrides.setdefault("vision_encoder_type", "qwen2_5_omni")
        overrides.setdefault("code2wav_type", "qwen2_5_omni")

        thinker = qwen2_config.thinker_config
        talker = qwen2_config.talker_config
        return cls(
            thinker_config=thinker,
            talker_config=talker,
            code2wav_config=qwen2_config.token2wav_config,
            im_start_token_id=getattr(qwen2_config, "im_start_token_id", 151644),
            im_end_token_id=getattr(qwen2_config, "im_end_token_id", 151645),
            tts_pad_token_id=getattr(talker, "tts_text_pad_token_id", 151859),
            tts_bos_token_id=getattr(talker, "tts_text_start_token_id", 151860),
            tts_eos_token_id=getattr(talker, "tts_text_end_token_id", 151861),
            system_token_id=getattr(qwen2_config, "system_token_id", 8948),
            user_token_id=getattr(
                thinker, "user_token_id", getattr(qwen2_config, "user_token_id", 872)
            ),
            assistant_token_id=getattr(qwen2_config, "assistant_token_id", 77091),
            **overrides,
        )
