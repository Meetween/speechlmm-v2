"""Native Omni generate() for SpeechLMM wraps."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers.generation import GenerationMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)

_Q25_DIT_ROPE_FIXED = False


def _deinterleave_head_dim(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)


def _llama_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """HF #47403: deinterleave checkpoint layout, then Llama half-split RoPE."""
    q = _deinterleave_head_dim(q)
    k = _deinterleave_head_dim(k)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    half = q.shape[-1] // 2
    q_embed = (q * cos) + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * sin
    k_embed = (k * cos) + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin
    return q_embed, k_embed


def _fix_qwen25_token2wav_rope() -> None:
    """Token2Wav DiT RoPE on transformers 5.2 (HF #47403). No-op if already fixed."""
    global _Q25_DIT_ROPE_FIXED
    if _Q25_DIT_ROPE_FIXED:
        return
    _Q25_DIT_ROPE_FIXED = True
    try:
        from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as q25
    except Exception:
        return
    if "rotate_half_codec" not in q25.apply_rotary_pos_emb.__code__.co_varnames:
        return
    q25.apply_rotary_pos_emb = _llama_apply_rotary_pos_emb


def _prepare_qwen25_vocoder(wrap) -> None:
    """Official Qwen2.5 Omni keeps token2wav in float32 (BigVGAN vs DiT)."""
    if str(getattr(getattr(wrap, "config", None), "backbone_type", "")) != "qwen2_5_omni":
        return
    vocoder = getattr(wrap, "code2wav", None)
    if vocoder is None or not callable(getattr(vocoder, "float", None)):
        return
    vocoder.float()


def maybe_load_spk_dict(wrap, pretrained_model_name_or_path) -> None:
    if getattr(wrap, "speaker_map", None):
        return
    path = Path(str(pretrained_model_name_or_path)) / "spk_dict.pt"
    if not path.is_file():
        return
    loader = getattr(wrap, "load_speakers", None)
    if callable(loader):
        loader(str(path))
        return
    wrap.speaker_map = dict(torch.load(path, map_location="cpu", weights_only=True))


def _sync_omni_from_wrap(wrap, omni) -> None:
    omni.talker = wrap.talker
    vocoder = wrap.code2wav
    if vocoder is not None:
        omni.token2wav = vocoder
        omni.code2wav = vocoder
    omni.has_talker = wrap.has_talker
    speakers = getattr(wrap, "speaker_map", None)
    if speakers:
        omni.speaker_map = speakers


def attach_omni_container(wrap) -> None:
    """Point a real Omni generate() host at the wrap's modules.

    Live wraps already keep the source Omni (not registered, so checkpoint keys
    stay flat). Saved SpeechLMM checkpoints load flat modules only; rebuild a
    Thinker+Omni shell so official ``generate()`` runs after ``from_pretrained``.
    """
    _prepare_qwen25_vocoder(wrap)
    omni = getattr(wrap, "omni", None)
    if omni is not None:
        _sync_omni_from_wrap(wrap, omni)
        return
    if not getattr(wrap, "has_talker", False):
        return
    thinker_cls = getattr(wrap, "native_thinker_cls", None)
    omni_cls = getattr(wrap, "native_generate_cls", None)
    make_config = getattr(wrap, "native_omni_config", None)
    if thinker_cls is None or omni_cls is None or make_config is None:
        return

    from transformers import PreTrainedModel

    thinker = thinker_cls.__new__(thinker_cls)
    PreTrainedModel.__init__(thinker, wrap.config.thinker_config)
    thinker.audio_tower = wrap.audio_tower
    thinker.visual = wrap.visual
    thinker.model = wrap.model
    thinker.lm_head = wrap.lm_head
    thinker.vocab_size = wrap.vocab_size
    thinker.spatial_merge_size = wrap.spatial_merge_size
    thinker.rope_deltas = getattr(wrap, "rope_deltas", None)
    for name in ("num_experts", "num_experts_per_tok", "router_aux_loss_coef"):
        if hasattr(wrap, name):
            setattr(thinker, name, getattr(wrap, name))
    gen_cfg = getattr(thinker, "generation_config", None)
    if gen_cfg is not None and getattr(gen_cfg, "eos_token_id", None) is None:
        gen_cfg.eos_token_id = int(getattr(wrap.config, "im_end_token_id", 151645))

    omni = omni_cls.__new__(omni_cls)
    PreTrainedModel.__init__(omni, make_config(wrap.config))
    omni.thinker = thinker
    omni.talker = wrap.talker
    omni.has_talker = wrap.has_talker
    vocoder = wrap.code2wav
    omni.token2wav = vocoder
    omni.code2wav = vocoder
    omni.speaker_map = getattr(wrap, "speaker_map", None) or {}

    object.__setattr__(wrap, "thinker", thinker)
    object.__setattr__(wrap, "omni", omni)


def _wants_audio(wrap, kwargs) -> bool:
    if kwargs.get("return_audio") is True:
        return True
    if kwargs.get("return_audio") is False:
        return False
    mode = kwargs.get("generation_mode", None)
    if mode in ("text", False):
        return False
    if mode == "audio":
        return True
    return bool(getattr(wrap, "has_talker", False))


def _remap_speaker(wrap, kwargs):
    speaker = kwargs.get("speaker", getattr(wrap.config, "talker_speaker", "ethan"))
    speaker_map = getattr(wrap, "speaker_map", None) or {}
    if not speaker_map:
        kwargs["speaker"] = speaker
        return
    if speaker in speaker_map:
        kwargs["speaker"] = speaker
        return
    for key in speaker_map:
        if str(key).lower() == str(speaker).lower():
            kwargs["speaker"] = key
            return
    raise ValueError(f"Unknown Talker speaker {speaker!r}; available: {list(speaker_map)}")


def _omni_generate(wrap, omni, *args, **kwargs):
    _prepare_qwen25_vocoder(wrap)
    _sync_omni_from_wrap(wrap, omni)
    return omni.generate(*args, **kwargs)


def speechlmm_generate(wrap, *args, **kwargs):
    want_audio = _wants_audio(wrap, kwargs)
    kwargs.pop("generation_mode", None)
    has_talker = bool(getattr(wrap, "has_talker", False))
    has_vocoder = getattr(wrap, "code2wav", None) is not None
    if want_audio and not (has_talker and has_vocoder):
        if kwargs.get("return_audio") is True:
            raise ValueError("Speech generate needs talker and code2wav/token2wav")
        if has_talker and not has_vocoder:
            warn = getattr(logger, "warning_once", logger.warning)
            warn("Talker is loaded but code2wav/token2wav is missing; generate() returns text only.")
        want_audio = False
    if not want_audio or not has_talker:
        kwargs.pop("speaker", None)
        kwargs.pop("return_audio", None)
        return GenerationMixin.generate(wrap, *args, **kwargs)

    _remap_speaker(wrap, kwargs)
    _fix_qwen25_token2wav_rope()
    attach_omni_container(wrap)
    omni = getattr(wrap, "omni", None)
    if omni is None:
        raise RuntimeError("Speech generate needs a native Omni container after attach_omni_container")
    return _omni_generate(wrap, omni, *args, **kwargs)
