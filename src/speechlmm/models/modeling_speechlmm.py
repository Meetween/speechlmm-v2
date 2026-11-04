"""SpeechLMM model: a customizable wrapper around Qwen Omni components.

Keeps the full architecture (Thinker + Talker + Code2Wav/Token2Wav) and allows
swapping encoders/decoders without modifying transformers.
"""

from __future__ import annotations

import importlib
import json
import re
import types
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerCausalLMOutputWithPast,
)
from transformers.utils import logging

from .configuration_speechlmm import SpeechLMMConfig
from .encoders.auto_avsr import AutoAVSR

logger = logging.get_logger(__name__)

_FAMILY_CLASSES = {
    "qwen2_5_omni": (".modeling_speechlmm_qwen2_5_omni", "SpeechLMMQwen2_5OmniForConditionalGeneration"),
    "qwen3_omni": (".modeling_speechlmm_qwen3_omni", "SpeechLMMQwen3OmniForConditionalGeneration"),
}


def _load_prefixed_weights(module: nn.Module, checkpoint: str, prefix: str) -> None:
    """Load the ``{prefix}*`` tensors of a safetensors checkpoint into ``module``."""
    import json
    from pathlib import Path

    from safetensors import safe_open

    path = Path(checkpoint)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted({path / shard for key, shard in weight_map.items() if key.startswith(prefix)})
    else:
        shards = [path / "model.safetensors"]
    tensors = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    tensors[key[len(prefix) :]] = handle.get_tensor(key)
    module.load_state_dict(tensors)


def scatter_or_keep_grad(
    inputs_embeds: torch.Tensor,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter features into placeholder positions, or keep a ZeRO-3 graph edge."""
    if not mask.any():
        return inputs_embeds + features.reshape(-1)[:1].sum() * 0
    return inputs_embeds.masked_scatter(mask, features)


_MOE_EXPERT_KEY = re.compile(
    r"^(?P<stem>.*mlp\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def _fuse_qwen_moe_expert_tensors(tensors: dict) -> dict:
    """Pack on-disk per-expert MoE weights into fused ``gate_up_proj`` / ``down_proj``."""
    grouped: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    fused = {}
    for key, value in tensors.items():
        match = _MOE_EXPERT_KEY.match(key)
        if match is None:
            fused[key] = value
            continue
        grouped.setdefault(match["stem"], {}).setdefault(match["proj"], {})[int(match["idx"])] = value
    for stem, projs in grouped.items():
        gates = projs.get("gate_proj", {})
        ups = projs.get("up_proj", {})
        downs = projs.get("down_proj", {})
        if set(gates) != set(ups):
            raise ValueError(f"Incomplete MoE expert gate/up tensors for {stem}")
        if gates:
            indices = list(range(max(gates) + 1))
            if indices != sorted(gates):
                raise ValueError(f"Non-contiguous MoE experts for {stem}")
            fused[f"{stem}.gate_up_proj"] = torch.cat(
                [
                    torch.stack([gates[i] for i in indices], dim=0),
                    torch.stack([ups[i] for i in indices], dim=0),
                ],
                dim=1,
            )
        if downs:
            indices = list(range(max(downs) + 1))
            if indices != sorted(downs):
                raise ValueError(f"Non-contiguous MoE down_proj experts for {stem}")
            fused[f"{stem}.down_proj"] = torch.stack([downs[i] for i in indices], dim=0)
    return fused


def load_prefixed_checkpoint(module: nn.Module, checkpoint, prefix: str) -> None:
    """Load `{prefix}*` tensors from a Hugging Face shard index into `module`."""
    from safetensors import safe_open

    path = Path(checkpoint)
    tensors = {}
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: dict[Path, list[str]] = {}
        for key, shard in weight_map.items():
            if key.startswith(prefix):
                shards.setdefault(path / shard, []).append(key)
        for shard, keys in shards.items():
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in keys:
                    tensors[key[len(prefix) :]] = handle.get_tensor(key)
    else:
        single = path / "model.safetensors" if path.is_dir() else path
        if not single.is_file():
            raise FileNotFoundError(f"No safetensors checkpoint at {checkpoint}")
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    tensors[key[len(prefix) :]] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"No tensors with prefix {prefix!r} in {checkpoint}")
    tensors = _fuse_qwen_moe_expert_tensors(tensors)
    missing, unexpected = module.load_state_dict(tensors, strict=False)
    if missing:
        raise ValueError(f"Missing {prefix} weights: {list(missing)[:20]}")
    if unexpected:
        logger.warning("Ignored unexpected %s keys: %s", prefix, list(unexpected)[:20])


# ---------------------------------------------------------------------------
# Registry for swappable components. New encoder/decoder types should be
# registered here.  Each entry maps a string key (used in config) to a
# callable ``(sub_config) -> nn.Module``.
# ---------------------------------------------------------------------------

_AUDIO_ENCODER_REGISTRY: dict[str, type[nn.Module]] = {}
_VISION_ENCODER_REGISTRY: dict[str, type[nn.Module]] = {}
_CODE2WAV_REGISTRY: dict[str, type[nn.Module]] = {}


def register_audio_encoder(name: str):
    """Decorator to register a custom audio encoder class."""

    def _wrap(cls):
        _AUDIO_ENCODER_REGISTRY[name] = cls
        return cls

    return _wrap


def register_vision_encoder(name: str):
    """Decorator to register a custom vision encoder class."""

    def _wrap(cls):
        _VISION_ENCODER_REGISTRY[name] = cls
        return cls

    return _wrap


def register_code2wav(name: str):
    """Decorator to register a custom code-to-waveform decoder class."""

    def _wrap(cls):
        _CODE2WAV_REGISTRY[name] = cls
        return cls

    return _wrap


class SpeechLMMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Shared SpeechLMM wrap: Thinker + Talker + Code2Wav/Token2Wav.

    Architecture-specific Omni families subclass this. Module names stay
    ``audio_tower`` / ``visual`` / ``model`` / ``lm_head`` / ``talker`` /
    ``code2wav`` so flat checkpoint keys and freeze/LoRA paths are unchanged.
    """

    config_class = SpeechLMMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    helper_class = None
    native_generate_cls = None
    supports_talker_loss = False
    talker_needs_multimodal_hidden_only = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _no_split_modules = ["AutoAVSR"]

    @classmethod
    def for_config(cls, config):
        backbone_type = getattr(config, "backbone_type", "qwen3_omni")
        spec = _FAMILY_CLASSES.get(backbone_type)
        if spec is None:
            raise ValueError(f"Unsupported backbone_type={backbone_type!r}")
        # importlib so HF remote-code does not regex-copy the unused family file.
        module = importlib.import_module(spec[0], __package__)
        return getattr(module, spec[1])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if cls is SpeechLMMForConditionalGeneration:
            config = kwargs.get("config")
            if config is None:
                config = SpeechLMMConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=kwargs.get("trust_remote_code", True),
                )
                kwargs["config"] = config
            return cls.for_config(config).from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        # Under deepspeed.zero.Init, the Qwen2.5 BigVGAN Kaiser filter mixes CPU and CUDA
        # tensors and crashes, so build the frozen vocoder after loading.
        config = kwargs.get("config")
        defer_code2wav = (
            config is not None
            and getattr(config, "backbone_type", None) == "qwen2_5_omni"
            and bool(getattr(config, "enable_code2wav", False))
            and is_deepspeed_zero3_enabled()
        )
        if defer_code2wav:
            config.enable_code2wav = False
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if defer_code2wav:
            config.enable_code2wav = True
            model.config.enable_code2wav = True
            model.has_code2wav = True
            # _from_config re-enters zero.Init while a DeepSpeed config is registered.
            import transformers.integrations.deepspeed as hf_ds

            saved_ds = getattr(hf_ds, "_hf_deepspeed_config_weak_ref", None)
            hf_ds._hf_deepspeed_config_weak_ref = None
            try:
                model.code2wav = model.build_code2wav(model.config)
            finally:
                hf_ds._hf_deepspeed_config_weak_ref = saved_ds
            _load_prefixed_weights(model.code2wav, pretrained_model_name_or_path, "code2wav.")
        from .talker_generate import attach_omni_container, maybe_load_spk_dict

        maybe_load_spk_dict(model, pretrained_model_name_or_path)
        attach_omni_container(model)
        return model

    @classmethod
    def _from_config(cls, config, **kwargs):
        if cls is SpeechLMMForConditionalGeneration:
            cls = cls.for_config(config)
        return super()._from_config(config, **kwargs)

    def __init__(self, config: SpeechLMMConfig):
        super().__init__(config)
        self._sync_nested_attn_implementation()

        thinker_cfg = config.thinker_config
        self.audio_tower = self._build_audio_encoder(config)
        self.visual = self._build_vision_encoder(config)
        # TODO(P1): Add explicit audio_projector and vision_projector modules
        # when swapping to custom encoders that output a different dimension.
        # Qwen3's encoders have built-in projectors (proj1/proj2, merger).
        self.vocab_size = thinker_cfg.text_config.vocab_size
        self.model = self.build_text_model(config)
        self.lm_head = nn.Linear(
            thinker_cfg.text_config.hidden_size,
            thinker_cfg.text_config.vocab_size,
            bias=False,
        )
        self.spatial_merge_size = thinker_cfg.vision_config.spatial_merge_size
        self._configure_model()

        # TODO(P1): Make codec_head/codec_embedding configurable for
        # different codec formats (RVQ vs FSQ) if we swap the audio codec.
        self.has_talker = config.enable_talker
        self.talker = self.build_talker(config) if config.enable_talker else None
        self.speaker_map = {}

        self.has_code2wav = config.enable_code2wav
        self.code2wav = self._build_code2wav(config) if config.enable_code2wav else None

        self.lipread_encoder = self._build_lipread_encoder()
        self.lipread_adapter = self._build_lipread_adapter(config)
        self._fix_lipread_bn_buffers()

        self.rope_deltas = None
        self._bind_backbone_helpers()
        self.post_init()

    def _bind_backbone_helpers(self) -> None:
        helper_class = self.helper_class
        if helper_class is None:
            raise NotImplementedError(f"{type(self).__name__} must set helper_class")
        for name in ("get_rope_index", "get_llm_pos_ids_for_vision", "get_chunked_index"):
            fn = getattr(helper_class, name, None)
            if fn is not None:
                setattr(self, name, types.MethodType(fn, self))

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        """Enable or disable gradient checkpointing, ensuring input require grads."""
        super()._set_gradient_checkpointing(enable=enable, gradient_checkpointing_func=gradient_checkpointing_func)
        if enable:
            self.enable_input_require_grads()

    def _sync_nested_attn_implementation(self) -> None:
        attn = getattr(self.config, "_attn_implementation", None)
        if not attn:
            return
        nested = []
        for cfg in (self.config.thinker_config, self.config.talker_config, self.config.code2wav_config):
            if cfg is None:
                continue
            nested.append(cfg)
            for name in ("text_config", "audio_config", "vision_config"):
                sub = getattr(cfg, name, None)
                if sub is not None:
                    nested.append(sub)
        for cfg in nested:
            cfg._attn_implementation = attn

    def _configure_model(self):
        self.num_experts = None
        self.num_experts_per_tok = None
        self.router_aux_loss_coef = 0.0

    @staticmethod
    def build_audio_encoder(config):
        raise NotImplementedError

    @staticmethod
    def build_vision_encoder(config):
        raise NotImplementedError

    @staticmethod
    def build_text_model(config):
        raise NotImplementedError

    @staticmethod
    def build_talker(config):
        raise NotImplementedError

    @staticmethod
    def build_code2wav(config):
        raise NotImplementedError

    @classmethod
    def take_code2wav(cls, source, enabled):
        raise NotImplementedError

    def audio_forward_kwargs(self, feature_lens):
        raise NotImplementedError

    def audio_output_length(self, feature_lens):
        raise NotImplementedError

    def visual_features(self, outputs):
        return outputs.pooler_output, None

    def merge_visual_features(self, image_mask, video_mask, image_features, video_features):
        return None, None

    def prepare_text_model_kwargs(self, kwargs, visual_features, visual_mask, output_router_logits):
        return kwargs

    def resolve_output_router_logits(self, requested):
        return False

    def compute_aux_loss(self, outputs, attention_mask, output_router_logits):
        return None

    def talker_accept_hidden_layer(self) -> int:
        raise NotImplementedError

    def talker_requires_output_hidden_states(self, has_multimodal: bool) -> bool:
        if not self.supports_talker_loss:
            return False
        if self.talker_needs_multimodal_hidden_only:
            return has_multimodal
        return False

    @staticmethod
    def _parse_chatml_segments(ids, im_start_id):
        starts = (ids == im_start_id).nonzero(as_tuple=True)[0]
        if len(starts) == 0:
            return []
        boundaries = torch.cat([starts, torch.tensor([ids.shape[0]], device=ids.device)])
        segments = []
        for index in range(len(boundaries) - 1):
            start = boundaries[index].item()
            end = boundaries[index + 1].item()
            if start + 1 < end:
                segments.append((ids[start + 1].item(), start, end))
        return segments

    def _multimodal_token_mask(self, input_ids):
        thinker_cfg = self.config.thinker_config
        audio_id = getattr(thinker_cfg, "audio_token_id", getattr(thinker_cfg, "audio_token_index", None))
        image_id = getattr(thinker_cfg, "image_token_id", getattr(thinker_cfg, "image_token_index", None))
        video_id = getattr(thinker_cfg, "video_token_id", getattr(thinker_cfg, "video_token_index", None))
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if audio_id is not None:
            mask = mask | (input_ids == audio_id)
        if image_id is not None:
            mask = mask | (input_ids == image_id)
        if video_id is not None:
            mask = mask | (input_ids == video_id)
        return mask

    def _thinker_branch_trainable(self) -> bool:
        for module in (
            self.model,
            self.lm_head,
            getattr(self, "audio_tower", None),
            getattr(self, "visual", None),
            getattr(self, "lipread_encoder", None),
            getattr(self, "lipread_adapter", None),
        ):
            if module is not None and any(param.requires_grad for param in module.parameters()):
                return True
        return False

    def resolve_talker_speaker_id(self, name: str | None = None) -> int:
        return self.config.resolve_talker_speaker_id(name, getattr(self, "speaker_map", None))

    def talker_is_loaded(self) -> bool:
        talker = getattr(self, "talker", None)
        if talker is None:
            return False
        # ZeRO-3 partitions params to torch.empty(0); ds_numel is the real size.
        return any(getattr(param, "ds_numel", param.numel()) > 0 for param in talker.parameters())

    def load_talker_weights(self, checkpoint) -> None:
        # Prefer Talker weights from native from_pretrained (audio output on,
        # vocoder dropped after wrap). Rebuild only when that Talker is missing.
        if self.talker_is_loaded():
            self.has_talker = True
            return
        warn = getattr(logger, "warning_once", logger.warning)
        warn(
            "Native Talker is empty; rebuilding from talker.* shards. "
            "Expected when enable_audio_output was false at from_pretrained."
        )
        # Fallback: native load had no Talker (audio output off). PreTrainedModel._from_config
        # re-enters deepspeed.zero.Init under ZeRO-3 and yields shape-[0] params,
        # so clear HF's flag only while constructing, then load talker.* shards.
        import transformers.integrations.deepspeed as hf_ds

        saved = getattr(hf_ds, "_hf_deepspeed_config_weak_ref", None)
        try:
            hf_ds._hf_deepspeed_config_weak_ref = None
            self.talker = self.build_talker(self.config)
        finally:
            hf_ds._hf_deepspeed_config_weak_ref = saved
        load_prefixed_checkpoint(self.talker, checkpoint, "talker.")
        ref = next(self.model.parameters())
        self.talker.to(device=ref.device, dtype=ref.dtype)
        self.has_talker = True

    def compute_talker_loss(
        self,
        codec_labels,
        word_embeds=None,
        input_ids=None,
        thinker_hidden=None,
        multimodal_mask=None,
        attention_mask=None,
        codec_mask=None,
        talker_speaker_ids=None,
        **rope_kwargs,
    ):
        if codec_labels is not None:
            warn = getattr(logger, "warning_once", logger.warning)
            warn(
                "codec_labels provided for SpeechLMM wrap; "
                "joint Talker training is not implemented on this backbone — ignoring codec_labels."
            )
        return None

    def generate(self, *args, **kwargs):
        from .talker_generate import speechlmm_generate

        return speechlmm_generate(self, *args, **kwargs)

    @staticmethod
    def _fix_bn_buffers(module: nn.Module) -> None:
        """Keep BatchNorm running stats in float32 for mixed-precision training.

        PyTorch requires running_mean/running_var to be float32 even when the
        rest of the model is in bfloat16.  Without this, training crashes with
        ``Expected running_mean to have type Float but got BFloat16``.

        Uses ``copy_`` instead of ``=`` assignment to preserve the buffer
        identity, which is required under DeepSpeed ZeRO-3 (buffers are
        registered tensors that DeepSpeed tracks by reference).
        """
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.running_mean is not None:
                    m.running_mean.copy_(m.running_mean.float())
                if m.running_var is not None:
                    m.running_var.copy_(m.running_var.float())

    def _fix_lipread_bn_buffers(self) -> None:
        """Re-cast AutoAVSR BatchNorm running stats to float32 after checkpoint load."""
        if getattr(self, "lipread_encoder", None) is not None:
            self._fix_bn_buffers(self.lipread_encoder)

    @classmethod
    def _load_pretrained_model(cls, model, state_dict, checkpoint_files, load_config):
        load_info, remaining = super()._load_pretrained_model(model, state_dict, checkpoint_files, load_config)
        if getattr(model, "lipread_encoder", None) is not None:
            cls._fix_bn_buffers(model.lipread_encoder)
        return load_info, remaining

    @staticmethod
    def _build_lipread_adapter(config: SpeechLMMConfig) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    ("ln_post", nn.LayerNorm(config.lipread_encoder_dim)),
                    (
                        "linear_1",
                        nn.Linear(
                            config.lipread_encoder_dim,
                            config.lipread_adapter_hidden,
                        ),
                    ),
                    ("act_1", nn.GELU()),
                    (
                        "linear_2",
                        nn.Linear(
                            config.lipread_adapter_hidden,
                            config.hidden_size,
                        ),
                    ),
                ]
            )
        )

    @staticmethod
    def _build_lipread_encoder() -> nn.Module:
        """Build the lipread encoder shell (weights loaded separately).

        Weights are never loaded here: under DeepSpeed ZeRO-3, ``__init__`` runs
        while parameters are partitioned (shape 0), so an in-place ``.pth`` load
        fails. They come from ``from_pretrained`` / checkpoint weights, or from
        ``load_auto_avsr_weights`` after the module is fully constructed (wrapper path).
        """
        return AutoAVSR().eval()

    @classmethod
    def _build_audio_encoder(cls, config: SpeechLMMConfig) -> nn.Module:
        enc_type = config.audio_encoder_type
        if enc_type in ("qwen3_omni", "qwen2_5_omni"):
            return cls.build_audio_encoder(config)
        if enc_type in _AUDIO_ENCODER_REGISTRY:
            return _AUDIO_ENCODER_REGISTRY[enc_type](config.custom_audio_config)
        raise ValueError(
            f"Unknown audio_encoder_type={enc_type!r}. "
            f"Available: 'qwen3_omni', 'qwen2_5_omni', {list(_AUDIO_ENCODER_REGISTRY)}"
        )

    @classmethod
    def _build_vision_encoder(cls, config: SpeechLMMConfig) -> nn.Module:
        enc_type = config.vision_encoder_type
        if enc_type in ("qwen3_omni", "qwen2_5_omni"):
            return cls.build_vision_encoder(config)
        if enc_type in _VISION_ENCODER_REGISTRY:
            return _VISION_ENCODER_REGISTRY[enc_type](config.custom_vision_config)
        raise ValueError(
            f"Unknown vision_encoder_type={enc_type!r}. "
            f"Available: 'qwen3_omni', 'qwen2_5_omni', {list(_VISION_ENCODER_REGISTRY)}"
        )

    @classmethod
    def _build_code2wav(cls, config: SpeechLMMConfig) -> nn.Module:
        c2w_type = config.code2wav_type
        if c2w_type in ("qwen3_omni", "qwen2_5_omni"):
            return cls.build_code2wav(config)
        if c2w_type in _CODE2WAV_REGISTRY:
            return _CODE2WAV_REGISTRY[c2w_type](config.custom_code2wav_config)
        raise ValueError(
            f"Unknown code2wav_type={c2w_type!r}. " f"Available: 'qwen3_omni', 'qwen2_5_omni', {list(_CODE2WAV_REGISTRY)}"
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_audio_features(
        self,
        input_features,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        **kwargs,
    ):
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)

        valid_output_len = None
        if is_deepspeed_zero3_enabled() and feature_lens is not None and hasattr(self.audio_tower, "n_window"):
            input_features, feature_lens, valid_output_len = self._z3_pad_audio(input_features, feature_lens)
        audio_outputs = self.audio_tower(input_features, **self.audio_forward_kwargs(feature_lens))

        if valid_output_len is not None:
            audio_outputs.last_hidden_state = audio_outputs.last_hidden_state[:valid_output_len]

        return audio_outputs

    def _z3_pad_audio(self, input_features, feature_lens):
        """Pad flattened audio so every Z3 rank runs identical conv iterations.

        The Qwen3-Omni audio encoder splits the flat mel tensor into chunks of
        ``n_window * 2`` frames, pads them, and runs conv layers in a loop of
        ``ceil(total_chunks / conv_chunksize)`` iterations.  When ranks have
        different audio lengths the loop count diverges, which makes Z3
        allgather operations go out-of-sync and crash.

        We fix this by:
        1. Computing ``total_chunks`` on each rank.
        2. All-reducing the max across ranks.
        3. Appending a zero-padded dummy sample whose length makes every rank
           reach the same ``total_chunks``.
        4. Returning the original valid output length so the caller can strip
           the dummy frames afterwards.
        """
        n_window = self.audio_tower.n_window
        chunk_window = n_window * 2

        local_chunks = torch.ceil(feature_lens / chunk_window).long().sum()
        max_chunks = local_chunks.clone()
        dist.all_reduce(max_chunks, op=dist.ReduceOp.MAX)

        pad_chunks = (max_chunks - local_chunks).item()
        if pad_chunks <= 0:
            return input_features, feature_lens, None

        valid_output_len = self.audio_output_length(feature_lens)
        dummy_len = pad_chunks * chunk_window
        dummy_features = input_features.new_zeros(input_features.shape[0], dummy_len)

        input_features = torch.cat([input_features, dummy_features], dim=1)
        dummy_feature_len = torch.tensor([dummy_len], device=feature_lens.device, dtype=feature_lens.dtype)
        feature_lens = torch.cat([feature_lens, dummy_feature_len])

        return input_features, feature_lens, valid_output_len

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        return self.visual(pixel_values_videos, grid_thw=video_grid_thw)

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        """Build boolean masks for each modality and validate feature/token alignment."""
        if input_ids is None:
            embed_fn = self.get_input_embeddings()

            def _tok_mask(token_id):
                ref = embed_fn(torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device))
                return (inputs_embeds == ref).all(-1)

            special_image_mask = _tok_mask(self.config.thinker_config.image_token_id)
            special_video_mask = _tok_mask(self.config.thinker_config.video_token_id)
            special_audio_mask = _tok_mask(self.config.thinker_config.audio_token_id)
        else:
            special_image_mask = input_ids == self.config.thinker_config.image_token_id
            special_video_mask = input_ids == self.config.thinker_config.video_token_id
            special_audio_mask = input_ids == self.config.thinker_config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: " f"tokens={n_image_tokens}, features={image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: " f"tokens={n_video_tokens}, features={video_features.shape[0]}"
            )

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits: bool | None = None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        codec_labels=None,
        lipread=None,
        lipread_mask=None,
        codec_mask=None,
        talker_speaker_ids=None,
        **kwargs,
    ) -> Qwen3OmniMoeThinkerCausalLMOutputWithPast:
        output_router_logits = self.resolve_output_router_logits(output_router_logits)
        if self.num_experts is None:
            output_router_logits = False

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        word_embeds = inputs_embeds.clone() if codec_labels is not None else None

        has_multimodal = input_features is not None or pixel_values is not None or pixel_values_videos is not None
        need_thinker_hidden = (
            codec_labels is not None
            and self.has_talker
            and self.supports_talker_loss
            and (has_multimodal or not self.talker_needs_multimodal_hidden_only)
        )
        multimodal_mask = None
        if codec_labels is not None and input_ids is not None:
            multimodal_mask = self._multimodal_token_mask(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None
        image_mask, video_mask = None, None
        image_embeds_multiscale = None
        video_embeds_multiscale = None

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            ).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = scatter_or_keep_grad(inputs_embeds, audio_features, audio_mask)

        if self.config.enable_vision and pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds, image_embeds_multiscale = self.visual_features(image_outputs)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = scatter_or_keep_grad(inputs_embeds, image_embeds, image_mask)

        if self.config.enable_vision and pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds, video_embeds_multiscale = self.visual_features(video_outputs)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = scatter_or_keep_grad(inputs_embeds, video_embeds, video_mask)

        if lipread is not None and lipread_mask is not None:
            if not any(param.requires_grad for param in self.lipread_encoder.parameters()):
                self.lipread_encoder.eval()
            lipread_feats = lipread.unsqueeze(2)  # [b, t, 1, 88, 88]
            lipread_placeholder_mask = input_ids == self.config.lipread_pad_token_id
            lengths = lipread_placeholder_mask.sum(1).tolist()
            if any(length == 0 for length in lengths):
                lengths = [int(lipread_mask[i, :, 0].sum().item()) for i in range(lipread_feats.shape[0])]
            if any(length == 0 for length in lengths):
                raise ValueError(
                    "Lipread placeholder count is zero; ensure lipread special tokens are in the "
                    "tokenizer (add_special_tokens) and match config lipread_*_token_id."
                )

            enc_feat, _ = self.lipread_encoder.encode(lipread_feats, lengths=lengths)
            lipread_embeddings = self.lipread_adapter(enc_feat)
            lipread_embeddings = lipread_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
            lipread_placeholder_mask = lipread_placeholder_mask.to(inputs_embeds.device)

            if not lipread_placeholder_mask.any():
                inputs_embeds = scatter_or_keep_grad(
                    inputs_embeds, lipread_embeddings, lipread_placeholder_mask
                )
            else:
                for i in range(lipread_feats.shape[0]):
                    inputs_embeds[i, lipread_placeholder_mask[i]] = lipread_embeddings[i, : lengths[i]]

        visual_embeds_multiscale, visual_pos_masks = self.merge_visual_features(
            image_mask,
            video_mask,
            image_embeds_multiscale,
            video_embeds_multiscale,
        )

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if past_key_values_length == 0 or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = (past_key_values_length + self.rope_deltas).to(input_ids.device)
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        output_hidden_states_flag = kwargs.pop("output_hidden_states", None)
        if need_thinker_hidden and self.talker_requires_output_hidden_states(has_multimodal):
            output_hidden_states_flag = True

        model_forward_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "output_hidden_states": output_hidden_states_flag,
            "cache_position": cache_position,
        }
        model_forward_kwargs = self.prepare_text_model_kwargs(
            model_forward_kwargs,
            visual_embeds_multiscale,
            visual_pos_masks,
            output_router_logits,
        )
        thinker_ctx = (
            torch.no_grad()
            if (
                codec_labels is not None
                and self.config.thinker_loss_weight == 0
                and not self._thinker_branch_trainable()
            )
            else nullcontext()
        )
        with thinker_ctx:
            outputs = self.model(**model_forward_kwargs, **kwargs)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        thinker_hidden = None
        if need_thinker_hidden:
            if self.talker_requires_output_hidden_states(has_multimodal) and getattr(
                outputs, "hidden_states", None
            ) is not None:
                thinker_hidden = outputs.hidden_states[self.talker_accept_hidden_layer()]
            else:
                thinker_hidden = hidden_states

        thinker_loss = None
        if labels is not None:
            thinker_loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.thinker_config.text_config.vocab_size,
            )

        aux_loss = self.compute_aux_loss(outputs, attention_mask, output_router_logits)
        if thinker_loss is not None and aux_loss is not None:
            thinker_loss = thinker_loss + self.router_aux_loss_coef * aux_loss.to(thinker_loss.device)

        talker_loss = self.compute_talker_loss(
            codec_labels=codec_labels,
            word_embeds=word_embeds,
            input_ids=input_ids,
            thinker_hidden=thinker_hidden,
            multimodal_mask=multimodal_mask,
            attention_mask=attention_mask,
            codec_mask=codec_mask,
            talker_speaker_ids=talker_speaker_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
        )

        loss = None
        if thinker_loss is not None or talker_loss is not None:
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            if thinker_loss is not None:
                loss = loss + self.config.thinker_loss_weight * thinker_loss
            if talker_loss is not None:
                loss = loss + self.config.talker_loss_weight * talker_loss

        return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def load_auto_avsr_weights(self, autoavsr_weights_path: str):
        """Load pretrained AutoAVSR weights into the lipread encoder from a ``.pth`` file.

        Use this on the wrapper path (fresh Qwen3-Omni → SpeechLMM) via
        ``SpeechLMMConfig.lipread_encoder_weights``. Do not load ``.pth`` inside
        ``_build_lipread_encoder``: DeepSpeed ZeRO-3 partitions parameters during
        ``__init__``. For exported checkpoints, weights come from
        ``from_pretrained`` (safetensors); set ``lipread_encoder_weights`` only on
        the config used at wrap/export time, not in saved ``config.json``.

        Safe to call before or after ``deepspeed.initialize()``: under ZeRO-3 the
        ``load_state_dict`` is wrapped with ``GatheredParameters``.
        """
        state_dict = torch.load(autoavsr_weights_path, map_location="cpu", weights_only=True)
        if is_deepspeed_zero3_enabled():
            try:
                import deepspeed
            except ImportError as exc:
                raise ImportError("DeepSpeed ZeRO-3 is enabled but the deepspeed package is not installed") from exc

            with deepspeed.zero.GatheredParameters(self.lipread_encoder.parameters(), modifier_rank=0):
                errors = self.lipread_encoder.load_state_dict(state_dict, strict=False)
        else:
            errors = self.lipread_encoder.load_state_dict(state_dict, strict=False)
        if errors.missing_keys:
            raise ValueError(f"Missing keys in lipread encoder: {errors.missing_keys}")
        self._fix_lipread_bn_buffers()

    @classmethod
    def from_qwen3_omni(
        cls,
        pretrained_model_name_or_path: str,
        config_overrides: dict | None = None,
        **kwargs,
    ) -> "SpeechLMMForConditionalGeneration":
        """Load a SpeechLMM model from a Qwen3-Omni checkpoint.

        Uses zero-copy module reassignment: loads the original model, creates
        a SpeechLMM shell, and directly moves nn.Module references from the
        original to the new structure.  Peak memory is ~1x model size.
        """
        from transformers import AutoConfig

        qwen3_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        speechlmm_config = SpeechLMMConfig.from_qwen3_omni_config(qwen3_config, **(config_overrides or {}))

        logger.info("Loading original Qwen3-Omni model...")
        qwen3_model = cls._load_native_omni(pretrained_model_name_or_path, qwen3_config, speechlmm_config, kwargs)

        model = cls.for_config(speechlmm_config)._wrap_omni(qwen3_model, speechlmm_config)
        if speechlmm_config.enable_talker:
            model.load_talker_weights(pretrained_model_name_or_path)
        logger.info("SpeechLMM wrapping complete (zero-copy).")
        return model

    @classmethod
    def from_qwen2_5_omni(
        cls,
        pretrained_model_name_or_path: str,
        config_overrides: dict | None = None,
        **kwargs,
    ) -> "SpeechLMMForConditionalGeneration":
        """Load a SpeechLMM model from a Qwen2.5-Omni checkpoint (zero-copy wrap)."""
        from transformers import AutoConfig

        qwen2_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        speechlmm_config = SpeechLMMConfig.from_qwen2_5_omni_config(qwen2_config, **(config_overrides or {}))

        logger.info("Loading original Qwen2.5-Omni model...")
        qwen2_model = cls._load_native_omni(pretrained_model_name_or_path, qwen2_config, speechlmm_config, kwargs)

        model = cls.for_config(speechlmm_config)._wrap_omni(qwen2_model, speechlmm_config)
        if speechlmm_config.enable_talker:
            model.load_talker_weights(pretrained_model_name_or_path)
        logger.info("SpeechLMM Qwen2.5 wrapping complete (zero-copy).")
        return model

    @staticmethod
    def _load_native_omni(pretrained_model_name_or_path, native_config, speechlmm_config, kwargs):
        from transformers import AutoModelForTextToWaveform

        load_kwargs = dict(kwargs)
        load_kwargs.pop("pretrained_model_name_or_path", None)
        native_config = load_kwargs.pop("config", native_config)
        # Audio output loads Talker+vocoder; wrap drops the vocoder when unused.
        native_config.enable_audio_output = bool(speechlmm_config.enable_talker or speechlmm_config.enable_code2wav)
        return AutoModelForTextToWaveform.from_pretrained(
            pretrained_model_name_or_path, config=native_config, **load_kwargs
        )

    @classmethod
    def _wrap_qwen3_omni(cls, qwen3_model: nn.Module, config: SpeechLMMConfig):
        return cls.for_config(config)._wrap_omni(qwen3_model, config)

    @classmethod
    def _wrap_qwen2_5_omni(cls, qwen2_model: nn.Module, config: SpeechLMMConfig):
        return cls.for_config(config)._wrap_omni(qwen2_model, config)

    @classmethod
    def _wrap_omni(cls, source, config):
        if cls is SpeechLMMForConditionalGeneration:
            cls = cls.for_config(config)
        model = cls.__new__(cls)
        PreTrainedModel.__init__(model, config)

        thinker = getattr(source, "thinker", source)
        model.audio_tower = thinker.audio_tower
        model.visual = thinker.visual
        model.model = thinker.model
        model.lm_head = thinker.lm_head

        model.vocab_size = config.thinker_config.text_config.vocab_size
        model.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
        model._configure_model()
        model.has_talker = config.enable_talker
        model.talker = getattr(source, "talker", None) if config.enable_talker else None
        model.has_code2wav = config.enable_code2wav
        model.code2wav = cls.take_code2wav(source, config.enable_code2wav)
        model.speaker_map = getattr(source, "speaker_map", None) or {}
        model.rope_deltas = None
        if not config.enable_code2wav:
            if hasattr(source, "token2wav"):
                source.token2wav = None
            if hasattr(source, "code2wav"):
                source.code2wav = None

        if hasattr(source, "generation_config"):
            model.generation_config = source.generation_config

        model.lipread_encoder = cls._build_lipread_encoder()
        model.lipread_adapter = cls._build_lipread_adapter(config)
        if config.lipread_encoder_weights:
            model.load_auto_avsr_weights(config.lipread_encoder_weights)

        model.lipread_encoder = model.lipread_encoder.to(model.visual.device, dtype=model.visual.dtype)
        model.lipread_adapter = model.lipread_adapter.to(model.visual.device, dtype=model.visual.dtype)
        model._fix_lipread_bn_buffers()
        model._bind_backbone_helpers()
        model.post_init()
        # Same modules as `source`; not registered so checkpoint keys stay flat.
        object.__setattr__(model, "thinker", getattr(source, "thinker", None))
        object.__setattr__(model, "omni", source)

        return model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        lipread=None,
        lipread_mask=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            is_first_iteration=is_first_iteration,
            lipread=lipread,
            lipread_mask=lipread_mask,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None
            model_inputs["lipread"] = None
            model_inputs["lipread_mask"] = None

        return model_inputs
