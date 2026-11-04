import torch
import torch.distributed as dist
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniTalkerForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniToken2WavModel,
    Qwen2_5OmniVisionEncoder,
)
from transformers.utils import logging

from .encoders.auto_avsr import AutoAVSR  # noqa: F401
from .modeling_speechlmm import SpeechLMMForConditionalGeneration
from .talker_generate import attach_omni_container  # noqa: F401
from .talker_loss import (
    build_streaming_text,
    codec_targets,
    continue_position_ids,
    dummy_talker_forward_loss,
    exclude_trailing_im_end,
    first_codebook,
    reduce_talker_losses,
    sequential_position_ids,
    token_ce_sum,
    valid_codec_codes,
)

logger = logging.get_logger(__name__)


class SpeechLMMQwen2_5OmniForConditionalGeneration(SpeechLMMForConditionalGeneration):
    helper_class = Qwen2_5OmniPreTrainedModelForConditionalGeneration
    native_generate_cls = Qwen2_5OmniForConditionalGeneration
    native_thinker_cls = Qwen2_5OmniThinkerForConditionalGeneration
    supports_talker_loss = True
    talker_needs_multimodal_hidden_only = False
    _no_split_modules = [
        "Qwen2_5OmniAudioEncoderLayer",
        "Qwen2_5OmniDecoderLayer",
        "Qwen2_5OmniTalkerDecoderLayer",
        "AutoAVSR",
    ]

    @staticmethod
    def native_omni_config(config):
        def as_dict(cfg):
            if cfg is None:
                return {}
            if isinstance(cfg, dict):
                return cfg
            return cfg.to_dict()

        omni_cfg = Qwen2_5OmniConfig(
            thinker_config=as_dict(config.thinker_config),
            talker_config=as_dict(config.talker_config),
            token2wav_config=as_dict(config.code2wav_config),
            enable_audio_output=bool(config.enable_talker),
        )
        for name in (
            "im_start_token_id",
            "im_end_token_id",
            "tts_pad_token_id",
            "tts_bos_token_id",
            "tts_eos_token_id",
            "system_token_id",
            "user_token_id",
            "assistant_token_id",
        ):
            if hasattr(config, name):
                setattr(omni_cfg, name, getattr(config, name))
        return omni_cfg

    @staticmethod
    def build_audio_encoder(config):
        return Qwen2_5OmniAudioEncoder._from_config(config.thinker_config.audio_config)

    @staticmethod
    def build_vision_encoder(config):
        return Qwen2_5OmniVisionEncoder._from_config(config.thinker_config.vision_config)

    @staticmethod
    def build_text_model(config):
        return Qwen2_5OmniThinkerTextModel._from_config(config.thinker_config.text_config)

    @staticmethod
    def build_talker(config):
        return Qwen2_5OmniTalkerForConditionalGeneration(config.talker_config)

    @staticmethod
    def build_code2wav(config):
        return Qwen2_5OmniToken2WavModel._from_config(config.code2wav_config)

    @classmethod
    def take_code2wav(cls, source, enabled):
        """Map ``token2wav`` → ``code2wav`` so freeze/LoRA keys stay consistent
        with the Qwen3 SpeechLMM surface. Dense text has no MoE aux fields.
        """
        return getattr(source, "token2wav", None) if enabled else None

    def audio_forward_kwargs(self, feature_lens):
        aftercnn_lens, _ = self.audio_tower._get_feat_extract_output_lengths(feature_lens)
        return {"feature_lens": feature_lens, "aftercnn_lens": aftercnn_lens}

    def audio_output_length(self, feature_lens):
        _, output_lengths = self.audio_tower._get_feat_extract_output_lengths(feature_lens)
        return output_lengths.sum()

    def _z3_pad_audio(self, input_features, feature_lens):
        """Append dummy clips so every ZeRO-3 rank has the same clip count.

        The audio tower runs ``ln_post`` and ``proj`` once per clip, so ranks with
        different clip counts issue different allgathers and deadlock.
        """
        local_count = torch.tensor([feature_lens.shape[0]], device=feature_lens.device)
        max_count = local_count.clone()
        dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
        extra = int((max_count - local_count).item())
        if extra <= 0:
            return input_features, feature_lens, None

        valid_output_len = self.audio_output_length(feature_lens)
        chunk_window = self.audio_tower.n_window * 2
        dummy_features = input_features.new_zeros(input_features.shape[0], extra * chunk_window)
        dummy_lens = feature_lens.new_full((extra,), chunk_window)
        input_features = torch.cat([input_features, dummy_features], dim=1)
        feature_lens = torch.cat([feature_lens, dummy_lens])
        return input_features, feature_lens, valid_output_len

    def talker_accept_hidden_layer(self):
        return -1

    def load_speakers(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.speaker_map = dict(payload)
        default = (self.config.talker_speaker or "ethan").lower()
        for name, value in self.speaker_map.items():
            if str(name).lower() == default:
                bos = value["bos_token"] if isinstance(value, dict) else value
                self.config.speaker_bos_token_id = int(bos)
                break

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
        if codec_labels is None or not self.has_talker or word_embeds is None:
            return None

        device = word_embeds.device
        dtype = word_embeds.dtype
        if codec_mask is not None and not codec_mask.any():
            return dummy_talker_forward_loss(self.talker, device, dtype)

        talker_config = self.config.talker_config
        codec_bos = talker_config.tts_codec_start_token_id
        codec_eos = talker_config.tts_codec_end_token_id
        codec_pad = talker_config.tts_codec_pad_token_id
        embed_tokens = self.get_input_embeddings()
        codec_embed = self.talker.model.embed_tokens
        text_eos_embed = embed_tokens(
            torch.tensor([[self.config.tts_eos_token_id]], device=device, dtype=torch.long)
        )
        text_pad_embed = embed_tokens(
            torch.tensor([[self.config.tts_pad_token_id]], device=device, dtype=torch.long)
        )
        codec_bos_embed = codec_embed(torch.tensor([codec_bos], device=device, dtype=torch.long))
        codec_pad_embed = codec_embed(torch.tensor([codec_pad], device=device, dtype=torch.long))

        if thinker_hidden is None:
            thinker_hidden = torch.zeros_like(word_embeds)
        if multimodal_mask is None:
            multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        embed_half = word_embeds.clone()
        embed_half.masked_fill_(multimodal_mask.unsqueeze(-1), 0)
        summed = thinker_hidden + embed_half

        sums = []
        counts = []
        for batch_index, ids in enumerate(input_ids):
            if codec_mask is not None and not bool(codec_mask[batch_index]):
                continue
            codes = valid_codec_codes(codec_labels[batch_index])
            if codes.numel() == 0:
                continue
            book0 = first_codebook(codes)
            segments = self._parse_chatml_segments(ids, self.config.im_start_token_id)
            assistant_segments = [seg for seg in segments if seg[0] == self.config.assistant_token_id]
            if not assistant_segments:
                continue
            _, assistant_start, assistant_end = assistant_segments[-1]
            assistant_end = exclude_trailing_im_end(ids, assistant_end, self.config.im_end_token_id)
            prompt_len = assistant_start + 3
            if prompt_len <= 0 or assistant_end <= prompt_len:
                continue
            prompt = summed[batch_index : batch_index + 1, :prompt_len]
            reply = summed[batch_index : batch_index + 1, prompt_len:assistant_end]
            if talker_speaker_ids is not None:
                speaker_bos = int(talker_speaker_ids[batch_index].item())
            else:
                speaker_bos = self.resolve_talker_speaker_id()
            speaker_embed = embed_tokens(
                torch.tensor([[speaker_bos]], device=device, dtype=torch.long)
            ) + codec_pad_embed.unsqueeze(0)
            codec_len = book0.numel()
            trailing = torch.cat([reply[:, 1:], text_eos_embed], dim=1) if reply.shape[1] > 1 else text_eos_embed
            stream = build_streaming_text(trailing, text_pad_embed, codec_len)
            first_slot = reply[:, :1] + codec_bos_embed.unsqueeze(0)
            rest_codec = codec_embed(book0)
            rest = (rest_codec + stream).unsqueeze(0)
            seq = torch.cat([prompt, speaker_embed, first_slot, rest], dim=1)
            target = codec_targets(prompt_len, 2, book0, codec_eos)
            if seq.shape[1] != target.shape[0]:
                logger.warning(
                    "Talker loss: shape mismatch embeds=%d target=%d, skipping sample.",
                    seq.shape[1],
                    target.shape[0],
                )
                continue
            seq = self.talker.thinker_to_talker_proj(seq)
            attention = torch.ones(1, seq.shape[1], device=device, dtype=torch.long)
            text_ids = torch.cat(
                [
                    ids[:prompt_len].unsqueeze(0),
                    torch.tensor([[speaker_bos]], device=device, dtype=ids.dtype),
                    ids[prompt_len : prompt_len + 1].unsqueeze(0),
                ],
                dim=-1,
            )
            get_rope = getattr(self.talker, "get_rope_index", None)
            position_ids = sequential_position_ids(1, seq.shape[1], device)
            if get_rope is not None:
                try:
                    prefix_pos, _ = get_rope(
                        text_ids,
                        rope_kwargs.get("image_grid_thw"),
                        rope_kwargs.get("video_grid_thw"),
                        None,
                        rope_kwargs.get("use_audio_in_video") or False,
                        rope_kwargs.get("audio_feature_lengths"),
                        rope_kwargs.get("video_second_per_grid"),
                    )
                    position_ids = continue_position_ids(prefix_pos, seq.shape[1] - text_ids.shape[1])
                except Exception:
                    pass
            hidden = self.talker.model(
                input_ids=None,
                inputs_embeds=seq,
                attention_mask=attention,
                position_ids=position_ids,
            )[0]
            logits = self.talker.codec_head(hidden)
            loss_sum, n_valid = token_ce_sum(logits, target.unsqueeze(0))
            sums.append(loss_sum)
            counts.append(n_valid)

        if not sums:
            dummy = dummy_talker_forward_loss(self.talker, device, dtype)
            if dummy is not None:
                return dummy
            return torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        return reduce_talker_losses(sums, counts, self.config.talker_loss_reduction)
