import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoePreTrainedModelForConditionalGeneration,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeVisionEncoder,
    _get_feat_extract_output_lengths,
    load_balancing_loss_func,
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
    first_codebook,
    reduce_talker_losses,
    rvq_codec_embeds,
    sequential_position_ids,
    token_ce_sum,
    valid_codec_codes,
)

logger = logging.get_logger(__name__)


class SpeechLMMQwen3OmniForConditionalGeneration(SpeechLMMForConditionalGeneration):
    helper_class = Qwen3OmniMoePreTrainedModelForConditionalGeneration
    native_generate_cls = Qwen3OmniMoeForConditionalGeneration
    native_thinker_cls = Qwen3OmniMoeThinkerForConditionalGeneration
    supports_talker_loss = True
    talker_needs_multimodal_hidden_only = True
    _no_split_modules = [
        "Qwen3OmniMoeAudioEncoderLayer",
        "Qwen3OmniMoeThinkerTextDecoderLayer",
        "Qwen3OmniMoeTalkerDecoderLayer",
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

        omni_cfg = Qwen3OmniMoeConfig(
            thinker_config=as_dict(config.thinker_config),
            talker_config=as_dict(config.talker_config),
            code2wav_config=as_dict(config.code2wav_config),
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
        return Qwen3OmniMoeAudioEncoder._from_config(config.thinker_config.audio_config)

    @staticmethod
    def build_vision_encoder(config):
        return Qwen3OmniMoeVisionEncoder._from_config(config.thinker_config.vision_config)

    @staticmethod
    def build_text_model(config):
        return Qwen3OmniMoeThinkerTextModel._from_config(config.thinker_config.text_config)

    @staticmethod
    def build_talker(config):
        return Qwen3OmniMoeTalkerForConditionalGeneration(config.talker_config)

    @staticmethod
    def build_code2wav(config):
        return Qwen3OmniMoeCode2Wav._from_config(config.code2wav_config)

    def _configure_model(self):
        text_config = self.config.thinker_config.text_config
        self.num_experts = getattr(text_config, "num_experts", None)
        self.num_experts_per_tok = getattr(text_config, "num_experts_per_tok", None)
        self.router_aux_loss_coef = getattr(text_config, "router_aux_loss_coef", 0.0)

    @classmethod
    def take_code2wav(cls, source, enabled):
        return getattr(source, "code2wav", None) if enabled else None

    def audio_forward_kwargs(self, feature_lens):
        return {"feature_lens": feature_lens}

    def audio_output_length(self, feature_lens):
        return _get_feat_extract_output_lengths(feature_lens).sum()

    def visual_features(self, outputs):
        return outputs.pooler_output, outputs.deepstack_features

    def merge_visual_features(self, image_mask, video_mask, image_features, video_features):
        if image_mask is not None and not image_mask.any():
            image_mask, image_features = None, None
        if video_mask is not None and not video_mask.any():
            video_mask, video_features = None, None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_mask = image_mask | video_mask
            image_positions = image_mask[visual_mask]
            video_positions = video_mask[visual_mask]
            merged = []
            for image_feature, video_feature in zip(image_features, video_features):
                feature = image_feature.new_zeros(visual_mask.sum(), image_feature.shape[-1])
                feature[image_positions] = image_feature
                feature[video_positions] = video_feature
                merged.append(feature)
            return tuple(merged), visual_mask
        if image_mask is not None:
            return image_features, image_mask[..., 0]
        if video_mask is not None:
            return video_features, video_mask[..., 0]
        return None, None

    def prepare_text_model_kwargs(self, kwargs, visual_features, visual_mask, output_router_logits):
        kwargs.update(
            output_router_logits=output_router_logits,
            deepstack_visual_embeds=visual_features,
            visual_pos_masks=visual_mask,
        )
        return kwargs

    def resolve_output_router_logits(self, requested):
        if requested is not None:
            return requested
        return getattr(self.config.thinker_config.text_config, "output_router_logits", False)

    def compute_aux_loss(self, outputs, attention_mask, output_router_logits):
        if not output_router_logits or self.num_experts is None or not hasattr(outputs, "router_logits"):
            return None
        return load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )

    def talker_accept_hidden_layer(self):
        return int(self.config.talker_config.accept_hidden_layer)

    def _talker_prefix_position_ids(self, talker_input_ids, seq_len, **rope_kwargs):
        get_rope = getattr(self.talker, "get_rope_index", None)
        if get_rope is None:
            return sequential_position_ids(1, seq_len, talker_input_ids.device)
        try:
            prefix_ids, _ = get_rope(
                talker_input_ids,
                rope_kwargs.get("image_grid_thw"),
                rope_kwargs.get("video_grid_thw"),
                None,
                rope_kwargs.get("use_audio_in_video") or False,
                rope_kwargs.get("audio_feature_lengths"),
                rope_kwargs.get("video_second_per_grid"),
            )
        except Exception:
            return sequential_position_ids(1, seq_len, talker_input_ids.device)
        return continue_position_ids(prefix_ids, seq_len - talker_input_ids.shape[1])

    def _code_predictor_loss(self, talker_hidden, codes, codec_positions):
        predictor = getattr(self.talker, "code_predictor", None)
        if predictor is None or codes is None or codes.dim() != 2 or codes.size(-1) < 2:
            return None, 0
        if not any(param.requires_grad for param in predictor.parameters()):
            return None, 0
        hidden = talker_hidden[0, codec_positions]
        book0 = self.talker.get_input_embeddings()(codes[:, 0])
        embeddings = getattr(predictor.model, "codec_embedding", None)
        if embeddings is None:
            return None, 0
        parts = [hidden, book0]
        residual_groups = codes.size(-1) - 1
        for group in range(residual_groups - 1):
            parts.append(embeddings[group](codes[:, group + 1]))
        seq = torch.stack(parts, dim=1)
        pred_hidden = predictor.model(inputs_embeds=seq)[0]
        loss_sum = pred_hidden.new_zeros(())
        valid = 0
        for group in range(residual_groups):
            logits = predictor.lm_head[group](pred_hidden[:, 1 + group])
            target = codes[:, group + 1]
            step_loss, step_n = token_ce_sum(logits, target)
            loss_sum = loss_sum + step_loss
            valid += step_n
        return loss_sum, valid

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
        special_ids = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            device=device,
            dtype=input_ids.dtype,
        )
        tts_bos, tts_eos, tts_pad = self.talker.text_projection(self.get_input_embeddings()(special_ids)).chunk(3, dim=1)
        if thinker_hidden is None:
            thinker_hidden = torch.zeros_like(word_embeds)
        if multimodal_mask is None:
            multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        sums = []
        counts = []
        for batch_index, ids in enumerate(input_ids):
            if codec_mask is not None and not bool(codec_mask[batch_index]):
                continue
            segments = self._parse_chatml_segments(ids, self.config.im_start_token_id)
            assistant_segments = [segment for segment in segments if segment[0] == self.config.assistant_token_id]
            if not assistant_segments:
                continue

            words = word_embeds[batch_index : batch_index + 1]
            hidden = thinker_hidden[batch_index : batch_index + 1]
            modality_mask = multimodal_mask[batch_index : batch_index + 1]
            user_parts = []
            talker_id_parts = []
            for role, start, end in segments:
                if role != self.config.user_token_id:
                    continue
                user_parts.append(
                    Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts(
                        self, start, end, modality_mask, hidden, words
                    )
                )
                talker_id_parts.append(ids[start:end].unsqueeze(0))

            _, assistant_start, assistant_end = assistant_segments[-1]
            end_tokens = (ids[assistant_start:assistant_end] == self.config.im_end_token_id).nonzero(as_tuple=True)[0]
            if len(end_tokens):
                assistant_end = assistant_start + end_tokens[0].item() + 1

            if talker_speaker_ids is not None:
                speaker_id = int(talker_speaker_ids[batch_index].item())
            else:
                speaker_id = self.resolve_talker_speaker_id()

            prefill, assistant_ids, trailing_text = Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts(
                self,
                assistant_start,
                assistant_end,
                speaker_id,
                words,
                tts_pad,
                tts_bos,
                tts_eos,
            )
            codes = valid_codec_codes(codec_labels[batch_index])
            if codes.numel() == 0:
                continue
            book0 = first_codebook(codes)
            speech = rvq_codec_embeds(self.talker, codes)
            stream = build_streaming_text(trailing_text, tts_pad, book0.numel())
            stream = (stream + speech).unsqueeze(0)
            talker_embeds = torch.cat(user_parts + [prefill, stream], dim=1)
            context_length = sum(part.shape[1] for part in user_parts)
            target = codec_targets(context_length, 9, book0, talker_config.codec_eos_token_id)
            if talker_embeds.shape[1] != target.shape[0]:
                logger.warning(
                    "Talker loss: shape mismatch embeds=%d target=%d, skipping sample.",
                    talker_embeds.shape[1],
                    target.shape[0],
                )
                continue

            sequence_length = talker_embeds.shape[1]
            attn = torch.ones(1, sequence_length, device=device, dtype=torch.long)
            prefix_ids = torch.cat(
                talker_id_parts
                + [
                    assistant_ids
                    if assistant_ids is not None
                    else torch.full(
                        (1, 9),
                        self.config.tts_pad_token_id,
                        device=device,
                        dtype=torch.long,
                    )
                ],
                dim=1,
            )
            position_ids = self._talker_prefix_position_ids(prefix_ids, sequence_length, **rope_kwargs)
            talker_output = self.talker.model(
                input_ids=None,
                inputs_embeds=talker_embeds,
                attention_mask=attn,
                position_ids=position_ids,
                cache_position=torch.arange(sequence_length, device=device),
                output_router_logits=True,
            )
            logits = self.talker.codec_head(talker_output[0])
            loss_sum, n_valid = token_ce_sum(logits, target.unsqueeze(0))
            router_logits = getattr(talker_output, "router_logits", None)
            if router_logits is not None:
                aux_loss = load_balancing_loss_func(
                    router_logits,
                    self.talker.num_experts,
                    self.talker.num_experts_per_tok,
                    attn,
                )
                loss_sum = loss_sum + self.talker.router_aux_loss_coef * aux_loss.to(loss_sum.device)
            first_valid = context_length + 8
            pred_loss, pred_n = self._code_predictor_loss(
                talker_output[0],
                codes if codes.dim() == 2 else None,
                torch.arange(first_valid, first_valid + book0.numel(), device=device),
            )
            if pred_loss is not None:
                loss_sum = loss_sum + pred_loss
                n_valid += pred_n
            sums.append(loss_sum)
            counts.append(n_valid)

        if not sums:
            dummy = dummy_talker_forward_loss(self.talker, device, dtype)
            if dummy is not None:
                return dummy
            return torch.zeros((), device=device, dtype=dtype, requires_grad=True)
        return reduce_talker_losses(sums, counts, self.config.talker_loss_reduction)
