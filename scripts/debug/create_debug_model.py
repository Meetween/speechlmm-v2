#!/usr/bin/env python3
"""Create a tiny Qwen3-Omni model with random weights for debugging.

Produces a model with the same architecture as the full Qwen3-Omni-30B but
with drastically reduced dimensions (~20M params instead of 30B).  The saved
checkpoint has ``model_type="qwen3_omni_moe"`` so it can be wrapped into
SpeechLMM at runtime via ``use_speechlmm_wrapper: true`` – exactly like the
full-size model.  This avoids copying ``modelling_speechlmm.py`` into the
output directory and keeps the debug model in sync with code changes.

Usage:
    python scripts/debug/create_debug_model.py --output_dir /path/to/debug_model
    python scripts/debug/create_debug_model.py --output_dir /tmp/debug_model --no-talker --no-code2wav
    python scripts/debug/create_debug_model.py --output_dir /tmp/debug_model --no-tokenizer

By default the tokenizer and processor are saved from the Qwen3-Omni model
(see DEFAULT_TOKENIZER_SOURCE).
Use --tokenizer_source to override or --no-tokenizer to skip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Default tokenizer source – HuggingFace model ID or local path.
DEFAULT_TOKENIZER_SOURCE = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def build_tiny_config(
    enable_audio_output: bool = True,
    enable_vision: bool = True,
    vocab_size: int = 152064,
) -> dict:
    """Build a minimal Qwen3OmniMoeConfig dict preserving the full architecture."""

    # ---- Thinker: Audio Encoder ----
    audio_config = {
        "model_type": "qwen3_omni_moe_audio_encoder",
        "d_model": 64,
        "encoder_layers": 2,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 128,
        "activation_function": "gelu",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "num_mel_bins": 128,
        "max_source_positions": 1500,
        "num_hidden_layers": 2,
        "output_dim": 64,
        "downsample_hidden_size": 32,
        "n_window": 50,
        "n_window_infer": 800,
        "conv_chunksize": 500,
        "scale_embedding": False,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
    }

    # ---- Thinker: Vision Encoder ----
    vision_config = {
        "model_type": "qwen3_omni_moe_vision_encoder",
        "hidden_size": 64,
        "intermediate_size": 128,
        "depth": 2,
        "num_heads": 2,
        "out_hidden_size": 64,
        "hidden_act": "gelu_pytorch_tanh",
        "image_size": 768,
        "patch_size": 16,
        "spatial_patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "in_channels": 3,
        "in_chans": 3,
        "tokens_per_second": 2,
        "apply_vit_abs_pos_embed": True,
        "deepstack_visual_indexes": [1],
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
    }

    # ---- Thinker: Text (MoE LLM backbone) ----
    text_config = {
        "model_type": "qwen3_omni_moe_text",
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "use_qk_norm": True,
        "use_sliding_window": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        # MoE params
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 0,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 0.001,
        "output_router_logits": False,
        "mlp_only_layers": [],
        "rope_theta": 1000000,
        "rope_scaling": {
            "rope_type": "default",
            "type": "default",
            "interleaved": True,
            "mrope_interleaved": True,
            "mrope_section": [6, 5, 5],
        },
        "tie_word_embeddings": False,
    }

    thinker_config = {
        "model_type": "qwen3_omni_moe_thinker",
        "audio_config": audio_config,
        "vision_config": vision_config,
        "text_config": text_config,
        "audio_token_id": 151675,
        "audio_start_token_id": 151669,
        "audio_end_token_id": 151670,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "position_id_per_seconds": 13,
        "seconds_per_chunk": 2,
        "initializer_range": 0.02,
        "dtype": "bfloat16",
        "user_token_id": 872,
    }

    # ---- Talker (MoE decoder for speech codec prediction) ----
    talker_text_config = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "use_sliding_window": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "decoder_sparse_step": 1,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 0.001,
        "output_router_logits": False,
        "mlp_only_layers": [],
        "vocab_size": 3072,
        "rope_theta": 1000000,
        "rope_scaling": {
            "rope_type": "default",
            "type": "default",
            "interleaved": True,
            "mrope_section": [6, 5, 5],
        },
        "tie_word_embeddings": False,
    }

    code_predictor_config = {
        "model_type": "qwen3_omni_moe_talker_code_predictor",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "use_sliding_window": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "num_code_groups": 16,
        "vocab_size": 2048,
        "layer_types": ["full_attention", "full_attention"],
        "rope_scaling": None,
        "rope_theta": 1000000,
        "tie_word_embeddings": False,
    }

    talker_config = {
        "model_type": "qwen3_omni_moe_talker",
        "text_config": talker_text_config,
        "code_predictor_config": code_predictor_config,
        "thinker_hidden_size": 64,
        "accept_hidden_layer": 1,
        "output_router_logits": False,
        "num_code_groups": 16,
        "position_id_per_seconds": 13,
        "seconds_per_chunk": 2,
        "spatial_merge_size": 2,
        "audio_token_id": 151675,
        "audio_start_token_id": 151669,
        "audio_end_token_id": 151670,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "codec_bos_id": 2149,
        "codec_eos_token_id": 2150,
        "codec_pad_id": 2148,
        "codec_nothink_id": 2155,
        "codec_think_bos_id": 2156,
        "codec_think_eos_id": 2157,
        "speaker_id": {"chelsie": 2301, "ethan": 2302, "aiden": 2303},
    }

    # ---- Code2Wav (codec-to-waveform decoder) ----
    code2wav_config = {
        "model_type": "",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_theta": 10000,
        "decoder_dim": 64,
        "codebook_dim": 32,
        "codebook_size": 2048,
        "semantic_codebook_size": 4096,
        "num_quantizers": 16,
        "num_semantic_quantizers": 1,
        "vector_quantization_hidden_dimension": 32,
        "layer_scale_initial_scale": 0.01,
        "sliding_window": 72,
        "upsample_rates": [8, 5, 4, 3],
        "upsampling_ratios": [2, 2],
        "tie_word_embeddings": False,
    }

    # ---- Top-level Qwen3OmniMoe config ----
    config = {
        "model_type": "qwen3_omni_moe",
        "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
        "thinker_config": thinker_config,
        "talker_config": talker_config if enable_audio_output else None,
        "code2wav_config": code2wav_config if enable_audio_output else None,
        "enable_audio_output": enable_audio_output,
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "tts_pad_token_id": 151671,
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "system_token_id": 8948,
        "user_token_id": 872,
        "assistant_token_id": 77091,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "pad_token_id": 151643,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    return config


def create_and_save_model(config_dict: dict, output_dir: Path) -> None:
    """Instantiate the tiny Qwen3-Omni model and save weights + config."""
    import torch
    from transformers import AutoModelForTextToWaveform
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeConfig,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building Qwen3OmniMoeConfig...")
    config = Qwen3OmniMoeConfig(**config_dict)

    print("Instantiating tiny Qwen3-Omni model (random weights)...")
    with torch.device("meta"):
        model = AutoModelForTextToWaveform.from_config(config)

    # Materialize on CPU with random init.
    model = model.to_empty(device="cpu")
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        elif "layer_norm" in name or "ln_" in name or "norm" in name:
            if "weight" in name:
                torch.nn.init.ones_(param)
            else:
                torch.nn.init.zeros_(param)
        else:
            torch.nn.init.normal_(param, mean=0.0, std=0.001)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.1f}M)")

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)

    print(f"Config saved to {output_dir / 'config.json'}")
    print("Done!")
    return model


def save_tokenizer(source: str, output_dir: Path) -> None:
    """Load tokenizer & processor from source and save into output_dir."""
    from transformers import AutoProcessor, AutoTokenizer

    print(f"Loading tokenizer from {source}...")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    try:
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        processor.save_pretrained(output_dir)
        print(f"Processor saved to {output_dir}")
    except Exception as e:
        print(f"Warning: could not save processor ({e}). "
              "Audio/vision preprocessing may not work.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a tiny debug Qwen3-Omni model with random weights."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the debug model.",
    )
    parser.add_argument(
        "--tokenizer_source",
        type=str,
        default=DEFAULT_TOKENIZER_SOURCE,
        help="HuggingFace model ID or local path to load the tokenizer from "
             f"(default: {DEFAULT_TOKENIZER_SOURCE}).",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip copying the tokenizer.",
    )
    parser.add_argument(
        "--no-talker",
        action="store_true",
        help="Disable the Talker and Code2Wav components.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable the Vision encoder.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=152064,
        help="Vocabulary size (default: 152064, matching Qwen3-Omni tokenizer).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    config_dict = build_tiny_config(
        enable_audio_output=not args.no_talker,
        enable_vision=not args.no_vision,
        vocab_size=args.vocab_size,
    )

    create_and_save_model(config_dict, output_dir)

    if not args.no_tokenizer:
        save_tokenizer(args.tokenizer_source, output_dir)


if __name__ == "__main__":
    main()
