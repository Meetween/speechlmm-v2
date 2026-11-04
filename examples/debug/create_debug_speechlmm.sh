#!/bin/bash
# Create a tiny Qwen3-Omni model for debugging with the SpeechLMM wrapper.
# The saved checkpoint has model_type="qwen3_omni_moe" and gets wrapped into
# SpeechLMM at runtime via use_speechlmm_wrapper: true in the training config.
python scripts/debug/create_debug_model.py --output_dir models/speechlmm_debug_model