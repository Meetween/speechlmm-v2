from transformers import AutoConfig, AutoModel

from .configuration_speechlmm import SpeechLMMConfig
from .modeling_speechlmm import SpeechLMMForConditionalGeneration
from .modeling_speechlmm_qwen2_5_omni import SpeechLMMQwen2_5OmniForConditionalGeneration
from .modeling_speechlmm_qwen3_omni import SpeechLMMQwen3OmniForConditionalGeneration

__all__ = [
    "SpeechLMMConfig",
    "SpeechLMMForConditionalGeneration",
    "SpeechLMMQwen2_5OmniForConditionalGeneration",
    "SpeechLMMQwen3OmniForConditionalGeneration",
]

AutoConfig.register("speechlmm", SpeechLMMConfig)
AutoModel.register(SpeechLMMConfig, SpeechLMMForConditionalGeneration)
