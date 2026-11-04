"""Self-distill Talker codec ids from a frozen official Omni ``generate()``."""

from __future__ import annotations

import re
from contextlib import contextmanager

import numpy as np

from speechlmm.data.codec_cache import N_CODEBOOKS

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def _messages(row: dict) -> tuple[list[dict], str, str]:
    messages = row.get("messages") or row.get("conversations") or []
    if not messages:
        return [], "role", "content"
    role_key = "role" if "role" in messages[0] else "from"
    content_key = "content" if "content" in messages[0] else "value"
    return messages, role_key, content_key


def assistant_text_from_row(row: dict) -> str:
    if row.get("text"):
        return str(row["text"]).strip()
    messages, role_key, content_key = _messages(row)
    for msg in reversed(messages):
        if str(msg.get(role_key, "")).lower() in {"assistant", "gpt"}:
            return str(msg.get(content_key, "")).strip()
    return ""


def distill_prompt(text: str) -> str:
    return f"Repeat exactly the following text and say nothing else:\n{text}"


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _official_speaker(model, speaker: str) -> str:
    names = getattr(model, "speaker_map", None)
    if not names:
        return speaker  # Qwen3-Omni lowercases the name itself
    for name in names:
        if name.lower() == speaker.lower():
            return name
    raise ValueError(f"Unknown speaker {speaker!r}; available: {list(names)}")


@contextmanager
def _capture_codes(model, backbone: str, captured: dict):
    if backbone == "qwen2_5_omni":

        def keep_codes(_module, args):
            captured["codes"] = args[0].detach().cpu()

        handle = model.token2wav.register_forward_pre_hook(keep_codes)
        try:
            yield
        finally:
            handle.remove()
        return

    vocoder = model.code2wav
    decode = vocoder.chunked_decode

    def keep_and_decode(codes, *args, **kwargs):
        captured["codes"] = codes.detach().cpu()
        return decode(codes, *args, **kwargs)

    vocoder.chunked_decode = keep_and_decode
    try:
        yield
    finally:
        del vocoder.chunked_decode


class OmniDistillEncoder:
    """Frozen official Omni: generate speech, keep the Talker codes and the Thinker's reply."""

    def __init__(
        self,
        backbone: str,
        model_path: str,
        *,
        device: str = "cuda",
        thinker_max_new_tokens: int = 256,
        talker_max_new_tokens: int = 512,
    ):
        if backbone not in N_CODEBOOKS:
            raise ValueError(f"Unknown backbone {backbone!r}")
        self.backbone = backbone
        self.model_path = model_path
        self.device = device
        self.thinker_max_new_tokens = thinker_max_new_tokens
        self.talker_max_new_tokens = talker_max_new_tokens
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        import torch
        from transformers import AutoModelForTextToWaveform, AutoProcessor

        dtype = torch.float32 if str(self.device) == "cpu" else torch.bfloat16
        load_kwargs = dict(torch_dtype=dtype, device_map=self.device, trust_remote_code=True)
        if str(self.device) == "auto" and self.backbone == "qwen3_omni":
            n_gpu = torch.cuda.device_count()
            if n_gpu >= 4:
                load_kwargs["max_memory"] = {0: "40GiB", 1: "60GiB", 2: "60GiB", 3: "60GiB"}
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForTextToWaveform.from_pretrained(self.model_path, **load_kwargs).eval()
        if str(self.device) == "auto" and self.backbone == "qwen3_omni":
            self._pin_qwen3_talker(self._model)
        return self._model, self._processor

    @staticmethod
    def _pin_qwen3_talker(model) -> None:
        """device_map=auto parks Talker on the last GPU; generate() stays on cuda:0."""
        import torch
        from accelerate.hooks import remove_hook_from_module

        hub = torch.device("cuda:0")
        if getattr(model, "talker", None) is not None:
            remove_hook_from_module(model.talker, recurse=True)
            model.talker.to(hub)
        if getattr(model, "code2wav", None) is not None:
            remove_hook_from_module(model.code2wav, recurse=True)
            model.code2wav.to(hub)
        if getattr(model, "hf_device_map", None) is not None:
            model.hf_device_map["talker"] = 0
            model.hf_device_map["code2wav"] = 0
        torch.cuda.empty_cache()

    def encode_text(self, prompt: str, speaker: str, *, sample: bool = True) -> tuple[list, str, np.ndarray]:
        """Return the Talker codes (``[T]`` or ``[T][16]``), the Thinker reply they voice and the 24 kHz waveform."""
        model, processor = self._load()
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        templated = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=templated, return_tensors="pt", padding=True).to(model.device)
        audio_flag = {"generation_mode": "audio"} if self.backbone == "qwen2_5_omni" else {"return_audio": True}
        captured: dict = {}
        with _capture_codes(model, self.backbone, captured):
            sequences, wav = model.generate(
                **inputs,
                **audio_flag,
                speaker=_official_speaker(model, speaker),
                thinker_max_new_tokens=self.thinker_max_new_tokens,
                talker_max_new_tokens=self.talker_max_new_tokens,
                talker_do_sample=sample,
            )
        reply = processor.batch_decode(sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        codes = captured["codes"][0]
        codes = codes.tolist() if self.backbone == "qwen2_5_omni" else codes.T.tolist()
        return codes, reply, wav.reshape(-1).float().cpu().numpy()
