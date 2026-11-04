#!/usr/bin/env python3
"""GPU smoke: SpeechLMM wrap generate() → wav on a local Qwen Omni checkpoint.

Sizes: m = Qwen2.5-Omni-3B, l = Qwen2.5-Omni-7B, xl = Qwen3-Omni-30B-A3B-Instruct.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch


MODELS = os.environ.get("MODELS_DIR", "models")
SIZE_CKPT = {
    "m": f"{MODELS}/Qwen/Qwen2.5-Omni-3B",
    "l": f"{MODELS}/Qwen/Qwen2.5-Omni-7B",
    "xl": f"{MODELS}/Qwen/Qwen3-Omni-30B-A3B-Instruct",
}

SYSTEM_TEXT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
DEFAULT_PHRASES = [
    ("codec", "Say only these words, nothing else: codec validation one two three."),
    ("hello", "Say only these words, nothing else: hello, how are you today?"),
    ("weather", "Say only these words, nothing else: the weather is nice this morning."),
    ("window", "Say only these words, nothing else: please open the window."),
    ("meeting", "Say only these words, nothing else: the meeting starts at three o'clock."),
]


def conversation_for(user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]



def audio_stats(audio) -> dict:
    import numpy as np

    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu()
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(x).all())
    if x.size == 0 or not finite:
        return {"n": int(x.size), "finite": finite, "rms": 0.0, "peak": 0.0, "nonsilent": False}
    rms = float((x * x).mean() ** 0.5)
    peak = float(abs(x).max())
    return {
        "n": int(x.size),
        "finite": finite,
        "rms": rms,
        "peak": peak,
        "nonsilent": rms > 1e-3 and peak > 1e-2,
    }


def write_wav(path: Path, audio, sr: int) -> None:
    import torchaudio

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    wav = audio.detach().float().cpu().reshape(1, -1).contiguous()
    peak = wav.abs().max().clamp(min=1e-8)
    if peak > 1.0:
        wav = wav / peak
    torchaudio.save(str(path), wav, sr)


def infer_size(ckpt: str) -> str:
    name = Path(ckpt).name.lower()
    if "qwen3" in name or "30b" in name:
        return "xl"
    if "7b" in name:
        return "l"
    return "m"


def pick_ckpt(explicit: str | None, size: str | None) -> str:
    if explicit:
        if not Path(explicit).is_dir():
            raise FileNotFoundError(explicit)
        return explicit
    if size:
        path = SIZE_CKPT[size]
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
        return path
    for path in (SIZE_CKPT["m"], SIZE_CKPT["l"]):
        if Path(path).is_dir():
            return path
    raise FileNotFoundError(f"No Qwen Omni checkpoint under {MODELS}/Qwen")


def load_processor(ckpt: str, size: str):
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(ckpt, local_files_only=True)
    except Exception:
        if size == "xl":
            from transformers import Qwen3OmniMoeProcessor

            return Qwen3OmniMoeProcessor.from_pretrained(ckpt, local_files_only=True)
        from transformers import Qwen2_5OmniProcessor

        return Qwen2_5OmniProcessor.from_pretrained(ckpt, local_files_only=True)


def vocoder_module(model):
    return getattr(model, "code2wav", None) or getattr(model, "token2wav", None)


def _tensor_fingerprint(weight) -> dict:
    x = weight.detach().float().reshape(-1)
    head = x[: min(x.numel(), 4096)].cpu()
    return {
        "shape": list(weight.shape),
        "dtype": str(weight.dtype).replace("torch.", ""),
        "mean": float(x.mean().cpu()),
        "std": float(x.std().cpu()),
        "absmax": float(x.abs().max().cpu()),
        "head_sum": float(head.sum()),
    }


def model_fingerprints(model) -> dict:
    talker = getattr(model, "talker", None)
    vocoder = vocoder_module(model)
    text = getattr(model, "model", None)
    if text is None and getattr(model, "thinker", None) is not None:
        text = model.thinker.model
    out = {}
    if talker is not None and getattr(talker, "codec_head", None) is not None:
        out["talker.codec_head"] = _tensor_fingerprint(talker.codec_head.weight)
        layers = getattr(getattr(talker, "model", None), "layers", None)
        if layers:
            out["talker.layer0_ln"] = _tensor_fingerprint(layers[0].input_layernorm.weight)
    if text is not None:
        layers = getattr(text, "layers", None)
        if layers:
            out["thinker.layer0_ln"] = _tensor_fingerprint(layers[0].input_layernorm.weight)
    if vocoder is not None:
        dit = getattr(vocoder, "code2wav_dit_model", vocoder)
        param = next(dit.parameters(), None)
        if param is not None:
            out["vocoder.first"] = _tensor_fingerprint(param)
    omni = getattr(model, "omni", None)
    out["omni_talker_is_checkpoint_talker"] = bool(
        omni is not None and talker is not None and omni.talker is talker
    )
    out["omni_vocoder_is_checkpoint_vocoder"] = bool(
        omni is not None
        and vocoder is not None
        and (getattr(omni, "token2wav", None) is vocoder or getattr(omni, "code2wav", None) is vocoder)
    )
    return out


def pin_talker_vocoder(model, device: torch.device) -> None:
    try:
        from accelerate.hooks import remove_hook_from_module
    except Exception:
        remove_hook_from_module = None
    for name in ("talker", "code2wav", "token2wav"):
        module = getattr(model, name, None)
        if module is None:
            continue
        if remove_hook_from_module is not None:
            remove_hook_from_module(module, recurse=True)
        module.to(device)
        print(f"pinned {name} -> {device}", flush=True)


def first_device(model, requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0")
    try:
        return torch.device(model.device)
    except Exception:
        return torch.device(requested if requested != "auto" else "cuda:0")


def extract_codes(talker_out, talker_kwargs, code2wav_codes):
    if code2wav_codes is not None:
        codes = code2wav_codes
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        return codes.detach().cpu().long()
    seq = talker_out.sequences if hasattr(talker_out, "sequences") else talker_out
    in_ids = talker_kwargs.get("input_ids")
    if in_ids is None and talker_kwargs.get("talker_input_ids") is not None:
        in_ids = talker_kwargs["talker_input_ids"]
    in_len = int(in_ids.shape[1]) if in_ids is not None and getattr(in_ids, "ndim", 0) == 2 else 0
    if seq.ndim == 2 and seq.shape[1] > in_len + 1:
        seq = seq[:, in_len:-1]
    return seq.detach().cpu().long()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--size", choices=("m", "l", "xl"), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speaker", default="ethan")
    parser.add_argument(
        "--talker-do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--thinker-max-new-tokens", type=int, default=32)
    parser.add_argument("--backend", choices=("wrap", "native", "speechlmm"), default="wrap")
    parser.add_argument(
        "--processor-ckpt",
        default=None,
        help="Tokenizer/processor dir. Default: --ckpt. Use the Qwen dir for matched compares.",
    )
    parser.add_argument(
        "--rebuild-omni",
        action="store_true",
        help="Drop a live Omni handle and rebuild it (saved-checkpoint path).",
    )
    parser.add_argument(
        "--export-reload",
        action="store_true",
        help="Export the wrap, then from_pretrained that dir before generate.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"ok": False}

    try:
        ckpt = pick_ckpt(args.ckpt, args.size)
        size = args.size or infer_size(ckpt)
        backend = args.backend
        print(f"ckpt={ckpt} size={size} backend={backend}", flush=True)
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto" if args.device == "auto" else args.device,
            local_files_only=True,
            trust_remote_code=True,
        )
        if size == "xl":
            load_kwargs["attn_implementation"] = "sdpa"
            load_kwargs["max_memory"] = {0: "40GiB", 1: "60GiB", 2: "60GiB", 3: "60GiB"}
        if backend == "native":
            from transformers import AutoModelForTextToWaveform

            from speechlmm.models.talker_generate import _fix_qwen25_token2wav_rope, _remap_speaker

            model = AutoModelForTextToWaveform.from_pretrained(ckpt, **load_kwargs)
            _fix_qwen25_token2wav_rope()
        elif backend == "speechlmm":
            from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration
            from speechlmm.models.talker_generate import _fix_qwen25_token2wav_rope

            model = SpeechLMMForConditionalGeneration.from_pretrained(ckpt, **load_kwargs)
            _fix_qwen25_token2wav_rope()
        else:
            from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration

            if size == "xl":
                model = SpeechLMMForConditionalGeneration.from_qwen3_omni(ckpt, **load_kwargs)
            else:
                model = SpeechLMMForConditionalGeneration.from_qwen2_5_omni(ckpt, **load_kwargs)
            if args.export_reload:
                import importlib.util

                from transformers import AutoTokenizer

                spec = importlib.util.spec_from_file_location(
                    "export_checkpoint",
                    Path(__file__).resolve().parents[1] / "scripts" / "export_checkpoint.py",
                )
                export = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(export)
                dest = out_dir / "export"
                dest.mkdir(parents=True, exist_ok=True)
                tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
                export._save_speechlmm(
                    model,
                    ckpt,
                    dest,
                    max_shard_size="5GB",
                    tokenizer=tokenizer,
                    inference_fallback=Path(ckpt),
                )
                export.verify_exported_inference_assets(dest)
                del model
                torch.cuda.empty_cache()
                model = SpeechLMMForConditionalGeneration.from_pretrained(str(dest), **load_kwargs)
                backend = "speechlmm"
                print(f"reloaded export has_omni={getattr(model, 'omni', None) is not None}", flush=True)
            elif args.rebuild_omni:
                from speechlmm.models.talker_generate import attach_omni_container

                object.__setattr__(model, "omni", None)
                object.__setattr__(model, "thinker", None)
                attach_omni_container(model)
                print(f"rebuilt omni class={type(model.omni).__name__}", flush=True)
        model.eval()
        input_device = first_device(model, args.device)
        spk_source = None
        ckpt_spk = Path(ckpt) / "spk_dict.pt"
        if ckpt_spk.is_file() and getattr(model, "speaker_map", None):
            spk_source = str(ckpt_spk)
        if not getattr(model, "speaker_map", None):
            from speechlmm.models.talker_generate import attach_omni_container, maybe_load_spk_dict

            for cand in (ckpt_spk, Path(SIZE_CKPT["l"]) / "spk_dict.pt", Path(SIZE_CKPT["m"]) / "spk_dict.pt"):
                if cand.is_file():
                    maybe_load_spk_dict(model, cand.parent)
                    attach_omni_container(model)
                    spk_source = str(cand)
                    print(f"loaded speakers from {cand}", flush=True)
                    break
        pin_talker_vocoder(model, input_device)
        torch.cuda.empty_cache()
        processor_ckpt = args.processor_ckpt or ckpt
        processor = load_processor(processor_ckpt, size)
        fingerprints = model_fingerprints(model)
        print(f"processor={processor_ckpt}", flush=True)
        print(f"spk_source={spk_source}", flush=True)
        print(f"fingerprints={json.dumps(fingerprints, default=str)}", flush=True)
        vocoder = vocoder_module(model)
        speakers = list(getattr(model, "speaker_map", {}).keys())
        speaker_ids = getattr(getattr(model.config, "talker_config", None), "speaker_id", None)
        import transformers

        summary.update(
            {
                "ckpt": ckpt,
                "size": size,
                "backend": backend,
                "class": type(model).__name__,
                "has_talker": bool(getattr(model, "has_talker", True) and getattr(model, "talker", None) is not None),
                "has_code2wav": vocoder is not None,
                "has_official_thinker": getattr(model, "thinker", None) is not None,
                "has_omni": getattr(model, "omni", None) is not None,
                "speakers": speakers,
                "speaker_id": speaker_ids,
                "spk_source": spk_source,
                "processor_ckpt": processor_ckpt,
                "fingerprints": fingerprints,
                "device": str(input_device),
                "transformers": transformers.__version__,
            }
        )
        if not summary["has_talker"] or not summary["has_code2wav"]:
            raise RuntimeError(f"missing talker/vocoder: {summary}")

        captured = {}
        orig_talker = model.talker.generate

        def hooked_talker(*a, **k):
            captured["talker_out"] = orig_talker(*a, **k)
            captured["talker_kwargs"] = k
            return captured["talker_out"]

        model.talker.generate = hooked_talker
        orig_c2w = None
        if hasattr(vocoder, "chunked_decode"):
            orig_c2w = vocoder.chunked_decode

            def hooked_c2w(codes, *a, **k):
                captured["c2w_codes"] = codes.detach().cpu()
                return orig_c2w(codes, *a, **k)

            vocoder.chunked_decode = hooked_c2w
        thinker_max = args.thinker_max_new_tokens
        talker_max = 1024
        gen_kwargs = dict(
            speaker=args.speaker,
            thinker_max_new_tokens=thinker_max,
            talker_max_new_tokens=talker_max,
            talker_do_sample=args.talker_do_sample,
            return_audio=True,
        )
        if backend == "native":
            _remap_speaker(model, gen_kwargs)
        sample_summaries = []
        try:
            for _phrase_i, (slug, user_text) in enumerate(DEFAULT_PHRASES):
                text = processor.apply_chat_template(
                    conversation_for(user_text), add_generation_prompt=True, tokenize=False
                )
                inputs = processor(text=text, return_tensors="pt", padding=True)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(input_device)
                else:
                    inputs = {k: v.to(input_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                for sample_i in range(max(1, args.num_samples)):
                    captured.clear()
                    if args.seed is not None:
                        torch.manual_seed(args.seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(args.seed)
                    with torch.inference_mode():
                        result = model.generate(**inputs, **gen_kwargs)
                    if not isinstance(result, (tuple, list)) or len(result) < 2:
                        raise RuntimeError(f"generate() did not return (text, wav); got {type(result)}")
                    sequences, wav = result[0], result[1]
                    if captured.get("talker_out") is None:
                        raise RuntimeError("talker.generate was not called")
                    wav_name = f"{backend}_generate_ethan_{slug}.wav"
                    if sample_i:
                        wav_name = f"{backend}_generate_ethan_{slug}_{sample_i}.wav"
                    wav_path = out_dir / wav_name
                    write_wav(wav_path, wav, 24000)
                    stats = audio_stats(wav)
                    codes = extract_codes(
                        captured["talker_out"],
                        captured.get("talker_kwargs") or {},
                        captured.get("c2w_codes"),
                    )
                    decoded = processor.batch_decode(sequences, skip_special_tokens=True)
                    n_codes = int(codes.numel() if codes.ndim < 2 else codes.shape[-1])
                    talker_clipped = n_codes >= talker_max - 1
                    sample_summaries.append(
                        {
                            "ok": bool(stats["nonsilent"] and stats["finite"] and not talker_clipped),
                            "slug": slug,
                            "prompt": user_text,
                            "wav": str(wav_path),
                            "audio": stats,
                            "duration_s": stats["n"] / 24000.0,
                            "n_codes": n_codes,
                            "codes_shape": list(codes.shape),
                            "talker_clipped": talker_clipped,
                            "text": decoded,
                            "seed": args.seed,
                            "input_len": int(inputs["input_ids"].shape[-1])
                            if isinstance(inputs, dict)
                            else int(inputs.input_ids.shape[-1]),
                        }
                    )
        finally:
            model.talker.generate = orig_talker
            if orig_c2w is not None:
                vocoder.chunked_decode = orig_c2w

        first = sample_summaries[0]
        summary.update(
            {
                "ok": all(item["ok"] for item in sample_summaries),
                "wav": first["wav"],
                "audio": first["audio"],
                "duration_s": first["duration_s"],
                "n_codes": first["n_codes"],
                "codes_shape": first["codes_shape"],
                "thinker_max_new_tokens": thinker_max,
                "talker_max_new_tokens": talker_max,
                "talker_do_sample": args.talker_do_sample,
                "talker_clipped": first["talker_clipped"],
                "text": first["text"],
                "seed": args.seed,
                "backend": backend,
                "samples": sample_summaries,
            }
        )
        if not summary["ok"]:
            summary["reason"] = "one or more phrases silent, non-finite, or talker-clipped"
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        print(summary["traceback"], flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
