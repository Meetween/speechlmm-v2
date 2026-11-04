"""CPU checks for the Qwen2.5 / Qwen3 SpeechLMM family split.

Unit tests cover dispatch, export auto_map, flatten, and wrap key mapping.
Integration tests (wrap, one train step, export, AutoModel reload) run only
when these env vars point at local checkpoints:

  SPEECHLMM_DEBUG_QWEN3     tiny Qwen3-Omni debug checkpoint
  SPEECHLMM_QWEN3_SOURCE    Qwen3-Omni dir with tokenizer/processor files
  SPEECHLMM_QWEN25_SOURCE   Qwen2.5-Omni config source (e.g. 3B)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_export_checkpoint():
    spec = importlib.util.spec_from_file_location(
        "export_checkpoint", ROOT / "scripts" / "export_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_dir(var: str) -> Path | None:
    raw = os.environ.get(var)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


class _Cfg:
    def __init__(self, backbone_type: str):
        self.backbone_type = backbone_type


class DispatchTests(unittest.TestCase):
    def test_for_config_selects_family_subclasses(self):
        from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration
        from speechlmm.models.modeling_speechlmm_qwen2_5_omni import (
            SpeechLMMQwen2_5OmniForConditionalGeneration,
        )
        from speechlmm.models.modeling_speechlmm_qwen3_omni import (
            SpeechLMMQwen3OmniForConditionalGeneration,
        )

        self.assertIs(
            SpeechLMMForConditionalGeneration.for_config(_Cfg("qwen3_omni")),
            SpeechLMMQwen3OmniForConditionalGeneration,
        )
        self.assertIs(
            SpeechLMMForConditionalGeneration.for_config(_Cfg("qwen2_5_omni")),
            SpeechLMMQwen2_5OmniForConditionalGeneration,
        )
        with self.assertRaises(ValueError):
            SpeechLMMForConditionalGeneration.for_config(_Cfg("not_a_backbone"))
        self.assertTrue(SpeechLMMQwen3OmniForConditionalGeneration._supports_sdpa)
        self.assertTrue(SpeechLMMQwen2_5OmniForConditionalGeneration._supports_sdpa)

    def test_take_code2wav_mapping(self):
        from speechlmm.models.modeling_speechlmm_qwen2_5_omni import (
            SpeechLMMQwen2_5OmniForConditionalGeneration,
        )
        from speechlmm.models.modeling_speechlmm_qwen3_omni import (
            SpeechLMMQwen3OmniForConditionalGeneration,
        )

        token2wav = object()
        code2wav = object()
        qwen25 = types.SimpleNamespace(token2wav=token2wav, code2wav=code2wav)
        qwen3 = types.SimpleNamespace(token2wav=token2wav, code2wav=code2wav)

        self.assertIs(SpeechLMMQwen2_5OmniForConditionalGeneration.take_code2wav(qwen25, True), token2wav)
        self.assertIsNone(SpeechLMMQwen2_5OmniForConditionalGeneration.take_code2wav(qwen25, False))
        self.assertIs(SpeechLMMQwen3OmniForConditionalGeneration.take_code2wav(qwen3, True), code2wav)
        self.assertIsNone(SpeechLMMQwen3OmniForConditionalGeneration.take_code2wav(qwen3, False))


class ExportHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = _load_export_checkpoint()

    def test_required_files_and_auto_map_are_per_backbone(self):
        qwen3_files = self.export._required_inference_files("qwen3_omni")
        qwen25_files = self.export._required_inference_files("qwen2_5_omni")
        self.assertIn("modeling_speechlmm_qwen3_omni.py", qwen3_files)
        self.assertNotIn("modeling_speechlmm_qwen2_5_omni.py", qwen3_files)
        self.assertIn("modeling_speechlmm_qwen2_5_omni.py", qwen25_files)
        self.assertNotIn("modeling_speechlmm_qwen3_omni.py", qwen25_files)
        for files in (qwen3_files, qwen25_files):
            self.assertIn("talker_generate.py", files)
            self.assertIn("talker_loss.py", files)

        qwen3_map = self.export._speechlmm_auto_map("qwen3_omni")
        qwen25_map = self.export._speechlmm_auto_map("qwen2_5_omni")
        self.assertEqual(qwen3_map["AutoConfig"], "configuration_speechlmm.SpeechLMMConfig")
        self.assertEqual(
            qwen3_map["AutoModel"],
            "modeling_speechlmm_qwen3_omni.SpeechLMMQwen3OmniForConditionalGeneration",
        )
        self.assertEqual(
            qwen25_map["AutoModel"],
            "modeling_speechlmm_qwen2_5_omni.SpeechLMMQwen2_5OmniForConditionalGeneration",
        )

    def test_remote_code_reaches_talker_generate(self):
        """HF copies only modules reached by top-level relative imports of the AutoModel file."""
        models = ROOT / "src" / "speechlmm" / "models"

        def top_level_relatives(path: Path) -> set[str]:
            names = set()
            for line in path.read_text().splitlines():
                if not line.startswith("from ."):
                    continue
                module = line[len("from .") :].split(" import ", 1)[0].strip()
                if module:
                    names.add(module.split(".", 1)[0])
            return names

        for entry in (
            "modeling_speechlmm_qwen2_5_omni.py",
            "modeling_speechlmm_qwen3_omni.py",
        ):
            seen: set[str] = set()
            stack = [entry]
            while stack:
                name = stack.pop()
                if name in seen:
                    continue
                seen.add(name)
                path = models / name
                if not path.is_file():
                    continue
                for rel in top_level_relatives(path):
                    stack.append(f"{rel}.py")
            self.assertIn("talker_generate.py", seen, entry)

    def test_unknown_backbone_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.export._backbone_modeling_file("nope")
        with self.assertRaises(RuntimeError):
            self.export._speechlmm_auto_map("nope")

    def test_missing_config_backbone_keeps_model_backbone(self):
        cfg: dict = {}
        model_backbone = "qwen2_5_omni"
        resolved = cfg.get("backbone_type") or model_backbone
        self.assertEqual(resolved, "qwen2_5_omni")
        self.assertEqual(self.export._backbone_type_from_config(cfg), "qwen3_omni")

    def test_flatten_collision_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pkg = tmp_p / "pkg"
            encoders = pkg / "encoders"
            encoders.mkdir(parents=True)
            (encoders / "auto_avsr.py").write_text("x = 1\n", encoding="utf-8")
            output = tmp_p / "export"
            output.mkdir()
            (output / "auto_avsr.py").write_text("y = 2\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self.export._flatten_nested_modules(output, pkg)

    def test_export_spk_dict_from_speaker_map(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            out.mkdir()
            model = types.SimpleNamespace(speaker_map={"Ethan": {"bos_token": 3}})
            self.export._export_spk_dict(model, out, [Path(tmp)])
            payload = torch.load(out / "spk_dict.pt", map_location="cpu", weights_only=True)
            self.assertEqual(payload["Ethan"]["bos_token"], 3)

    def test_flatten_rewrites_encoders_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pkg = tmp_p / "pkg"
            encoders = pkg / "encoders"
            encoders.mkdir(parents=True)
            (encoders / "auto_avsr.py").write_text("class AutoAVSR:\n    pass\n", encoding="utf-8")
            output = tmp_p / "export"
            output.mkdir()
            (output / "modeling_speechlmm.py").write_text(
                "from .encoders.auto_avsr import AutoAVSR\n",
                encoding="utf-8",
            )
            (output / "modeling_speechlmm_qwen3_omni.py").write_text(
                "from .encoders.auto_avsr import AutoAVSR\n"
                "from .modeling_speechlmm import SpeechLMMForConditionalGeneration\n",
                encoding="utf-8",
            )
            copied = self.export._flatten_nested_modules(output, pkg)
            self.assertIn("auto_avsr.py", copied)
            self.assertFalse((output / "encoders").exists())
            for name in ("modeling_speechlmm.py", "modeling_speechlmm_qwen3_omni.py"):
                text = (output / name).read_text(encoding="utf-8")
                self.assertIn("from .auto_avsr import AutoAVSR", text)
                self.assertNotIn("from .encoders.auto_avsr", text)


class _FamilyIntegration(unittest.TestCase):
    def _one_step(self, model) -> None:
        import torch

        model.train()
        vocab = int(getattr(model.config.thinker_config.text_config, "vocab_size", 1024))
        input_ids = torch.randint(0, min(vocab, 1000), (1, 8))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertIsNotNone(outputs.loss)
        outputs.loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads, "expected at least one gradient after backward")

    def _export_and_reload(
        self,
        model,
        base_model_path: Path,
        tmp: Path,
        inference_fallback: Path | None = None,
    ) -> None:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        export = _load_export_checkpoint()
        out = tmp / "export"
        out.mkdir()
        tokenizer = None
        if (base_model_path / "tokenizer.json").is_file():
            tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
        export._save_speechlmm(
            model,
            str(base_model_path),
            out,
            max_shard_size="5GB",
            tokenizer=tokenizer,
            inference_fallback=inference_fallback,
        )
        export.verify_exported_inference_assets(out)

        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        backbone = cfg["backbone_type"]
        expected = export._speechlmm_auto_map(backbone)
        self.assertEqual(cfg["auto_map"], expected)
        self.assertTrue((out / export._backbone_modeling_file(backbone)).is_file())
        unused = [
            name
            for name in export.BACKBONE_MODELING_FILES.values()
            if name != export._backbone_modeling_file(backbone)
        ]
        for name in unused:
            self.assertFalse((out / name).exists(), f"export shipped unused {name}")

        loaded = AutoModel.from_pretrained(str(out), trust_remote_code=True)
        self.assertEqual(type(loaded).__name__, type(model).__name__)
        AutoConfig.from_pretrained(str(out), trust_remote_code=True)


class Qwen3DebugIntegrationTests(_FamilyIntegration):
    def test_wrap_step_export_automodel(self):
        debug = _env_dir("SPEECHLMM_DEBUG_QWEN3")
        if debug is None:
            self.skipTest("set SPEECHLMM_DEBUG_QWEN3 to a tiny Qwen3-Omni checkpoint")

        from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration
        from speechlmm.models.modeling_speechlmm_qwen3_omni import (
            SpeechLMMQwen3OmniForConditionalGeneration,
        )

        model = SpeechLMMForConditionalGeneration.from_qwen3_omni(
            str(debug),
            config_overrides={"enable_talker": False, "enable_code2wav": False},
            torch_dtype="float32",
        )
        self.assertIsInstance(model, SpeechLMMQwen3OmniForConditionalGeneration)
        self.assertIsNotNone(model.omni)
        self.assertIs(model.thinker, model.omni.thinker)
        self._one_step(model)
        with tempfile.TemporaryDirectory() as tmp:
            self._export_and_reload(
                model,
                debug,
                Path(tmp),
                inference_fallback=_env_dir("SPEECHLMM_QWEN3_SOURCE"),
            )


class Qwen25ConfigIntegrationTests(_FamilyIntegration):
    def test_from_config_step_export_automodel(self):
        source = _env_dir("SPEECHLMM_QWEN25_SOURCE")
        if source is None:
            self.skipTest("set SPEECHLMM_QWEN25_SOURCE to a Qwen2.5-Omni checkpoint")

        from transformers import AutoConfig, AutoModelForTextToWaveform

        from speechlmm.models.configuration_speechlmm import SpeechLMMConfig
        from speechlmm.models.modeling_speechlmm import SpeechLMMForConditionalGeneration
        from speechlmm.models.modeling_speechlmm_qwen2_5_omni import (
            SpeechLMMQwen2_5OmniForConditionalGeneration,
        )

        qwen2_config = AutoConfig.from_pretrained(str(source), trust_remote_code=True)
        _shrink_qwen25_config(qwen2_config)
        qwen2_model = AutoModelForTextToWaveform.from_config(qwen2_config)
        speechlmm_config = SpeechLMMConfig.from_qwen2_5_omni_config(
            qwen2_config,
            enable_talker=False,
            enable_code2wav=False,
        )
        model = SpeechLMMForConditionalGeneration._wrap_qwen2_5_omni(qwen2_model, speechlmm_config)
        self.assertIsInstance(model, SpeechLMMQwen2_5OmniForConditionalGeneration)
        self.assertEqual(model.config.backbone_type, "qwen2_5_omni")
        self.assertIs(model.omni, qwen2_model)
        self.assertIs(model.thinker, qwen2_model.thinker)
        self._one_step(model)
        with tempfile.TemporaryDirectory() as tmp:
            self._export_and_reload(model, source, Path(tmp), inference_fallback=source)


def _shrink_qwen25_config(config) -> None:
    text = config.thinker_config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 2
    text.num_attention_heads = 2
    text.num_key_value_heads = 2
    if hasattr(text, "head_dim"):
        text.head_dim = 32

    audio = config.thinker_config.audio_config
    audio.d_model = 64
    audio.encoder_layers = 2
    audio.num_hidden_layers = 2
    audio.encoder_attention_heads = 2
    audio.encoder_ffn_dim = 128
    if hasattr(audio, "output_dim"):
        audio.output_dim = 64

    vision = config.thinker_config.vision_config
    vision.hidden_size = 64
    vision.intermediate_size = 128
    if hasattr(vision, "depth"):
        vision.depth = 2
    vision.num_heads = 2
    if hasattr(vision, "out_hidden_size"):
        vision.out_hidden_size = 64

    if getattr(config, "talker_config", None) is not None:
        talker = config.talker_config
        if hasattr(talker, "hidden_size"):
            talker.hidden_size = 64
        if hasattr(talker, "num_hidden_layers"):
            talker.num_hidden_layers = 2
        if hasattr(talker, "num_attention_heads"):
            talker.num_attention_heads = 2
        if hasattr(talker, "num_key_value_heads"):
            talker.num_key_value_heads = 2


class DummyScatterTests(unittest.TestCase):
    def test_empty_mask_keeps_shape_and_connects_grad(self):
        import torch

        from speechlmm.models.modeling_speechlmm import scatter_or_keep_grad

        embeds = torch.zeros(2, 4, 8, requires_grad=True)
        features = torch.ones(3, 8, requires_grad=True)
        mask = torch.zeros(2, 4, 8, dtype=torch.bool)
        out = scatter_or_keep_grad(embeds, features, mask)
        self.assertEqual(out.shape, embeds.shape)
        out.sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features)))
        self.assertTrue(torch.equal(out.detach(), embeds.detach()))

    def test_nonzero_mask_scatters_features(self):
        import torch

        from speechlmm.models.modeling_speechlmm import scatter_or_keep_grad

        embeds = torch.zeros(1, 2, 3)
        features = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.zeros(1, 2, 3, dtype=torch.bool)
        mask[0, 0] = True
        out = scatter_or_keep_grad(embeds, features, mask)
        self.assertTrue(torch.equal(out[0, 0], features[0]))
        self.assertTrue(torch.equal(out[0, 1], torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
