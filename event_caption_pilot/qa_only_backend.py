"""Offline Qwen QA with frozen externally prepared frame selections.

The question encoder is intentionally absent. Callers must independently
validate their saved candidate features, scores and selection provenance.
Prompt rendering, image processing and option likelihood scoring are inherited
unchanged from the existing real backend.
"""

from __future__ import annotations

import re
from importlib.metadata import version
from typing import Any

import numpy as np

from .config import Config
from .encoders import encoder_family
from .hf_backend import TransformersBackend
from .types import CaptionOutput


class QwenOnlyBackend(TransformersBackend):
    """Load the pinned BF16 Qwen model without allocating an encoder."""

    def __init__(self, config: Config) -> None:
        config.validate()
        if config.vlm_model != "Qwen/Qwen3.5-4B":
            raise ValueError("Qwen-only QA requires Qwen/Qwen3.5-4B")
        if not re.fullmatch(r"[0-9a-f]{40}", config.vlm_revision):
            raise ValueError("Qwen-only QA requires a pinned VLM commit")
        if not config.local_files_only:
            raise ValueError("Qwen-only QA requires local_files_only=True")
        if (
            config.model_dtype != "bfloat16"
            or config.attention_implementation != "sdpa"
        ):
            raise ValueError("Qwen-only QA requires bfloat16 and sdpa")
        if config.pilot == "a":
            raise ValueError("Qwen-only QA cannot run the encoder-only pilot")
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen-only QA requires the optional real dependencies and "
                "transformers==5.17.0. No synthetic fallback will be used."
            ) from exc
        self.config = config
        self.torch = torch
        self.image_class = Image
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.model_dtype)
        self.last_processor_info: dict[str, Any] = {}
        self.runtime_availability = {
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
        }
        if (
            self.device.type == "cuda"
            and not self.runtime_availability["cuda_available"]
        ):
            raise RuntimeError("CUDA was requested but is unavailable")
        if (
            self.device.type == "mps"
            and not self.runtime_availability["mps_available"]
        ):
            raise RuntimeError("MPS was requested but is unavailable")
        self.encoder_family = encoder_family(config.encoder_model)
        self.clip = None
        self.clip_processor = None
        self.encoder_commit = None
        self.encoder_parameter_info = {
            "devices": [], "dtypes": [], "parameter_count": 0,
            "all_frozen": True,
        }
        # Match the established loader and placement. No accelerate,
        # quantization, CPU offload or encoder model/processor is used.
        self.vlm = AutoModelForMultimodalLM.from_pretrained(
            config.vlm_model,
            revision=config.vlm_revision,
            dtype=self.dtype,
            cache_dir=config.model_cache_dir,
            local_files_only=True,
            attn_implementation=config.attention_implementation,
            trust_remote_code=False,
        )
        self.vlm_commit = getattr(self.vlm.config, "_commit_hash", None)
        if self.vlm_commit != config.vlm_revision:
            raise ValueError("Loaded Qwen commit does not match pinned commit")
        self.vlm = (
            self.vlm.to(self.device).eval().requires_grad_(False)
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.vlm_model,
            revision=self.vlm_commit,
            cache_dir=config.model_cache_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.vlm_parameter_info = self._parameter_info(self.vlm)
        self._synchronize()

    def metadata(self) -> dict[str, Any]:
        """Report actual QA loading and distinguish external encoder data."""
        return {
            "backend": "transformers",
            "adapter_version": "qwen-qa-only-v1",
            "qa_scoring_adapter": "TransformersBackend.answer",
            "synthetic": False,
            "frozen": True,
            "encoder_not_loaded": True,
            "encoder_loaded": False,
            "encoder_model": self.config.encoder_model,
            "encoder_requested_revision": self.config.encoder_revision,
            "encoder_resolved_commit": None,
            "encoder_family": self.encoder_family,
            "encoder_architecture": None,
            "encoder_attention_implementation": None,
            "encoder_parameters": dict(self.encoder_parameter_info),
            "frozen_feature_source": {
                "model": self.config.encoder_model,
                "revision": self.config.encoder_revision,
                "origin": "external frozen candidate features and scores",
                "validation": "caller validates provenance and hashes",
                "computed_by_this_backend": False,
            },
            "vlm_model": self.config.vlm_model,
            "vlm_requested_revision": self.config.vlm_revision,
            "vlm_resolved_commit": self.vlm_commit,
            "vlm_loaded": True,
            "vlm_architecture": type(self.vlm).__name__,
            "device": str(self.device),
            "dtype": self.config.model_dtype,
            "attention_implementation": self.config.attention_implementation,
            "vlm_attention_implementation": getattr(
                self.vlm.config, "_attn_implementation", None
            ),
            "runtime_availability": dict(self.runtime_availability),
            "vlm_parameters": dict(self.vlm_parameter_info),
            "model_cache_dir": self.config.model_cache_dir,
            "local_files_only": self.config.local_files_only,
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
            "caption_enabled": False,
            "enable_thinking": False,
            "qa_prompt_version": self.config.qa_prompt_version,
            "qa_scoring": "full-option conditional token log likelihood",
            "qa_length_normalize": self.config.qa_length_normalize,
            "qa_includes_eos": False,
            "qa_includes_event_captions": False,
            "qa_logits_scope": "continuation predictors only",
            "vlm_min_pixels": self.config.vlm_min_pixels,
            "vlm_max_pixels": self.config.vlm_max_pixels,
        }

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "Qwen-only QA has no encoder; use validated frozen frame features"
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError(
            "Qwen-only QA has no encoder; use validated frozen question scores"
        )

    def encode_visual_question(self, question: str) -> np.ndarray:
        raise RuntimeError(
            "Qwen-only QA has no encoder; use validated frozen question scores"
        )

    def caption(
        self, frames: np.ndarray, timestamps: np.ndarray
    ) -> CaptionOutput:
        raise RuntimeError("Qwen-only QA does not generate event captions")
