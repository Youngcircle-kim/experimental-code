"""Optional frozen CLIP + Qwen2.5-VL integration; loaded only on request.

Official API references consulted for this implementation:
https://huggingface.co/docs/transformers/model_doc/clip
https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl
https://huggingface.co/docs/transformers/chat_templating

The repository's offline fixture tests do not execute these model weights.
Real integration must be validated with the chosen weights/device before a
research run. Images are supplied chronologically with explicit timestamps.
Option scores include every continuation token, excluding the prompt and EOS.
"""

from __future__ import annotations

import logging
from importlib.metadata import version
from typing import Any

import numpy as np

from .backends import validate_frames, validate_observations
from .config import Config
from .types import CaptionOutput, QaOutput

LOGGER = logging.getLogger(__name__)


class TransformersBackend:
    """Frozen visual/text encoders, captioner and identical frame-only QA."""

    def __init__(self, config: Config) -> None:
        """Load explicitly requested models and freeze all parameters.

        Args:
            config: Model names, revisions, device and precision settings.

        Raises:
            ImportError: Optional real-model dependencies are not installed.
            RuntimeError: The selected accelerator is unavailable.
        """
        try:
            import torch
            from PIL import Image
            from transformers import (
                AutoProcessor,
                CLIPModel,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "The transformers backend requires the optional 'real' "
                "dependencies. No synthetic fallback will be used."
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
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        self.clip = (
            CLIPModel.from_pretrained(
                config.encoder_model,
                revision=config.encoder_revision,
                dtype=self.dtype,
                cache_dir=config.model_cache_dir,
                local_files_only=config.local_files_only,
                attn_implementation=config.attention_implementation,
                trust_remote_code=False,
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.encoder_commit = getattr(self.clip.config, "_commit_hash", None)
        self.clip_processor = AutoProcessor.from_pretrained(
            config.encoder_model,
            revision=self.encoder_commit or config.encoder_revision,
            cache_dir=config.model_cache_dir,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        self.vlm: Any = None
        self.vlm_processor: Any = None
        self.vlm_commit: str | None = None
        if config.pilot != "a":
            self.vlm = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    config.vlm_model,
                    revision=config.vlm_revision,
                    dtype=self.dtype,
                    cache_dir=config.model_cache_dir,
                    local_files_only=config.local_files_only,
                    attn_implementation=config.attention_implementation,
                    trust_remote_code=False,
                )
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )
            self.vlm_commit = getattr(self.vlm.config, "_commit_hash", None)
            self.vlm_processor = AutoProcessor.from_pretrained(
                config.vlm_model,
                revision=self.vlm_commit or config.vlm_revision,
                min_pixels=config.vlm_min_pixels,
                max_pixels=config.vlm_max_pixels,
                cache_dir=config.model_cache_dir,
                local_files_only=config.local_files_only,
                trust_remote_code=False,
            )
        self.encoder_parameter_info = self._parameter_info(self.clip)
        self.vlm_parameter_info = (
            self._parameter_info(self.vlm) if self.vlm is not None else None
        )
        if config.encoder_revision == "main" or (
            self.vlm is not None and config.vlm_revision == "main"
        ):
            LOGGER.warning(
                "A model revision is 'main', which is mutable. Resolved model "
                "commits are logged; pin them in Config for subsequent runs."
            )
        self._synchronize()

    def metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata for cache keys and run artifacts.

        Returns:
            Model revisions, software versions and fixed inference settings.
        """
        return {
            "backend": "transformers",
            "synthetic": False,
            "frozen": True,
            "encoder_model": self.config.encoder_model,
            "encoder_requested_revision": self.config.encoder_revision,
            "encoder_resolved_commit": self.encoder_commit,
            "vlm_model": self.config.vlm_model,
            "vlm_requested_revision": self.config.vlm_revision,
            "vlm_resolved_commit": self.vlm_commit,
            "vlm_loaded": self.vlm is not None,
            "device": str(self.device),
            "dtype": self.config.model_dtype,
            "attention_implementation": self.config.attention_implementation,
            "encoder_attention_implementation": getattr(
                self.clip.config, "_attn_implementation", None
            ),
            "vlm_attention_implementation": (
                getattr(self.vlm.config, "_attn_implementation", None)
                if self.vlm is not None
                else None
            ),
            "runtime_availability": dict(self.runtime_availability),
            "encoder_parameters": dict(self.encoder_parameter_info),
            "vlm_parameters": (
                dict(self.vlm_parameter_info)
                if self.vlm_parameter_info is not None
                else None
            ),
            "model_cache_dir": self.config.model_cache_dir,
            "local_files_only": self.config.local_files_only,
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
            "caption_decoding": "greedy",
            "caption_do_sample": False,
            "caption_max_new_tokens": self.config.caption_max_new_tokens,
            "caption_prompt_version": self.config.caption_prompt_version,
            "qa_prompt_version": self.config.qa_prompt_version,
            "qa_scoring": "full-option conditional token log likelihood",
            "qa_length_normalize": self.config.qa_length_normalize,
            "qa_includes_eos": False,
            "qa_includes_event_captions": False,
            "qa_logits_scope": "continuation predictors only",
            "vlm_min_pixels": self.config.vlm_min_pixels,
            "vlm_max_pixels": self.config.vlm_max_pixels,
            "clip_text_truncation": "fixed context length; warning emitted",
        }

    def _parameter_info(self, model: Any) -> dict[str, Any]:
        """Snapshot actual parameter placement without copying model weights.

        Args:
            model: Loaded frozen Torch module.

        Returns:
            Stable device/dtype sets, parameter count and frozen-state flag.
        """
        devices: set[str] = set()
        dtypes: set[str] = set()
        parameter_count = 0
        all_frozen = True
        for parameter in model.parameters():
            devices.add(str(parameter.device))
            dtypes.add(str(parameter.dtype))
            parameter_count += int(parameter.numel())
            all_frozen = all_frozen and not parameter.requires_grad
        return {
            "devices": sorted(devices),
            "dtypes": sorted(dtypes),
            "parameter_count": parameter_count,
            "all_frozen": all_frozen,
        }

    def _synchronize(self) -> None:
        """Wait for accelerator operations so external timings include work."""
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def _require_vlm(self) -> None:
        """Reject caption or QA calls when detector-only loading was requested.

        Raises:
            RuntimeError: Pilot A intentionally omitted the caption/QA model.
        """
        if self.vlm is None or self.vlm_processor is None:
            raise RuntimeError(
                "Pilot A loads only CLIP for detector comparison. Set "
                "Config.pilot='both' to load the caption/QA model."
            )

    def _images(self, frames: np.ndarray) -> list[Any]:
        """Convert original RGB observations without extra spatial resizing.

        Args:
            frames: RGB values on the [0, 255] scale.

        Returns:
            PIL images; each model's fixed processor handles normalization.
        """
        validate_frames(frames)
        return [
            self.image_class.fromarray(np.rint(frame).astype(np.uint8))
            for frame in frames
        ]

    def _to_device(self, inputs: Any) -> dict[str, Any]:
        """Move model inputs while preserving integer token and grid indices.

        Args:
            inputs: Processor output with tensor-valued fields.

        Returns:
            Device-local input dictionary using the configured floating dtype.
        """
        moved = {
            key: value.to(
                device=self.device,
                dtype=self.dtype if value.is_floating_point() else value.dtype,
            )
            for key, value in inputs.items()
        }
        for key, value in moved.items():
            if value.is_floating_point():
                if not bool(self.torch.isfinite(value).all()):
                    raise ValueError(
                        f"Processed model input {key} is nonfinite"
                    )
                assert bool(self.torch.isfinite(value).all())
        return moved

    def _feature_array(self, features: Any, expected_rows: int) -> np.ndarray:
        """Convert projected features and validate shape and finiteness.

        Args:
            features: Projected Torch tensor.
            expected_rows: Number of frames or texts in this batch.

        Returns:
            Finite feature matrix on CPU in float64.
        """
        result = features.detach().float().cpu().numpy().astype(np.float64)
        if result.ndim != 2 or result.shape[0] != expected_rows:
            raise ValueError("Encoder returned an unexpected feature shape")
        if not np.isfinite(result).all():
            raise ValueError("Encoder returned NaN or infinite features")
        assert result.ndim == 2 and result.shape[0] == expected_rows
        assert np.isfinite(result).all()
        return result

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """Batch frozen CLIP visual encoding using its learned projection.

        Args:
            frames: Original RGB candidate pool, shape (N, H, W, 3).

        Returns:
            Projected finite visual features with shape (N, D).
        """
        validate_frames(frames)
        self._synchronize()
        batches: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), self.config.encoder_batch_size):
                images = self._images(
                    frames[start : start + self.config.encoder_batch_size]
                )
                inputs = self._to_device(
                    self.clip_processor(
                        images=images,
                        return_tensors="pt",
                    )
                )
                # Explicit projection avoids get_*_features return-type changes
                # between Transformers 4 and 5.
                pooled = self.clip.vision_model(**inputs).pooler_output
                projected = self.clip.visual_projection(pooled)
                batches.append(self._feature_array(projected, len(images)))
        self._synchronize()
        result = np.concatenate(batches, axis=0)
        assert result.shape[0] == len(frames) and np.isfinite(result).all()
        return result

    def encode_visual_question(self, question: str) -> np.ndarray:
        """Place question text in the same CLIP space as visual features.

        Args:
            question: Query string, without labels or event captions.

        Returns:
            A finite projected query vector with shape (D,).
        """
        return self.encode_texts([question])[0]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode captions and questions with the identical frozen text model.

        Args:
            texts: Nonempty strings; long texts trigger a truncation warning.

        Returns:
            Projected finite CLIP text matrix with shape (N, D).
        """
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Text encoder requires nonempty input strings")
        max_length = self.clip.config.text_config.max_position_embeddings
        token_lengths = self.clip_processor.tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_length=True,
        )["length"]
        if any(length > max_length for length in token_lengths):
            LOGGER.warning("CLIP text truncated to %d tokens", max_length)
        batches: list[np.ndarray] = []
        self._synchronize()
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.config.encoder_batch_size):
                batch = texts[start : start + self.config.encoder_batch_size]
                inputs = self._to_device(
                    self.clip_processor(
                        text=batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    )
                )
                pooled = self.clip.text_model(**inputs).pooler_output
                projected = self.clip.text_projection(pooled)
                batches.append(self._feature_array(projected, len(batch)))
        self._synchronize()
        result = np.concatenate(batches, axis=0)
        assert result.shape[0] == len(texts) and np.isfinite(result).all()
        return result

    def _prompt(self, timestamps: np.ndarray, instruction: str) -> str:
        """Render the model's chat template with chronological image markers.

        Args:
            timestamps: Exact observed timestamps in seconds.
            instruction: Fixed caption instruction or frame-only QA question.

        Returns:
            Chat-formatted text ending at the assistant generation prefix.
        """
        content: list[dict[str, str]] = []
        for timestamp in timestamps:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Frame at {timestamp:.6f} seconds:",
                    },
                    {"type": "image"},
                ]
            )
        content.append({"type": "text", "text": instruction})
        return str(
            self.vlm_processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _vlm_inputs(self, prompt: str, images: list[Any]) -> dict[str, Any]:
        """Process a multimodal prompt without truncating observations.

        Args:
            prompt: Chat-templated text, optionally followed by an answer.
            images: Original observations corresponding to image markers.

        Returns:
            Validated model input tensors on the configured device.

        Raises:
            ValueError: The expanded image/text sequence exceeds model context.
        """
        inputs = self._to_device(
            self.vlm_processor(
                text=[prompt],
                images=images,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )
        )
        text_config = getattr(self.vlm.config, "text_config", self.vlm.config)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit and inputs["input_ids"].shape[1] > context_limit:
            raise ValueError("VLM input exceeds the configured model context")
        if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
            raise ValueError(
                "VLM processor returned an unexpected input shape"
            )
        assert inputs["input_ids"].ndim == 2
        image_grid = inputs.get("image_grid_thw")
        if image_grid is None or image_grid.shape != (len(images), 3):
            raise ValueError("VLM image grids do not match supplied images")
        assert image_grid.ndim == 2 and image_grid.shape[0] == len(images)
        # Per-call input diagnostics are intentionally excluded from metadata:
        # metadata participates in caption cache keys and must remain stable.
        self.last_processor_info = {
            "n_frames": len(images),
            "image_sizes_width_height": [list(image.size) for image in images],
            "image_grid_thw": image_grid.detach().cpu().tolist(),
            "expanded_input_tokens": int(inputs["input_ids"].shape[1]),
            "input_device": str(inputs["input_ids"].device),
        }
        return inputs

    def caption(
        self, frames: np.ndarray, timestamps: np.ndarray
    ) -> CaptionOutput:
        """Generate a greedy caption using only fixed observations and times.

        Args:
            frames: Question-independent chronological observations.
            timestamps: Exact observation times in seconds.

        Returns:
            Caption and actual expanded input/output token counts.
        """
        self._require_vlm()
        validate_observations(frames, timestamps)
        if len(frames) > self.config.caption_max_frames:
            raise ValueError(
                "Caption observation count exceeds fixed input budget"
            )
        self._synchronize()
        prompt = self._prompt(timestamps, self.config.caption_prompt)
        inputs = self._vlm_inputs(prompt, self._images(frames))
        input_tokens = int(inputs["input_ids"].shape[1])
        self.last_processor_info["task"] = "caption"
        LOGGER.info(
            "Caption start: n_frames=%d input_tokens=%d max_new_tokens=%d",
            len(frames),
            input_tokens,
            self.config.caption_max_new_tokens,
        )
        with self.torch.inference_mode():
            output_ids = self.vlm.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.config.caption_max_new_tokens,
            )
        self._synchronize()
        continuation = output_ids[:, input_tokens:]
        text = self.vlm_processor.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if not text:
            raise ValueError("Caption model returned an empty caption")
        assert continuation.ndim == 2 and continuation.shape[0] == 1
        LOGGER.info(
            "Caption complete: n_frames=%d input_tokens=%d output_tokens=%d",
            len(frames),
            input_tokens,
            int(continuation.shape[1]),
        )
        return CaptionOutput(text, input_tokens, int(continuation.shape[1]))

    def answer(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        question: str,
        options: tuple[str, ...],
    ) -> QaOutput:
        """Score complete option text conditioned on selected original frames.

        Args:
            frames: Final original frames; no event captions are accepted.
            timestamps: Their chronological timestamps.
            question: Question text, without ground-truth information.
            options: Candidate answer strings, in dataset order.

        Returns:
            Deterministic argmax and all conditional option log-likelihoods.

        Raises:
            ValueError: Invalid inputs, continuation boundaries or scores.
        """
        self._require_vlm()
        validate_observations(frames, timestamps)
        if not question.strip() or len(options) < 2:
            raise ValueError("QA requires a question and at least two options")
        if any(not option.strip() for option in options):
            raise ValueError("QA options must be nonempty")
        self._synchronize()
        option_lines = "\n".join(
            f"{index + 1}. {option}" for index, option in enumerate(options)
        )
        instruction = (
            f"{self.config.qa_prompt}\nQuestion: {question}\nOptions:\n"
            f"{option_lines}\nAnswer with the full text of one option."
        )
        prompt = self._prompt(timestamps, instruction)
        images = self._images(frames)
        prompt_inputs = self._vlm_inputs(prompt, images)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_length = int(prompt_ids.shape[1])
        self.last_processor_info["task"] = "qa_prompt"
        LOGGER.info(
            "QA start: n_frames=%d input_tokens=%d n_options=%d",
            len(frames),
            prompt_length,
            len(options),
        )
        option_scores: list[float] = []
        with self.torch.inference_mode():
            for option_index, option in enumerate(options):
                # Tokenize the exact joint continuation: independently encoded
                # option tokens can differ at a BPE boundary.
                inputs = self._vlm_inputs(prompt + option, images)
                full_ids = inputs["input_ids"]
                if not self.torch.equal(
                    full_ids[:, :prompt_length], prompt_ids
                ):
                    raise ValueError(
                        "Tokenizer changed the prompt at the option boundary; "
                        "cannot score this chat template without token leakage"
                    )
                target_ids = full_ids[0, prompt_length:]
                option_length = int(target_ids.numel())
                if option_length == 0:
                    raise ValueError("Option has no continuation tokens")
                self.last_processor_info.update(
                    {
                        "task": "qa_option",
                        "option_index": option_index,
                        "prompt_tokens": prompt_length,
                        "continuation_tokens": option_length,
                    }
                )
                LOGGER.info(
                    "QA option %d/%d: input_tokens=%d continuation_tokens=%d",
                    option_index + 1,
                    len(options),
                    int(full_ids.shape[1]),
                    option_length,
                )
                output = self.vlm(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=option_length + 1,
                )
                # Every token is predicted by the preceding position. Only the
                # continuation positions contribute; the full prompt is masked.
                # Retain the final prompt position plus continuation positions;
                # drop the last unused prediction before scoring target tokens.
                logits = output.logits[0, :-1].float()
                if logits.shape[0] != option_length:
                    raise ValueError(
                        "QA logits and continuation lengths disagree"
                    )
                log_probabilities = self.torch.log_softmax(logits, dim=-1)
                token_log_likelihood = log_probabilities.gather(
                    dim=-1,
                    index=target_ids[:, None],
                ).squeeze(-1)
                if not self.torch.isfinite(token_log_likelihood).all():
                    raise ValueError(
                        "QA produced nonfinite token log likelihoods"
                    )
                score = (
                    token_log_likelihood.mean()
                    if self.config.qa_length_normalize
                    else token_log_likelihood.sum()
                )
                option_scores.append(float(score.item()))
        self._synchronize()
        scores = np.asarray(option_scores, dtype=float)
        assert scores.shape == (len(options),) and np.isfinite(scores).all()
        LOGGER.info(
            "QA complete: n_frames=%d input_tokens=%d n_options=%d",
            len(frames),
            prompt_length,
            len(options),
        )
        return QaOutput(
            int(np.argmax(scores)),
            tuple(float(score) for score in scores),
        )
