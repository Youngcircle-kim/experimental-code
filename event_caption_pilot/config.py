"""Serializable, validated settings for both research pilots."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

DEFAULT_SEED = 42


@dataclass(frozen=True)
class Config:
    """Keep all experimental choices in one JSON-serializable object."""

    seed: int = DEFAULT_SEED
    output_dir: str = "outputs"
    cache_dir: str = ".cache/event_caption"
    mode: str = "demo"
    manifest_path: str | None = None
    backend: str = "demo"
    deterministic: bool = True
    seed_torch: bool = False
    seed_tensorflow: bool = False
    cublas_workspace_config: str = ":4096:8"
    pilot: str = "both"
    detector: str = "D2"
    uniform_segments: int = 6
    boundary_threshold: float = 0.15
    window_size: int = 3
    min_segment_frames: int = 2
    max_segments: int = 24
    diagnostic_max_boundaries: int = 12
    observation_strides: tuple[int, ...] = (1, 2, 4)
    frame_budget: int = 16
    caption_max_frames: int = 4
    caption_max_new_tokens: int = 80
    caption_prompt_version: str = "observed-frames-v1"
    caption_prompt: str = (
        "Summarize only the supplied frames, shown in chronological order, in "
        "one or two short sentences. Describe visible objects, "
        "observed actions "
        "and verifiable state changes. Do not infer unseen actions between "
        "frames, causes, intentions, emotions, exact counts or temporal order "
        "that is not visible. State uncertainty when observations "
        "are insufficient."
    )
    qa_prompt_version: str = "original-frames-mcq-v1"
    qa_prompt: str = (
        "Use only these original video frames in chronological order "
        "to answer "
        "the multiple-choice question. Select the best supported option."
    )
    text_weight: float = 0.5
    allocation_temperature: float = 1.0
    normalization_epsilon: float = 1e-8
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    candidate_fps: float = 2.0
    max_candidate_frames: int = 4096
    frame_height: int = 224
    frame_width: int = 224
    encoder_batch_size: int = 16
    encoder_model: str = "openai/clip-vit-base-patch32"
    encoder_revision: str = "main"
    model_cache_dir: str = ".cache/huggingface"
    local_files_only: bool = False
    attention_implementation: str = "eager"
    vlm_model: str = "Qwen/Qwen3.8-27B"
    vlm_revision: str = "main"
    device: str = "cpu"
    model_dtype: str = "float32"
    vlm_min_pixels: int = 50176
    vlm_max_pixels: int = 200704
    qa_length_normalize: bool = True
    randomize_condition_order: bool = True
    demo_dev_videos: int = 2
    demo_eval_videos: int = 4
    demo_questions_per_video: int = 6
    demo_frames_per_video: int = 96
    demo_event_count: int = 6
    demo_noise_std: float = 0.025
    demo_frame_height: int = 32
    demo_frame_width: int = 32

    def validate(self) -> None:
        """Reject invalid or incompatible experiment settings.

        Raises:
            ValueError: A parameter is outside its supported range.
        """
        defaults = Config()
        for parameter in fields(self):
            value = getattr(self, parameter.name)
            default = getattr(defaults, parameter.name)
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{parameter.name} must be a boolean")
            if type(default) is int and (type(value) is not int):
                raise ValueError(f"{parameter.name} must be an integer")
            if type(default) is float and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"{parameter.name} must be numeric")
            if isinstance(default, str) and (
                not isinstance(value, str) or not value
            ):
                raise ValueError(f"{parameter.name} must be a nonempty string")
        positive_names = (
            "uniform_segments",
            "window_size",
            "min_segment_frames",
            "max_segments",
            "diagnostic_max_boundaries",
            "frame_budget",
            "caption_max_frames",
            "caption_max_new_tokens",
            "bootstrap_samples",
            "candidate_fps",
            "max_candidate_frames",
            "frame_height",
            "frame_width",
            "encoder_batch_size",
            "allocation_temperature",
            "normalization_epsilon",
            "vlm_min_pixels",
            "vlm_max_pixels",
            "demo_dev_videos",
            "demo_eval_videos",
            "demo_questions_per_video",
            "demo_frames_per_video",
            "demo_event_count",
            "demo_frame_height",
            "demo_frame_width",
        )
        for name in positive_names:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not 0 <= self.text_weight <= 1:
            raise ValueError("text_weight must be in [0, 1]")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if (
            not math.isfinite(self.boundary_threshold)
            or not 0 <= self.boundary_threshold <= 2
        ):
            raise ValueError("boundary_threshold must be in [0, 2]")
        if not math.isfinite(self.demo_noise_std) or self.demo_noise_std < 0:
            raise ValueError("demo_noise_std must be finite and nonnegative")
        if not self.observation_strides or any(
            not isinstance(stride, int)
            or isinstance(stride, bool)
            or stride < 1
            for stride in self.observation_strides
        ):
            raise ValueError(
                "observation_strides must contain positive integers"
            )
        if self.mode not in {"demo", "real"} or self.pilot not in {
            "a",
            "both",
        }:
            raise ValueError("mode must be demo/real and pilot must be a/both")
        if self.detector not in {"D0", "D1", "D2"}:
            raise ValueError("detector must be D0, D1 or D2")
        if self.mode == "real" and (
            not self.manifest_path or self.backend == "demo"
        ):
            raise ValueError(
                "Real runs require a manifest and a non-demo backend"
            )
        if self.vlm_min_pixels > self.vlm_max_pixels:
            raise ValueError("vlm_min_pixels must not exceed vlm_max_pixels")
        if self.demo_event_count > self.demo_frames_per_video:
            raise ValueError(
                "demo_event_count must not exceed demo_frames_per_video"
            )
        if min(self.demo_frame_height, self.demo_frame_width) < 8:
            raise ValueError("Demo frame height/width must be at least 8")
        if self.model_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "model_dtype must be float32, float16 or bfloat16"
            )
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("attention_implementation must be eager or sdpa")

    def to_dict(self) -> dict[str, Any]:
        """Return serializable settings for artifact provenance.

        Returns:
            All configured parameters, without omitting defaults.
        """
        return asdict(self)
