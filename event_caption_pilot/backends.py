"""Frozen backend boundary and an explicitly synthetic executable fixture.

The demo is a deterministic pixel/word rule system, not a pretrained model and
not scientific evidence for caption utility. Real backend failures never fall
back to this fixture.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, cast

import numpy as np

from .config import Config
from .types import Backend, CaptionOutput, QaOutput, Question, Video

COLOR_NAMES = ("red", "green", "blue", "yellow", "magenta", "cyan")
COLOR_RGB = np.asarray(
    [
        (230, 35, 35),
        (35, 230, 35),
        (35, 35, 230),
        (230, 230, 35),
        (230, 35, 230),
        (35, 230, 230),
    ],
    dtype=np.float64,
)
POSITION_NAMES = ("left", "right")
SHAPE_NAMES = ("square", "circle")
DEMO_VOCABULARY = COLOR_NAMES + POSITION_NAMES + SHAPE_NAMES
RGB_CHANNELS = 3
PIXEL_MAX = 255
DEMO_BACKGROUND = 18
DEMO_FOREGROUND_THRESHOLD = 100
DEMO_CIRCLE_FILL_THRESHOLD = 0.9
DEMO_POSITION_FRACTIONS = (0.25, 0.75)
DEMO_OBJECT_RADIUS_FRACTION = 0.18
DEMO_SCHEMA_VERSION = "synthetic-pixels-v1"


def validate_frames(frames: np.ndarray) -> None:
    """Validate nonempty RGB inputs before a model or pixel operation.

    Args:
        frames: Original RGB frames with shape (N, H, W, 3).

    Raises:
        ValueError: Frames have an invalid shape, range or numeric dtype.
    """
    if frames.ndim != 4 or frames.shape[-1] != RGB_CHANNELS:
        raise ValueError("Expected RGB frames with shape (N, H, W, 3)")
    if any(size <= 0 for size in frames.shape):
        raise ValueError("Frame dimensions must be nonempty")
    if not np.issubdtype(frames.dtype, np.number):
        raise ValueError("Frames must have a numeric dtype")
    if not np.isfinite(frames).all():
        raise ValueError("Frames contain NaN or infinity")
    if frames.min() < 0 or frames.max() > PIXEL_MAX:
        raise ValueError("Frames must use the RGB [0, 255] pixel convention")
    assert frames.ndim == 4 and np.isfinite(frames).all()


def validate_observations(
    frames: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """Reject missing timestamps or observations out of chronological order.

    Args:
        frames: Selected original RGB observations.
        timestamps: One finite timestamp per frame, in seconds.

    Raises:
        ValueError: Observation and timestamp shapes or ordering disagree.
    """
    validate_frames(frames)
    if timestamps.shape != (frames.shape[0],):
        raise ValueError("Each observation requires exactly one timestamp")
    if not np.isfinite(timestamps).all() or np.any(timestamps < 0):
        raise ValueError("Timestamps must be finite and nonnegative")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Observation timestamps must be strictly increasing")
    assert timestamps.ndim == 1 and np.isfinite(timestamps).all()


def _draw_scene(
    config: Config,
    color_index: int,
    position_index: int,
    shape_index: int,
) -> np.ndarray:
    """Render a visible colored object, without embedding labels in pixels.

    Args:
        config: Synthetic image dimensions and drawing configuration.
        color_index: Palette entry used to render the object.
        position_index: Left/right location.
        shape_index: Square/circle geometry.

    Returns:
        RGB image with a dark background.
    """
    height, width = config.demo_frame_height, config.demo_frame_width
    yy, xx = np.mgrid[:height, :width]
    center_x = DEMO_POSITION_FRACTIONS[position_index] * (width - 1)
    center_y = (height - 1) / 2
    radius = max(1.0, min(height, width) * DEMO_OBJECT_RADIUS_FRACTION)
    if SHAPE_NAMES[shape_index] == "square":
        mask = (np.abs(xx - center_x) <= radius) & (
            np.abs(yy - center_y) <= radius
        )
    else:
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    frame = np.full((height, width, RGB_CHANNELS), DEMO_BACKGROUND, float)
    frame[mask] = COLOR_RGB[color_index]
    assert mask.any() and frame.shape == (height, width, RGB_CHANNELS)
    return frame


def make_demo_dataset(config: Config) -> list[Video]:
    """Build disjoint development/evaluation videos with pixel-derived labels.

    Args:
        config: Seed, synthetic scene sizes and dataset counts.

    Returns:
        Explicitly synthetic videos; evidence intervals use seconds.
    """
    config.validate()
    if min(config.demo_frame_height, config.demo_frame_width) < 8:
        raise ValueError(
            "Synthetic scene dimensions must be at least 8 pixels"
        )
    rng = np.random.default_rng(config.seed)
    videos: list[Video] = []
    for split, video_count in (
        ("dev", config.demo_dev_videos),
        ("eval", config.demo_eval_videos),
    ):
        for video_index in range(video_count):
            video_id = f"synthetic_{split}_{video_index:03d}"
            frame_indices = np.array_split(
                np.arange(config.demo_frames_per_video),
                config.demo_event_count,
            )
            color_order = rng.permutation(len(COLOR_NAMES))
            event_states = [
                (
                    int(color_order[index % len(COLOR_NAMES)]),
                    int(rng.integers(len(POSITION_NAMES))),
                    int(rng.integers(len(SHAPE_NAMES))),
                )
                for index in range(config.demo_event_count)
            ]
            frames = np.empty(
                (
                    config.demo_frames_per_video,
                    config.demo_frame_height,
                    config.demo_frame_width,
                    RGB_CHANNELS,
                ),
                dtype=np.uint8,
            )
            for indices, state in zip(
                frame_indices, event_states, strict=True
            ):
                scene = _draw_scene(config, *state)
                noise = rng.normal(
                    scale=config.demo_noise_std * PIXEL_MAX,
                    size=(len(indices), *scene.shape),
                )
                frames[indices] = np.clip(scene + noise, 0, PIXEL_MAX).astype(
                    np.uint8,
                )
            timestamps = (
                np.arange(len(frames), dtype=float) / config.candidate_fps
            )
            duration_seconds = len(frames) / config.candidate_fps
            questions: list[Question] = []
            for question_index in range(config.demo_questions_per_video):
                event_index = question_index % config.demo_event_count
                color_index, position_index, shape_index = event_states[
                    event_index
                ]
                indices = frame_indices[event_index]
                start_seconds = float(timestamps[indices[0]])
                stop_seconds = float((indices[-1] + 1) / config.candidate_fps)
                target_seconds = (start_seconds + stop_seconds) / 2
                prefix = (
                    f"At {target_seconds:.6f} seconds, the "
                    f"{COLOR_NAMES[color_index]} object is visible. "
                )
                if question_index % len(POSITION_NAMES) == 0:
                    question_text = prefix + "Where is it located?"
                    options = POSITION_NAMES
                    answer_index = position_index
                else:
                    question_text = prefix + "What shape is it?"
                    options = SHAPE_NAMES
                    answer_index = shape_index
                questions.append(
                    Question(
                        question_id=f"{video_id}_q{question_index:03d}",
                        text=question_text,
                        options=options,
                        answer_index=answer_index,
                        evidence_intervals=(
                            (
                                start_seconds,
                                float(timestamps[indices[-1]]),
                            ),
                        ),
                    )
                )
            validate_observations(frames, timestamps)
            videos.append(
                Video(
                    video_id=video_id,
                    split=split,
                    frames=frames,
                    timestamps=timestamps,
                    duration_seconds=duration_seconds,
                    questions=tuple(questions),
                    source_id=f"synthetic://{video_id}",
                )
            )
    assert len({video.video_id for video in videos}) == len(videos)
    assert len(videos) == config.demo_dev_videos + config.demo_eval_videos
    return videos


def _decode_scene(frame: np.ndarray) -> tuple[str, str, str] | None:
    """Read color, position and shape from visible foreground pixels only.

    Args:
        frame: One validated RGB image.

    Returns:
        Visible object properties, or None if no foreground is observed.
    """
    mask = frame.max(axis=-1) > DEMO_FOREGROUND_THRESHOLD
    if not mask.any():
        return None
    yy, xx = np.nonzero(mask)
    color = frame[mask].astype(float).mean(axis=0)
    color_index = int(np.argmin(np.square(COLOR_RGB - color).sum(axis=-1)))
    position_index = int(float(xx.mean()) >= (frame.shape[1] - 1) / 2)
    box_area = (int(xx.max()) - int(xx.min()) + 1) * (
        int(yy.max()) - int(yy.min()) + 1
    )
    fill_fraction = len(xx) / box_area
    shape_index = int(fill_fraction < DEMO_CIRCLE_FILL_THRESHOLD)
    return (
        COLOR_NAMES[color_index],
        POSITION_NAMES[position_index],
        SHAPE_NAMES[shape_index],
    )


class DemoBackend:
    """Frozen synthetic rules with no ground-truth model inputs."""

    def __init__(self, config: Config) -> None:
        """Retain the run configuration for provenance.

        Args:
            config: Experiment configuration.
        """
        self.config = config

    def metadata(self) -> dict[str, Any]:
        """Describe the fixture honestly in every exported run.

        Returns:
            Serializable synthetic backend identity and fixed rule constants.
        """
        return {
            "backend": "demo",
            "synthetic": True,
            "model": DEMO_SCHEMA_VERSION,
            "frozen": True,
            "encoder": "pixel-properties-and-lexical-overlap",
            "captioner": "observed-pixel-template",
            "qa_model": "selected-pixels-and-timestamp-rule",
            "qa_scoring": "property indicator; first-option tie break",
            "foreground_threshold": DEMO_FOREGROUND_THRESHOLD,
            "circle_fill_threshold": DEMO_CIRCLE_FILL_THRESHOLD,
            "scientific_performance_claim_allowed": False,
        }

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """Map pixel-visible properties into a fixed shared vocabulary.

        Args:
            frames: Original RGB frames, shape (N, H, W, 3).

        Returns:
            Finite property matrix with shape (N, vocabulary size).
        """
        validate_frames(frames)
        features = np.zeros((len(frames), len(DEMO_VOCABULARY)), dtype=float)
        for index, frame in enumerate(frames):
            properties = _decode_scene(frame)
            if properties is not None:
                for word in properties:
                    features[index, DEMO_VOCABULARY.index(word)] = 1.0
        assert features.shape == (len(frames), len(DEMO_VOCABULARY))
        assert np.isfinite(features).all()
        return features

    def encode_visual_question(self, question: str) -> np.ndarray:
        """Encode the query in the exact same space as frame properties.

        Args:
            question: Question text without evaluation labels.

        Returns:
            One finite vocabulary vector.
        """
        return self.encode_texts([question])[0]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode caption and query words using one fixed lexical mapping.

        Args:
            texts: Nonempty caption or question strings.

        Returns:
            Finite matrix with shape (N, vocabulary size).
        """
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Text encoder requires nonempty input strings")
        features = np.zeros((len(texts), len(DEMO_VOCABULARY)), dtype=float)
        for index, text in enumerate(texts):
            words = set(re.findall(r"[a-z]+", text.lower()))
            for column, word in enumerate(DEMO_VOCABULARY):
                features[index, column] = float(word in words)
        assert features.shape == (len(texts), len(DEMO_VOCABULARY))
        assert np.isfinite(features).all()
        return features

    def caption(
        self, frames: np.ndarray, timestamps: np.ndarray
    ) -> CaptionOutput:
        """Summarize only properties actually seen in supplied observations.

        Args:
            frames: Fixed chronological caption observations.
            timestamps: Their original timestamps in seconds.

        Returns:
            Short template caption; tokenizer counts are unavailable.
        """
        validate_observations(frames, timestamps)
        seen = list(
            dict.fromkeys(
                properties
                for frame in frames
                if (properties := _decode_scene(frame)) is not None
            )
        )
        descriptions = [
            f"a {color} {shape} on the {position}"
            for color, position, shape in seen
        ]
        text = (
            "Observed " + "; ".join(descriptions) + "."
            if descriptions
            else "No clear foreground object is observed."
        )
        return CaptionOutput(text=text)

    def answer(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        question: str,
        options: tuple[str, ...],
    ) -> QaOutput:
        """Answer from the closest selected observation of the queried object.

        Args:
            frames: Selected original frames, with no event captions.
            timestamps: Selected chronological timestamps.
            question: Query containing a color and an explicit time.
            options: Candidate answer strings; no correct-option index.

        Returns:
            Fixed indicator scores and a deterministic argmax prediction.
        """
        validate_observations(frames, timestamps)
        if len(options) < 2 or any(not option.strip() for option in options):
            raise ValueError("QA requires at least two nonempty options")
        words = set(re.findall(r"[a-z]+", question.lower()))
        colors = set(COLOR_NAMES).intersection(words)
        time_match = re.search(r"At ([0-9.]+) seconds", question)
        target_time = float(time_match.group(1)) if time_match else 0.0
        observations = [
            (abs(float(timestamp) - target_time), properties)
            for frame, timestamp in zip(frames, timestamps, strict=True)
            if (properties := _decode_scene(frame)) is not None
            and properties[0] in colors
        ]
        scores = np.zeros(len(options), dtype=float)
        if observations:
            _, properties = min(observations, key=lambda item: item[0])
            for index, option in enumerate(options):
                scores[index] = float(option.lower() in properties)
        assert scores.shape == (len(options),) and np.isfinite(scores).all()
        return QaOutput(
            int(np.argmax(scores)), tuple(float(x) for x in scores)
        )


def build_backend(config: Config) -> Backend:
    """Construct the requested implementation without synthetic fallback.

    Args:
        config: Backend selector: demo, transformers, or module:factory.

    Returns:
        A validated backend exposing all required model interfaces.

    Raises:
        ValueError: A real run requests demo or a backend selector is invalid.
        TypeError: A plugin factory does not satisfy the backend interface.
    """
    if config.mode == "real" and config.backend == "demo":
        raise ValueError("A real run cannot use the synthetic backend")
    if config.backend == "demo":
        return DemoBackend(config)
    if config.backend == "transformers":
        from .hf_backend import TransformersBackend

        return TransformersBackend(config)
    if ":" not in config.backend:
        raise ValueError(
            "Backend must be demo, transformers or module:factory"
        )
    module_name, factory_name = config.backend.split(":", maxsplit=1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    backend = factory(config)
    for name in (
        "metadata",
        "encode_frames",
        "encode_visual_question",
        "encode_texts",
        "caption",
        "answer",
    ):
        if not callable(getattr(backend, name, None)):
            raise TypeError(f"Backend plugin lacks callable {name}")
    if (
        config.mode == "real"
        and backend.metadata().get("synthetic") is not False
    ):
        raise ValueError(
            "Real backend metadata must explicitly set synthetic=False"
        )
    return cast(Backend, backend)
