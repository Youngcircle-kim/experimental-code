"""Load real candidates and keep evaluation annotations outside models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import Config
from .types import Question, Video


def array_digest(*arrays: np.ndarray) -> str:
    """Hash array content, dtype and shape without object serialization.

    Args:
        arrays: Numeric arrays contributing to a content identity.

    Returns:
        A SHA-256 hexadecimal content identifier.
    """
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.shape, contiguous.dtype.str)).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def validate_video(video: Video, config: Config) -> None:
    """Check input shapes, chronology, labels and evidence before inference.

    Args:
        video: Candidate frames and separate evaluation annotations.
        config: Active experimental constraints.

    Raises:
        ValueError: Input data is malformed or the exact budget is infeasible.
    """
    frames, timestamps = video.frames, video.timestamps
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            f"{video.video_id}: frames must be uint8 RGB (N, H, W, 3)"
        )
    if min(frames.shape) < 1 or timestamps.shape != (len(frames),):
        raise ValueError(
            f"{video.video_id}: empty frames or timestamp shape mismatch"
        )
    if not np.isfinite(timestamps).all() or not np.all(
        np.diff(timestamps) > 0
    ):
        raise ValueError(
            f"{video.video_id}: timestamps must be finite and increasing"
        )
    if not np.isfinite(video.duration_seconds) or not (
        0 <= timestamps[0] <= timestamps[-1] < video.duration_seconds
    ):
        raise ValueError(
            f"{video.video_id}: duration must exceed the final timestamp"
        )
    if (
        video.split not in {"dev", "eval"}
        or not video.video_id
        or not video.source_id
    ):
        raise ValueError(
            "Each video needs video_id, original source_id and dev/eval split"
        )
    if len(frames) > config.max_candidate_frames:
        raise ValueError(
            "Candidate pool exceeds max_candidate_frames; lower candidate_fps"
        )
    if config.pilot == "both" and len(frames) < config.frame_budget:
        raise ValueError(
            f"{video.video_id}: not enough unique candidates for frame_budget"
        )
    if config.pilot == "both" and not video.questions:
        raise ValueError(f"{video.video_id}: pilot B requires questions")
    question_ids: set[str] = set()
    for question in video.questions:
        if not question.question_id or question.question_id in question_ids:
            raise ValueError(
                "Question IDs must be nonempty and unique within a video"
            )
        question_ids.add(question.question_id)
        if not question.text.strip() or len(question.options) < 2:
            raise ValueError(
                "Questions require text and at least two answer options"
            )
        if any(not option.strip() for option in question.options):
            raise ValueError("Empty options are not allowed")
        if type(
            question.answer_index
        ) is not int or not 0 <= question.answer_index < len(question.options):
            raise ValueError("answer_index must identify a valid option")
        for start, stop in question.evidence_intervals:
            if (
                not np.isfinite([start, stop]).all()
                or not 0 <= start <= stop <= video.duration_seconds
            ):
                raise ValueError(
                    "Evidence intervals must be finite and inside the video"
                )
    assert frames.shape[0] == timestamps.size
    assert np.isfinite(frames).all() and np.isfinite(timestamps).all()


def validate_dataset(videos: list[Video], config: Config) -> None:
    """Require video-disjoint development and evaluation inputs.

    Args:
        videos: Fully loaded dataset.
        config: Experiment settings.

    Raises:
        ValueError: Dataset is empty, duplicated or split incorrectly.
    """
    if not videos:
        raise ValueError("The dataset is empty")
    seen_ids: set[str] = set()
    seen_sources: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    for video in videos:
        validate_video(video, config)
        if video.video_id in seen_ids:
            raise ValueError(f"Duplicate video_id: {video.video_id}")
        seen_ids.add(video.video_id)
        content_digest = array_digest(video.frames, video.timestamps)
        if content_digest in seen_content:
            raise ValueError(
                "Duplicate video content; copies cannot be independent "
                "video groups"
            )
        seen_content[content_digest] = video.split
        previous = seen_sources.setdefault(video.source_id, video.split)
        if previous != video.split:
            raise ValueError(
                "The same original video/content crosses dev and eval splits"
            )
    if config.pilot == "both" and {video.split for video in videos} != {
        "dev",
        "eval",
    }:
        raise ValueError("Pilot B requires separate dev and eval videos")
    # Bootstrap groups are video IDs. Separate clips from one source must be
    # consolidated so that the resampling unit remains the original video.
    if len(seen_sources) != len(videos):
        raise ValueError(
            "Use one Video per source_id; consolidate clips "
            "from one original video"
        )


def decode_video(
    path: Path, config: Config
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode constant-FPS video into chronologically sampled RGB frames.

    Args:
        path: Local video path.
        config: Candidate FPS, resolution and memory guard settings.

    Returns:
        RGB array, frame-index/FPS timestamps, and duration in seconds.

    Raises:
        ValueError: Decoder metadata is invalid or the pool is too large.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Video decoding requires the optional 'video' dependencies"
        ) from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(
                "Invalid video FPS; supply an NPZ with verified timestamps"
            )
        frame_index, next_observation = 0, 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            if timestamp >= next_observation:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(
                    rgb, (config.frame_width, config.frame_height)
                )
                frames.append(rgb)
                timestamps.append(timestamp)
                next_observation += 1.0 / min(config.candidate_fps, fps)
                if len(frames) > config.max_candidate_frames:
                    raise ValueError(
                        "Video exceeds candidate limit; reduce candidate_fps"
                    )
            frame_index += 1
        if not frames:
            raise ValueError("No decodable frames")
        result = np.stack(frames)
        assert result.shape == (
            len(timestamps),
            config.frame_height,
            config.frame_width,
            3,
        )
        return result, np.asarray(timestamps, dtype=float), frame_index / fps
    finally:
        capture.release()


def load_manifest(config: Config) -> list[Video]:
    """Load JSON metadata with NPZ candidate pools or local video files.

    Args:
        config: Settings containing manifest_path and decoding options.

    Returns:
        Validated videos; NPZ paths are resolved relative to the manifest.
    """
    if config.manifest_path is None:
        raise ValueError("manifest_path is required")
    manifest_path = Path(config.manifest_path).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest: dict[str, Any] = json.load(stream)
    videos: list[Video] = []
    for item in manifest["videos"]:
        started = perf_counter()
        source_path = (manifest_path.parent / item["path"]).resolve()
        if source_path.suffix.lower() == ".npz":
            with np.load(source_path, allow_pickle=False) as archive:
                frames = archive["frames"].copy()
                timestamps = archive["timestamps"].astype(float)
                duration = float(archive["duration_seconds"].item())
        else:
            if item.get("timestamp_mode") != "constant_fps":
                raise ValueError(
                    "Video files require timestamp_mode='constant_fps'; "
                    "use NPZ for VFR"
                )
            frames, timestamps, duration = decode_video(source_path, config)
        decode_seconds = perf_counter() - started
        questions = tuple(
            Question(
                question_id=question["question_id"],
                text=question["text"],
                options=tuple(question["options"]),
                answer_index=question["answer_index"],
                evidence_intervals=tuple(
                    tuple(interval)
                    for interval in question.get("evidence_intervals", [])
                ),
            )
            for question in item.get("questions", [])
        )
        videos.append(
            Video(
                video_id=item["video_id"],
                split=item["split"],
                frames=frames,
                timestamps=timestamps,
                duration_seconds=duration,
                questions=questions,
                source_id=item["source_id"],
                decode_seconds=decode_seconds,
            )
        )
    validate_dataset(videos, config)
    return videos
