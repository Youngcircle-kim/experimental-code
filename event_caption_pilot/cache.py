"""Content-addressed, question-independent caption provenance and cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import Config
from .data import array_digest
from .reproducibility import fix_seed
from .types import Backend, Event, Video


def write_json(path: Path, payload: Any) -> None:
    """Atomically write strict JSON, never nonstandard NaN or Infinity values.

    Args:
        path: Destination file.
        payload: JSON-serializable artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    )
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = stream.name
            stream.write(encoded)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def event_caption(
    video: Video,
    event: Event,
    indices: np.ndarray,
    backend: Backend,
    config: Config,
) -> dict[str, Any]:
    """Caption only fixed observations, with a cache independent of questions.

    Args:
        video: Original candidate pool; annotations are never sent to backend.
        event: Fixed interval; captions cannot alter it.
        indices: Unique chronological caption candidates inside the event.
        backend: Frozen caption model.
        config: Prompt, generation and seed settings.

    Returns:
        Caption, original provenance and measured cache/generation cost.
    """
    started = perf_counter()
    if (
        indices.ndim != 1
        or len(indices) == 0
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("Caption indices must be a nonempty integer vector")
    if (
        not np.all(np.diff(indices) > 0)
        or indices[0] < event.start
        or indices[-1] >= event.stop
    ):
        raise ValueError(
            "Caption observations must be unique, chronological "
            "and inside the event"
        )
    frames, timestamps = video.frames[indices], video.timestamps[indices]
    if len(indices) > config.caption_max_frames:
        raise ValueError("Caption observations exceed caption_max_frames")
    assert frames.shape[0] == len(indices) <= config.caption_max_frames
    assert np.isfinite(timestamps).all()
    observation_digest = array_digest(frames, timestamps)
    seed_identity = (
        f"{config.seed}:{video.source_id}:{event.start}:"
        f"{event.stop}:{observation_digest}"
    )
    generation_seed = int.from_bytes(
        hashlib.sha256(seed_identity.encode()).digest()[:4],
        "big",
    )
    provenance = {
        "cache_schema_version": 2,
        "source_id": video.source_id,
        "event_start_index": event.start,
        "event_stop_index": event.stop,
        "event_start_seconds": float(video.timestamps[event.start])
        if event.start
        else 0.0,
        "event_end_seconds": float(video.timestamps[event.stop])
        if event.stop < len(video.frames)
        else video.duration_seconds,
        "input_indices": indices.tolist(),
        "input_timestamps": timestamps.tolist(),
        "observations_sha256": observation_digest,
        "backend": backend.metadata(),
        "prompt": config.caption_prompt,
        "prompt_version": config.caption_prompt_version,
        "max_new_tokens": config.caption_max_new_tokens,
        "seed": config.seed,
        "generation_seed": generation_seed,
        "deterministic": config.deterministic,
        "seed_torch": config.seed_torch or config.backend == "transformers",
        "seed_tensorflow": config.seed_tensorflow,
        "cublas_workspace_config": config.cublas_workspace_config,
    }
    # Canonicalize plugin metadata such as tuples before comparing JSON reads.
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    key = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    path = Path(config.cache_dir) / f"{key}.json"
    cache_hit = path.exists()
    if cache_hit:
        with path.open(encoding="utf-8") as stream:
            stored = json.load(stream)
        if stored.get("provenance") != provenance or not isinstance(
            stored.get("text"), str
        ):
            raise ValueError(f"Invalid caption cache entry: {path}")
        if not stored["text"].strip():
            raise ValueError(f"Empty cached caption: {path}")
    else:
        fix_seed(
            generation_seed,
            config.deterministic,
            config.seed_torch or config.backend == "transformers",
            config.seed_tensorflow,
            config.cublas_workspace_config,
        )
        generation_started = perf_counter()
        # The backend gets only sampled RGB frames and their timestamps.
        generated = backend.caption(frames.copy(), timestamps.copy())
        generation_seconds = perf_counter() - generation_started
        if not isinstance(generated.text, str) or not generated.text.strip():
            raise ValueError("The caption backend returned an empty caption")
        for token_count in (generated.input_tokens, generated.output_tokens):
            if token_count is not None and (
                type(token_count) is not int or token_count < 0
            ):
                raise ValueError(
                    "Tokenizer counts must be nonnegative integers or None"
                )
        stored = {
            "provenance": provenance,
            "text": generated.text.strip(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation_seconds": generation_seconds,
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
        }
        write_json(path, stored)
    return {
        **stored,
        "cache_key": key,
        "cache_hit": cache_hit,
        "input_frame_count": len(indices),
        "current_input_frame_count": 0 if cache_hit else len(indices),
        "current_input_tokens": 0 if cache_hit else stored["input_tokens"],
        "current_output_tokens": 0 if cache_hit else stored["output_tokens"],
        "current_generation_seconds": 0.0
        if cache_hit
        else stored["generation_seconds"],
        "current_caption_wall_seconds": perf_counter() - started,
    }
