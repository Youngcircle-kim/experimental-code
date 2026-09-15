"""Data boundaries that prevent labels or captions entering model calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Question:
    """Evaluation labels live here and are never passed to the QA backend."""

    question_id: str
    text: str
    options: tuple[str, ...]
    answer_index: int
    evidence_intervals: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Video:
    """One original-frame candidate pool; RGB shape is (N, H, W, 3)."""

    video_id: str
    split: str
    frames: np.ndarray
    timestamps: np.ndarray
    duration_seconds: float
    questions: tuple[Question, ...]
    source_id: str
    decode_seconds: float = 0.0


@dataclass(frozen=True)
class Event:
    """Half-open candidate-index interval [start, stop)."""

    start: int
    stop: int


@dataclass(frozen=True)
class CaptionOutput:
    """Backend-generated text and actual tokenizer counts, if available."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class QaOutput:
    """Prediction and frozen option scores, without access to answer labels."""

    predicted_index: int
    option_scores: tuple[float, ...]


class Backend(Protocol):
    """Models must be frozen; methods must synchronize accelerator timings."""

    def metadata(self) -> dict[str, Any]:
        """Describe the frozen model implementation.

        Returns:
            JSON-compatible model IDs, revisions and decoding settings.
        """
        ...

    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """Encode original RGB observations.

        Args:
            frames: RGB candidate array with shape (N, H, W, 3).

        Returns:
            Finite, nonzero visual feature rows with shape (N, D).
        """
        ...

    def encode_visual_question(self, question: str) -> np.ndarray:
        """Encode a question in the visual feature space.

        Args:
            question: Query text without options or evaluation labels.

        Returns:
            A finite, nonzero vector with shape (D,).
        """
        ...

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode captions and questions in one shared text space.

        Args:
            texts: Nonempty caption or question strings.

        Returns:
            Finite, nonzero embeddings with shape (N, T).
        """
        ...

    def caption(
        self, frames: np.ndarray, timestamps: np.ndarray
    ) -> CaptionOutput:
        """Caption fixed observations independently of queries.

        Args:
            frames: Fixed-budget, chronological original observations.
            timestamps: Their source-video times in seconds.

        Returns:
            Caption text and tokenizer counts when available.
        """
        ...

    def answer(
        self,
        frames: np.ndarray,
        timestamps: np.ndarray,
        question: str,
        options: tuple[str, ...],
    ) -> QaOutput:
        """Score options using only selected original frames.

        Args:
            frames: Final original-frame observations, without captions.
            timestamps: Their source-video times in seconds.
            question: Query text, without evaluation annotations.
            options: Candidate answer strings in original order.

        Returns:
            Predicted option index and all finite option scores.
        """
        ...
