"""Predefined hypotheses and deterministic frame-level controls."""

import numpy as np

from .algorithms import sample_indices

CONDITIONS = (
    "uniform", "frame_top", "temporal_bin", "uniform_event_v",
    "uniform_event_vt", "control_v", "treatment_t", "treatment_vt",
)
HYPOTHESES = {
    "H1_question_conditioning": (
        ("frame_top", "uniform"), ("temporal_bin", "uniform"),
    ),
    "H2_event_grouping": (
        ("control_v", "uniform_event_v"),
        ("control_v", "frame_top"), ("control_v", "temporal_bin"),
    ),
    "H3_caption_relevance": (
        ("treatment_vt", "control_v"),
        ("uniform_event_vt", "uniform_event_v"),
    ),
}
COMPARISONS = tuple(
    pair for pairs in HYPOTHESES.values() for pair in pairs
) + (("treatment_t", "control_v"), ("treatment_vt", "treatment_t"))


def select_frame_control(
    condition: str, scores: np.ndarray, budget: int
) -> np.ndarray:
    """Select exactly B unique indices; ties favor earlier candidates.

    BIN uses B nonempty equal-count chronological bins, one maximum per bin.
    On irregular input timestamps these are not equal-duration bins.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Frame scores must be a finite vector")
    if not 1 <= budget <= len(scores):
        raise ValueError("Budget must be within the candidate count")
    if condition == "uniform":
        return sample_indices(0, len(scores), budget)
    if condition == "frame_top":
        return np.sort(np.argsort(-scores, kind="stable")[:budget])
    if condition == "temporal_bin":
        bins = np.array_split(np.arange(len(scores)), budget)
        return np.array([part[np.argmax(scores[part])] for part in bins])
    raise ValueError(f"Unknown frame control: {condition}")
