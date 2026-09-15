"""Deterministic event detection, frame allocation and clustered evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config
from .types import Event


def _finite_array(values: np.ndarray, name: str, ndim: int) -> np.ndarray:
    """Validate a nonempty numeric array before research computations.

    Args:
        values: Numeric observations to validate.
        name: Field name for an actionable error message.
        ndim: Required number of dimensions.

    Returns:
        A finite float64 array with the required dimensions.
    """
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if result.ndim != ndim or result.size == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    assert result.ndim == ndim and result.size > 0
    assert np.isfinite(result).all()
    return result


def _positive_epsilon(epsilon: float) -> None:
    """Validate the numerical stability tolerance.

    Args:
        epsilon: Smallest usable positive normalization denominator.

    Returns:
        None.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")


def normalize_rows(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Normalize nonzero feature rows for cosine comparisons.

    Near-zero rows are rejected because their cosine direction is undefined.
    Rescaling before computing a norm avoids overflow for large finite inputs.

    Args:
        values: Finite feature matrix of shape (observations, dimensions).
        epsilon: Minimum admissible row norm.

    Returns:
        Float64 feature matrix with unit-length rows and unchanged shape.
    """
    _positive_epsilon(epsilon)
    matrix = _finite_array(values, "values", ndim=2)
    row_scale = np.max(np.abs(matrix), axis=1, keepdims=True)
    if np.any(row_scale == 0):
        raise ValueError("Cannot normalize a zero feature row")
    scaled_matrix = matrix / row_scale
    scaled_norms = np.linalg.norm(scaled_matrix, axis=1, keepdims=True)
    if np.any(row_scale <= epsilon / scaled_norms):
        raise ValueError("Cannot normalize a feature row with norm <= epsilon")
    normalized = scaled_matrix / scaled_norms
    assert normalized.shape == matrix.shape
    assert np.isfinite(normalized).all()
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    return normalized


def segment(
    features: np.ndarray,
    config: Config,
    detector: str,
    segment_count: int | None = None,
) -> list[Event]:
    """Partition observed frames using uniform or cosine-change boundaries.

    D1 compares adjacent normalized observations. D2 compares the normalized
    mean features before and after each boundary, using prefix sums and
    truncated windows near the video ends. Candidate boundaries exceeding the
    configured threshold are accepted strongest-first, with earlier positions
    breaking ties. A video shorter than the minimum segment length is retained
    as one segment. Explicit D0 counts must be feasible and match exactly.
    Distances within normalization_epsilon of zero are treated as roundoff.

    Args:
        features: Observed frame features with shape (frames, dimensions).
        config: Validated detector and numerical settings.
        detector: One of D0, D1 or D2.
        segment_count: Optional exact segment count, supported only for D0.

    Returns:
        Ordered, nonempty half-open events covering every input observation.
    """
    config.validate()
    normalized = normalize_rows(features, config.normalization_epsilon)
    frame_count = normalized.shape[0]
    if detector not in {"D0", "D1", "D2"}:
        raise ValueError("detector must be D0, D1 or D2")
    if segment_count is not None and detector != "D0":
        raise ValueError("segment_count is supported only by D0")
    feasible_count = max(1, frame_count // config.min_segment_frames)
    feasible_count = min(feasible_count, config.max_segments)
    if detector == "D0":
        if segment_count is None:
            count = min(config.uniform_segments, feasible_count)
        else:
            if (
                isinstance(segment_count, bool)
                or not isinstance(segment_count, (int, np.integer))
                or not 1 <= segment_count <= feasible_count
            ):
                raise ValueError("Requested D0 segment_count is not feasible")
            count = int(segment_count)
        boundaries = [
            index * frame_count // count for index in range(count + 1)
        ]
    else:
        candidate_positions = np.arange(1, frame_count, dtype=np.int64)
        if frame_count == 1 or feasible_count == 1:
            boundaries = [0, frame_count]
        else:
            if detector == "D1":
                cosine_similarity = np.einsum(
                    "ij,ij->i", normalized[:-1], normalized[1:]
                )
            else:
                prefix_sums = np.vstack(
                    (
                        np.zeros((1, normalized.shape[1])),
                        normalized.cumsum(axis=0),
                    )
                )
                left_positions = np.maximum(
                    0, candidate_positions - config.window_size
                )
                right_positions = np.minimum(
                    frame_count, candidate_positions + config.window_size
                )
                preceding_means = (
                    prefix_sums[candidate_positions]
                    - prefix_sums[left_positions]
                ) / (candidate_positions - left_positions)[:, None]
                following_means = (
                    prefix_sums[right_positions]
                    - prefix_sums[candidate_positions]
                ) / (right_positions - candidate_positions)[:, None]
                preceding_features = normalize_rows(
                    preceding_means, config.normalization_epsilon
                )
                following_features = normalize_rows(
                    following_means, config.normalization_epsilon
                )
                cosine_similarity = np.einsum(
                    "ij,ij->i", preceding_features, following_features
                )
            boundary_scores = 1.0 - np.clip(cosine_similarity, -1.0, 1.0)
            boundary_scores[
                boundary_scores <= config.normalization_epsilon
            ] = 0.0
            assert boundary_scores.shape == (frame_count - 1,)
            assert np.isfinite(boundary_scores).all()
            ranking = np.lexsort((candidate_positions, -boundary_scores))
            boundaries = [0, frame_count]
            for candidate_index in ranking:
                if (
                    boundary_scores[candidate_index]
                    <= config.boundary_threshold
                ):
                    break
                position = int(candidate_positions[candidate_index])
                if all(
                    abs(position - boundary) >= config.min_segment_frames
                    for boundary in boundaries
                ):
                    boundaries.append(position)
                    if len(boundaries) - 1 == feasible_count:
                        break
            boundaries.sort()
    events = [
        Event(start=int(start), stop=int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]
    assert events and events[0].start == 0 and events[-1].stop == frame_count
    assert all(event.stop > event.start for event in events)
    assert all(
        left.stop == right.start for left, right in zip(events, events[1:])
    )
    assert len(events) <= config.max_segments
    assert frame_count < config.min_segment_frames or all(
        event.stop - event.start >= config.min_segment_frames
        for event in events
    )
    return events


def sample_indices(start: int, stop: int, count: int) -> np.ndarray:
    """Select unique, evenly spaced integer indices deterministically.

    A single observation uses the interval midpoint. Multiple observations
    include both endpoints of the available half-open interval.

    Args:
        start: Inclusive nonnegative candidate index.
        stop: Exclusive candidate index.
        count: Number of unique observations, no greater than interval length.

    Returns:
        Sorted int64 indices with exactly count elements.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in (start, stop, count)
    ):
        raise ValueError("start, stop and count must be integers")
    if start < 0 or stop < start or count < 0 or count > stop - start:
        raise ValueError(
            "Require 0 <= start <= stop and 0 <= count <= stop-start"
        )
    if count == 0:
        result = np.empty(0, dtype=np.int64)
    elif count == 1:
        result = np.asarray([start + (stop - start - 1) // 2], dtype=np.int64)
    else:
        result = np.asarray(
            [start + index * (stop - start - 1) // (count - 1)
             for index in range(count)],
            dtype=np.int64,
        )
    assert result.shape == (count,)
    assert np.unique(result).size == count
    assert np.all((result >= start) & (result < stop))
    return result


def allocate_frames(
    scores: np.ndarray,
    capacities: np.ndarray,
    budget: int,
    temperature: float,
) -> np.ndarray:
    """Allocate an exact frame budget with capacity-bounded softmax weights.

    Full events are saturated and the remaining budget is redistributed.
    Fractional quotas are rounded by largest remainder, with the earlier event
    breaking ties. Events can receive zero frames when budget is scarce.

    Args:
        scores: Finite one-dimensional event relevance scores.
        capacities: Nonnegative integer candidate counts for each event.
        budget: Exact total frame count, bounded by total candidate capacity.
        temperature: Positive softmax temperature controlling concentration.

    Returns:
        Integer allocation with the input shape, exact sum and no overflow.
    """
    event_scores = _finite_array(scores, "scores", ndim=1)
    capacity_values = np.asarray(capacities)
    if capacity_values.shape != event_scores.shape:
        raise ValueError("capacities and scores must have the same shape")
    if capacity_values.dtype.kind not in {"i", "u"}:
        raise ValueError("capacities must contain integers")
    if np.any(capacity_values < 0) or np.any(
        capacity_values > np.iinfo(np.int64).max
    ):
        raise ValueError("capacities must be nonnegative int64 values")
    capacity_values = capacity_values.astype(np.int64, copy=False)
    total_capacity = sum(int(capacity) for capacity in capacity_values)
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, np.integer))
        or not 0 <= budget <= total_capacity
    ):
        raise ValueError("budget must be an integer within total capacity")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    allocation = np.zeros(event_scores.shape, dtype=np.int64)
    remaining_budget = int(budget)
    while remaining_budget:
        available = capacity_values - allocation
        active_indices = np.flatnonzero(available > 0)
        active_scores = event_scores[active_indices]
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(
                (active_scores - np.max(active_scores)) / temperature
            )
        quotas = remaining_budget * (weights / weights.sum())
        saturated = quotas >= available[active_indices]
        if np.any(saturated):
            saturated_indices = active_indices[saturated]
            allocation[saturated_indices] += available[saturated_indices]
            remaining_budget -= sum(
                int(value) for value in available[saturated_indices]
            )
            continue
        integer_quotas = np.floor(quotas).astype(np.int64)
        allocation[active_indices] += integer_quotas
        remaining_budget -= sum(int(value) for value in integer_quotas)
        fractional_quotas = quotas - integer_quotas
        remainder_order = np.lexsort((active_indices, -fractional_quotas))
        for position in remainder_order:
            if remaining_budget == 0:
                break
            event_index = int(active_indices[position])
            if allocation[event_index] < capacity_values[event_index]:
                allocation[event_index] += 1
                remaining_budget -= 1
    assert allocation.shape == event_scores.shape
    assert sum(int(value) for value in allocation) == budget
    assert np.all((allocation >= 0) & (allocation <= capacity_values))
    return allocation


def event_features(
    features: np.ndarray, events: list[Event], epsilon: float
) -> np.ndarray:
    """Pool normalized observations into normalized event feature vectors.

    Args:
        features: Nonzero, finite frame feature matrix (frames, dimensions).
        events: Nonempty, ordered, nonoverlapping half-open event intervals.
        epsilon: Minimum admissible normalization denominator.

    Returns:
        Unit-normalized mean features with shape (events, dimensions).
    """
    normalized = normalize_rows(features, epsilon)
    if not events:
        raise ValueError("events must not be empty")
    preceding_stop = 0
    for event in events:
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                for value in (event.start, event.stop)
            )
            or not 0 <= event.start < event.stop <= normalized.shape[0]
            or event.start < preceding_stop
        ):
            raise ValueError(
                "Events must be ordered, nonoverlapping valid intervals"
            )
        preceding_stop = event.stop
    means = np.stack(
        [normalized[event.start:event.stop].mean(axis=0) for event in events]
    )
    assert means.shape == (len(events), normalized.shape[1])
    assert np.isfinite(means).all()
    result = normalize_rows(means, epsilon)
    assert result.shape == means.shape and np.isfinite(result).all()
    return result


@dataclass(frozen=True)
class ScoreCalibrator:
    """Immutable development standardization for visual and text scores."""

    visual_mean: float
    visual_std: float
    text_mean: float
    text_std: float
    epsilon: float

    @classmethod
    def fit(
        cls: type[ScoreCalibrator],
        control_visual_scores: np.ndarray,
        treatment_text_scores: np.ndarray,
        epsilon: float,
    ) -> ScoreCalibrator:
        """Fit channel statistics from development scores only.

        The caller must enforce the development/evaluation split. Population
        standard deviations are floored at epsilon for constant channels.

        Args:
            control_visual_scores: One-dimensional development visual scores.
            treatment_text_scores: Paired development caption-text scores.
            epsilon: Positive lower bound on standard deviations.

        Returns:
            Frozen calibration reusable without evaluation refitting.
        """
        _positive_epsilon(epsilon)
        visual_scores = _finite_array(
            control_visual_scores, "visual scores", ndim=1
        )
        text_scores = _finite_array(
            treatment_text_scores, "text scores", ndim=1
        )
        if visual_scores.shape != text_scores.shape:
            raise ValueError(
                "Development visual and text scores must be paired"
            )
        statistics = np.asarray(
            [visual_scores.mean(), visual_scores.std(),
             text_scores.mean(), text_scores.std()]
        )
        if not np.isfinite(statistics).all():
            raise ValueError("Development calibration statistics overflowed")
        assert statistics.shape == (4,) and np.isfinite(statistics).all()
        return cls(
            visual_mean=float(statistics[0]),
            visual_std=max(float(statistics[1]), epsilon),
            text_mean=float(statistics[2]),
            text_std=max(float(statistics[3]), epsilon),
            epsilon=float(epsilon),
        )

    def transform(
        self,
        control_visual_scores: np.ndarray,
        treatment_text_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply fixed channel calibration without updating its parameters.

        Args:
            control_visual_scores: One-dimensional event visual scores.
            treatment_text_scores: Paired event caption-text scores.

        Returns:
            Standardized visual and text score arrays in the same event order.
        """
        visual_scores = _finite_array(
            control_visual_scores, "visual scores", ndim=1
        )
        text_scores = _finite_array(
            treatment_text_scores, "text scores", ndim=1
        )
        if visual_scores.shape != text_scores.shape:
            raise ValueError("Visual and text scores must have the same shape")
        _positive_epsilon(self.epsilon)
        if (
            not np.isfinite(
                [
                    self.visual_mean, self.visual_std,
                    self.text_mean, self.text_std,
                ]
            ).all()
            or self.visual_std < self.epsilon
            or self.text_std < self.epsilon
        ):
            raise ValueError(
                "Calibration must be finite with positive scales"
            )
        standardized_visual = (
            visual_scores - self.visual_mean
        ) / self.visual_std
        standardized_text = (text_scores - self.text_mean) / self.text_std
        if not (
            np.isfinite(standardized_visual).all()
            and np.isfinite(standardized_text).all()
        ):
            raise ValueError("Calibrated scores overflowed")
        assert (
            standardized_visual.shape
            == standardized_text.shape
            == visual_scores.shape
        )
        assert np.isfinite(standardized_visual).all()
        assert np.isfinite(standardized_text).all()
        return standardized_visual, standardized_text


def evidence_metrics(
    selected_timestamps: np.ndarray,
    intervals: tuple[tuple[float, float], ...],
) -> dict[str, float | bool | None]:
    """Measure whether selected frames intersect annotated evidence intervals.

    Evidence timestamps use closed intervals [start, stop]; overlapping
    annotations are evaluated individually. No annotations means unavailable
    evidence metrics, not a negative result.

    Args:
        selected_timestamps: Finite nonnegative selected frame timestamps.
        intervals: Ground-truth evidence intervals in seconds.

    Returns:
        Fraction of intervals hit and whether every annotated interval is hit.
    """
    timestamps = np.asarray(selected_timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all():
        raise ValueError(
            "selected_timestamps must be a finite one-dimensional array"
        )
    if np.any(timestamps < 0):
        raise ValueError("selected_timestamps must be nonnegative")
    if not intervals:
        return {"interval_hit": None, "all_interval_hit": None}
    interval_array = _finite_array(np.asarray(intervals), "intervals", ndim=2)
    if (
        interval_array.shape[1] != 2
        or np.any(interval_array < 0)
        or np.any(interval_array[:, 1] < interval_array[:, 0])
    ):
        raise ValueError(
            "Evidence intervals require finite 0 <= start <= stop"
        )
    hits = np.any(
        (timestamps[None, :] >= interval_array[:, :1])
        & (timestamps[None, :] <= interval_array[:, 1:]),
        axis=1,
    )
    assert hits.shape == (len(intervals),)
    interval_hit = float(hits.mean())
    assert 0 <= interval_hit <= 1
    return {"interval_hit": interval_hit, "all_interval_hit": bool(hits.all())}


def clustered_comparison(
    rows: list[dict[str, Any]], config: Config
) -> dict[str, Any]:
    """Compare paired QA arms using video-cluster percentile bootstrap CIs.

    Each replicate resamples videos with replacement and retains all questions
    in each sampled video, including repeated copies. Accuracy and paired
    differences remain question-weighted, including unequal video sizes.

    Args:
        rows: Evaluation rows with IDs and three binary correctness fields.
        config: Seed, bootstrap replicate count and confidence level.

    Returns:
        Accuracies, paired differences and confidence intervals. Intervals are
        unavailable for fewer than two distinct video clusters.
    """
    config.validate()
    if not rows:
        raise ValueError("At least one evaluation question is required")
    arm_names = ("control_v", "treatment_t", "treatment_vt")
    question_keys: set[tuple[str, str]] = set()
    clustered_rows: dict[str, list[list[int]]] = {}
    for row in rows:
        video_id = row.get("video_id")
        question_id = row.get("question_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("Every evaluation row needs a nonempty video_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(
                "Every evaluation row needs a nonempty question_id"
            )
        question_key = (video_id, question_id)
        if question_key in question_keys:
            raise ValueError(f"Duplicate evaluation question: {question_key}")
        question_keys.add(question_key)
        correctness = [
            row.get(f"{arm_name}_correct") for arm_name in arm_names
        ]
        if any(
            not isinstance(
                value, (bool, int, float, np.bool_, np.integer, np.floating)
            )
            or value not in (0, 1)
            for value in correctness
        ):
            raise ValueError(
                "All paired correctness values must be binary and present"
            )
        clustered_rows.setdefault(video_id, []).append(
            [int(value) for value in correctness]
        )
    video_ids = sorted(clustered_rows)
    cluster_sizes = np.asarray(
        [len(clustered_rows[video_id]) for video_id in video_ids],
        dtype=np.int64,
    )
    cluster_correct_counts = np.stack(
        [np.asarray(clustered_rows[video_id], dtype=np.int64).sum(axis=0)
         for video_id in video_ids]
    )
    assert cluster_correct_counts.shape == (len(video_ids), len(arm_names))
    assert np.isfinite(cluster_correct_counts).all()
    assert int(cluster_sizes.sum()) == len(rows)
    arm_accuracies = cluster_correct_counts.sum(axis=0) / cluster_sizes.sum()
    assert np.all((arm_accuracies >= 0) & (arm_accuracies <= 1))
    bootstrap_accuracies: np.ndarray | None = None
    if len(video_ids) >= 2:
        generator = np.random.default_rng(config.seed)
        bootstrap_cluster_multiplicities = generator.multinomial(
            len(video_ids),
            np.full(len(video_ids), 1.0 / len(video_ids)),
            size=config.bootstrap_samples,
        )
        bootstrap_question_counts = (
            bootstrap_cluster_multiplicities @ cluster_sizes
        )
        # These are exact counts. Keep all accumulation in integer arithmetic
        # and enter floating point only for the final accuracy division.
        bootstrap_correct_counts = (
            bootstrap_cluster_multiplicities @ cluster_correct_counts
        )
        if np.any(bootstrap_question_counts <= 0) or np.any(
            (bootstrap_correct_counts < 0)
            | (bootstrap_correct_counts > bootstrap_question_counts[:, None])
        ):
            raise ValueError("Bootstrap integer counts are out of bounds")
        bootstrap_accuracies = (
            bootstrap_correct_counts / bootstrap_question_counts[:, None]
        )
        assert bootstrap_accuracies.shape == (
            config.bootstrap_samples, len(arm_names)
        )
        assert np.isfinite(bootstrap_accuracies).all()
        assert np.all(bootstrap_question_counts > 0)
    paired_comparisons: dict[str, dict[str, Any]] = {}
    tail_probability = (1.0 - config.confidence_level) / 2.0
    for treatment_index, control_index in ((1, 0), (2, 0), (2, 1)):
        comparison_name = (
            f"{arm_names[treatment_index]}_minus_{arm_names[control_index]}"
        )
        confidence_interval: list[float] | None = None
        if bootstrap_accuracies is not None:
            bootstrap_effect_sizes = (
                bootstrap_accuracies[:, treatment_index]
                - bootstrap_accuracies[:, control_index]
            )
            confidence_interval = np.quantile(
                bootstrap_effect_sizes,
                [tail_probability, 1.0 - tail_probability],
            ).tolist()
            assert np.isfinite(confidence_interval).all()
            assert -1 <= confidence_interval[0] <= confidence_interval[1] <= 1
        paired_comparisons[comparison_name] = {
            "mean_difference": float(
                arm_accuracies[treatment_index] - arm_accuracies[control_index]
            ),
            "confidence_interval": confidence_interval,
        }
    return {
        "n_questions": len(rows),
        "n_videos": len(video_ids),
        "accuracies": {
            arm_name: float(arm_accuracies[index])
            for index, arm_name in enumerate(arm_names)
        },
        "paired_comparisons": paired_comparisons,
        "confidence_level": config.confidence_level,
        "bootstrap_samples": config.bootstrap_samples,
        "resampling_unit": "video",
        "confidence_interval_method": "percentile",
        "confidence_interval_status": (
            "available"
            if bootstrap_accuracies is not None
            else "insufficient_video_clusters"
        ),
    }
