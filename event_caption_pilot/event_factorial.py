"""Count-matched partition x allocation; selectors never see gold."""

import numpy as np

from .algorithms import (
    allocate_frames,
    event_features,
    normalize_rows,
    sample_indices,
    segment,
)
from .paired_reanalysis import clustered_means

ARMS = {
    "A": ("D0", "candidate_count"),
    "B": ("D2", "candidate_count"),
    "C": ("D0", "question_relevance"),
    "D": ("D2", "question_relevance"),
}
COMPARISONS = {
    "D_minus_C": ("D", "C"),
    "B_minus_A": ("B", "A"),
    "C_minus_A": ("C", "A"),
    "D_minus_B": ("D", "B"),
    "D_minus_uniform": ("D", "uniform"),
}


def make_partitions(features, config):
    detected = segment(features, config, "D2")
    return {
        "D2": detected,
        "D0": segment(features, config, "D0", segment_count=len(detected)),
    }


def select_plans(features, query, timestamps, partitions, config):
    """No answer, option text, or captions enter allocation or selection."""
    features = normalize_rows(features, config.normalization_epsilon)
    query = normalize_rows(
        np.asarray(query)[None, :], config.normalization_epsilon
    )[0]
    times = np.asarray(timestamps, dtype=float)
    if (
        times.shape != (len(features),)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or times[0] < 0
    ):
        raise ValueError("Require chronological finite candidate timestamps")
    if len(partitions["D0"]) != len(partitions["D2"]):
        raise ValueError("Partition counts must match")
    plans = {}
    for arm, (detector, policy) in ARMS.items():
        events = partitions[detector]
        if (
            events[0].start != 0
            or events[-1].stop != len(features)
            or any(a.stop != b.start for a, b in zip(events, events[1:]))
        ):
            raise ValueError("Partitions must cover all candidates")
        pooled = event_features(features, events, config.normalization_epsilon)
        relevance = pooled @ query
        capacities = np.array([e.stop - e.start for e in events])
        # exp(log(capacity)) yields candidate-count proportional quotas.
        scores = (
            np.log(capacities) if policy == "candidate_count" else relevance
        )
        temperature = (
            1.0
            if policy == "candidate_count"
            else config.allocation_temperature
        )
        counts = allocate_frames(
            scores, capacities, config.frame_budget, temperature
        )
        indices = np.concatenate(
            [
                sample_indices(e.start, e.stop, int(n))
                for e, n in zip(events, counts)
            ]
        )
        if (
            len(indices) != config.frame_budget
            or len(np.unique(indices)) != config.frame_budget
        ):
            raise ValueError("Unique exact frame budget violated")
        weights = np.exp((scores - scores.max()) / temperature)
        weights /= weights.sum()
        selected = features[indices]
        n = len(indices)
        redundancy = (
            float(((selected @ selected.T).sum() - n) / (n * (n - 1)))
            if n > 1
            else None
        )
        bin_ids = np.empty(len(features), dtype=int)
        for i, part in enumerate(np.array_split(np.arange(len(features)), n)):
            bin_ids[part] = i
        plans[arm] = {
            "detector": detector,
            "allocation_policy": policy,
            "events": [[e.start, e.stop] for e in events],
            "capacities": capacities.tolist(),
            "relevance_raw_cosine": relevance.tolist(),
            "allocation_scores": scores.tolist(),
            "temperature": temperature,
            "weights_before_capacity_rounding": weights.tolist(),
            "allocation": counts.tolist(),
            "selected_indices": indices.tolist(),
            "selected_timestamps": times[indices].tolist(),
            "diagnostics": {
                "zero_allocation_events": int((counts == 0).sum()),
                "weight_entropy": float(
                    -np.sum(
                        weights[weights > 0] * np.log(weights[weights > 0])
                    )
                ),
                "selected_feature_redundancy": redundancy,
                "fixed_bin_coverage": len(np.unique(bin_ids[indices])) / n,
                "max_candidate_time_gap_seconds": float(
                    np.diff(np.r_[times[0], times[indices], times[-1]]).max()
                ),
            },
        }
    return plans


def summarize(rows, seed=42):
    if not rows:
        raise ValueError("No completed questions")
    arms = list(rows[0]["conditions"])
    comparisons = dict(COMPARISONS)
    if "temporal_bin" in arms:
        comparisons["D_minus_temporal_bin"] = ("D", "temporal_bin")
    means = []
    for row in rows:
        if set(row["conditions"]) != set(arms):
            raise ValueError("Incomplete conditions")
        reference = [
            t["order"] for t in row["conditions"]["uniform"]["trials"]
        ]
        if not reference or len({tuple(o) for o in reference}) != len(
            reference
        ):
            raise ValueError("Empty or duplicate orders")
        values = {}
        for arm in arms:
            trials = row["conditions"][arm]["trials"]
            if [t["order"] for t in trials] != reference:
                raise ValueError("Mismatched paired orders")
            if any(type(t["scoring_correct"]) is not bool for t in trials):
                raise ValueError("Invalid correctness")
            values[arm] = float(
                np.mean([t["scoring_correct"] for t in trials])
            )
        values.update(
            {k: values[a] - values[b] for k, (a, b) in comparisons.items()}
        )
        values["interaction"] = values["D_minus_B"] - values["C_minus_A"]
        means.append(
            {
                **{
                    k: row[k]
                    for k in ("video_id", "question_id", "duration_group")
                },
                **values,
            }
        )
    keys = [*arms, *comparisons, "interaction"]
    return {
        "primary_comparison": "D_minus_C",
        "overall": clustered_means(means, keys, seed=seed),
        "by_duration": {
            g: clustered_means(
                [r for r in means if r["duration_group"] == g], keys, seed=seed
            )
            for g in sorted({r["duration_group"] for r in means})
        },
        "paired_question_changes": {
            k: {
                "improved": sum(r[k] > 1e-12 for r in means),
                "degraded": sum(r[k] < -1e-12 for r in means),
                "unchanged": sum(abs(r[k]) <= 1e-12 for r in means),
            }
            for k in comparisons
        },
        "interpretation": "Exploratory dev; no multiplicity correction. "
        "D-C includes changes in segment lengths and pooled representations. "
        "Question repeats are not independent samples.",
    }
