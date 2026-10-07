"""Fixed D2 partitions: allocation concentration x within-event retrieval."""

import numpy as np

from .algorithms import allocate_frames, normalize_rows, sample_indices
from .event_factorial_dev import validate_indices
from .paired_reanalysis import clustered_means

ARMS = ("D", "D_topk", "focus_uniform", "focus_topk")
COMPARISONS = {
    "within_only": ("D_topk", "D"),
    "allocation_only": ("focus_uniform", "D"),
    "both": ("focus_topk", "D"),
    "within_under_focus": ("focus_topk", "focus_uniform"),
    "allocation_under_topk": ("focus_topk", "D_topk"),
}


def focus_allocation(plan, budget, top_events):
    """Mask low-ranked events, retaining the original softmax temperature.

    Expand the eligible set only when its candidate capacity cannot meet the
    budget. Ties use chronological event order, including allocator ties.
    """
    if type(top_events) is not int or top_events < 1:
        raise ValueError("top-events must be a positive integer")
    scores = np.asarray(plan["relevance_raw_cosine"], dtype=float)
    capacities = np.asarray(plan["capacities"], dtype=int)
    ranking = np.lexsort((np.arange(len(scores)), -scores))
    count = min(top_events, len(scores))
    while capacities[ranking[:count]].sum() < budget:
        count += 1
        if count > len(scores):
            raise ValueError("Insufficient candidate capacity")
    eligible = np.sort(ranking[:count])
    masked = np.zeros_like(capacities)
    masked[eligible] = capacities[eligible]
    allocation = allocate_frames(scores, masked, budget, plan["temperature"])
    return allocation.tolist(), {
        "requested_top_events": top_events,
        "eligible_events": eligible.tolist(),
        "capacity_expansion_events": count - min(top_events, len(scores)),
    }


def select_inside(scores, times, events, allocation, min_gap=0.0):
    """Greedy cosine top-k per event; optional hard-first temporal spacing.

    If spacing prevents an exact budget, fill remaining slots by descending
    score without spacing. Report every such fallback; never duplicate a
    frame. All ties use the earlier candidate index.
    """
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError("min-gap-seconds must be finite and nonnegative")
    selected, fallback = [], 0
    for (start, stop), count in zip(events, allocation):
        candidates = np.arange(start, stop)
        ranking = candidates[np.lexsort((candidates, -scores[start:stop]))]
        chosen = []
        for candidate in ranking:
            if len(chosen) == count:
                break
            if all(
                abs(times[candidate] - times[i]) >= min_gap for i in chosen
            ):
                chosen.append(int(candidate))
        for candidate in ranking:
            if len(chosen) == count:
                break
            if candidate not in chosen:
                chosen.append(int(candidate))
                fallback += 1
        if len(chosen) != count:
            raise ValueError("Event does not have enough unique candidates")
        selected.extend(chosen)
    return sorted(selected), fallback


def make_plans(features, query, times, original, config, top_events, min_gap):
    """Selectors receive only question embedding, images, and saved D plan.

    No choices, answers, captions, or human evidence labels enter selection.
    """
    features = normalize_rows(features, config.normalization_epsilon)
    query = normalize_rows(
        np.asarray(query)[None, :], config.normalization_epsilon
    )[0]
    times = np.asarray(times, dtype=float)
    if (
        times.shape != (len(features),)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or times[0] < 0
    ):
        raise ValueError("Require chronological finite timestamps")
    events = original["events"]
    if events[-1][1] != len(features):
        raise ValueError("Original events do not match candidate count")
    budget = config.frame_budget
    scores = features @ query
    focused, focus_info = focus_allocation(original, budget, top_events)
    plans = {}
    for arm in ARMS:
        allocation = (
            focused if arm.startswith("focus") else original["allocation"]
        )
        if arm.endswith("topk"):
            indices, fallback = select_inside(
                scores, times, events, allocation, min_gap
            )
        else:
            indices = [
                int(i)
                for (a, b), n in zip(events, allocation)
                for i in sample_indices(a, b, n)
            ]
            fallback = 0
        validate_indices(indices, len(features), budget)
        if arm == "D" and indices != original["selected_indices"]:
            raise ValueError("Original D selection does not reproduce")
        selected = features[indices]
        redundancy = float(
            ((selected @ selected.T).sum() - budget) / (budget * (budget - 1))
        )
        plans[arm] = {
            "events": events,
            "allocation": allocation,
            "selected_indices": indices,
            "selected_timestamps": times[indices].tolist(),
            "allocation_policy": (
                "top_events_mask_then_original_softmax"
                if arm.startswith("focus")
                else "original_D"
            ),
            "within_event": (
                "question_cosine_topk" if arm.endswith("topk") else "uniform"
            ),
            "focus": focus_info if arm.startswith("focus") else None,
            "diagnostics": {
                "zero_allocation_events": sum(n == 0 for n in allocation),
                "multi_frame_events": sum(n > 1 for n in allocation),
                "max_event_allocation": max(allocation),
                "selected_feature_redundancy": redundancy,
                "selected_mean_question_cosine": float(scores[indices].mean()),
                "overlap_with_D": len(
                    set(indices) & set(original["selected_indices"])
                )
                / budget,
                "gap_fallback_frames": fallback,
            },
        }
    return {"frame_question_cosines": scores.tolist(), "conditions": plans}


def summarize(rows, seed=42):
    if not rows:
        raise ValueError("No completed questions")
    means = []
    for row in rows:
        if set(row["conditions"]) != set(ARMS):
            raise ValueError("Incomplete conditions")
        orders = [t["order"] for t in row["conditions"]["D"]["trials"]]
        if not orders or len({tuple(o) for o in orders}) != len(orders):
            raise ValueError("Invalid reference orders")
        values = {}
        for arm in ARMS:
            trials = row["conditions"][arm]["trials"]
            if [t["order"] for t in trials] != orders or any(
                type(t["scoring_correct"]) is not bool for t in trials
            ):
                raise ValueError("Mismatched paired trials")
            values[arm] = float(
                np.mean([t["scoring_correct"] for t in trials])
            )
        values.update(
            {k: values[a] - values[b] for k, (a, b) in COMPARISONS.items()}
        )
        values["interaction"] = (
            values["within_under_focus"] - values["within_only"]
        )
        means.append(
            {
                **{
                    k: row[k]
                    for k in ("video_id", "question_id", "duration_group")
                },
                **values,
            }
        )
    keys = [*ARMS, *COMPARISONS, "interaction"]
    summary = {
        "primary_comparison": "both (focus_topk minus D)",
        "overall": clustered_means(means, keys, seed=seed),
        "by_duration": {
            group: clustered_means(
                [r for r in means if r["duration_group"] == group],
                keys,
                seed=seed,
            )
            for group in sorted({r["duration_group"] for r in means})
        },
        "paired_question_changes": {
            key: {
                "improved": sum(r[key] > 1e-12 for r in means),
                "degraded": sum(r[key] < -1e-12 for r in means),
                "unchanged": sum(abs(r[key]) <= 1e-12 for r in means),
            }
            for key in COMPARISONS
        },
        "selection_diagnostics_mean": {
            arm: {
                key: float(
                    np.mean(
                        [
                            row["conditions"][arm]["diagnostics"][key]
                            for row in rows
                        ]
                    )
                )
                for key in rows[0]["conditions"][arm]["diagnostics"]
            }
            for arm in ARMS
        },
        "interpretation": (
            "Exploratory dev study after inspecting a failure case. "
            "Question order repeats are not independent samples. "
            "CI uses paired video clusters; no multiplicity correction. "
            "Cosine and redundancy are proxies, not evidence recall. "
            "No captions; no gold or options in frame selection."
        ),
    }
    return summary, means
