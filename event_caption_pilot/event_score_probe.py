"""Event-score ablations and question-independent long-interval review."""

import html
from pathlib import Path

import numpy as np

from .algorithms import (
    allocate_frames,
    event_features,
    normalize_rows,
    sample_indices,
)
from .event_factorial_dev import validate_indices
from .reporting import png_data_url
from .types import Event

ARMS = ("pooled_mean", "frame_max", "frame_topn_mean")


def make_plans(features, query, times, original, config, top_n=3):
    """Change event scores only; never select individual high-score frames.

    Inputs deliberately exclude answer options, gold, and evidence labels.
    The baseline is cosine(normalized mean frame feature, question), not the
    arithmetic mean of frame-question cosines.
    """
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top-n must be a positive integer")
    features = normalize_rows(features, config.normalization_epsilon)
    query = normalize_rows(
        np.asarray(query)[None, :], config.normalization_epsilon
    )[0]
    times = np.asarray(times, dtype=float)
    if (
        times.shape != (len(features),)
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Require finite chronological timestamps")
    events = [Event(a, b) for a, b in original["events"]]
    if (
        events[0].start != 0
        or events[-1].stop != len(features)
        or any(a.stop != b.start for a, b in zip(events, events[1:]))
    ):
        raise ValueError("Events must partition the original candidates")
    pooled = (
        event_features(features, events, config.normalization_epsilon) @ query
    )
    if not np.allclose(
        pooled, original["relevance_raw_cosine"], atol=1e-6, rtol=1e-6
    ):
        raise ValueError("Original pooled event scores do not reproduce")
    raw = features @ query
    local_rankings = []
    for e in events:
        ids = np.arange(e.start, e.stop)
        local_rankings.append(ids[np.lexsort((ids, -raw[ids]))])
    scores = {
        "pooled_mean": np.asarray(original["relevance_raw_cosine"]),
        "frame_max": np.array([raw[ids[0]] for ids in local_rankings]),
        "frame_topn_mean": np.array(
            [raw[ids[:top_n]].mean() for ids in local_rankings]
        ),
    }
    capacities = np.array([e.stop - e.start for e in events])
    plans = {}
    for arm, values in scores.items():
        allocation = allocate_frames(
            values,
            capacities,
            config.frame_budget,
            config.allocation_temperature,
        )
        indices = [
            int(i)
            for e, n in zip(events, allocation)
            for i in sample_indices(e.start, e.stop, int(n))
        ]
        validate_indices(indices, len(features), config.frame_budget)
        if arm == "pooled_mean" and (
            allocation.tolist() != original["allocation"]
            or indices != original["selected_indices"]
        ):
            raise ValueError(
                "Baseline allocation/selection does not reproduce"
            )
        ranking = np.lexsort((np.arange(len(events)), -values))
        ranks = np.empty(len(events), dtype=int)
        ranks[ranking] = np.arange(1, len(events) + 1)
        plans[arm] = {
            "events": original["events"],
            "capacities": capacities.tolist(),
            "event_scores": values.tolist(),
            "event_ranks_1based": ranks.tolist(),
            "allocation": allocation.tolist(),
            "selected_indices": indices,
            "selected_timestamps": times[indices].tolist(),
            "temperature": config.allocation_temperature,
            "within_event": "uniform (midpoint for one frame)",
            "diagnostics": {
                "zero_allocation_events": int((allocation == 0).sum()),
                "multi_frame_events": int((allocation > 1).sum()),
                "max_event_allocation": int(allocation.max()),
                "overlap_with_baseline": len(
                    set(indices) & set(original["selected_indices"])
                )
                / config.frame_budget,
            },
        }
    return {
        "conditions": plans,
        "highest_score_candidate_per_event": [
            int(ids[0]) for ids in local_rankings
        ],
        "topn_scoring_candidates_per_event": [
            ids[:top_n].tolist() for ids in local_rankings
        ],
    }


def boundary_scores(features, config):
    """Exact D2 window-cosine scores, indexed by boundary before frame i."""
    z = normalize_rows(features, config.normalization_epsilon)
    positions = np.arange(1, len(z))
    if not len(positions):
        return positions, np.empty(0)
    prefix = np.vstack((np.zeros((1, z.shape[1])), z.cumsum(axis=0)))
    left = np.maximum(0, positions - config.window_size)
    right = np.minimum(len(z), positions + config.window_size)
    before = (prefix[positions] - prefix[left]) / (positions - left)[:, None]
    after = (prefix[right] - prefix[positions]) / (right - positions)[:, None]
    a = normalize_rows(before, config.normalization_epsilon)
    b = normalize_rows(after, config.normalization_epsilon)
    values = 1 - np.clip(np.einsum("ij,ij->i", a, b), -1, 1)
    values[values <= config.normalization_epsilon] = 0
    return positions, values


def longest_intervals(source_plans, threshold=60, limit=10):
    """Outcome-independent diagnostic sample, not a prevalence estimate."""
    rows = []
    for vid, payload in source_plans.items():
        stats = payload["statistics"]
        lengths = stats["event_lengths_seconds"]
        start = max(0, stats["duration_seconds"] - sum(lengths))
        first = next(iter(payload["questions"].values()))["D"]
        for i, length in enumerate(lengths):
            if length > threshold:
                rows.append(
                    {
                        "video_id": vid,
                        "event_index_zero_based": i,
                        "start_seconds": start,
                        "end_seconds": start + length,
                        "length_seconds": length,
                        "candidate_range": first["events"][i],
                    }
                )
            start += length
    return sorted(
        rows,
        key=lambda r: (
            -r["length_seconds"],
            r["video_id"],
            r["event_index_zero_based"],
        ),
    )[:limit]


def review_html(video, intervals, features, config, preview_count=12):
    """Offline visual review; no questions, answers, or predictions shown."""
    positions, values = boundary_scores(features, config)
    parts = [
        "<!doctype html><html lang='ko'><meta charset='utf-8'>",
        "<title>긴 구간 경계 검토</title><style>body{font:16px sans-serif;",
        "max-width:1250px;margin:24px auto;padding:16px;background:#f5f5f5}",
        "section{background:white;padding:20px;margin:20px 0}figure{margin:0}",
        ".frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}",
        "img{width:100%}svg{width:100%;height:auto}textarea{width:95%;min-height:90px}",
        "figcaption{font-size:13px}a{color:#12499f}</style>",
        f"<h1>영상 {html.escape(video.video_id)}: 긴 구간 검토</h1>",
        "<p>질문·정답을 가린 균등 미리보기입니다. ",
        "프레임 사이 변화는 놓칠 수 있습니다. ",
        "파란 선은 D2 변화 점수, 빨간 점선은 임계값입니다. ",
        "양 끝은 저장된 구간 경계입니다. ",
        "내부 점수가 임계값을 넘어도 최소 간격·전역 상한 때문에 ",
        "채택되지 않을 수 있습니다.</p>",
    ]
    for interval in intervals:
        start, stop = interval["candidate_range"]
        keep = (positions >= start) & (positions <= stop)
        local_positions, local_values = positions[keep], values[keep]
        if len(local_values) == 0:
            continue
        top = max(
            config.boundary_threshold * 1.2, float(local_values.max()), 0.01
        )
        points = " ".join(
            f"{50 + 900 * (p - start) / (stop - start):.2f},"
            f"{175 - 145 * v / top:.2f}"
            for p, v in zip(local_positions, local_values)
        )
        y = 175 - 145 * config.boundary_threshold / top
        event_id = interval["event_index_zero_based"]
        indices = sample_indices(start, stop, min(preview_count, stop - start))
        inner = values[(positions > start) & (positions < stop)]
        peaks = int((inner > config.boundary_threshold).sum())
        parts.extend(
            [
                f"<section><h2>구간 {event_id} (0부터 시작): "
                f"{interval['start_seconds']:.1f}–{interval['end_seconds']:.1f}초</h2>",
                f"<p>길이 {interval['length_seconds']:.1f}초 · "
                f"후보 [{start}, {stop}) · "
                f"내부 임계값 초과 후보 {peaks}개</p>",
                "<svg viewBox='0 0 1000 215' role='img' "
                "aria-label='D2 변화 점수'>",
                f"<text x='5' y='22'>점수 0–{top:.3f}</text>",
                "<path d='M50 25 V175 H950' fill='none' stroke='#555'/>",
                f"<path d='M50 {y:.2f} H950' stroke='#b22' "
                "stroke-dasharray='7 4'/>",
                f"<polyline points='{points}' fill='none' "
                "stroke='#2462a0' stroke-width='1.4'/>",
                f"<text x='50' y='202'>후보 {start}</text>"
                f"<text x='870' y='202'>{stop}</text></svg>",
                "<div class='frames'>",
            ]
        )
        for i in indices:
            parts.append(
                "<figure><img loading='lazy' "
                f"src='{png_data_url(video.frames[i])}' "
                f"alt='후보 {i}'><figcaption>후보 {i} · "
                f"{video.timestamps[i]:.2f}초</figcaption></figure>"
            )
        parts.append(
            "</div><p>변화가 없다고 단정하기 전에 "
            "원 영상도 확인하세요.</p></section>"
        )
    return "\n".join(parts) + "</html>"


def write_review_index(output, files):
    links = "\n".join(
        f"<li><a href='{html.escape(name, quote=True)}'>"
        f"{html.escape(Path(name).stem)}</a></li>"
        for name in files
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>긴 구간 검토</title>"
        "<h1>긴 구간 검토</h1><p>길이 기준 표본. "
        "QA 예측은 표시하지 않습니다.</p>"
        f"<ul>{links}</ul>",
        encoding="utf-8",
    )
