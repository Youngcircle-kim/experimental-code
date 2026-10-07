"""Offline factorial diagnostics: no decoding, model loading, or QA calls."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .algorithms import allocate_frames, sample_indices
from .qa_diagnostics import report_digest

ARMS = ("A", "B", "C", "D")
PAIRS = (("C", "A"), ("D", "B"), ("D", "C"), ("B", "A"), ("D", "uniform"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def overlap(left, right):
    """Shared fraction of an equal, unique frame budget (not Jaccard)."""
    if len(left) != len(right) or not left or len(set(left)) != len(left):
        raise ValueError("Expected equal nonempty unique frame budgets")
    if len(set(right)) != len(right):
        raise ValueError("Duplicate frames")
    return len(set(left) & set(right)) / len(left)


def score(condition):
    trials = condition["trials"]
    orders = [tuple(t["order"]) for t in trials]
    if not orders or len(set(orders)) != len(orders):
        raise ValueError("Empty or duplicate option orders")
    if any(type(t["scoring_correct"]) is not bool for t in trials):
        raise ValueError("Invalid correctness")
    return sum(t["scoring_correct"] for t in trials) / len(trials)


def replay(plan, temperature):
    """Replay allocation and sampling without estimating QA accuracy."""
    counts = allocate_frames(
        np.array(plan["relevance_raw_cosine"]),
        np.array(plan["capacities"]),
        len(plan["selected_indices"]),
        temperature,
    )
    indices = [
        int(i)
        for (start, stop), n in zip(plan["events"], counts)
        for i in sample_indices(start, stop, int(n))
    ]
    return counts.tolist(), indices


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, dict))
                    else v
                    for k, v in row.items()
                }
            )


def validate_plan(plan, budget, policy):
    events = plan["events"]
    cap = np.asarray(plan["capacities"])
    allocation = np.asarray(plan["allocation"])
    if (
        not events
        or cap.shape != (len(events),)
        or allocation.shape != cap.shape
    ):
        raise ValueError("Invalid event/count dimensions")
    if (
        events[0][0] != 0
        or any(a >= b for a, b in events)
        or any(a[1] != b[0] for a, b in zip(events, events[1:]))
        or cap.tolist() != [b - a for a, b in events]
    ):
        raise ValueError("Invalid contiguous partitions/capacities")
    if (
        allocation.dtype.kind not in "iu"
        or allocation.sum() != budget
        or np.any(allocation < 0)
        or np.any(allocation > cap)
    ):
        raise ValueError("Invalid frame allocation")
    relevance = np.asarray(plan["relevance_raw_cosine"])
    weights = np.asarray(plan["weights_before_capacity_rounding"])
    if (
        relevance.shape != cap.shape
        or weights.shape != cap.shape
        or not np.isfinite(relevance).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Invalid relevance/weight arrays")
    scores = np.log(cap) if policy == "candidate_count" else relevance
    temperature = 1.0 if policy == "candidate_count" else plan["temperature"]
    counts = allocate_frames(scores, cap, budget, temperature)
    indices = [
        int(i)
        for (start, stop), n in zip(events, counts)
        for i in sample_indices(start, stop, int(n))
    ]
    expected_weights = np.exp((scores - scores.max()) / temperature)
    expected_weights /= expected_weights.sum()
    if (
        counts.tolist() != plan["allocation"]
        or indices != plan["selected_indices"]
        or not np.allclose(weights, expected_weights, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("Current allocator cannot reproduce saved plan")
    times = np.asarray(plan["selected_timestamps"])
    if (
        times.shape != (budget,)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Invalid selected timestamps")


def run(source, output, temperatures=(1.0, 0.3, 0.1)):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError(
            "Use a separate output directory outside the source run"
        )
    if not temperatures or any(
        not np.isfinite(t) or t <= 0 for t in temperatures
    ):
        raise ValueError("Temperatures must be finite and positive")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("Duplicate temperatures")
    protocol = read(source / "protocol.json")
    digest = report_digest(protocol)
    budget = protocol["config"]["frame_budget"]
    plans = {}
    for path in sorted((source / "plans").glob("*.json")):
        plan = read(path)
        vid = plan["identity"]["video_id"]
        if vid in plans or plan["identity"]["protocol_sha256"] != digest:
            raise ValueError("Duplicate video or mismatched plan protocol")
        plans[vid] = plan
    rows, seen = [], set()
    metadata = {}
    manifest_path = Path(
        protocol["config"].get("manifest_path") or "__missing_manifest__"
    )
    if manifest_path.is_file():
        for video in read(manifest_path)["videos"]:
            for question in video.get("questions", []):
                metadata[(video["video_id"], question["question_id"])] = {
                    "question_text": question["text"],
                    "options": question["options"],
                    "answer_index": question["answer_index"],
                    "video_path": video["path"],
                }
    for path in sorted((source / "questions").glob("*.json")):
        row = read(path)
        key = (row["video_id"], row["question_id"])
        if key in seen:
            raise ValueError("Duplicate question")
        seen.add(key)
        plan = plans[key[0]]
        if row["candidate_sha256"] != plan["identity"]["candidate_sha256"]:
            raise ValueError("Candidate identity mismatch")
        reference = [
            t["order"] for t in row["conditions"]["uniform"]["trials"]
        ]
        if not set(ARMS).issubset(row["conditions"]):
            raise ValueError("Incomplete factorial conditions")
        for arm, condition in row["conditions"].items():
            score(condition)
            if [t["order"] for t in condition["trials"]] != reference:
                raise ValueError("Unpaired option orders")
            if arm in ARMS:
                saved = plan["questions"][key[1]][arm]
                for field in (
                    "events",
                    "capacities",
                    "allocation",
                    "selected_indices",
                    "relevance_raw_cosine",
                    "temperature",
                    "selected_timestamps",
                    "weights_before_capacity_rounding",
                    "diagnostics",
                ):
                    if saved[field] != condition[field]:
                        raise ValueError("Plan/question mismatch: " + field)
                validate_plan(
                    saved,
                    budget,
                    "candidate_count"
                    if arm in ("A", "B")
                    else "question_relevance",
                )
        cc = row["conditions"]
        if (
            cc["A"]["events"] != cc["C"]["events"]
            or cc["B"]["events"] != cc["D"]["events"]
            or len(cc["A"]["events"]) != len(cc["B"]["events"])
        ):
            raise ValueError("Unmatched factorial partitions")
        rows.append(row)
    if not rows:
        raise ValueError("No completed questions")
    progress = read(source / "progress.json")
    if len(rows) != progress["completed_questions"]:
        raise ValueError(
            "Progress/question count mismatch; stop concurrent runs"
        )
    if len(rows) > progress["total_questions"]:
        raise ValueError("Completed count exceeds expected count")

    question_rows, event_rows, sensitivity, reviews, video_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    for row in rows:
        base = {
            k: row[k] for k in ("video_id", "question_id", "duration_group")
        }
        conditions = row["conditions"]
        for arm in ARMS:
            p = conditions[arm]
            cap, allocation = (
                np.array(p["capacities"]),
                np.array(p["allocation"]),
            )
            weights = np.array(p["weights_before_capacity_rounding"])
            relevance = np.array(p["relevance_raw_cosine"])
            question_rows.append(
                {
                    **base,
                    "arm": arm,
                    "accuracy": score(p),
                    "n_events": len(cap),
                    "zero_events": int(sum(allocation == 0)),
                    "zero_event_fraction": float(np.mean(allocation == 0)),
                    "omitted_candidate_fraction": float(
                        cap[allocation == 0].sum() / cap.sum()
                    ),
                    "one_frame_events": int(sum(allocation == 1)),
                    "max_frames_per_event": int(allocation.max()),
                    "relevance_range": float(np.ptp(relevance)),
                    "rank_only_regime": bool(budget * weights.max() < 1),
                    "weight_underflow": bool(weights.min() == 0),
                    "weight_max_min_ratio": float(
                        weights.max() / weights.min()
                    )
                    if weights.min() > 0
                    else None,
                    **p["diagnostics"],
                }
            )
            for i, ((start, stop), n) in enumerate(
                zip(p["events"], allocation)
            ):
                event_rows.append(
                    {
                        **base,
                        "arm": arm,
                        "event": i,
                        "start_candidate": start,
                        "stop_candidate_exclusive": stop,
                        "candidate_count": stop - start,
                        "allocation": int(n),
                        "relevance": float(relevance[i]),
                        "weight": float(weights[i]),
                        "selected_timestamps": [
                            t
                            for j, t in zip(
                                p["selected_indices"], p["selected_timestamps"]
                            )
                            if start <= j < stop
                        ],
                    }
                )
            if arm in ("C", "D"):
                for temperature in temperatures:
                    counts, indices = replay(p, temperature)
                    old_events = {
                        i for i, n in enumerate(p["allocation"]) if n
                    }
                    new_events = {i for i, n in enumerate(counts) if n}
                    sensitivity.append(
                        {
                            **base,
                            "arm": arm,
                            "temperature": temperature,
                            "allocation_changed": counts != p["allocation"],
                            "shared_frame_fraction": overlap(
                                indices, p["selected_indices"]
                            ),
                            "selected_event_jaccard": (
                                len(old_events & new_events)
                                / len(old_events | new_events)
                            ),
                            "zero_events": counts.count(0),
                            "one_frame_events": counts.count(1),
                            "max_frames_per_event": max(counts),
                            "allocation": counts,
                            "selected_indices": indices,
                        }
                    )
        for treatment, control in PAIRS:
            a, b = conditions[treatment], conditions[control]
            delta = score(a) - score(b)
            reviews.append(
                {
                    **base,
                    "comparison": treatment + "_minus_" + control,
                    **metadata.get(
                        (row["video_id"], row["question_id"]),
                        {
                            "question_text": "",
                            "options": [],
                            "answer_index": None,
                            "video_path": "",
                        },
                    ),
                    "delta_pp": 100 * delta,
                    "direction": "improved"
                    if delta > 1e-12
                    else "degraded"
                    if delta < -1e-12
                    else "unchanged",
                    "treatment_accuracy": score(a),
                    "control_accuracy": score(b),
                    "shared_frame_fraction": overlap(
                        a["selected_indices"], b["selected_indices"]
                    ),
                    "same_allocation": a["allocation"] == b["allocation"]
                    if (treatment, control) in (("C", "A"), ("D", "B"))
                    else None,
                    "treatment_indices": a["selected_indices"],
                    "control_indices": b["selected_indices"],
                    "treatment_timestamps": a.get("selected_timestamps", []),
                    "control_timestamps": b.get("selected_timestamps", []),
                    "evidence_in_candidates": "",
                    "treatment_evidence_segment_received_frames": "",
                    "control_evidence_segment_received_frames": "",
                    "evidence_in_treatment": "",
                    "evidence_in_control": "",
                    "failure_category": "",
                    "notes": "",
                }
            )
    for vid in sorted({r["video_id"] for r in rows}):
        group = [r for r in rows if r["video_id"] == vid]
        for arm in ARMS:
            pp = [r["conditions"][arm] for r in group]
            if any(p["events"] != pp[0]["events"] for p in pp):
                raise ValueError(
                    "Partitions must be question-independent within a video"
                )
            overlaps = [
                overlap(a["selected_indices"], b["selected_indices"])
                for a, b in itertools.combinations(pp, 2)
            ]
            lengths = pp[0]["capacities"]
            video_rows.append(
                {
                    "video_id": vid,
                    "duration_group": group[0]["duration_group"],
                    "arm": arm,
                    "n_questions": len(pp),
                    "n_events": len(lengths),
                    "hit_max_segments": len(lengths)
                    == protocol["config"]["max_segments"],
                    "events_exceed_budget": len(lengths) > budget,
                    "min_segment_candidates": min(lengths),
                    "median_segment_candidates": float(np.median(lengths)),
                    "max_segment_candidates": max(lengths),
                    "unique_allocations": len(
                        {tuple(p["allocation"]) for p in pp}
                    ),
                    "unique_selections": len(
                        {tuple(p["selected_indices"]) for p in pp}
                    ),
                    "between_question_shared_fraction": float(
                        np.mean(overlaps)
                    )
                    if overlaps
                    else None,
                }
            )

    summaries = {}
    for group in ["overall", *sorted({r["duration_group"] for r in rows})]:

        def take(data):
            return [
                r
                for r in data
                if group == "overall" or r["duration_group"] == group
            ]

        arm_summary = {}
        for arm in ARMS:
            qq = [r for r in take(question_rows) if r["arm"] == arm]
            vv = [r for r in take(video_rows) if r["arm"] == arm]
            arm_summary[arm] = {
                "n_videos": len(vv),
                "n_questions": len(qq),
                "hit_max_segments_video_fraction": float(
                    np.mean([r["hit_max_segments"] for r in vv])
                ),
                "single_allocation_video_fraction": float(
                    np.mean(
                        [
                            r["unique_allocations"] == 1
                            for r in vv
                            if r["n_questions"] > 1
                        ]
                    )
                )
                if any(r["n_questions"] > 1 for r in vv)
                else None,
                **{
                    k + "_question_mean": float(
                        np.mean([r[k] for r in qq if r[k] is not None])
                    )
                    if any(r[k] is not None for r in qq)
                    else None
                    for k in (
                        "zero_event_fraction",
                        "omitted_candidate_fraction",
                        "one_frame_events",
                        "max_frames_per_event",
                        "relevance_range",
                        "weight_max_min_ratio",
                        "rank_only_regime",
                        "weight_underflow",
                    )
                },
            }
        pairs = {}
        for a, b in PAIRS:
            name = a + "_minus_" + b
            rr = [r for r in take(reviews) if r["comparison"] == name]
            pairs[name] = {
                "same_selection_fraction": float(
                    np.mean([r["shared_frame_fraction"] == 1 for r in rr])
                ),
                "shared_frame_fraction_mean": float(
                    np.mean([r["shared_frame_fraction"] for r in rr])
                ),
                "delta_pp_mean": float(np.mean([r["delta_pp"] for r in rr])),
            }
        temperature_summary = []
        for arm in ("C", "D"):
            for temperature in temperatures:
                ss = [
                    r
                    for r in take(sensitivity)
                    if r["arm"] == arm and r["temperature"] == temperature
                ]
                temperature_summary.append(
                    {
                        "arm": arm,
                        "temperature": temperature,
                        "allocation_changed_fraction": float(
                            np.mean([r["allocation_changed"] for r in ss])
                        ),
                        "shared_frame_fraction_mean": float(
                            np.mean([r["shared_frame_fraction"] for r in ss])
                        ),
                        "selected_event_jaccard_mean": float(
                            np.mean([r["selected_event_jaccard"] for r in ss])
                        ),
                        "zero_events_mean": float(
                            np.mean([r["zero_events"] for r in ss])
                        ),
                        "one_frame_only_question_fraction": float(
                            np.mean(
                                [r["max_frames_per_event"] == 1 for r in ss]
                            )
                        ),
                    }
                )
        summaries[group] = {
            "arms": arm_summary,
            "comparisons": pairs,
            "temperature_replay": temperature_summary,
        }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("questions", question_rows),
        ("events", event_rows),
        ("videos", video_rows),
        ("temperature_sensitivity", sensitivity),
        (
            "review_queue_template",
            sorted(reviews, key=lambda r: -abs(r["delta_pp"])),
        ),
    ):
        write_csv(output / (name + ".csv"), data)
    if not (output / "review_queue.csv").exists():
        write_csv(
            output / "review_queue.csv",
            sorted(reviews, key=lambda r: -abs(r["delta_pp"])),
        )
    report = {
        "source": str(source),
        "protocol_sha256": digest,
        "completed_questions": len(rows),
        "expected_questions": progress["total_questions"],
        "partial": len(rows) != progress["total_questions"],
        "groups": summaries,
        "limitations": (
            "Descriptive diagnostics only. No evidence recall, causal failure "
            "labels, or counterfactual QA accuracy. Temperature replay is "
            "selection-only; do not select temperature by these metrics alone."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# 프레임 선택 병목 진단",
        "",
        f"완료 문항: {len(rows)}/{progress['total_questions']}",
        "",
        "기존 기록의 기술통계입니다. "
        "근거 누락 여부와 QA 실패 원인은 수동 검토가 필요합니다.",
        "",
    ]
    for group, summary in summaries.items():
        lines += [
            f"## {group}",
            "",
            "| 조건 | 상한 도달 영상 | 질문 간 동일 배분 영상 | "
            "미배정 구간 비율 | 미배정 구간의 후보 비율 |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, s in summary["arms"].items():
            same = s["single_allocation_video_fraction"]
            lines.append(
                f"| {arm} | {s['hit_max_segments_video_fraction']:.1%} | "
                f"{format(same, '.1%') if same is not None else 'N/A'} | "
                f"{s['zero_event_fraction_question_mean']:.1%} | "
                f"{s['omitted_candidate_fraction_question_mean']:.1%} |"
            )
        lines += [
            "",
            "| 비교 | 동일 프레임 선택 | 평균 공유 프레임 비율 | "
            "정확도 차이 (%p) |",
            "|---|---:|---:|---:|",
        ]
        for name, s in summary["comparisons"].items():
            lines.append(
                f"| {name} | {s['same_selection_fraction']:.1%} | "
                f"{s['shared_frame_fraction_mean']:.1%} | "
                f"{s['delta_pp_mean']:+.2f} |"
            )
        lines += [
            "",
            "| 조건 / 온도 | 배분 변경 문항 | 공유 프레임 비율 | "
            "선택 구간 Jaccard | 평균 미배정 구간 수 | "
            "배정 구간마다 1장인 문항 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in summary["temperature_replay"]:
            lines.append(
                f"| {s['arm']} / {s['temperature']:g} | "
                f"{s['allocation_changed_fraction']:.1%} | "
                f"{s['shared_frame_fraction_mean']:.1%} | "
                f"{s['selected_event_jaccard_mean']:.1%} | "
                f"{s['zero_events_mean']:.2f} | "
                f"{s['one_frame_only_question_fraction']:.1%} |"
            )
        lines.append("")
    lines += [
        "온도 민감도는 재추론 없이 배분과 균등 샘플링을 재계산합니다. "
        "변경된 선택의 정확도는 측정하지 않았습니다.",
        "rank_only_regime은 모든 실수 할당량이 1 미만인 경우입니다. "
        "이 범위에서는 관련성 상위 구간에 1장씩 배정하므로 "
        "온도 변화에도 선택이 유지될 수 있습니다.",
        "review_queue_template.csv는 차이 절댓값 순의 최신 검토 목록입니다. "
        "review_queue.csv는 최초 실행 때 생성하며 재실행 시 보존합니다.",
        "미배정 후보 비율은 해당 구간의 후보 수 비중이며 "
        "정답 근거 recall이 아닙니다.",
        "상한 도달 비율은 A/C에도 D2와 맞춘 구간 수를 표시합니다. "
        "질문 간 비교는 완료 문항이 2개 이상인 영상만 사용합니다.",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/videomme_event_factorial"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/videomme_bottleneck_diagnostics"),
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[1.0, 0.3, 0.1]
    )
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir, args.temperatures)
    print(
        json.dumps(
            {
                "completed_questions": report["completed_questions"],
                "output_dir": str(args.output_dir.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
