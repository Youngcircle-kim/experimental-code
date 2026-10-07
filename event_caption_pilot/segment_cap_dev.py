"""Two-stage relaxed segment-cap diagnostic with frozen C/D QA."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .backends import build_backend
from .bottleneck_diagnostics import read, validate_plan, write_csv
from .cache import write_json
from .data import array_digest, validate_video
from .event_factorial import make_partitions, select_plans
from .event_factorial_dev import cached_trial, measured_score, validate_indices
from .event_refinement_dev import (
    checked_payload,
    question_from,
    save_payload,
)
from .event_refinement_dev import (
    load_inputs as load_source,
)
from .frame_replacement_probe import strict_trials
from .midpoint_review import load_candidates
from .order_experiments import seed_call
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .types import Video

ARMS = ("C_original", "D_original", "C", "D")
COMPARISONS = {
    "D_minus_C": ("D", "C"),
    "D_minus_original_D": ("D", "D_original"),
    "C_minus_original_C": ("C", "C_original"),
    "original_D_minus_C": ("D_original", "C_original"),
}
VLM_ONLY_FIELDS = {
    "vlm_loaded",
    "vlm_resolved_commit",
    "vlm_architecture",
    "vlm_attention_implementation",
    "vlm_parameters",
}


def validate_metadata(actual, expected, encoder_only=False):
    if encoder_only:
        if actual.get("vlm_loaded") is not False or any(
            actual.get(k) is not None for k in VLM_ONLY_FIELDS - {"vlm_loaded"}
        ):
            raise ValueError("Preparation must load only the encoder")
        actual = {k: v for k, v in actual.items() if k not in VLM_ONLY_FIELDS}
        expected = {
            k: v for k, v in expected.items() if k not in VLM_ONLY_FIELDS
        }
    if actual != expected:
        raise ValueError("Runtime/model metadata differs from source")


def load_inputs(args):
    if args.max_segments < 1:
        raise ValueError("max-segments must be positive")
    original, manifest, dev, sources, metadata, prior = load_source(
        SimpleNamespace(
            source_dir=args.source_dir,
            baseline_dir=args.baseline_dir,
            output_dir=args.output_dir,
            top_events=8,
            min_gap_seconds=0,
            video_id=args.video_id,
            max_videos=None,
        )
    )
    dev = [
        v
        for v in dev
        if args.duration == "all" or v["duration"] == args.duration
    ]
    if args.max_videos is not None:
        if args.max_videos < 1:
            raise ValueError("max-videos must be positive")
        dev = dev[: args.max_videos]
    if not dev:
        raise ValueError("No matching development videos")
    for item in dev:
        for annotation in item["questions"]:
            q = question_from(annotation)
            old = sources[(item["video_id"], q.question_id)]["conditions"]
            validate_plan(
                old["C"], original.frame_budget, "question_relevance"
            )
            strict_trials(old["C"]["trials"], q, original)
            if (
                len(old["C"]["events"]) != len(old["D"]["events"])
                or old["C"]["events"][-1][1] != old["D"]["events"][-1][1]
            ):
                raise ValueError("Source C/D partition mismatch")
    config = replace(original, max_segments=args.max_segments)
    config.validate()
    root = Path(__file__).parent
    protocol = {
        "version": 1,
        "experiment": "relaxed_segment_cap_C_D",
        **{
            k: prior[k]
            for k in (
                "source_dir",
                "source_results_sha256",
                "source_protocol_sha256",
                "manifest_sha256",
                "backend_metadata_sha256",
            )
        },
        "original_max_segments": original.max_segments,
        "config": config.to_dict(),
        "video_ids": [v["video_id"] for v in dev],
        "duration_filter": args.duration,
        "arms": list(ARMS),
        "primary_comparison": "D_minus_C",
        "selection": "Original D2 threshold/minimum length; C has the same "
        "actual event count as D. Original query allocation and uniform "
        "within-event selection; 16 unique chronological frames; no captions.",
        "preparation": "Decode + CLIP only; persist C/D selections, no QA",
        "interpretation": "Exploratory dev. Changing the cap can change "
        "boundaries, pooled features and allocation, not boundary quality "
        "alone. Six option orders are paired repeats, not six questions.",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))
        },
    }
    return original, config, manifest, dev, sources, metadata, protocol


def video_identity(item, sources, digest):
    hashes = {
        sources[(item["video_id"], str(q["question_id"]))]["candidate_sha256"]
        for q in item["questions"]
    }
    if len(hashes) != 1:
        raise ValueError("Source questions disagree on candidate content")
    return {
        "protocol_sha256": digest,
        "video_id": item["video_id"],
        "candidate_sha256": hashes.pop(),
    }


def plan_path(output, video_id):
    return output / "plans" / (report_digest(video_id) + ".json")


def decode_item(item, manifest, config, identity):
    frames, times, duration = load_candidates(item, manifest, config)
    video = Video(
        item["video_id"],
        "dev",
        frames,
        times,
        duration,
        tuple(question_from(q) for q in item["questions"]),
        item["source_id"],
    )
    validate_video(video, config)
    if array_digest(frames, times) != identity["candidate_sha256"]:
        raise ValueError("Decoded candidates differ from original experiment")
    return video


def video_statistics(
    video, partitions, original_count, config, duration_group
):
    events = partitions["D2"]
    # Duration of candidate-index intervals, not annotated semantic events.
    edges = np.r_[video.timestamps, video.duration_seconds]
    lengths = [float(edges[e.stop] - edges[e.start]) for e in events]
    counts = [e.stop - e.start for e in events]
    return {
        "video_id": video.video_id,
        "duration_group": duration_group,
        "duration_seconds": video.duration_seconds,
        "candidate_count": len(video.frames),
        "original_n_events": original_count,
        "n_events": len(events),
        "hit_max_segments": len(events) == config.max_segments,
        "n_events_above_frame_budget": len(events) > config.frame_budget,
        "event_lengths_seconds": lengths,
        "event_candidate_counts": counts,
    }


def validate_prepared(prepared, item, source, config):
    expected = {str(q["question_id"]) for q in item["questions"]}
    if set(prepared["questions"]) != expected:
        raise ValueError("Incomplete prepared questions")
    reference = None
    for qid, plans in prepared["questions"].items():
        if set(plans) != {"C", "D"}:
            raise ValueError("Preparation must contain C and D")
        for arm, plan in plans.items():
            validate_plan(plan, config.frame_budget, "question_relevance")
            if (
                plan["events"][-1][1]
                != source["conditions"]["D"]["events"][-1][1]
            ):
                raise ValueError("Candidate count differs in prepared plan")
            if plan["detector"] != ("D0" if arm == "C" else "D2"):
                raise ValueError("Wrong partition policy")
        if len(plans["C"]["events"]) != len(plans["D"]["events"]):
            raise ValueError("C/D event counts differ")
        boundaries = {arm: p["events"] for arm, p in plans.items()}
        if reference is not None and boundaries != reference:
            raise ValueError("Partitions must not depend on question")
        reference = boundaries
    stats = prepared["statistics"]
    if (
        stats["video_id"] != item["video_id"]
        or stats["n_events"] != len(reference["D"])
        or not 1 <= stats["n_events"] <= config.max_segments
    ):
        raise ValueError("Prepared statistics/partition mismatch")


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.quantile(a, 0.9)),
        "max": float(a.max()),
    }


def preparation_report(output, prepared, protocol):
    stats = [p["statistics"] for p in prepared]

    def group_summary(rows):
        lengths = [s for row in rows for s in row["event_lengths_seconds"]]
        return {
            "n_videos": len(rows),
            "n_events_per_video": distribution([r["n_events"] for r in rows]),
            "hit_max_segments_fraction": float(
                np.mean([r["hit_max_segments"] for r in rows])
            ),
            "event_lengths_seconds_pooled": distribution(lengths),
            "events_under_2_seconds_fraction": float(
                np.mean(np.asarray(lengths) < 2)
            ),
            "events_over_60_seconds_fraction": float(
                np.mean(np.asarray(lengths) > 60)
            ),
        }

    allocation_rows = []
    for prepared_video in prepared:
        for qid, plans in prepared_video["questions"].items():
            for arm, plan in plans.items():
                counts = plan["allocation"]
                allocation_rows.append(
                    {
                        "video_id": prepared_video["statistics"]["video_id"],
                        "question_id": qid,
                        "condition": arm,
                        "n_events": len(counts),
                        "zero_allocation_events": sum(n == 0 for n in counts),
                        "multi_frame_events": sum(n > 1 for n in counts),
                        "max_event_allocation": max(counts),
                        "unallocated_candidate_fraction": sum(
                            c
                            for c, n in zip(plan["capacities"], counts)
                            if n == 0
                        )
                        / sum(plan["capacities"]),
                        "allocation": counts,
                        "selected_indices": plan["selected_indices"],
                    }
                )
    summary = {
        "stage": "prepared_no_QA",
        "protocol_sha256": report_digest(protocol),
        "max_segments": protocol["config"]["max_segments"],
        "overall": group_summary(stats),
        "by_duration": {
            g: group_summary([r for r in stats if r["duration_group"] == g])
            for g in sorted({r["duration_group"] for r in stats})
        },
        "interpretation": "Durations use candidate-index intervals. "
        "No evidence recall or QA accuracy measured in preparation.",
    }
    write_json(output / "segmentation_summary.json", summary)
    write_csv(output / "segmentation_videos.csv", stats)
    write_csv(output / "prepared_selections.csv", allocation_rows)
    (output / "segmentation_summary.md").write_text(
        "# Segmentation preparation (no QA)\n\n```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return summary


def prepare(args, inputs):
    original, config, manifest, dev, sources, metadata, protocol = inputs
    digest = report_digest(protocol)
    encoder = None
    payloads, plan_hashes = [], {}
    for item in dev:
        identity = video_identity(item, sources, digest)
        path = plan_path(args.output_dir, item["video_id"])
        source = sources[
            (item["video_id"], str(item["questions"][0]["question_id"]))
        ]
        if path.exists():
            prepared = checked_payload(path, identity)
        else:
            print(f"Preparing {item['video_id']}: decode + CLIP", flush=True)
            video = decode_item(item, manifest, config, identity)
            if encoder is None:
                encoder_config = replace(config, pilot="a")
                seed_call(encoder_config)
                encoder = build_backend(encoder_config)
                validate_metadata(encoder.metadata(), metadata, True)
                write_json(
                    args.output_dir / "encoder_backend.json",
                    encoder.metadata(),
                )
            seed_call(config)
            features = encoder.encode_frames(video.frames)
            # Verify the old boundary selection before changing its cap.
            old_parts = make_partitions(features, original)
            for arm, detector in (("C", "D0"), ("D", "D2")):
                if [[e.start, e.stop] for e in old_parts[detector]] != source[
                    "conditions"
                ][arm]["events"]:
                    raise ValueError("Original partitions do not reproduce")
            partitions = make_partitions(features, config)
            plans = {}
            for q in video.questions:
                all_plans = select_plans(
                    features,
                    encoder.encode_visual_question(q.text),
                    video.timestamps,
                    partitions,
                    config,
                )
                plans[q.question_id] = {
                    arm: all_plans[arm] for arm in ("C", "D")
                }
            prepared = {
                "statistics": video_statistics(
                    video,
                    partitions,
                    len(old_parts["D2"]),
                    config,
                    item["duration"],
                ),
                "questions": plans,
            }
            save_payload(path, identity, prepared)
            del video, features
        validate_prepared(prepared, item, source, config)
        payloads.append(prepared)
        plan_hashes[item["video_id"]] = report_digest(prepared)
        write_json(
            args.output_dir / "prepare_progress.json",
            {
                "completed_videos": len(payloads),
                "total_videos": len(dev),
                "status": "running",
            },
        )
        print(
            f"Prepared {len(payloads)}/{len(dev)}: "
            f"K={prepared['statistics']['n_events']}",
            flush=True,
        )
    summary = preparation_report(args.output_dir, payloads, protocol)
    write_json(
        args.output_dir / "prepare_complete.json",
        {
            "protocol_sha256": digest,
            "plan_sha256": plan_hashes,
        },
    )
    write_json(
        args.output_dir / "prepare_progress.json",
        {
            "completed_videos": len(dev),
            "total_videos": len(dev),
            "status": "complete_no_QA",
        },
    )
    print(json.dumps(summary, indent=2))


def validate_result(row, plans, source, question, config):
    if set(row["conditions"]) != set(ARMS):
        raise ValueError("Incomplete result conditions")
    for arm in ARMS:
        c = row["conditions"][arm]
        if arm.endswith("_original"):
            if c != source["conditions"][arm[0]]:
                raise ValueError("Original control changed")
        elif any(c[k] != v for k, v in plans[arm].items()):
            raise ValueError("QA selection differs from preparation")
        validate_indices(c["selected_indices"], c["events"][-1][1], 16)
        strict_trials(c["trials"], question, config)


def qa_report(rows, seed):
    means = []
    for row in rows:
        values = {
            arm: float(
                np.mean(
                    [
                        t["scoring_correct"]
                        for t in row["conditions"][arm]["trials"]
                    ]
                )
            )
            for arm in ARMS
        }
        values.update(
            {k: values[a] - values[b] for k, (a, b) in COMPARISONS.items()}
        )
        values["change_in_D_minus_C"] = (
            values["D_minus_C"] - values["original_D_minus_C"]
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
    keys = [*ARMS, *COMPARISONS, "change_in_D_minus_C"]
    summary = {
        "primary_comparison": "D_minus_C",
        "overall": clustered_means(means, keys, seed=seed),
        "by_duration": {
            g: clustered_means(
                [r for r in means if r["duration_group"] == g], keys, seed=seed
            )
            for g in sorted({r["duration_group"] for r in means})
        },
        "paired_question_changes": {
            key: {
                "improved": sum(r[key] > 1e-12 for r in means),
                "degraded": sum(r[key] < -1e-12 for r in means),
                "unchanged": sum(abs(r[key]) <= 1e-12 for r in means),
            }
            for key in COMPARISONS
        },
    }
    return summary, means


def evaluate(args, inputs):
    _, config, manifest, dev, sources, metadata, protocol = inputs
    digest = report_digest(protocol)
    complete = read(args.output_dir / "prepare_complete.json")
    if complete["protocol_sha256"] != digest or set(
        complete["plan_sha256"]
    ) != {v["video_id"] for v in dev}:
        raise ValueError("Complete preparation required before QA")
    prepared_by_video = {}
    # Validate every plan before initializing the QA model.
    for item in dev:
        identity = video_identity(item, sources, digest)
        p = checked_payload(
            plan_path(args.output_dir, item["video_id"]), identity
        )
        if report_digest(p) != complete["plan_sha256"][item["video_id"]]:
            raise ValueError("Prepared plan changed after completion")
        validate_prepared(
            p,
            item,
            sources[
                (item["video_id"], str(item["questions"][0]["question_id"]))
            ],
            config,
        )
        prepared_by_video[item["video_id"]] = p
    model = None

    def get_backend():
        nonlocal model
        if model is None:
            seed_call(config)
            model = build_backend(config)
            validate_metadata(model.metadata(), metadata)
            write_json(args.output_dir / "backend.json", model.metadata())
        return model

    rows = []
    total = sum(len(v["questions"]) for v in dev)
    for item in dev:
        identity = video_identity(item, sources, digest)
        pending = []
        for annotation in item["questions"]:
            q = question_from(annotation)
            source = sources[(item["video_id"], q.question_id)]
            plans = prepared_by_video[item["video_id"]]["questions"][
                q.question_id
            ]
            q_identity = {**identity, "question_id": q.question_id}
            path = (
                args.output_dir
                / "questions"
                / (report_digest([item["video_id"], q.question_id]) + ".json")
            )
            if path.exists():
                row = checked_payload(path, q_identity)
                validate_result(row, plans, source, q, config)
                rows.append(row)
            else:
                pending.append((q, source, plans, q_identity, path))
        if not pending:
            print(f"Reused completed video {item['video_id']}", flush=True)
            continue
        video = decode_item(item, manifest, config, identity)
        for q, source, plans, q_identity, path in pending:
            conditions = {
                f"{a}_original": source["conditions"][a] for a in ("C", "D")
            }
            reusable = {
                tuple(c["selected_indices"]): c["trials"]
                for c in conditions.values()
            }
            for arm in ("C", "D"):
                indices = plans[arm]["selected_indices"]
                trials = reusable.get(tuple(indices))
                if trials is None:
                    trials = []
                    for i, old in enumerate(
                        source["conditions"]["D"]["trials"]
                    ):
                        order = old["order"]
                        trial_id = {
                            **q_identity,
                            "indices": indices,
                            "order": order,
                        }
                        trial, reused = cached_trial(
                            args.output_dir
                            / "trials"
                            / (report_digest(trial_id) + ".json"),
                            trial_id,
                            lambda video=video: measured_score(
                                get_backend(), video, q, indices, order, config
                            ),
                        )
                        trials.append(trial)
                        print(
                            f"{video.video_id}/{q.question_id}/{arm} "
                            f"order {i + 1}/6 cached={reused}",
                            flush=True,
                        )
                strict_trials(trials, q, config)
                reusable[tuple(indices)] = trials
                conditions[arm] = {**plans[arm], "trials": trials}
            row = {
                "video_id": item["video_id"],
                "question_id": q.question_id,
                "candidate_sha256": identity["candidate_sha256"],
                "duration_group": item["duration"],
                "conditions": conditions,
            }
            validate_result(row, plans, source, q, config)
            save_payload(path, q_identity, row)
            rows.append(row)
            write_json(
                args.output_dir / "qa_progress.json",
                {
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "status": "running",
                },
            )
        del video
    rows.sort(key=lambda r: (r["video_id"], r["question_id"]))
    summary, means = qa_report(rows, config.seed)
    write_json(
        args.output_dir / "results.json",
        {
            "protocol_sha256": digest,
            "questions": rows,
            "summary": summary,
        },
    )
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "question_metrics.csv", means)
    write_json(
        args.output_dir / "handoff.json",
        {
            "protocol": protocol,
            "summary": summary,
            "segmentation": read(
                args.output_dir / "segmentation_summary.json"
            ),
            "question_metrics": means,
        },
    )
    (args.output_dir / "summary.md").write_text(
        "# Relaxed segment cap C/D QA\n\n```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "qa_progress.json",
        {
            "completed_questions": total,
            "total_questions": total,
            "status": "complete",
        },
    )
    print(json.dumps(summary, indent=2))


def run(args):
    inputs = load_inputs(args)
    _, _, _, dev, _, _, protocol = inputs
    if args.check_inputs:
        total = sum(len(v["questions"]) for v in dev)
        print(
            json.dumps(
                {
                    "status": "validated_no_inference",
                    "phase": args.phase,
                    "n_videos": len(dev),
                    "n_questions": total,
                    "max_segments": args.max_segments,
                    "maximum_new_QA_calls": total * 2 * 6,
                }
            )
        )
        return
    if args.phase == "evaluate" or args.resume:
        if report_digest(
            read(args.output_dir / "protocol.json")
        ) != report_digest(protocol):
            raise ValueError(
                "Protocol changed; use a separate output directory"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "protocol.json", protocol)
    if args.phase == "prepare":
        prepare(args, inputs)
    else:
        evaluate(args, inputs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("prepare", "evaluate"), required=True)
    p.add_argument("--max-segments", type=int, default=128)
    p.add_argument(
        "--duration",
        choices=("long", "medium", "short", "all"),
        default="long",
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("outputs/videomme_event_factorial"),
    )
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/videomme_dev_frozen"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--video-id")
    p.add_argument("--max-videos", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-inputs", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
