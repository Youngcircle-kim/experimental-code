"""Run four fixed-budget D refinements, with resumable per-order scoring."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .backends import build_backend
from .bottleneck_diagnostics import read, validate_plan, write_csv
from .cache import write_json
from .config import Config
from .data import array_digest, validate_video
from .event_factorial_dev import cached_trial, measured_score, validate_indices
from .event_refinement import ARMS, make_plans, summarize
from .frame_replacement_probe import FROZEN_CODE, strict_trials
from .midpoint_review import load_candidates
from .order_experiments import seed_call
from .qa_diagnostics import report_digest
from .types import Question, Video


def question_from(annotation):
    return Question(
        str(annotation["question_id"]),
        annotation["text"],
        tuple(annotation["options"]),
        annotation["answer_index"],
    )


def load_inputs(args):
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if (
        source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be separate from the source directory")
    if args.top_events < 1 or args.min_gap_seconds < 0:
        raise ValueError("Require positive top-events and nonnegative gap")
    if not math.isfinite(args.min_gap_seconds):
        raise ValueError("Gap must be finite")
    original = read(source / "protocol.json")
    result = read(source / "results.json")
    if result["protocol_sha256"] != report_digest(original):
        raise ValueError("Source results/protocol mismatch")
    root = Path(__file__).parent
    for name in FROZEN_CODE:
        if (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            != original["code_sha256"][name]
        ):
            raise ValueError(f"Frozen QA/decode code changed: {name}")
    settings = dict(original["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    if (
        config.frame_budget != 16
        or config.mode != "real"
        or config.backend != "transformers"
        or config.pilot != "both"
        or config.vlm_revision == "main"
        or config.encoder_revision == "main"
    ):
        raise ValueError("Require pinned real backend and original 16 frames")
    manifest_path = Path(config.manifest_path).resolve()
    manifest = read(manifest_path)
    baseline = read(args.baseline_dir / "protocol.json")
    if (
        report_digest(baseline) != original["baseline_protocol_sha256"]
        or report_digest(manifest) != baseline["manifest_sha256"]
    ):
        raise ValueError("Original baseline protocol/manifest changed")
    metadata = read(source / "backend.json")
    if report_digest(metadata) != original["backend_metadata_sha256"]:
        raise ValueError("Source backend metadata changed")
    lookup = {
        (r["video_id"], r["question_id"]): r for r in result["questions"]
    }
    if len(lookup) != len(result["questions"]):
        raise ValueError("Duplicate source questions")
    dev = [
        v for v in manifest["videos"] if v["video_id"] in original["video_ids"]
    ]
    if {v["video_id"] for v in dev} != set(original["video_ids"]):
        raise ValueError("Missing source videos")
    expected = {
        (v["video_id"], str(q["question_id"]))
        for v in dev
        for q in v["questions"]
    }
    if set(lookup) != expected:
        raise ValueError("Incomplete source results")
    if args.video_id:
        dev = [v for v in dev if v["video_id"] == args.video_id]
    if args.max_videos is not None:
        if args.max_videos < 1:
            raise ValueError("max-videos must be positive")
        dev = dev[: args.max_videos]
    if not dev:
        raise ValueError("No matching videos")
    for item in dev:
        if item["split"] != "dev":
            raise ValueError("Only development videos are allowed")
        if not (manifest_path.parent / item["path"]).is_file():
            raise ValueError(f"Missing video: {item['path']}")
        prepared = read(
            source / "plans" / (report_digest(item["video_id"]) + ".json")
        )
        for annotation in item["questions"]:
            q = question_from(annotation)
            row = lookup[(item["video_id"], q.question_id)]
            if prepared["identity"] != {
                "video_id": item["video_id"],
                "protocol_sha256": report_digest(original),
                "candidate_sha256": row["candidate_sha256"],
            }:
                raise ValueError("Source plan identity mismatch")
            plan = prepared["questions"][q.question_id]["D"]
            if any(row["conditions"]["D"][k] != v for k, v in plan.items()):
                raise ValueError("Source D plan/results mismatch")
            validate_plan(plan, config.frame_budget, "question_relevance")
            strict_trials(row["conditions"]["D"]["trials"], q, config)
    protocol = {
        "version": 1,
        "source_dir": str(source),
        "source_results_sha256": report_digest(result),
        "source_protocol_sha256": report_digest(original),
        "manifest_sha256": report_digest(manifest),
        "backend_metadata_sha256": report_digest(metadata),
        "config": config.to_dict(),
        "video_ids": [v["video_id"] for v in dev],
        "arms": list(ARMS),
        "top_events": args.top_events,
        "min_gap_seconds": args.min_gap_seconds,
        "partition": "Exact saved D2 event boundaries",
        "allocation": "Mask to top events by saved D pooled question cosine; "
        "original temperature/allocator; expand mask only for capacity",
        "within": "Frame-question cosine top-k; earlier index breaks ties; "
        "optional greedy time gap with ranked fill fallback",
        "selection_inputs": "Images, timestamps, question text only",
        "baseline": "Reuse verified original D QA; same-input trials reused",
        "primary_comparison": "focus_topk minus D",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))
        },
    }
    return config, manifest_path, dev, lookup, metadata, protocol


def checked_payload(path, identity):
    record = read(path)
    if record["identity"] != identity or record[
        "payload_sha256"
    ] != report_digest(record["payload"]):
        raise ValueError(f"Checkpoint mismatch: {path}")
    return record["payload"]


def save_payload(path, identity, payload):
    write_json(
        path,
        {
            "identity": identity,
            "payload": payload,
            "payload_sha256": report_digest(payload),
        },
    )


def validate_row(row, source, question, config):
    if (
        row["candidate_sha256"] != source["candidate_sha256"]
        or row["video_id"] != source["video_id"]
        or row["question_id"] != question.question_id
        or set(row["conditions"]) != set(ARMS)
    ):
        raise ValueError("Question checkpoint identity differs")
    original = source["conditions"]["D"]
    for arm, condition in row["conditions"].items():
        indices = condition["selected_indices"]
        validate_indices(indices, original["events"][-1][1], 16)
        if condition["events"] != original["events"]:
            raise ValueError("Event boundaries changed")
        counts = [
            sum(a <= i < b for i in indices) for a, b in original["events"]
        ]
        if counts != condition["allocation"]:
            raise ValueError("Selection/allocation mismatch")
        strict_trials(condition["trials"], question, config)
        if arm in ("D", "D_topk") and counts != original["allocation"]:
            raise ValueError("Original allocation changed")
    if (
        row["conditions"]["D"]["selected_indices"]
        != original["selected_indices"]
        or row["conditions"]["D"]["trials"] != original["trials"]
    ):
        raise ValueError("Saved D control changed")
    if (
        row["conditions"]["focus_uniform"]["allocation"]
        != row["conditions"]["focus_topk"]["allocation"]
    ):
        raise ValueError("Focused allocation differs between arms")


def write_reports(output, rows, protocol, seed):
    summary, means = summarize(rows, seed)
    result = {
        "protocol_sha256": report_digest(protocol),
        "questions": rows,
        "summary": summary,
    }
    write_json(output / "results.json", result)
    write_json(output / "summary.json", summary)
    write_csv(output / "question_metrics.csv", means)
    selections = []
    for row in rows:
        for arm, c in row["conditions"].items():
            selections.append(
                {
                    "video_id": row["video_id"],
                    "question_id": row["question_id"],
                    "condition": arm,
                    "allocation": c["allocation"],
                    "selected_indices": c["selected_indices"],
                    "selected_timestamps": c["selected_timestamps"],
                    **c["diagnostics"],
                }
            )
    write_csv(output / "selections.csv", selections)
    lines = [
        "# Event allocation x within-event top-k",
        "",
        "Exploratory dev; same 16 frames and paired option orders.",
        "",
        "| Condition | Order-mean accuracy |",
        "|---|---:|",
    ]
    for arm in ARMS:
        value = summary["overall"]["metrics"][arm]["mean"]
        lines.append(f"| {arm} | {value:.2%} |")
    lines += ["", "```json", json.dumps(summary, indent=2), "```"]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "handoff.json",
        {
            "protocol": protocol,
            "summary": summary,
            "question_metrics": means,
            "known_case_856_1_posthoc": [
                {
                    "condition": s["condition"],
                    "selected_indices": s["selected_indices"],
                    "selected_human_evidence_1035_1036": sorted(
                        set(s["selected_indices"]) & {1035, 1036}
                    ),
                }
                for s in selections
                if s["video_id"] == "856" and s["question_id"] == "856-1"
            ],
        },
    )
    return summary


def run(args):
    config, manifest_path, dev, sources, metadata, protocol = load_inputs(args)
    total = sum(len(v["questions"]) for v in dev)
    if args.check_inputs:
        print(
            json.dumps(
                {
                    "status": "validated_no_inference",
                    "n_videos": len(dev),
                    "n_questions": total,
                    "maximum_new_qa_calls": total * 3 * 6,
                    "arms": list(ARMS),
                    "top_events": args.top_events,
                    "min_gap_seconds": args.min_gap_seconds,
                }
            )
        )
        return
    output = args.output_dir
    digest = report_digest(protocol)
    if args.resume:
        if read(output / "protocol.json") != protocol:
            raise ValueError("Resume protocol changed; use a new directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", protocol)
    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            seed_call(config)
            backend = build_backend(config)
            if backend.metadata() != metadata:
                raise ValueError("Backend differs from original experiment")
            write_json(output / "backend.json", metadata)
        return backend

    rows = []

    def progress():
        write_json(
            output / "progress.json",
            {
                "completed_questions": len(rows),
                "total_questions": total,
                "status": "running",
            },
        )

    for item in dev:
        pending = []
        for annotation in item["questions"]:
            q = question_from(annotation)
            source = sources[(item["video_id"], q.question_id)]
            identity = {
                "protocol_sha256": digest,
                "video_id": item["video_id"],
                "question_id": q.question_id,
                "candidate_sha256": source["candidate_sha256"],
            }
            path = (
                output
                / "questions"
                / (report_digest([item["video_id"], q.question_id]) + ".json")
            )
            if path.exists():
                row = checked_payload(path, identity)
                validate_row(row, source, q, config)
                rows.append(row)
                progress()
            else:
                pending.append((q, source, identity, path))
        if not pending:
            print(f"Reused completed video {item['video_id']}", flush=True)
            continue
        print(f"Decoding video {item['video_id']}", flush=True)
        frames, times, duration = load_candidates(item, manifest_path, config)
        video = Video(
            item["video_id"],
            "dev",
            frames,
            times,
            duration,
            tuple(p[0] for p in pending),
            item["source_id"],
        )
        validate_video(video, config)
        content = array_digest(frames, times)
        features = None
        for q, source, identity, path in pending:
            if content != source["candidate_sha256"]:
                raise ValueError("Decoded candidate content changed")
            plan_path = output / "plans" / (report_digest(identity) + ".json")
            if plan_path.exists():
                prepared = checked_payload(plan_path, identity)
            else:
                model = get_backend()
                if features is None:
                    print("Encoding candidate frames", flush=True)
                    seed_call(config)
                    features = model.encode_frames(frames)
                prepared = make_plans(
                    features,
                    model.encode_visual_question(q.text),
                    times,
                    source["conditions"]["D"],
                    config,
                    args.top_events,
                    args.min_gap_seconds,
                )
                save_payload(plan_path, identity, prepared)
            original = source["conditions"]["D"]
            reusable = {
                tuple(original["selected_indices"]): original["trials"]
            }
            conditions = {}
            for arm in ARMS:
                plan = prepared["conditions"][arm]
                indices = plan["selected_indices"]
                validate_indices(indices, len(frames), config.frame_budget)
                trials = reusable.get(tuple(indices))
                if trials is None:
                    trials = []
                    for number, saved in enumerate(original["trials"]):
                        order = saved["order"]
                        trial_id = {
                            **identity,
                            "indices": indices,
                            "order": order,
                        }
                        trial, reused = cached_trial(
                            output
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
                            f"order {number + 1}/6 cached={reused}",
                            flush=True,
                        )
                strict_trials(trials, q, config)
                reusable[tuple(indices)] = trials
                conditions[arm] = {**plan, "trials": trials}
            row = {
                "video_id": video.video_id,
                "question_id": q.question_id,
                "duration_group": source["duration_group"],
                "question_type": source["question_type"],
                "candidate_sha256": content,
                "conditions": conditions,
            }
            validate_row(row, source, q, config)
            save_payload(path, identity, row)
            rows.append(row)
            progress()
            print(f"Completed questions {len(rows)}/{total}", flush=True)
        del video, frames, times, features
    rows.sort(key=lambda r: (r["video_id"], r["question_id"]))
    summary = write_reports(output, rows, protocol, config.seed)
    write_json(
        output / "progress.json",
        {
            "completed_questions": len(rows),
            "total_questions": total,
            "status": "complete",
        },
    )
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--top-events", type=int, default=8)
    p.add_argument("--min-gap-seconds", type=float, default=0.0)
    p.add_argument("--max-videos", type=int)
    p.add_argument("--video-id")
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
