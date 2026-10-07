"""Streaming, per-order resumable A/B/C/D evaluation under frozen QA."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .ablation import select_frame_control
from .backends import build_backend
from .cache import write_json
from .config import Config
from .data import array_digest, decode_video, validate_video
from .event_factorial import ARMS, make_partitions, select_plans, summarize
from .frozen_dev import check_manifest
from .order_experiments import option_orders, score_call, seed_call
from .qa_diagnostics import report_digest
from .selection_dev import validate_baseline
from .types import Question, Video


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_questions(rows):
    result = {(r["video_id"], r["question_id"]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate source questions")
    return result


def validate_trials(trials, question, config):
    orders = option_orders(len(question.options), config.seed)
    if [t["order"] for t in trials] != [list(o) for o in orders]:
        raise ValueError("Option order mismatch")
    for t, order in zip(trials, orders):
        predicted = t["scoring"]["predicted_index"]
        scores = np.asarray(t["scoring"]["option_scores"], dtype=float)
        if (
            type(predicted) is not int
            or not 0 <= predicted < len(order)
            or scores.shape != (len(order),)
            or not np.isfinite(scores).all()
            or t["presented_options"] != [question.options[i] for i in order]
            or t["scored_original_index"] != order[predicted]
            or type(t["scoring_correct"]) is not bool
            or t["scoring_correct"]
            != (order[predicted] == question.answer_index)
        ):
            raise ValueError("Invalid cached scoring trial")


def validate_indices(indices, count, budget):
    if (
        len(indices) != budget
        or any(type(i) is not int for i in indices)
        or indices != sorted(set(indices))
        or not 0 <= indices[0] <= indices[-1] < count
    ):
        raise ValueError("Invalid cached frame indices")


def cached_trial(path, identity, compute):
    """Atomic per-order persistence; identical inputs share a key."""
    if path.exists():
        entry = read_json(path)
        if entry["identity"] != identity:
            raise ValueError("Trial checkpoint identity mismatch")
        return entry["trial"], True
    trial = compute()
    write_json(path, {"identity": identity, "trial": trial})
    return trial, False


def measured_score(backend, video, question, indices, order, config):
    started = perf_counter()
    trial = score_call(backend, video, question, indices, order, config)
    trial["qa_wall_seconds"] = perf_counter() - started
    trial["processor_info"] = dict(getattr(backend, "last_processor_info", {}))
    return trial


def load_inputs(args):
    baseline = read_json(args.baseline_dir / "results.json")
    locked = read_json(args.baseline_dir / "protocol.json")
    settings = dict(locked["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    original_config = Config(**settings)
    original_config.validate()
    manifest_path = Path(original_config.manifest_path).resolve()
    manifest = read_json(manifest_path)
    validate_baseline(original_config, manifest, baseline, locked)
    dev = check_manifest(manifest)
    expected = {
        (v["video_id"], str(q["question_id"]))
        for v in dev
        for q in v["questions"]
    }
    lookup = index_questions(baseline["questions"])
    if set(lookup) != expected:
        raise ValueError("Incomplete baseline")
    for v in dev:
        if v.get("timestamp_mode") != "constant_fps":
            raise ValueError("Require constant-FPS input")
        if not (manifest_path.parent / v["path"]).is_file():
            raise ValueError(f"Missing dev video: {v['path']}")
        for annotation in v["questions"]:
            q = Question(
                str(annotation["question_id"]),
                annotation["text"],
                tuple(annotation["options"]),
                annotation["answer_index"],
            )
            source = lookup[(v["video_id"], q.question_id)]
            if source["question"] != q.text:
                raise ValueError("Baseline question differs")
            validate_trials(source["trials"], q, original_config)
    backend_metadata = read_json(args.baseline_dir / "backend.json")
    selection, selection_source = {}, None
    if args.selection_dir:
        selection_source = read_json(args.selection_dir / "results.json")
        prior = read_json(args.selection_dir / "protocol.json")
        if (
            selection_source["protocol_sha256"] != report_digest(prior)
            or prior["baseline_results_sha256"] != report_digest(baseline)
            or prior["baseline_protocol_sha256"] != report_digest(locked)
            or read_json(args.selection_dir / "backend.json")
            != backend_metadata
        ):
            raise ValueError("Selection source protocol/backend differs")
        for name, digest in prior["code_sha256"].items():
            if (
                hashlib.sha256(
                    (Path(__file__).parent / name).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError(f"Selection code changed: {name}")
        selection = index_questions(selection_source["questions"])
        if set(selection) != expected:
            raise ValueError("Incomplete selection baseline")
    config = (
        original_config
        if args.temperature is None
        else replace(original_config, allocation_temperature=args.temperature)
    )
    config.validate()
    if args.max_videos is not None:
        if args.max_videos < 1:
            raise ValueError("max-videos must be positive")
        dev = dev[: args.max_videos]
    protocol = {
        "version": 1,
        "baseline_sha256": report_digest(baseline),
        "baseline_protocol_sha256": report_digest(locked),
        "backend_metadata_sha256": report_digest(backend_metadata),
        "selection_sha256": report_digest(selection_source),
        "config": config.to_dict(),
        "video_ids": [v["video_id"] for v in dev],
        "arms": ARMS,
        "primary_comparison": "D_minus_C",
        "relevance": "raw cosine; no score calibration; fixed temperature",
        "within_event": "uniform; exact unique budget; chronological",
        "bootstrap": "5000 video clusters; question-weighted order means",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    return (
        config,
        manifest_path,
        dev,
        lookup,
        selection,
        backend_metadata,
        protocol,
    )


def run(args):
    config, manifest_path, dev, baseline, selection, metadata, protocol = (
        load_inputs(args)
    )
    total = sum(len(v["questions"]) for v in dev)
    if args.check_inputs:
        print(
            json.dumps(
                {
                    "status": "inputs_validated_no_inference",
                    "n_videos": len(dev),
                    "n_questions": total,
                    "new_arms": list(ARMS),
                    "allocation_temperature": config.allocation_temperature,
                }
            )
        )
        return
    digest = report_digest(protocol)
    if args.resume:
        if (
            report_digest(read_json(args.output_dir / "protocol.json"))
            != digest
        ):
            raise ValueError("Protocol changed; use a new output directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "protocol.json", protocol)
    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            seed_call(config)  # Must precede CUDA initialization.
            backend = build_backend(config)
            if backend.metadata() != metadata:
                raise ValueError(
                    "Runtime/model metadata differs from baseline"
                )
            write_json(args.output_dir / "backend.json", metadata)
        return backend

    rows, seen_content, video_diagnostics = [], set(), []
    for item in dev:
        started = perf_counter()
        frames, times, duration = decode_video(
            manifest_path.parent / item["path"], config
        )
        decode_seconds = perf_counter() - started
        questions = tuple(
            Question(
                str(q["question_id"]),
                q["text"],
                tuple(q["options"]),
                q["answer_index"],
            )
            for q in item["questions"]
        )
        video = Video(
            item["video_id"],
            "dev",
            frames,
            times,
            duration,
            questions,
            item["source_id"],
        )
        validate_video(video, config)
        content = array_digest(frames, times)
        if content in seen_content:
            raise ValueError("Duplicate decoded video")
        seen_content.add(content)
        for q in questions:
            source = baseline[(video.video_id, q.question_id)]
            uniform = select_frame_control(
                "uniform", np.zeros(len(frames)), config.frame_budget
            ).tolist()
            if (
                source["candidate_sha256"] != content
                or source["selected_indices"] != uniform
            ):
                raise ValueError(
                    "Baseline candidate content/selection differs"
                )
        plan_path = (
            args.output_dir
            / "plans"
            / (report_digest(video.video_id) + ".json")
        )
        identity = {
            "protocol_sha256": digest,
            "candidate_sha256": content,
            "video_id": video.video_id,
        }
        if plan_path.exists():
            prepared = read_json(plan_path)
            if prepared["identity"] != identity:
                raise ValueError("Plan checkpoint mismatch")
        else:
            model = get_backend()
            started = perf_counter()
            features = model.encode_frames(frames)
            partitions = make_partitions(features, config)
            plans = {
                q.question_id: select_plans(
                    features,
                    model.encode_visual_question(q.text),
                    times,
                    partitions,
                    config,
                )
                for q in questions
            }
            prepared = {
                "identity": identity,
                "questions": plans,
                "decode_seconds": decode_seconds,
                "feature_partition_selection_seconds": perf_counter()
                - started,
                "n_events": len(partitions["D2"]),
                "hit_max_segments": len(partitions["D2"])
                == config.max_segments,
                "allocation_varies_across_questions": {
                    a: len({tuple(p[a]["allocation"]) for p in plans.values()})
                    > 1
                    for a in ARMS
                },
            }
            write_json(plan_path, prepared)
            del features
        video_diagnostics.append(
            {k: v for k, v in prepared.items() if k != "questions"}
        )
        for q in questions:
            source = baseline[(video.video_id, q.question_id)]
            conditions = {
                "uniform": {
                    "selected_indices": source["selected_indices"],
                    "trials": source["trials"],
                }
            }
            if selection:
                old = selection[(video.video_id, q.question_id)]
                if old["candidate_sha256"] != content:
                    raise ValueError("Selection candidate mismatch")
                for arm in ("frame_top", "temporal_bin"):
                    entry = old["conditions"][arm]
                    validate_indices(
                        entry["selected_indices"],
                        len(frames),
                        config.frame_budget,
                    )
                    validate_trials(entry["trials"], q, config)
                    conditions[arm] = entry
            reusable = {
                tuple(c["selected_indices"]): c["trials"]
                for c in conditions.values()
            }
            for arm in ARMS:
                plan = prepared["questions"][q.question_id][arm]
                indices = plan["selected_indices"]
                validate_indices(indices, len(frames), config.frame_budget)
                started = perf_counter()
                hits = 0
                if tuple(indices) in reusable:
                    trials = reusable[tuple(indices)]
                    hits = len(trials)
                else:
                    trials = []
                    for order in option_orders(len(q.options), config.seed):
                        trial_id = {
                            **identity,
                            "question_id": q.question_id,
                            "indices": indices,
                            "order": list(order),
                        }
                        path = (
                            args.output_dir
                            / "trials"
                            / (report_digest(trial_id) + ".json")
                        )
                        trial, reused = cached_trial(
                            path,
                            trial_id,
                            lambda video=video: measured_score(
                                get_backend(), video, q, indices, order, config
                            ),
                        )
                        trials.append(trial)
                        hits += int(reused)
                validate_trials(trials, q, config)
                reusable[tuple(indices)] = trials
                conditions[arm] = {
                    **plan,
                    "trials": trials,
                    "reused_orders_this_run": hits,
                    "qa_wall_seconds_this_run": perf_counter() - started,
                }
                print(
                    f"Done {video.video_id}/{q.question_id}/{arm}", flush=True
                )
            row = {
                "video_id": video.video_id,
                "question_id": q.question_id,
                "duration_group": item["duration"],
                "question_type": source["question_type"],
                "candidate_sha256": content,
                "conditions": conditions,
            }
            rows.append(row)
            write_json(
                args.output_dir
                / "questions"
                / (report_digest([video.video_id, q.question_id]) + ".json"),
                row,
            )
            write_json(
                args.output_dir / "progress.json",
                {
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "status": "running",
                    "primary_comparison": "D_minus_C",
                },
            )
        del video, frames, times
    result = {
        "protocol_sha256": digest,
        "questions": rows,
        "video_diagnostics": video_diagnostics,
        "summary": summarize(rows, config.seed),
    }
    write_json(args.output_dir / "results.json", result)
    (args.output_dir / "summary.md").write_text(
        "# Event partition x allocation\n\n"
        "Primary: D-C. Exploratory dev; fixed QA, orders and budget.\n"
        "\n```json\n" + json.dumps(result["summary"], indent=2) + "\n```\n",
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "progress.json",
        {
            "completed_questions": total,
            "total_questions": total,
            "status": "complete",
        },
    )
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Default: locked baseline temperature; never auto-tuned",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Smoke test only; separate output/protocol",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
