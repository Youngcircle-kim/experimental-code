"""One GPU's resumable, frozen-source Long B32 factorial ablation."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .aggregate_budget_shards import rooted
from .algorithms import allocate_frames, normalize_rows, sample_indices
from .backends import build_backend
from .compare_budgets import (
    normalized_backend,
    normalized_runtime,
    validate_backend,
)
from .config import Config
from .data import array_digest, decode_video, validate_video
from .event_factorial_dev import validate_indices
from .event_refinement_dev import question_from
from .frame_replacement_probe import strict_trials
from .long_ablation_analysis import CONDITIONS, make_conditions
from .long_ablation_source import read_frozen
from .order_experiments import option_orders, seed_call
from .portable_budget import (
    checkpoint,
    file_identity,
    output_lock,
    read_json,
    runtime_metadata,
    save_json,
)
from .qa_diagnostics import report_digest
from .resolution_dev import (
    decode_selected,
    score_with_memory,
    validate_processor,
)
from .types import Video

QA_SIDE = 320


def job_directory(root, shard_index):
    root = Path(root).resolve()
    if type(shard_index) is not int or shard_index < 0:
        raise ValueError("Invalid Long ablation shard index")
    result = (root / "jobs" / f"dev_long_b32_s{shard_index}").resolve()
    if root not in result.parents:
        raise ValueError("Worker directory must stay within output root")
    return result


def _shard(plan, index):
    matches = [s for s in plan["shards"] if s["index"] == index]
    if type(index) is not int or len(matches) != 1:
        raise ValueError("Unknown Long ablation shard")
    return matches[0]


def worker_protocol(plan, shard_index):
    shard = _shard(plan, shard_index)
    return {
        "version": 1,
        "experiment": "a4500_long_ablation_worker_v1",
        "suite_plan_sha256": report_digest(plan),
        "snapshot_sha256": plan["snapshot_sha256"],
        "shard_index": shard_index,
        "gpu": shard["gpu"],
        "video_ids": shard["video_ids"],
        "conditions": list(CONDITIONS),
        "frame_budget": 32,
        "qa_side": QA_SIDE,
        "split": "dev",
        "duration": "long",
    }


def _config(snapshot):
    settings = dict(snapshot["config"])
    if "observation_strides" in settings:
        settings["observation_strides"] = tuple(
            settings["observation_strides"]
        )
    config = Config(**settings)
    config.validate()
    if (
        config.frame_budget != 32
        or config.vlm_min_pixels != QA_SIDE**2
        or config.vlm_max_pixels != QA_SIDE**2
    ):
        raise ValueError("The frozen Long ablation requires B32 and 320 QA")
    return config


def _videos(snapshot, shard):
    by_id = {v["video_id"]: v for v in snapshot["videos"]}
    if len(by_id) != len(snapshot["videos"]):
        raise ValueError("Duplicate frozen video IDs")
    return [by_id[vid] for vid in shard["video_ids"]]


def _baseline_rows(snapshot):
    rows = {
        (r["video_id"], r["question_id"]): r for r in snapshot["baseline_rows"]
    }
    if len(rows) != len(snapshot["baseline_rows"]):
        raise ValueError("Duplicate frozen baseline questions")
    return rows


def _question_path(output, vid, qid):
    return rooted(output, "questions", report_digest([vid, qid]) + ".json")


def _plan_path(output, vid):
    return rooted(output, "plans", report_digest(vid) + ".json")


def _trial_path(output, identity):
    return rooted(output, "trials", report_digest(identity) + ".json")


def _question_identity(digest, vid, qid):
    return {"protocol_sha256": digest, "video_id": vid, "question_id": qid}


def _plan_identity(digest, vid, candidate_hash):
    return {
        "protocol_sha256": digest,
        "video_id": vid,
        "candidate_sha256": candidate_hash,
    }


def _trial_identity(digest, vid, qid, name, indices, order, pixel_hash):
    return {
        "protocol_sha256": digest,
        "video_id": vid,
        "question_id": qid,
        "condition": name,
        "indices": indices,
        "order": list(order),
        "qa_pixels_sha256": pixel_hash,
    }


def _orders(baseline, question, config):
    orders = [
        trial["order"] for trial in baseline["conditions"]["D"]["trials"]
    ]
    expected = [
        list(order)
        for order in option_orders(len(question.options), config.seed)
    ]
    if len(orders) != 6 or orders != expected:
        raise ValueError(
            "Source D must have the same six frozen option orders"
        )
    return orders


def _validate_backend_identity(metadata, snapshot, config):
    validate_backend(metadata, config)
    if normalized_backend(metadata) != snapshot["backend"]:
        raise ValueError(
            "Backend/model metadata differs from the frozen baseline"
        )
    if metadata.get("device") != "cuda:0" or any(
        metadata.get(key, {}).get("devices") != ["cuda:0"]
        for key in ("encoder_parameters", "vlm_parameters")
    ):
        raise ValueError("Each worker must place both frozen models on cuda:0")


def _validate_runtime(runtime, snapshot, shard):
    if runtime.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != str(
        shard["gpu"]
    ):
        raise ValueError("Worker GPU mask differs from its frozen shard")
    if normalized_runtime(runtime) != snapshot["runtime"]:
        raise ValueError(
            "Runtime environment differs from the frozen baseline"
        )


def _validate_selection(selection, frozen_d, name, times, config):
    indices = selection["selected_indices"]
    validate_indices(indices, len(times), config.frame_budget)
    if selection["selected_timestamps"] != times[indices].tolist():
        raise ValueError("Selected timestamps differ from the candidate grid")
    events = frozen_d["events"]
    capacities = [b - a for a, b in events]
    allocation = (
        frozen_d["allocation"]
        if name == "D_topk"
        else allocate_frames(
            np.log(np.asarray(capacities, dtype=float)),
            capacities,
            config.frame_budget,
            1.0,
        ).tolist()
    )
    if (
        selection["events"] != events
        or selection["capacities"] != capacities
        or selection["allocation"] != allocation
    ):
        raise ValueError(
            "Frozen D2 boundaries or factorial allocations changed"
        )
    actual = [sum(a <= i < b for i in indices) for a, b in events]
    if actual != allocation:
        raise ValueError("Selection does not match its per-event allocation")
    if name == "B":
        expected = [
            int(i)
            for (start, stop), count in zip(events, allocation)
            for i in sample_indices(start, stop, count)
        ]
        if indices != expected:
            raise ValueError("B must use the original uniform event sampler")


def _validate_video_plan(saved, item, snapshot, config):
    vid = item["video_id"]
    frozen = snapshot["baseline_plans"][vid]
    for key in ("candidate_sha256", "candidate_count", "duration_seconds"):
        if saved[key] != frozen[key]:
            raise ValueError("Candidate plan differs from the frozen baseline")
    times = np.asarray(saved["candidate_timestamps"], dtype=float)
    if (
        times.shape != (saved["candidate_count"],)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not 0 <= times[0] < times[-1] < saved["duration_seconds"]
    ):
        raise ValueError("Invalid saved candidate timestamps")
    if set(saved["questions"]) != {
        str(q["question_id"]) for q in item["questions"]
    }:
        raise ValueError(
            "Saved selection plan has a different question cohort"
        )
    for annotation in item["questions"]:
        qid = str(annotation["question_id"])
        frozen_d = frozen["questions"][qid]["D"]
        if (
            times[frozen_d["selected_indices"]].tolist()
            != frozen_d["selected_timestamps"]
        ):
            raise ValueError("Candidate grid differs from frozen D timestamps")
        selections = saved["questions"][qid]
        if set(selections) != set(CONDITIONS):
            raise ValueError("Saved selection plan has different conditions")
        for name, selection in selections.items():
            _validate_selection(selection, frozen_d, name, times, config)
    return times


def _validate_completed_row(
    row, item, question, config, digest, output, snapshot, baseline
):
    expected = {
        "video_id": item["video_id"],
        "source_id": item["source_id"],
        "question_id": question.question_id,
        "duration_group": "long",
        "question": question.text,
        "options": list(question.options),
        "answer_index_evaluator_only": question.answer_index,
        "candidate_sha256": baseline["candidate_sha256"],
        "candidate_count": baseline["candidate_count"],
        "verified_D_qa_pixels_sha256": baseline["conditions"]["D"][
            "qa_pixels_sha256"
        ],
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Completed question differs from frozen source identity/pixels"
        )
    if set(row["conditions"]) != set(CONDITIONS) or sorted(
        row["execution_order"]
    ) != sorted(CONDITIONS):
        raise ValueError(
            "Completed question conditions differ from the protocol"
        )
    saved = checkpoint(
        _plan_path(output, item["video_id"]),
        _plan_identity(digest, item["video_id"], row["candidate_sha256"]),
    )
    _validate_video_plan(saved, item, snapshot, config)
    orders = _orders(baseline, question, config)
    for name in CONDITIONS:
        condition = row["conditions"][name]
        selection = saved["questions"][question.question_id][name]
        if any(
            condition.get(key) != value for key, value in selection.items()
        ):
            raise ValueError(
                "Completed question selection differs from its plan"
            )
        pixel_hash = condition["qa_pixels_sha256"]
        if not isinstance(pixel_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", pixel_hash
        ):
            raise ValueError("Invalid QA pixels digest")
        trials = condition["trials"]
        strict_trials(trials, question, config)
        if [t["order"] for t in trials] != orders:
            raise ValueError("New QA option orders differ from frozen D")
        for trial in trials:
            validate_processor(trial, QA_SIDE, config.frame_budget)
            identity = _trial_identity(
                digest,
                item["video_id"],
                question.question_id,
                name,
                condition["selected_indices"],
                trial["order"],
                pixel_hash,
            )
            recorded = checkpoint(_trial_path(output, identity), identity)
            if recorded != trial:
                raise ValueError("Completed row/trial checkpoint mismatch")


def validate_worker(root, plan, snapshot, shard_index):
    """Validate a complete shard without opening videos or loading a model."""
    shard = _shard(plan, shard_index)
    output = job_directory(root, shard_index)
    protocol = worker_protocol(plan, shard_index)
    digest = report_digest(protocol)
    if read_json(rooted(output, "protocol.json")) != protocol:
        raise ValueError("Worker protocol differs from the frozen suite")
    config = _config(snapshot)
    _validate_runtime(
        read_json(rooted(output, "runtime.json")), snapshot, shard
    )
    metadata = read_json(rooted(output, "backend.json"))
    _validate_backend_identity(metadata, snapshot, config)
    if (
        checkpoint(
            rooted(output, "backend_identity.json"),
            {"protocol_sha256": digest},
        )
        != metadata
    ):
        raise ValueError("Worker backend checkpoint mismatch")
    videos = _videos(snapshot, shard)
    baselines = _baseline_rows(snapshot)
    results = read_json(rooted(output, "results.json"))
    total = sum(len(item["questions"]) for item in videos)
    if (
        results.get("completed") is not True
        or results.get("protocol_sha256") != digest
        or results.get("config") != snapshot["config"]
        or results.get("expected_new_qa_trials") != total * 6 * len(CONDITIONS)
    ):
        raise ValueError("Worker is incomplete or has incompatible results")
    rows = results["questions"]
    indexed = {(r["video_id"], r["question_id"]): r for r in rows}
    expected = {
        (v["video_id"], str(q["question_id"]))
        for v in videos
        for q in v["questions"]
    }
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError(
            "Worker results do not cover their exact question cohort"
        )
    checked = []
    for item in videos:
        for annotation in item["questions"]:
            question = question_from(annotation)
            key = (item["video_id"], question.question_id)
            row = indexed[key]
            recorded = checkpoint(
                _question_path(output, *key), _question_identity(digest, *key)
            )
            if recorded != row:
                raise ValueError("Worker results/question checkpoint mismatch")
            _validate_completed_row(
                row,
                item,
                question,
                config,
                digest,
                output,
                snapshot,
                baselines[key],
            )
            checked.append(row)
    progress = read_json(rooted(output, "progress.json"))
    if progress != {
        "status": "complete",
        "completed_questions": total,
        "total_questions": total,
        "protocol_sha256": digest,
    }:
        raise ValueError("Worker completion marker differs from its results")
    return checked


def run(args):
    plan_path = Path(args.plan).resolve()
    plan, snapshot = read_frozen(
        plan_path,
        getattr(args, "expected_plan_sha256", None),
        verify_code=True,
    )
    shard = _shard(plan, args.shard_index)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(shard["gpu"]):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must equal this worker's frozen GPU"
        )
    config = _config(snapshot)
    videos = _videos(snapshot, shard)
    if not videos:
        raise ValueError("Cannot launch an empty Long ablation shard")
    total = sum(len(item["questions"]) for item in videos)
    protocol = worker_protocol(plan, args.shard_index)
    digest = report_digest(protocol)
    output = job_directory(plan_path.parent, args.shard_index)
    baselines = _baseline_rows(snapshot)
    rooted(output, ".run.lock")
    with output_lock(output):
        protocol_path = rooted(output, "protocol.json")
        if protocol_path.exists():
            if read_json(protocol_path) != protocol:
                raise ValueError(
                    "Worker protocol changed; use a new output directory"
                )
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError(
                "Worker output contains files without a frozen protocol"
            )
        else:
            save_json(protocol_path, protocol)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
        runtime = runtime_metadata()
        _validate_runtime(runtime, snapshot, shard)
        runtime_path = rooted(output, "runtime.json")
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError("Worker runtime changed during resume")
        save_json(runtime_path, runtime)
        result_path = rooted(output, "results.json")
        progress_path = rooted(output, "progress.json")
        if (
            result_path.exists()
            and read_json(result_path).get("completed") is True
            and progress_path.exists()
            and read_json(progress_path).get("status") == "complete"
        ):
            validate_worker(plan_path.parent, plan, snapshot, args.shard_index)
            return read_json(result_path)
        metadata_path = rooted(output, "backend.json")
        backend_identity_path = rooted(output, "backend_identity.json")
        backend_identity = {"protocol_sha256": digest}
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            _validate_backend_identity(metadata, snapshot, config)
            if checkpoint(backend_identity_path, backend_identity) != metadata:
                raise ValueError("Worker backend metadata/checkpoint mismatch")
        model = None

        def get_model():
            nonlocal model
            if model is None:
                seed_call(config)
                model = build_backend(config)
                metadata = model.metadata()
                _validate_backend_identity(metadata, snapshot, config)
                if (
                    metadata_path.exists()
                    and read_json(metadata_path) != metadata
                ):
                    raise ValueError("Backend metadata changed during resume")
                recorded = checkpoint(
                    backend_identity_path, backend_identity, lambda: metadata
                )
                if recorded != metadata:
                    raise ValueError("Backend identity changed during resume")
                save_json(metadata_path, metadata)
            return model

        rows = []
        save_json(
            progress_path,
            {
                "status": "running",
                "completed_questions": 0,
                "total_questions": total,
                "protocol_sha256": digest,
            },
        )
        try:
            for item in videos:
                vid = item["video_id"]
                questions = tuple(question_from(q) for q in item["questions"])
                pending = [
                    q
                    for q in questions
                    if not _question_path(output, vid, q.question_id).exists()
                ]
                if pending:
                    source_identity = snapshot["inputs"][vid]
                    source_path = Path(source_identity["path"])
                    current_identity = file_identity(source_path)
                    if any(
                        current_identity[key] != source_identity[key]
                        for key in ("path", "size_bytes", "sha256")
                    ):
                        raise ValueError(f"Frozen source video changed: {vid}")
                    started = perf_counter()
                    frames, times, duration = decode_video(source_path, config)
                    video = Video(
                        vid,
                        "dev",
                        frames,
                        times,
                        duration,
                        questions,
                        item["source_id"],
                    )
                    validate_video(video, config)
                    candidate_hash = array_digest(frames, times)
                    frozen_plan = snapshot["baseline_plans"][vid]
                    if (
                        candidate_hash != frozen_plan["candidate_sha256"]
                        or len(times) != frozen_plan["candidate_count"]
                        or duration != frozen_plan["duration_seconds"]
                    ):
                        raise ValueError(
                            "Decoded candidates differ from the frozen "
                            f"baseline: {vid}"
                        )

                    def prepare():
                        backend = get_model()
                        features = normalize_rows(
                            backend.encode_frames(frames),
                            config.normalization_epsilon,
                        )
                        selections = {}
                        for question in questions:
                            selections[question.question_id] = make_conditions(
                                features,
                                backend.encode_visual_question(question.text),
                                times,
                                frozen_plan["questions"][question.question_id][
                                    "D"
                                ],
                                config,
                            )
                        return {
                            "candidate_sha256": candidate_hash,
                            "candidate_count": len(times),
                            "candidate_timestamps": times.tolist(),
                            "duration_seconds": duration,
                            "questions": selections,
                            "prepare_wall_seconds": perf_counter() - started,
                        }

                    selection_plan = checkpoint(
                        _plan_path(output, vid),
                        _plan_identity(digest, vid, candidate_hash),
                        prepare,
                    )
                    _validate_video_plan(
                        selection_plan, item, snapshot, config
                    )
                    if (
                        selection_plan["candidate_timestamps"]
                        != times.tolist()
                    ):
                        raise ValueError(
                            "Decoded candidate timestamps changed "
                            "during resume"
                        )
                    del video, frames
                    for question in pending:
                        baseline = baselines[(vid, question.question_id)]
                        selections = selection_plan["questions"][
                            question.question_id
                        ]
                        frozen_d = baseline["conditions"]["D"]
                        union = sorted(
                            {
                                i
                                for selection in list(selections.values())
                                + [frozen_d]
                                for i in selection["selected_indices"]
                            }
                        )
                        (
                            low,
                            pixels,
                            replay_times,
                            replay_duration,
                            decode_info,
                        ) = decode_selected(
                            source_path,
                            config,
                            len(times),
                            union,
                            QA_SIDE,
                            candidate_hash,
                        )
                        del low
                        if (
                            replay_times.tolist() != times.tolist()
                            or replay_duration != duration
                        ):
                            raise ValueError(
                                "QA replay differs from the frozen "
                                "candidate grid"
                            )
                        original_indices = frozen_d["selected_indices"]
                        original_pixels = np.stack(
                            [pixels[i] for i in original_indices]
                        )
                        original_hash = array_digest(
                            original_pixels, replay_times[original_indices]
                        )
                        del original_pixels
                        if original_hash != frozen_d["qa_pixels_sha256"]:
                            raise ValueError(
                                "Original D QA pixels changed "
                                "at 320 resolution"
                            )
                        conditions = {}
                        execution = list(CONDITIONS)
                        if config.randomize_condition_order:
                            np.random.default_rng(
                                config.seed
                                + int(
                                    report_digest([vid, question.question_id])[
                                        :8
                                    ],
                                    16,
                                )
                            ).shuffle(execution)
                        for name in execution:
                            selected = selections[name]
                            indices = selected["selected_indices"]
                            qa_frames = np.stack([pixels[i] for i in indices])
                            qa_times = replay_times[indices]
                            qa_video = Video(
                                vid,
                                "dev",
                                qa_frames,
                                qa_times,
                                replay_duration,
                                (),
                                item["source_id"],
                            )
                            pixel_hash = array_digest(qa_frames, qa_times)
                            trials = []
                            for order in _orders(baseline, question, config):
                                trial_id = _trial_identity(
                                    digest,
                                    vid,
                                    question.question_id,
                                    name,
                                    indices,
                                    order,
                                    pixel_hash,
                                )
                                trial = checkpoint(
                                    _trial_path(output, trial_id),
                                    trial_id,
                                    lambda: score_with_memory(
                                        get_model(),
                                        qa_video,
                                        question,
                                        order,
                                        config,
                                        QA_SIDE,
                                    ),
                                )
                                validate_processor(
                                    trial, QA_SIDE, config.frame_budget
                                )
                                trials.append(trial)
                            strict_trials(trials, question, config)
                            conditions[name] = {
                                **selected,
                                "qa_pixels_sha256": pixel_hash,
                                "trials": trials,
                            }
                            del qa_video, qa_frames
                        del pixels
                        row = {
                            "video_id": vid,
                            "source_id": item["source_id"],
                            "question_id": question.question_id,
                            "duration_group": "long",
                            "candidate_sha256": candidate_hash,
                            "candidate_count": len(times),
                            "question": question.text,
                            "options": list(question.options),
                            "answer_index_evaluator_only": (
                                question.answer_index
                            ),
                            "execution_order": execution,
                            "decode_info": decode_info,
                            "verified_D_qa_pixels_sha256": original_hash,
                            "conditions": {
                                name: conditions[name] for name in CONDITIONS
                            },
                        }
                        checkpoint(
                            _question_path(output, vid, question.question_id),
                            _question_identity(
                                digest, vid, question.question_id
                            ),
                            lambda: row,
                        )
                        print(
                            f"Saved Long B32 {vid}/{question.question_id} "
                            "(3 new conditions, 18 QA trials)",
                            flush=True,
                        )
                for question in questions:
                    row = checkpoint(
                        _question_path(output, vid, question.question_id),
                        _question_identity(digest, vid, question.question_id),
                    )
                    _validate_completed_row(
                        row,
                        item,
                        question,
                        config,
                        digest,
                        output,
                        snapshot,
                        baselines[(vid, question.question_id)],
                    )
                    rows.append(row)
                save_json(
                    progress_path,
                    {
                        "status": "running",
                        "completed_questions": len(rows),
                        "total_questions": total,
                        "protocol_sha256": digest,
                    },
                )
            results = {
                "protocol_sha256": digest,
                "completed": True,
                "config": snapshot["config"],
                "expected_new_qa_trials": total * 6 * len(CONDITIONS),
                "questions": rows,
            }
            save_json(result_path, results)
            save_json(
                progress_path,
                {
                    "status": "complete",
                    "completed_questions": total,
                    "total_questions": total,
                    "protocol_sha256": digest,
                },
            )
            validate_worker(plan_path.parent, plan, snapshot, args.shard_index)
            return results
        except BaseException as error:
            save_json(
                progress_path,
                {
                    "status": "failed",
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "protocol_sha256": digest,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            if result_path.exists():
                save_json(
                    result_path,
                    {
                        "protocol_sha256": digest,
                        "completed": False,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--shard-index", type=int, required=True)
    result.add_argument("--expected-plan-sha256", required=True)
    return result


def main(argv=None):
    result = run(parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": "complete",
                "n_questions": len(result["questions"]),
                "expected_new_qa_trials": result["expected_new_qa_trials"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
