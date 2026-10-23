"""Resumable count-controlled event/fixed selection and QA on one GPU shard."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .a4500_segment_count import read_frozen, validate_frozen
from .aggregate_budget_shards import rooted
from .algorithms import normalize_rows
from .backends import build_backend
from .data import array_digest, decode_video, validate_video
from .event_factorial import make_partitions
from .event_refinement_dev import question_from
from .frame_replacement_probe import strict_trials
from .long_ablation_analysis import make_conditions as replay_source_d
from .long_ablation_worker import (
    QA_SIDE,
    _baseline_rows,
    _config,
    _orders,
    _plan_identity,
    _plan_path,
    _question_identity,
    _question_path,
    _shard,
    _trial_identity,
    _trial_path,
    _validate_backend_identity,
    _validate_runtime,
    _videos,
)
from .order_experiments import seed_call
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
from .segment_count_analysis import (
    CONDITIONS,
    condition_spec,
    make_conditions,
    prepare_partitions,
    summarize_selections,
    validate_selection,
)
from .types import Video


def job_directory(root, index):
    if type(index) is not int or index < 0:
        raise ValueError("Invalid segment-count shard index")
    return rooted(root, "jobs", f"dev_long_segment_count_b32_s{index}")


def worker_protocol(plan, index):
    shard = _shard(plan, index)
    return {
        "version": 1,
        "experiment": "a4500_segment_count_worker_v1",
        "suite_plan_sha256": report_digest(plan),
        "snapshot_sha256": plan["snapshot_sha256"],
        "algorithm": plan["algorithm"],
        "shard_index": index,
        "gpu": shard["gpu"],
        "video_ids": shard["video_ids"],
        "conditions": list(CONDITIONS),
        "frame_budget": 32,
        "qa_side": QA_SIDE,
        "split": "dev",
        "duration": "long",
    }


def validate_video_plan(saved, item, snapshot, config):
    """Audit persisted selection recipes without models or source videos."""
    vid = item["video_id"]
    frozen = snapshot["baseline_plans"][vid]
    for key in ("candidate_sha256", "candidate_count", "duration_seconds"):
        if saved.get(key) != frozen[key]:
            raise ValueError(
                "Segment-count candidates differ from frozen source"
            )
    if (
        saved.get("video_id") != vid
        or saved.get("source_id") != item["source_id"]
    ):
        raise ValueError("Segment-count video identity changed")
    times = np.asarray(saved["candidate_timestamps"], dtype=float)
    if (
        times.shape != (saved["candidate_count"],)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not 0 <= times[0] < times[-1] < saved["duration_seconds"]
    ):
        raise ValueError("Invalid segment-count candidate timestamps")
    qids = {str(q["question_id"]) for q in item["questions"]}
    if set(saved["questions"]) != qids:
        raise ValueError("Segment-count plan question cohort changed")
    expected_replay = {
        qid: report_digest(frozen["questions"][qid]["D"]) for qid in qids
    }
    if saved.get("source_D_replay") != expected_replay:
        raise ValueError("Original D replay verification changed")
    baselines = _baseline_rows(snapshot)
    if saved.get("verified_D_qa_pixels_sha256") != {
        qid: baselines[(vid, qid)]["conditions"]["D"]["qa_pixels_sha256"]
        for qid in qids
    }:
        raise ValueError("Original D QA pixel verification changed")
    shared_partitions, shared_feature_hash = {}, None
    original_events = None
    for qid in qids:
        frozen_d = frozen["questions"][qid]["D"]
        if (
            original_events is not None
            and original_events != frozen_d["events"]
        ):
            raise ValueError("Original D2 events differ across questions")
        original_events = frozen_d["events"]
        if (
            times[frozen_d["selected_indices"]].tolist()
            != frozen_d["selected_timestamps"]
        ):
            raise ValueError("Segment-count candidate grid differs from D")
        selections = saved["questions"][qid]
        if set(selections) != set(CONDITIONS):
            raise ValueError("Missing or unexpected segment-count selection")
        frame_scores = None
        partition_relevance = {}
        for name, selection in selections.items():
            validate_selection(selection, name, times, config)
            partition = condition_spec(name)["partition"]
            relevance = selection["relevance_raw_cosine"]
            if partition in partition_relevance and (
                partition_relevance[partition] != relevance
            ):
                raise ValueError(
                    "Event relevance changed across condition variants"
                )
            partition_relevance[partition] = relevance
            partition_identity = {
                key: selection[key]
                for key in ("events", "complexity", "source_features_sha256")
            }
            if partition in shared_partitions and (
                shared_partitions[partition] != partition_identity
            ):
                raise ValueError(
                    "Video partition/complexity changed across questions"
                )
            shared_partitions[partition] = partition_identity
            if shared_feature_hash is not None and (
                shared_feature_hash != selection["source_features_sha256"]
            ):
                raise ValueError(
                    "Feature identity changed across partitions/questions"
                )
            shared_feature_hash = selection["source_features_sha256"]
            if frame_scores is not None and (
                selection["frame_scores_raw_cosine"] != frame_scores
            ):
                raise ValueError("Frame relevance changed across conditions")
            frame_scores = selection["frame_scores_raw_cosine"]
    seconds = saved.get("prepare_wall_seconds")
    if (
        not isinstance(seconds, (float, int))
        or not np.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError("Invalid preparation duration")
    return times


def _validate_worker_metadata(root, plan, snapshot, index):
    shard, config = _shard(plan, index), _config(snapshot)
    output = job_directory(root, index)
    protocol = worker_protocol(plan, index)
    digest = report_digest(protocol)
    if read_json(rooted(output, "protocol.json")) != protocol:
        raise ValueError("Segment-count worker protocol changed")
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
        raise ValueError("Segment-count backend identity checkpoint mismatch")
    return output, config, digest


def _prepared_marker(plan, digest, plans):
    return {
        "status": "prepared",
        "protocol_sha256": digest,
        "suite_plan_sha256": report_digest(plan),
        "prepared_plan_sha256": report_digest(plans),
        "n_videos": len(plans),
        "n_questions": sum(len(p["questions"]) for p in plans),
        "qa_executed_by_prepare": False,
        "diagnostics": summarize_selections(plans),
    }


def validate_prepared(root, plan, snapshot, index):
    """Audit one complete selection-only shard without loading a backend."""
    validate_frozen(plan, snapshot)
    output, config, digest = _validate_worker_metadata(
        root, plan, snapshot, index
    )
    plans = []
    for item in _videos(snapshot, _shard(plan, index)):
        vid = item["video_id"]
        saved = checkpoint(
            _plan_path(output, vid),
            _plan_identity(
                digest,
                vid,
                snapshot["baseline_plans"][vid]["candidate_sha256"],
            ),
        )
        validate_video_plan(saved, item, snapshot, config)
        plans.append(saved)
    if read_json(
        rooted(output, "selection_preflight.json")
    ) != _prepared_marker(plan, digest, plans):
        raise ValueError(
            "Incomplete or changed segment-count selection preflight"
        )
    return plans


def validate_suite_preflight(root, plan, snapshot):
    """Require both prepared GPU shards and the exact coordinator gate."""
    plans = [
        saved
        for shard in plan["shards"]
        for saved in validate_prepared(root, plan, snapshot, shard["index"])
    ]
    expected = {
        "status": "prepared",
        "suite_plan_sha256": report_digest(plan),
        "snapshot_sha256": plan["snapshot_sha256"],
        "prepared_plan_sha256": report_digest(plans),
        "diagnostics": summarize_selections(plans),
    }
    if read_json(rooted(root, "selection_preflight.json")) != expected:
        raise ValueError(
            "Both-shard preflight gate is absent or stale; run prepare first"
        )
    return plans


def validate_row(row, item, question, config, digest, output, snapshot):
    baseline = _baseline_rows(snapshot)[
        (item["video_id"], question.question_id)
    ]
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
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Segment-count row differs from frozen source identity"
        )
    if set(row["conditions"]) != set(CONDITIONS) or sorted(
        row["execution_order"]
    ) != sorted(CONDITIONS):
        raise ValueError("Segment-count row conditions differ from its plan")
    saved = checkpoint(
        _plan_path(output, item["video_id"]),
        _plan_identity(digest, item["video_id"], row["candidate_sha256"]),
    )
    validate_video_plan(saved, item, snapshot, config)
    for name in CONDITIONS:
        condition = row["conditions"][name]
        selection = saved["questions"][question.question_id][name]
        if any(condition.get(k) != v for k, v in selection.items()):
            raise ValueError("Segment-count row selection/plan mismatch")
        pixel_hash = condition["qa_pixels_sha256"]
        if not isinstance(pixel_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", pixel_hash
        ):
            raise ValueError("Invalid segment-count QA pixel digest")
        trials = condition["trials"]
        strict_trials(trials, question, config)
        if [t["order"] for t in trials] != _orders(baseline, question, config):
            raise ValueError("Segment-count QA orders differ from original D")
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
            if checkpoint(_trial_path(output, identity), identity) != trial:
                raise ValueError(
                    "Segment-count row/raw trial checkpoint mismatch"
                )


def validate_worker(root, plan, snapshot, index):
    validate_frozen(plan, snapshot)
    validate_suite_preflight(root, plan, snapshot)
    shard = _shard(plan, index)
    output, config, digest = _validate_worker_metadata(
        root, plan, snapshot, index
    )
    videos = _videos(snapshot, shard)
    expected = {
        (v["video_id"], str(q["question_id"]))
        for v in videos
        for q in v["questions"]
    }
    result = read_json(rooted(output, "results.json"))
    if (
        result.get("completed") is not True
        or result.get("protocol_sha256") != digest
        or result.get("config") != snapshot["config"]
        or result.get("expected_new_qa_trials")
        != len(expected) * len(CONDITIONS) * 6
    ):
        raise ValueError("Segment-count worker incomplete or incompatible")
    rows = result["questions"]
    indexed = {(r["video_id"], r["question_id"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError(
            "Segment-count worker lacks its exact question cohort"
        )
    for item in videos:
        for annotation in item["questions"]:
            question = question_from(annotation)
            key = item["video_id"], question.question_id
            row = indexed[key]
            if (
                checkpoint(
                    _question_path(output, *key),
                    _question_identity(digest, *key),
                )
                != row
            ):
                raise ValueError("Segment-count results/question mismatch")
            validate_row(row, item, question, config, digest, output, snapshot)
    if read_json(rooted(output, "progress.json")) != {
        "status": "complete",
        "completed_questions": len(expected),
        "total_questions": len(expected),
        "protocol_sha256": digest,
    }:
        raise ValueError("Segment-count worker completion marker mismatch")
    return rows


def prepare_video(item, snapshot, config, digest, output, get_model):
    """Replay the original D, then freeze all 24 selection plans without QA."""
    vid = item["video_id"]
    identity = snapshot["inputs"][vid]
    path = Path(identity["path"])
    current = file_identity(path)
    if any(
        current[k] != identity[k] for k in ("path", "size_bytes", "sha256")
    ):
        raise ValueError(f"Frozen segment-count source video changed: {vid}")
    frozen = snapshot["baseline_plans"][vid]
    plan_identity = _plan_identity(digest, vid, frozen["candidate_sha256"])
    saved_path = _plan_path(output, vid)
    if saved_path.exists():
        saved = checkpoint(saved_path, plan_identity)
        times = validate_video_plan(saved, item, snapshot, config)
        return saved, times, saved["duration_seconds"]
    started = perf_counter()
    frames, times, duration = decode_video(path, config)
    questions = tuple(question_from(q) for q in item["questions"])
    validate_video(
        Video(
            vid, "dev", frames, times, duration, questions, item["source_id"]
        ),
        config,
    )
    candidate_hash = array_digest(frames, times)
    if (
        candidate_hash != frozen["candidate_sha256"]
        or len(times) != frozen["candidate_count"]
        or duration != frozen["duration_seconds"]
    ):
        raise ValueError(
            "Decoded segment-count candidates differ from baseline"
        )

    def compute():
        model = get_model()
        features = normalize_rows(
            model.encode_frames(frames), config.normalization_epsilon
        )
        original_events = [
            [event.start, event.stop]
            for event in make_partitions(features, config)["D2"]
        ]
        if any(
            frozen["questions"][q.question_id]["D"]["events"]
            != original_events
            for q in questions
        ):
            raise ValueError("Original D2 segmentation does not reproduce")
        partitions = prepare_partitions(features, config)
        planned_questions, replays = {}, {}
        for question in questions:
            qid = question.question_id
            query = model.encode_visual_question(question.text)
            frozen_d = frozen["questions"][qid]["D"]
            # This independently replays the historical relevance, integer
            # allocation, uniform indices and timestamps before new selection.
            replay_source_d(features, query, times, frozen_d, config)
            replays[qid] = report_digest(frozen_d)
            planned_questions[qid] = make_conditions(
                features, query, times, config, partitions=partitions
            )
        # Validate the historical 320-pixel inputs during selection preflight,
        # before either GPU is allowed to begin any new answer-model calls.
        baselines = _baseline_rows(snapshot)
        union = sorted(
            {
                index
                for question in questions
                for index in frozen["questions"][question.question_id]["D"][
                    "selected_indices"
                ]
            }
        )
        low, pixels, replay_times, replay_duration, _ = decode_selected(
            path, config, len(times), union, QA_SIDE, candidate_hash
        )
        del low
        if (
            not np.array_equal(replay_times, times)
            or replay_duration != duration
        ):
            raise ValueError(
                "Original D QA replay changed frozen candidate grid"
            )
        pixel_hashes = {}
        for question in questions:
            qid = question.question_id
            indices = frozen["questions"][qid]["D"]["selected_indices"]
            digest_pixels = array_digest(
                np.stack([pixels[index] for index in indices]), times[indices]
            )
            if (
                digest_pixels
                != baselines[(vid, qid)]["conditions"]["D"]["qa_pixels_sha256"]
            ):
                raise ValueError(
                    "Original D QA pixels changed during preflight"
                )
            pixel_hashes[qid] = digest_pixels
        return {
            "video_id": vid,
            "source_id": item["source_id"],
            "candidate_sha256": candidate_hash,
            "candidate_count": len(times),
            "candidate_timestamps": times.tolist(),
            "duration_seconds": duration,
            "source_D_replay": replays,
            "verified_D_qa_pixels_sha256": pixel_hashes,
            "questions": planned_questions,
            "prepare_wall_seconds": perf_counter() - started,
        }

    saved = checkpoint(saved_path, plan_identity, compute)
    if not np.array_equal(
        validate_video_plan(saved, item, snapshot, config), times
    ):
        raise ValueError("Segment-count timestamps changed on replay")
    return saved, times, duration


def score_question(
    item,
    question,
    saved,
    times,
    duration,
    baseline,
    config,
    digest,
    output,
    source_path,
    get_model,
):
    vid, qid = item["video_id"], question.question_id
    selections = saved["questions"][qid]
    frozen_d = baseline["conditions"]["D"]
    union = sorted(
        {
            i
            for selection in [*selections.values(), frozen_d]
            for i in selection["selected_indices"]
        }
    )
    low, pixels, replay_times, replay_duration, decode_info = decode_selected(
        source_path,
        config,
        len(times),
        union,
        QA_SIDE,
        saved["candidate_sha256"],
    )
    del low
    if not np.array_equal(replay_times, times) or replay_duration != duration:
        raise ValueError(
            "Segment-count QA replay differs from frozen candidates"
        )
    original_indices = frozen_d["selected_indices"]
    original_hash = array_digest(
        np.stack([pixels[i] for i in original_indices]),
        times[original_indices],
    )
    if original_hash != frozen_d["qa_pixels_sha256"]:
        raise ValueError("Original D QA pixels changed at 320 resolution")
    execution = list(CONDITIONS)
    if config.randomize_condition_order:
        np.random.default_rng(
            config.seed + int(report_digest([vid, qid])[:8], 16)
        ).shuffle(execution)
    conditions = {}
    for name in execution:
        selection = selections[name]
        indices = selection["selected_indices"]
        images, stamps = np.stack([pixels[i] for i in indices]), times[indices]
        video = Video(
            vid, "dev", images, stamps, duration, (), item["source_id"]
        )
        pixel_hash = array_digest(images, stamps)
        trials = []
        for order in _orders(baseline, question, config):
            identity = _trial_identity(
                digest, vid, qid, name, indices, order, pixel_hash
            )
            trial = checkpoint(
                _trial_path(output, identity),
                identity,
                lambda: score_with_memory(
                    get_model(), video, question, order, config, QA_SIDE
                ),
            )
            validate_processor(trial, QA_SIDE, 32)
            trials.append(trial)
        strict_trials(trials, question, config)
        conditions[name] = {
            **selection,
            "qa_pixels_sha256": pixel_hash,
            "trials": trials,
        }
    return {
        "video_id": vid,
        "source_id": item["source_id"],
        "question_id": qid,
        "duration_group": "long",
        "question": question.text,
        "options": list(question.options),
        "answer_index_evaluator_only": question.answer_index,
        "candidate_sha256": saved["candidate_sha256"],
        "candidate_count": saved["candidate_count"],
        "execution_order": execution,
        "decode_info": decode_info,
        "verified_D_qa_pixels_sha256": original_hash,
        "conditions": {name: conditions[name] for name in CONDITIONS},
    }


def run(args):
    plan_path = Path(args.plan).resolve()
    root = plan_path.parent
    plan, snapshot = read_frozen(
        plan_path, getattr(args, "expected_plan_sha256", None)
    )
    shard = _shard(plan, args.shard_index)
    config, videos = _config(snapshot), _videos(snapshot, shard)
    if not videos:
        raise ValueError("Cannot launch an empty segment-count shard")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(shard["gpu"]):
        raise ValueError("Segment-count GPU mask differs from frozen GPU")
    prepare_only = bool(getattr(args, "prepare_only", False))
    # Run this before output mutation and, in particular, before model loading.
    if not prepare_only:
        validate_suite_preflight(root, plan, snapshot)
    output = job_directory(root, args.shard_index)
    protocol = worker_protocol(plan, args.shard_index)
    digest = report_digest(protocol)
    total = sum(len(v["questions"]) for v in videos)
    baselines = _baseline_rows(snapshot)
    with output_lock(output):
        protocol_path = rooted(output, "protocol.json")
        if protocol_path.exists():
            if read_json(protocol_path) != protocol:
                raise ValueError("Segment-count worker protocol changed")
        elif any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("Segment-count output has no compatible protocol")
        else:
            save_json(protocol_path, protocol)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
        runtime = runtime_metadata()
        _validate_runtime(runtime, snapshot, shard)
        runtime_path = rooted(output, "runtime.json")
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError("Segment-count worker runtime changed")
        save_json(runtime_path, runtime)
        result_path, progress_path = (
            rooted(output, name) for name in ("results.json", "progress.json")
        )
        if (
            not prepare_only
            and result_path.exists()
            and progress_path.exists()
            and read_json(result_path).get("completed") is True
            and read_json(progress_path).get("status") == "complete"
        ):
            validate_worker(root, plan, snapshot, args.shard_index)
            return read_json(result_path)
        metadata_path = rooted(output, "backend.json")
        identity_path = rooted(output, "backend_identity.json")
        backend_identity = {"protocol_sha256": digest}
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            _validate_backend_identity(metadata, snapshot, config)
            if checkpoint(identity_path, backend_identity) != metadata:
                raise ValueError("Segment-count backend metadata changed")
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
                ) or checkpoint(
                    identity_path, backend_identity, lambda: metadata
                ) != metadata:
                    raise ValueError("Segment-count backend identity changed")
                save_json(metadata_path, metadata)
            return model

        if prepare_only:
            prepare_progress = rooted(output, "prepare_progress.json")
            preflight_path = rooted(output, "selection_preflight.json")
            plans = []
            try:
                for item in videos:
                    saved, _, _ = prepare_video(
                        item, snapshot, config, digest, output, get_model
                    )
                    plans.append(saved)
                    save_json(
                        prepare_progress,
                        {
                            "status": "preparing",
                            "prepared_videos": len(plans),
                            "total_videos": len(videos),
                            "protocol_sha256": digest,
                            "qa_executed_by_prepare": False,
                        },
                    )
                    print(
                        f"Prepared segment-count B32 {item['video_id']} "
                        f"({len(CONDITIONS)} selections per question; no QA)",
                        flush=True,
                    )
                marker = _prepared_marker(plan, digest, plans)
                save_json(preflight_path, marker)
                validate_prepared(root, plan, snapshot, args.shard_index)
                save_json(
                    prepare_progress,
                    {
                        "status": "prepared",
                        "prepared_videos": len(plans),
                        "total_videos": len(videos),
                        "protocol_sha256": digest,
                        "qa_executed_by_prepare": False,
                    },
                )
                return marker
            except BaseException as error:
                failure = {
                    "status": "failed",
                    "protocol_sha256": digest,
                    "error": f"{type(error).__name__}: {error}",
                    "qa_executed_by_prepare": False,
                }
                save_json(prepare_progress, failure)
                save_json(preflight_path, failure)
                raise

        prepared = {
            saved["video_id"]: saved
            for saved in validate_prepared(
                root, plan, snapshot, args.shard_index
            )
        }
        rows = []

        def progress(status, **extra):
            save_json(
                progress_path,
                {
                    "status": status,
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "protocol_sha256": digest,
                    **extra,
                },
            )

        progress("running")
        try:
            for item in videos:
                vid = item["video_id"]
                saved = prepared[vid]
                times = np.asarray(saved["candidate_timestamps"], dtype=float)
                duration = saved["duration_seconds"]
                for annotation in item["questions"]:
                    question = question_from(annotation)
                    question_path = _question_path(
                        output, vid, question.question_id
                    )
                    if not question_path.exists():
                        row = score_question(
                            item,
                            question,
                            saved,
                            times,
                            duration,
                            baselines[(vid, question.question_id)],
                            config,
                            digest,
                            output,
                            Path(snapshot["inputs"][vid]["path"]),
                            get_model,
                        )
                        checkpoint(
                            question_path,
                            _question_identity(
                                digest, vid, question.question_id
                            ),
                            lambda: row,
                        )
                        print(
                            f"Saved segment-count B32 {vid}/"
                            f"{question.question_id} "
                            f"({len(CONDITIONS)} conditions, "
                            f"{6 * len(CONDITIONS)} QA trials)",
                            flush=True,
                        )
                    row = checkpoint(
                        question_path,
                        _question_identity(digest, vid, question.question_id),
                    )
                    validate_row(
                        row, item, question, config, digest, output, snapshot
                    )
                    rows.append(row)
                    progress("running")
            result = {
                "protocol_sha256": digest,
                "completed": True,
                "config": snapshot["config"],
                "expected_new_qa_trials": total * len(CONDITIONS) * 6,
                "questions": rows,
            }
            save_json(result_path, result)
            progress("complete")
            validate_worker(root, plan, snapshot, args.shard_index)
            return result
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            progress("failed", error=message)
            save_json(
                result_path,
                {
                    "protocol_sha256": digest,
                    "completed": False,
                    "error": message,
                },
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(args)
    if args.prepare_only:
        status = {
            "status": "prepared",
            "n_videos": result["n_videos"],
            "n_questions": result["n_questions"],
            "qa_executed_by_prepare": False,
        }
    else:
        status = {
            "status": "complete",
            "n_questions": len(result["questions"]),
            "expected_new_qa_trials": result["expected_new_qa_trials"],
        }
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
