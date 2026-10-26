"""Resumable diversity QA with an exact-input frozen baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from .aggregate_budget_shards import rooted
from .backends import build_backend
from .data import array_digest
from .event_refinement_dev import question_from
from .frame_replacement_probe import strict_trials
from .long_ablation_worker import (
    QA_SIDE,
    _baseline_rows,
    _config,
    _orders,
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
from .query_diversity_analysis import CONDITIONS
from .resolution_dev import (
    decode_selected,
    score_with_memory,
    validate_processor,
)
from .types import Video

BASELINE_CONDITION = "A_R_topk"
SOURCE_BASELINE_CONDITION = "legacy_R_topk"


def job_directory(root, index):
    if type(index) is not int or index < 0:
        raise ValueError("Invalid query diversity shard index")
    return rooted(root, "jobs", f"dev_long_query_diversity_b32_s{index}")


def worker_protocol(plan, index):
    shard = _shard(plan, index)
    return {
        "version": 1,
        "experiment": "a4500_query_diversity_worker_v1",
        "suite_plan_sha256": report_digest(plan),
        "snapshot_sha256": plan["snapshot_sha256"],
        "algorithm": plan["algorithm"],
        "diversity_strength": plan["diversity_strength"],
        "diversity_temperature": plan["diversity_temperature"],
        "shard_index": index,
        "gpu": shard["gpu"],
        "video_ids": shard["video_ids"],
        "conditions": list(CONDITIONS),
        "frame_budget": 32,
        "qa_side": QA_SIDE,
        "split": "dev",
        "duration": "long",
        "reuse_policy": {
            "identity": (
                "same question, indices, QA pixel/timestamp digest and order"
            ),
            "priority": [
                "frozen_source_legacy_R_topk",
                "other_frozen_source",
                "earlier_current_condition",
                "new",
            ],
            "baseline": "A_R_topk must reuse frozen legacy_R_topk",
            "seed_per_call": "frozen config.seed; condition independent",
            "source_timing_is_new_compute": False,
        },
    }


def _source_rows(snapshot):
    rows = {
        (r["video_id"], r["question_id"]): r for r in snapshot["source_rows"]
    }
    if len(rows) != len(snapshot["source_rows"]):
        raise ValueError("Duplicate frozen Diversity source questions")
    return rows


def _execution_order(config, vid, qid):
    execution = list(CONDITIONS)
    if config.randomize_condition_order:
        np.random.default_rng(
            config.seed + int(report_digest([vid, qid])[:8], 16)
        ).shuffle(execution)
    return execution


def _matching_condition(conditions, indices, pixel_hash):
    # Source maps use the explicit baseline-first order below; current maps
    # retain execution order so reuse provenance is deterministic.
    for name, condition in conditions.items():
        if (
            condition["selected_indices"] == indices
            and condition.get("qa_pixels_sha256") == pixel_hash
        ):
            return name, condition
    return None


def _validate_baseline_selection(source, selection, pixel_hash=None):
    reference = source["conditions"].get(SOURCE_BASELINE_CONDITION)
    if reference is None:
        raise ValueError("Frozen source lacks legacy_R_topk baseline")
    for key in ("selected_indices", "selected_timestamps"):
        if selection.get(key) != reference.get(key):
            raise ValueError(f"A_R_topk {key} differs from legacy_R_topk")
    if pixel_hash is not None and pixel_hash != reference["qa_pixels_sha256"]:
        raise ValueError("A_R_topk QA pixels differ from legacy_R_topk")


def _reuse_for_trial(
    source, previous, indices, pixel_hash, order, condition_name=None
):
    if SOURCE_BASELINE_CONDITION not in source["conditions"]:
        raise ValueError("Frozen source lacks legacy_R_topk baseline")
    # A is the prior experiment's exact-input control. Prefer that provenance
    # for every identical input, even if other source arms also match.
    names = [SOURCE_BASELINE_CONDITION] + sorted(
        name
        for name in source["conditions"]
        if name != SOURCE_BASELINE_CONDITION
    )
    source_conditions = {name: source["conditions"][name] for name in names}
    matched = _matching_condition(source_conditions, indices, pixel_hash)
    if condition_name == BASELINE_CONDITION and (
        matched is None or matched[0] != SOURCE_BASELINE_CONDITION
    ):
        raise ValueError("A_R_topk must reuse the exact legacy_R_topk input")
    kind = "source"
    if matched is None:
        matched = _matching_condition(previous, indices, pixel_hash)
        kind = "current_condition"
    if matched is None:
        return None
    name, condition = matched
    trials = [
        trial for trial in condition["trials"] if trial["order"] == order
    ]
    if len(trials) != 1:
        raise ValueError("Reusable condition lacks the exact frozen order")
    trial = deepcopy(trials[0])
    provenance = {
        "kind": kind,
        "condition": name,
        "trial_sha256": report_digest(trial),
    }
    if kind == "source":
        provenance["source_row_sha256"] = report_digest(source)
    return {"trial": trial, "provenance": provenance}


def _counts(rows):
    kinds = [
        provenance["kind"]
        for row in rows
        for condition in row["conditions"].values()
        for provenance in condition["trial_provenance"]
    ]
    return {
        "expected_qa_trials": len(kinds),
        "new_qa_trials": kinds.count("new"),
        "reused_qa_trials": len(kinds) - kinds.count("new"),
        "source_qa_trials_reused": kinds.count("source"),
        "current_condition_qa_trials_reused": kinds.count("current_condition"),
    }


def _validate_source_question(source, baseline, item, question, config):
    expected = {
        "video_id": item["video_id"],
        "question_id": question.question_id,
        "candidate_sha256": baseline["candidate_sha256"],
        "candidate_count": baseline["candidate_count"],
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Diversity QA source question/candidate identity changed"
        )
    if SOURCE_BASELINE_CONDITION not in source["conditions"]:
        raise ValueError("Frozen source lacks legacy_R_topk baseline")
    orders = _orders(baseline, question, config)
    for condition in source["conditions"].values():
        strict_trials(condition["trials"], question, config)
        if [trial["order"] for trial in condition["trials"]] != orders:
            raise ValueError("Diversity QA source orders changed")
        for trial in condition["trials"]:
            # validate_processor annotates visual_tokens; do not mutate the
            # frozen snapshot or any reusable source trial during validation.
            validate_processor(deepcopy(trial), QA_SIDE, config.frame_budget)


def _validate_trial_record(record, reusable, question, config):
    trial, provenance = record["trial"], record["provenance"]
    validate_processor(deepcopy(trial), QA_SIDE, config.frame_budget)
    if provenance.get("trial_sha256") != report_digest(trial):
        raise ValueError(
            "Diversity trial provenance digest differs from its trial"
        )
    if reusable is not None:
        if record != reusable:
            raise ValueError("Diversity reusable trial/provenance changed")
    elif provenance != {"kind": "new", "trial_sha256": report_digest(trial)}:
        raise ValueError("Diversity trial has invalid new-compute provenance")
    # Full six-order scoring checks are performed after every condition.
    if trial["presented_options"] != [
        question.options[i] for i in trial["order"]
    ]:
        raise ValueError(
            "Diversity trial presented options differ from the question"
        )


def validate_row(
    row, item, question, saved, baseline, source, config, digest, output
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
        "prepared_video_plan_sha256": report_digest(saved),
        "source_row_sha256": report_digest(source),
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Diversity result differs from its frozen question/source/plan"
        )
    execution = _execution_order(
        config, item["video_id"], question.question_id
    )
    if row.get("execution_order") != execution or set(
        row["conditions"]
    ) != set(CONDITIONS):
        raise ValueError("Diversity result condition execution order changed")
    _validate_source_question(source, baseline, item, question, config)
    _validate_baseline_selection(
        source,
        saved["questions"][question.question_id][BASELINE_CONDITION],
        row["conditions"][BASELINE_CONDITION]["qa_pixels_sha256"],
    )
    previous = {}
    for name in execution:
        condition = row["conditions"][name]
        selection = saved["questions"][question.question_id][name]
        if any(
            condition.get(key) != value for key, value in selection.items()
        ):
            raise ValueError(
                "Diversity result selection differs from prepared plan"
            )
        pixel_hash = condition["qa_pixels_sha256"]
        if not isinstance(pixel_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", pixel_hash
        ):
            raise ValueError("Invalid Diversity QA pixel digest")
        trials = condition["trials"]
        strict_trials(trials, question, config)
        if [trial["order"] for trial in trials] != _orders(
            baseline, question, config
        ):
            raise ValueError(
                "Diversity result orders differ from the frozen orders"
            )
        if len(condition["trial_provenance"]) != len(trials):
            raise ValueError(
                "Diversity trial provenance count differs from QA count"
            )
        for trial, provenance in zip(
            trials, condition["trial_provenance"], strict=True
        ):
            identity = _trial_identity(
                digest,
                item["video_id"],
                question.question_id,
                name,
                selection["selected_indices"],
                trial["order"],
                pixel_hash,
            )
            record = {"trial": trial, "provenance": provenance}
            if checkpoint(_trial_path(output, identity), identity) != record:
                raise ValueError(
                    "Diversity question/raw trial checkpoint mismatch"
                )
            reusable = _reuse_for_trial(
                source,
                previous,
                selection["selected_indices"],
                pixel_hash,
                trial["order"],
                name,
            )
            _validate_trial_record(record, reusable, question, config)
        previous[name] = condition
    if row.get("qa_counts") != _counts([row]):
        raise ValueError("Diversity result QA counts changed")


def _validate_metadata(root, plan, snapshot, index, require_backend=False):
    baseline, shard = snapshot["baseline"], _shard(plan, index)
    config = _config(baseline)
    output = job_directory(root, index)
    protocol = worker_protocol(plan, index)
    digest = report_digest(protocol)
    if read_json(rooted(output, "protocol.json")) != protocol:
        raise ValueError("Diversity worker protocol changed")
    _validate_runtime(
        read_json(rooted(output, "runtime.json")), baseline, shard
    )
    if (
        read_json(rooted(output, "backend_reference.json"))
        != baseline["backend"]
    ):
        raise ValueError("Diversity frozen backend binding changed")
    metadata_path = rooted(output, "backend.json")
    identity_path = rooted(output, "backend_identity.json")
    if require_backend or metadata_path.exists() or identity_path.exists():
        metadata = read_json(metadata_path)
        _validate_backend_identity(metadata, baseline, config)
        if checkpoint(identity_path, {"protocol_sha256": digest}) != metadata:
            raise ValueError("Diversity backend identity checkpoint mismatch")
    return output, config, digest


def validate_worker(root, plan, snapshot, index):
    from .a4500_query_diversity import load_prepared

    prepared = load_prepared(root, plan, snapshot)
    shard = _shard(plan, index)
    videos = _videos(snapshot["baseline"], shard)
    expected = {
        (item["video_id"], str(annotation["question_id"]))
        for item in videos
        for annotation in item["questions"]
    }
    output, config, digest = _validate_metadata(root, plan, snapshot, index)
    result = read_json(rooted(output, "results.json"))
    if (
        result.get("completed") is not True
        or result.get("protocol_sha256") != digest
        or result.get("config") != snapshot["baseline"]["config"]
        or result.get("expected_qa_trials")
        != len(expected) * len(CONDITIONS) * 6
    ):
        raise ValueError("Diversity worker incomplete or incompatible")
    rows = result["questions"]
    indexed = {(r["video_id"], r["question_id"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError(
            "Diversity worker lacks its exact frozen question cohort"
        )
    baselines, sources = (
        _baseline_rows(snapshot["baseline"]),
        _source_rows(snapshot),
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
                raise ValueError(
                    "Diversity result/question checkpoint mismatch"
                )
            validate_row(
                row,
                item,
                question,
                prepared[item["video_id"]],
                baselines[key],
                sources[key],
                config,
                digest,
                output,
            )
    counts = _counts(rows)
    if any(result.get(key) != value for key, value in counts.items()):
        raise ValueError("Diversity worker reuse/new QA counts changed")
    _validate_metadata(
        root,
        plan,
        snapshot,
        index,
        require_backend=bool(counts["new_qa_trials"]),
    )
    if read_json(rooted(output, "progress.json")) != {
        "status": "complete",
        "completed_questions": len(expected),
        "total_questions": len(expected),
        "protocol_sha256": digest,
        **counts,
    }:
        raise ValueError("Diversity worker completion marker mismatch")
    return rows


def score_question(
    item,
    question,
    saved,
    baseline,
    source,
    config,
    digest,
    output,
    source_path,
    get_model,
):
    vid, qid = item["video_id"], question.question_id
    _validate_source_question(source, baseline, item, question, config)
    times = np.asarray(saved["candidate_timestamps"], dtype=float)
    selections = saved["questions"][qid]
    _validate_baseline_selection(source, selections[BASELINE_CONDITION])
    frozen_d = baseline["conditions"]["D"]
    union = sorted(
        {
            index
            for selection in [*selections.values(), frozen_d]
            for index in selection["selected_indices"]
        }
    )
    low, pixels, replay_times, duration, decode_info = decode_selected(
        source_path,
        config,
        len(times),
        union,
        QA_SIDE,
        saved["candidate_sha256"],
    )
    del low
    if (
        not np.array_equal(replay_times, times)
        or duration != saved["duration_seconds"]
    ):
        raise ValueError(
            "Diversity QA candidate grid differs from frozen source"
        )
    original = frozen_d["selected_indices"]
    original_hash = array_digest(
        np.stack([pixels[i] for i in original]), times[original]
    )
    if original_hash != frozen_d["qa_pixels_sha256"]:
        raise ValueError("Original D QA pixels changed at 320 resolution")
    baseline_indices = selections[BASELINE_CONDITION]["selected_indices"]
    _validate_baseline_selection(
        source,
        selections[BASELINE_CONDITION],
        array_digest(
            np.stack([pixels[i] for i in baseline_indices]),
            times[baseline_indices],
        ),
    )
    execution = _execution_order(config, vid, qid)
    conditions = {}
    for name in execution:
        selection = selections[name]
        indices = selection["selected_indices"]
        images, stamps = np.stack([pixels[i] for i in indices]), times[indices]
        pixel_hash = array_digest(images, stamps)
        video = Video(
            vid, "dev", images, stamps, duration, (), item["source_id"]
        )
        trials, provenance = [], []
        for order in _orders(baseline, question, config):
            reusable = _reuse_for_trial(
                source, conditions, indices, pixel_hash, order, name
            )
            identity = _trial_identity(
                digest, vid, qid, name, indices, order, pixel_hash
            )

            def compute():
                if reusable is not None:
                    return reusable
                trial = score_with_memory(
                    get_model(), video, question, order, config, QA_SIDE
                )
                return {
                    "trial": trial,
                    "provenance": {
                        "kind": "new",
                        "trial_sha256": report_digest(trial),
                    },
                }

            record = checkpoint(
                _trial_path(output, identity), identity, compute
            )
            _validate_trial_record(record, reusable, question, config)
            trials.append(record["trial"])
            provenance.append(record["provenance"])
        strict_trials(trials, question, config)
        conditions[name] = {
            **selection,
            "qa_pixels_sha256": pixel_hash,
            "trials": trials,
            "trial_provenance": provenance,
        }
    row = {
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
        "prepared_video_plan_sha256": report_digest(saved),
        "source_row_sha256": report_digest(source),
        "conditions": {name: conditions[name] for name in CONDITIONS},
    }
    row["qa_counts"] = _counts([row])
    return row


def run(args):
    from .a4500_query_diversity import load_prepared, read_frozen

    plan_path = Path(args.plan).resolve()
    root = plan_path.parent
    plan, snapshot = read_frozen(
        plan_path, getattr(args, "expected_plan_sha256", None)
    )
    # All shards' selection plans must be complete and validated before any
    # worker can load a model or write QA output.
    prepared = load_prepared(root, plan, snapshot)
    baseline = snapshot["baseline"]
    shard = _shard(plan, args.shard_index)
    config, videos = _config(baseline), _videos(baseline, shard)
    if not videos:
        raise ValueError("Cannot launch an empty query diversity shard")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(shard["gpu"]):
        raise ValueError("Diversity GPU mask differs from frozen source GPU")
    output = job_directory(root, args.shard_index)
    protocol = worker_protocol(plan, args.shard_index)
    digest = report_digest(protocol)
    total = sum(len(item["questions"]) for item in videos)
    baselines, sources = _baseline_rows(baseline), _source_rows(snapshot)
    with output_lock(output):
        protocol_path = rooted(output, "protocol.json")
        if protocol_path.exists():
            if read_json(protocol_path) != protocol:
                raise ValueError("Diversity worker protocol changed")
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError("Diversity output has no compatible protocol")
        else:
            save_json(protocol_path, protocol)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
        runtime = runtime_metadata()
        _validate_runtime(runtime, baseline, shard)
        runtime_path = rooted(output, "runtime.json")
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise ValueError("Diversity worker runtime changed")
        save_json(runtime_path, runtime)
        reference = rooted(output, "backend_reference.json")
        if reference.exists() and read_json(reference) != baseline["backend"]:
            raise ValueError("Diversity source backend reference changed")
        save_json(reference, baseline["backend"])
        result_path, progress_path = (
            rooted(output, name) for name in ("results.json", "progress.json")
        )
        if (
            result_path.exists()
            and progress_path.exists()
            and read_json(result_path).get("completed") is True
            and read_json(progress_path).get("status") == "complete"
        ):
            validate_worker(root, plan, snapshot, args.shard_index)
            return read_json(result_path)
        metadata_path, identity_path = (
            rooted(output, name)
            for name in ("backend.json", "backend_identity.json")
        )
        backend_identity = {"protocol_sha256": digest}
        if metadata_path.exists() or identity_path.exists():
            _validate_metadata(root, plan, snapshot, args.shard_index)
        model = None

        def get_model():
            nonlocal model
            if model is None:
                seed_call(config)
                model = build_backend(config)
                metadata = model.metadata()
                _validate_backend_identity(metadata, baseline, config)
                if (
                    metadata_path.exists()
                    and read_json(metadata_path) != metadata
                ) or checkpoint(
                    identity_path, backend_identity, lambda: metadata
                ) != metadata:
                    raise ValueError("Diversity backend identity changed")
                save_json(metadata_path, metadata)
            return model

        rows = []

        def progress(status, **extra):
            save_json(
                progress_path,
                {
                    "status": status,
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "protocol_sha256": digest,
                    **_counts(rows),
                    **extra,
                },
            )

        progress("running")
        try:
            for item in videos:
                vid = item["video_id"]
                saved = prepared[vid]
                identity = baseline["inputs"][vid]
                source_path = Path(identity["path"])
                # Completed question checkpoints can be audited offline.
                # Any incomplete question must recheck the actual source bytes.
                pending = any(
                    not _question_path(
                        output, vid, str(q["question_id"])
                    ).exists()
                    for q in item["questions"]
                )
                if pending:
                    current = file_identity(source_path)
                    if any(
                        current[key] != identity[key]
                        for key in ("path", "size_bytes", "sha256")
                    ):
                        raise ValueError(
                            f"Frozen Diversity source video changed: {vid}"
                        )
                for annotation in item["questions"]:
                    question = question_from(annotation)
                    key = vid, question.question_id
                    question_path = _question_path(output, *key)
                    if not question_path.exists():
                        row = score_question(
                            item,
                            question,
                            saved,
                            baselines[key],
                            sources[key],
                            config,
                            digest,
                            output,
                            source_path,
                            get_model,
                        )
                        checkpoint(
                            question_path,
                            _question_identity(digest, *key),
                            lambda: row,
                        )
                        print(
                            f"Saved ABC B32 {vid}/{question.question_id}: "
                            f"{row['qa_counts']['new_qa_trials']} new, "
                            f"{row['qa_counts']['reused_qa_trials']} reused "
                            "QA trials",
                            flush=True,
                        )
                    row = checkpoint(
                        question_path, _question_identity(digest, *key)
                    )
                    validate_row(
                        row,
                        item,
                        question,
                        saved,
                        baselines[key],
                        sources[key],
                        config,
                        digest,
                        output,
                    )
                    rows.append(row)
                    progress("running")
            result = {
                "protocol_sha256": digest,
                "completed": True,
                "config": baseline["config"],
                **_counts(rows),
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
                    **_counts(rows),
                },
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    result = run(parser.parse_args(argv))
    print(
        json.dumps(
            {
                "status": "complete",
                "n_questions": len(result["questions"]),
                **{key: result[key] for key in _counts(result["questions"])},
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
