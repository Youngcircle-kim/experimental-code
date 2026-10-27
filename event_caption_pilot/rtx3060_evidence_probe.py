"""Fixed-budget evidence replacement diagnosis, never a method benchmark.

The reviewed case manifest and replacement indices are frozen before QA.
Only original images, the question and reordered options reach the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from . import rtx3060_fixed_events as fixed
from .data import array_digest
from .event_refinement_dev import question_from
from .paired_reanalysis import clustered_means
from .portable_budget import (
    checkpoint,
    file_identity,
    output_lock,
    read_json,
    save_json,
)
from .qa_diagnostics import report_digest
from .resolution_dev import score_with_memory
from .types import Video

EXPERIMENT = "rtx3060_evidence_replacement_diagnostic_v1"
ARMS = ("baseline", "repair", "sham")
CONTRASTS = (
    "repair_minus_baseline",
    "sham_minus_baseline",
    "repair_minus_sham",
)
INTERPRETATION = (
    "Exploratory diagnosis on reviewed development cases. AI/human evidence "
    "selection may use the question and answer label; labels and review text "
    "never enter QA. These are diagnostic interventions, not an automated "
    "method score. Six option orders are repeated measures within a question. "
    "Bootstrap uncertainty is conditional on the reviewed, eligible cases."
)


def code_hashes():
    result = fixed.code_hashes()
    result[Path(__file__).name] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return result


def resolved(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def condition_indices(case, source_selection, count):
    if set(case["conditions"]) != set(ARMS):
        raise ValueError(
            "Every planned case requires baseline, repair and sham"
        )
    indices = {}
    for name in ARMS:
        selected = case["conditions"][name]["selected_indices"]
        if (
            len(selected) != 32
            or any(type(i) is not int for i in selected)
            or selected != sorted(set(selected))
            or not 0 <= selected[0] <= selected[-1] < count
        ):
            raise ValueError(
                "Require exactly 32 chronological source candidates"
            )
        indices[name] = selected
    if indices["baseline"] != source_selection["selected_indices"]:
        raise ValueError("Baseline must be the frozen G_topk selection")
    baseline, repair, sham = (set(indices[name]) for name in ARMS)
    removed, inserted = baseline - repair, repair - baseline
    sham_removed, sham_inserted = baseline - sham, sham - baseline
    if (
        not 1 <= len(removed) <= 4
        or len(removed) != len(inserted)
        or removed != sham_removed
        or len(inserted) != len(sham_inserted)
        or inserted & sham_inserted
    ):
        raise ValueError(
            "Repair/sham require 1-4 matched, distinct replacements"
        )
    events = source_selection["events"]

    def event_counts(selected):
        result = [
            sum(start <= i < stop for i in selected) for start, stop in events
        ]
        if sum(result) != len(selected):
            raise ValueError("Replacement is outside source events")
        return result

    if event_counts(inserted) != event_counts(sham_inserted):
        raise ValueError("Repair/sham added frames must match by source event")
    return indices, {
        "removed_indices": sorted(removed),
        "repair_added_indices": sorted(inserted),
        "sham_added_indices": sorted(sham_inserted),
        "matching": "same removed frames and added-frame source-event counts",
    }


def load_pixel_pack(cache, plan, required):
    path = Path(cache["npz_path"])
    if file_identity(path)["sha256"] != cache["file_sha256"]:
        raise ValueError("Frozen pixel file changed")
    with np.load(path, allow_pickle=False) as saved:
        indices, frames, times = (
            saved["indices"],
            saved["frames"],
            saved["timestamps"],
        )
    values = indices.tolist()
    if (
        indices.dtype != np.int64
        or indices.ndim != 1
        or values != sorted(set(values))
        or not values
        or not 0 <= values[0] <= values[-1] < plan["candidate_count"]
        or frames.dtype != np.uint8
        or frames.shape != (len(values), 224, 224, 3)
        or times.dtype != np.float64
        or times.shape != (len(values),)
        or times.tolist() != [plan["candidate_timestamps"][i] for i in values]
        or not set(required).issubset(values)
        or array_digest(indices, frames, times) != cache["array_sha256"]
    ):
        raise ValueError("Pixel array/indices/timestamps/source-grid mismatch")
    return {i: n for n, i in enumerate(values)}, frames, times


def source_case(source, protocol, runtime_hash, vid, qid):
    items = [v for v in protocol["videos"] if v["video_id"] == vid]
    if len(items) != 1:
        raise ValueError("Unknown source video")
    annotations = [
        q for q in items[0]["questions"] if str(q["question_id"]) == qid
    ]
    if len(annotations) != 1:
        raise ValueError("Unknown source question")
    question = question_from(annotations[0])
    plan = checkpoint(
        fixed.plan_path(source, vid), fixed.plan_identity(protocol, vid)
    )
    if report_digest(plan) != protocol["prepared_sha256"][vid]:
        raise ValueError("Source selection plan changed")
    row = checkpoint(
        fixed.question_path(source, vid, qid),
        {
            **fixed.plan_identity(protocol, vid),
            "question_id": qid,
            "runtime_sha256": runtime_hash,
        },
    )
    if row["video_id"] != vid or row["question_id"] != qid:
        raise ValueError("Source question checkpoint identity mismatch")
    orders = protocol["orders"][f"{vid}/{qid}"]
    if len(orders) != 6 or len({tuple(o) for o in orders}) != 6:
        raise ValueError("Source must contain six distinct orders")
    for name in ("G_topk", "G_diverse"):
        condition = row["conditions"][name]
        selection = plan["questions"][qid][name]
        if (
            any(condition.get(k) != v for k, v in selection.items())
            or len(condition["trials"]) != 6
        ):
            raise ValueError(
                "Source condition differs from prepared selection"
            )
        for order, trial in zip(orders, condition["trials"]):
            fixed.validate_trial(trial, question, order, 224)
            identity = fixed.trial_identity(
                protocol,
                runtime_hash,
                vid,
                qid,
                selection,
                condition["qa_pixels_sha256"],
                order,
            )
            record = checkpoint(fixed.trial_path(source, identity), identity)
            if record["trial"] != trial:
                raise ValueError("Source trial checkpoint changed")
    return items[0], annotations[0], plan, row, orders


def prepare(args):
    manifest_path, output = args.manifest.resolve(), args.output_root.resolve()
    manifest = read_json(manifest_path)
    source = resolved(manifest_path.parent, manifest["source_root"])
    if (
        source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Use a separate diagnostic output")
    source_protocol = read_json(source / "protocol.json")
    source_runtime = read_json(source / "runtime.json")
    handoff = read_json(source / "handoff.json")
    if (
        source_protocol["experiment"] != fixed.EXPERIMENT
        or source_protocol["scope"] != "full"
        or source_protocol["qa_side"] != 224
        or source_protocol["frame_budget"] != 32
        or source_protocol["code_sha256"] != fixed.code_hashes()
        or handoff.get("status") != "complete"
        or handoff.get("completed") is not True
        or handoff["protocol_sha256"] != report_digest(source_protocol)
        or handoff["runtime_sha256"] != report_digest(source_runtime)
    ):
        raise ValueError(
            "Require the verified completed fixed-event B32/224 source"
        )
    review_ref = manifest["review_manifest"]
    review_path = resolved(manifest_path.parent, review_ref["path"])
    if file_identity(review_path)["sha256"] != review_ref["file_sha256"]:
        raise ValueError("Frozen review manifest changed")
    reviewed = read_json(review_path)
    reviewed_cases = {
        c["case_id"]: (c["video_id"], c["question_id"])
        for c in reviewed["cases"]
    }
    cases, excluded = manifest["cases"], manifest["excluded_cases"]
    selected_ids = [c["case_id"] for c in cases]
    excluded_ids = [c["case_id"] for c in excluded]
    if (
        len(reviewed_cases) != 20
        or len(reviewed["cases"]) != 20
        or not 1 <= len(cases) <= 8
        or len(set(selected_ids + excluded_ids)) != 20
        or set(selected_ids + excluded_ids) != set(reviewed_cases)
        or any(
            not c.get("reason") or not c.get("review_status") for c in excluded
        )
        or manifest["reviewer"]["kind"] not in ("ai", "human", "mixed")
    ):
        raise ValueError(
            "Declare all 20 reviewed cases and at most 8 eligible QA cases"
        )
    prepared, seen = [], set()
    for case in cases:
        vid, qid = case["video_id"], case["question_id"]
        if (
            (vid, qid) != reviewed_cases[case["case_id"]]
            or (vid, qid) in seen
            or case["review_status"] != "evidence_confirmed"
        ):
            raise ValueError("Eligible case identity/status mismatch")
        seen.add((vid, qid))
        item, annotation, plan, row, orders = source_case(
            source,
            source_protocol,
            report_digest(source_runtime),
            vid,
            qid,
        )
        selected, intervention = condition_indices(
            case, row["conditions"]["G_topk"], plan["candidate_count"]
        )
        cache = dict(case["pixel_cache"])
        cache["npz_path"] = str(
            resolved(manifest_path.parent, cache["npz_path"])
        )
        original = source_protocol["inputs"][vid]
        if (
            cache["source_video_sha256"] != original["sha256"]
            or cache["source_candidate_sha256"] != plan["candidate_sha256"]
            or not isinstance(cache.get("decoder"), str)
            or not cache["decoder"].strip()
        ):
            raise ValueError("Pixel cache lacks source/decode provenance")
        if file_identity(original["path"])["sha256"] != original["sha256"]:
            raise ValueError("Original source video changed")
        positions, frames, times = load_pixel_pack(
            cache,
            plan,
            {i for values in selected.values() for i in values},
        )
        conditions = {}
        for name, indices in selected.items():
            local = [positions[i] for i in indices]
            conditions[name] = {
                "selected_indices": indices,
                "selected_timestamps": times[local].tolist(),
                "qa_pixels_sha256": array_digest(frames[local], times[local]),
            }
        prepared.append(
            {
                "case_id": case["case_id"],
                "video_id": vid,
                "question_id": qid,
                "source_id": item["source_id"],
                "question": annotation,
                "review_status": case["review_status"],
                "intervention": intervention,
                "pixel_cache": cache,
                "conditions": conditions,
                "orders": orders,
                "candidate_count": plan["candidate_count"],
                "candidate_timestamps": plan["candidate_timestamps"],
                "duration_seconds": plan["duration_seconds"],
                "source_row_sha256": report_digest(row),
                "source_baseline": {
                    key: row["conditions"]["G_topk"][key]
                    for key in (
                        "selected_indices",
                        "selected_timestamps",
                        "qa_pixels_sha256",
                        "trials",
                    )
                },
                "source_diverse_reference": {
                    "accuracy": float(
                        np.mean(
                            [
                                t["scoring_correct"]
                                for t in row["conditions"]["G_diverse"][
                                    "trials"
                                ]
                            ]
                        )
                    ),
                    "qa_pixels_sha256": row["conditions"]["G_diverse"][
                        "qa_pixels_sha256"
                    ],
                    "selected_indices": row["conditions"]["G_diverse"][
                        "selected_indices"
                    ],
                },
            }
        )
    protocol = {
        "version": 1,
        "experiment": EXPERIMENT,
        "manifest_path": str(manifest_path),
        "manifest_sha256": report_digest(manifest),
        "review_manifest": {
            "path": str(review_path),
            "file_sha256": review_ref["file_sha256"],
        },
        "reviewer": manifest["reviewer"],
        "excluded_cases": excluded,
        "reviewed_case_count": 20,
        "planned_case_count": len(cases),
        "source_root": str(source),
        "source_protocol_sha256": report_digest(source_protocol),
        "source_runtime": source_runtime,
        "qa_config": source_protocol["qa_config"],
        "conditions": list(ARMS),
        "reuse_source": args.reuse_source,
        "cases": prepared,
        "code_sha256": code_hashes(),
        "interpretation": INTERPRETATION,
    }
    fixed.atomic_fixed(output / "protocol.json", protocol)
    fixed.atomic_fixed(
        output / "prepare_complete.json",
        {
            "protocol_sha256": report_digest(protocol),
            "planned_case_count": len(cases),
        },
    )
    return protocol


def read_protocol(output):
    protocol = read_json(output / "protocol.json")
    if (
        protocol["experiment"] != EXPERIMENT
        or protocol["code_sha256"] != code_hashes()
    ):
        raise ValueError("Diagnostic implementation changed")
    complete = read_json(output / "prepare_complete.json")
    if (
        complete["protocol_sha256"] != report_digest(protocol)
        or complete["planned_case_count"] != protocol["planned_case_count"]
    ):
        raise ValueError("Prepared diagnostic protocol changed")
    if (
        report_digest(read_json(Path(protocol["manifest_path"])))
        != protocol["manifest_sha256"]
    ):
        raise ValueError("Replacement manifest changed after freezing")
    review = protocol["review_manifest"]
    if file_identity(review["path"])["sha256"] != review["file_sha256"]:
        raise ValueError("Review manifest changed after freezing")
    return protocol


def identity(protocol, runtime_hash, case, name, order):
    condition = case["conditions"][name]
    return {
        "protocol_sha256": report_digest(protocol),
        "runtime_sha256": runtime_hash,
        "case_id": case["case_id"],
        "video_id": case["video_id"],
        "question_id": case["question_id"],
        "selected_indices": condition["selected_indices"],
        "qa_pixels_sha256": condition["qa_pixels_sha256"],
        "order": order,
    }


def reusable_source(protocol, runtime, case, name, order):
    if (
        not protocol["reuse_source"]
        or name != "baseline"
        or runtime != protocol["source_runtime"]
    ):
        return None
    current, old = case["conditions"][name], case["source_baseline"]
    if any(
        current[k] != old[k]
        for k in (
            "selected_indices",
            "selected_timestamps",
            "qa_pixels_sha256",
        )
    ):
        return None
    matches = [t for t in old["trials"] if t["order"] == order]
    if len(matches) != 1:
        raise ValueError("Source baseline lacks an exact option order")
    return {
        "trial": matches[0],
        "origin": "source_G_topk",
        "source_row_sha256": case["source_row_sha256"],
    }


def execute(output, protocol, timeout_seconds=7200, backend_factory=None):
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Require a positive timeout")
    started, done = perf_counter(), 0
    status = {
        "protocol_sha256": report_digest(protocol),
        "status": "running",
        "started_at": fixed.now(),
        "planned_cases": len(protocol["cases"]),
        "planned_qa_records": 18 * len(protocol["cases"]),
        "pid": os.getpid(),
    }

    def progress(**extra):
        save_json(
            output / "progress.json",
            {
                **status,
                "completed_qa_records": done,
                "elapsed_seconds": perf_counter() - started,
                "updated_at": fixed.now(),
                **extra,
            },
        )

    def limit():
        if perf_counter() - started >= timeout_seconds:
            raise TimeoutError(
                "Evidence probe limit reached; no automatic restart"
            )

    progress()
    try:
        if backend_factory is None:
            from .qa_only_backend import QwenOnlyBackend

            backend_factory = QwenOnlyBackend
        config = fixed.config_from(protocol["qa_config"])
        model = backend_factory(config)
        runtime = fixed.model_runtime(model)
        fixed.atomic_fixed(output / "runtime.json", runtime)
        runtime_hash = report_digest(runtime)
        limit()
        for case in protocol["cases"]:
            question = question_from(case["question"])
            required = {
                i
                for c in case["conditions"].values()
                for i in c["selected_indices"]
            }
            positions, frames, times = load_pixel_pack(
                case["pixel_cache"], case, required
            )
            execution = list(ARMS)
            np.random.default_rng(
                config.seed + int(report_digest(case["case_id"])[:8], 16)
            ).shuffle(execution)
            for name in execution:
                indices = case["conditions"][name]["selected_indices"]
                local = [positions[i] for i in indices]
                pixels, timestamps = frames[local], times[local]
                if (
                    array_digest(pixels, timestamps)
                    != case["conditions"][name]["qa_pixels_sha256"]
                ):
                    raise ValueError("Prepared QA pixel identity changed")
                video = Video(
                    case["video_id"],
                    "dev",
                    pixels,
                    timestamps,
                    case["duration_seconds"],
                    (),
                    case["source_id"],
                )
                for order in case["orders"]:
                    limit()
                    key = identity(protocol, runtime_hash, case, name, order)
                    path = fixed.trial_path(output, key)
                    reusable = reusable_source(
                        protocol, runtime, case, name, order
                    )

                    def compute():
                        if reusable is not None:
                            return reusable
                        value = score_with_memory(
                            model, video, question, order, config, 224
                        )
                        fixed.validate_trial(value, question, order, 224)
                        return {"trial": value, "origin": "new"}

                    record = checkpoint(path, key, compute)
                    limit()
                    fixed.validate_trial(record["trial"], question, order, 224)
                    if (
                        (reusable is not None and record != reusable)
                        or (
                            reusable is None
                            and set(record) != {"trial", "origin"}
                        )
                        or (reusable is None and record["origin"] != "new")
                    ):
                        raise ValueError("Trial reuse provenance changed")
                    done += 1
                    progress(case_id=case["case_id"], condition=name)
            print(f"Completed evidence probe: {case['case_id']}", flush=True)
        limit()
        result = aggregate(output)
        progress(status="complete", finished_at=fixed.now())
        return result
    except Exception as error:
        progress(
            status="timeout" if isinstance(error, TimeoutError) else "failed",
            error=f"{type(error).__name__}: {error}",
            finished_at=fixed.now(),
        )
        raise


def aggregate(output):
    protocol = read_protocol(output)
    runtime = read_json(output / "runtime.json")
    runtime_hash = report_digest(runtime)
    rows, metrics, new_trials = [], [], []
    reused = 0
    for case in protocol["cases"]:
        question = question_from(case["question"])
        row = {
            "case_id": case["case_id"],
            "video_id": case["video_id"],
            "question_id": case["question_id"],
            "intervention": case["intervention"],
            "conditions": {},
            "historical_diverse_reference": case["source_diverse_reference"],
        }
        metric = {
            "video_id": case["video_id"],
            "question_id": case["question_id"],
        }
        for name in ARMS:
            trials, provenance = [], []
            for order in case["orders"]:
                key = identity(protocol, runtime_hash, case, name, order)
                record = checkpoint(fixed.trial_path(output, key), key)
                fixed.validate_trial(record["trial"], question, order, 224)
                expected = reusable_source(
                    protocol, runtime, case, name, order
                )
                if expected is not None:
                    if record != expected:
                        raise ValueError("Historical reuse checkpoint changed")
                    reused += 1
                else:
                    if (
                        set(record) != {"trial", "origin"}
                        or record["origin"] != "new"
                    ):
                        raise ValueError("New trial provenance changed")
                    new_trials.append((name, record["trial"]))
                trials.append(record["trial"])
                provenance.append(
                    {
                        "kind": record["origin"],
                        "trial_key": report_digest(key),
                        "trial_sha256": report_digest(record["trial"]),
                    }
                )
            row["conditions"][name] = {
                **case["conditions"][name],
                "trials": trials,
                "trial_provenance": provenance,
            }
            metric[name] = float(
                np.mean([t["scoring_correct"] for t in trials])
            )
        metric.update(
            {
                "repair_minus_baseline": metric["repair"] - metric["baseline"],
                "sham_minus_baseline": metric["sham"] - metric["baseline"],
                "repair_minus_sham": metric["repair"] - metric["sham"],
            }
        )
        rows.append(row)
        metrics.append(metric)
    if len(new_trials) + reused != 18 * protocol["planned_case_count"]:
        raise ValueError("Incomplete planned diagnostic; no partial score")
    resources = {}
    for name in ARMS:
        values = [t for condition, t in new_trials if condition == name]
        resources[name] = {
            "new_qa_trials": len(values),
            "new_qa_seconds": sum(t["qa_wall_seconds"] for t in values),
            "peak_allocated_bytes": max(
                (
                    t.get("cuda_memory", {}).get("peak_allocated_bytes", 0)
                    for t in values
                ),
                default=0,
            ),
            "peak_reserved_bytes": max(
                (
                    t.get("cuda_memory", {}).get("peak_reserved_bytes", 0)
                    for t in values
                ),
                default=0,
            ),
        }
    result = {
        "version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "protocol_sha256": report_digest(protocol),
        "runtime_sha256": runtime_hash,
        "reviewed_case_count": 20,
        "qa_eligible_case_count": len(rows),
        "excluded_before_qa": protocol["excluded_cases"],
        "reviewer": protocol["reviewer"],
        "interpretation": INTERPRETATION,
        "qa_counts": {
            "records": len(new_trials) + reused,
            "new": len(new_trials),
            "source_reused": reused,
        },
        "resources": resources,
        "summary": clustered_means(
            metrics, [*ARMS, *CONTRASTS], seed=42, samples=5000
        ),
        "question_metrics": metrics,
        "cases": rows,
    }
    save_json(output / "comparison.json", result)
    save_json(
        output / "handoff.json",
        {k: v for k, v in result.items() if k != "cases"},
    )
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("prepare", "run", "aggregate"))
    result.add_argument("--manifest", type=Path)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--reuse-source", action="store_true")
    result.add_argument("--timeout-seconds", type=float, default=7200)
    return result


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    with output_lock(output):
        if args.command == "aggregate":
            result = aggregate(output)
        elif (
            args.command == "prepare"
            or not (output / "protocol.json").exists()
        ):
            if args.manifest is None:
                raise ValueError("--manifest is required before preparation")
            protocol = prepare(args)
            result = (
                {"status": "prepared", "planned_cases": len(protocol["cases"])}
                if args.command == "prepare"
                else execute(output, protocol, args.timeout_seconds)
            )
        else:
            protocol = read_protocol(output)
            if protocol["reuse_source"] != args.reuse_source:
                raise ValueError("Historical reuse policy changed")
            result = execute(output, protocol, args.timeout_seconds)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("status", "planned_cases", "qa_counts", "summary")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
