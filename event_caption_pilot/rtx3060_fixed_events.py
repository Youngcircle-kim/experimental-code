"""Fixed-event allocation/selection factorial; every baseline reruns locally.

Preparation uses frozen source features and questions only. GPU inference
starts only in run/smoke, after every selection is prepared and audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import Config
from .data import array_digest
from .event_refinement_dev import question_from
from .fixed_event_selection import ALGORITHM_SPEC, CONDITIONS, make_conditions
from .frame_replacement_probe import strict_trials
from .paired_reanalysis import clustered_means
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

EXPERIMENT = "rtx3060_fixed_event_allocation_selection_v1"
CODE_FILES = (
    "rtx3060_fixed_events.py",
    "fixed_event_selection.py",
    "qa_only_backend.py",
    "hf_backend.py",
    "backends.py",
    "config.py",
    "data.py",
    "types.py",
    "event_factorial_dev.py",
    "event_refinement_dev.py",
    "order_experiments.py",
    "reproducibility.py",
    "resolution_dev.py",
    "video_sampling.py",
    "portable_budget.py",
    "paired_reanalysis.py",
    "frame_replacement_probe.py",
    "qa_diagnostics.py",
    "legacy_complexity_analysis.py",
    "query_diversity_analysis.py",
    "encoders.py",
)
COMPARISONS = (
    "G_topk_minus_R_topk",
    "R_diverse_minus_R_topk",
    "G_diverse_minus_R_topk",
    "interaction",
    "allocation_main_effect",
    "selection_main_effect",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def config_from(settings):
    settings = dict(settings)
    settings["observation_strides"] = tuple(settings["observation_strides"])
    result = Config(**settings)
    result.validate()
    return result


def code_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in CODE_FILES
    }


def atomic_fixed(path, value):
    """Create once; subsequent invocations must have identical settings."""
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen settings changed: {path}")
    else:
        save_json(path, value)


def plan_identity(protocol, vid):
    return {"protocol_sha256": report_digest(protocol), "video_id": vid}


def plan_path(output, vid):
    return output / "plans" / (report_digest(vid) + ".json")


def question_path(output, vid, qid):
    return output / "questions" / (report_digest([vid, qid]) + ".json")


def load_source(source):
    """Bind the snapshot, completed suite and all frozen video plans."""
    suite = read_json(source / "suite_plan.json")
    handoff = read_json(source / "handoff.json")
    snapshot = read_json(source / "source_snapshot.json")
    suite_hash = report_digest(suite)
    if (
        suite.get("snapshot_sha256") != report_digest(snapshot)
        or handoff.get("suite_plan_sha256") != suite_hash
        or handoff.get("status") != "complete"
    ):
        raise ValueError("Source suite/snapshot/completion identity mismatch")
    baseline = snapshot["baseline"]
    videos = baseline["videos"]
    cohort = {
        v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
        for v in videos
    }
    if (
        len(videos) != 20
        or sum(map(len, cohort.values())) != 60
        or cohort != baseline["cohort"]
        or len(cohort) != len(videos)
    ):
        raise ValueError(
            "Require the complete frozen 20-video/60-question cohort"
        )
    config = config_from(baseline["config"])
    if (
        config.frame_budget != 32
        or config.vlm_model != "Qwen/Qwen3.5-4B"
        or config.encoder_revision == "main"
        or config.vlm_revision == "main"
    ):
        raise ValueError("Require the pinned B32 source models")
    rows = {
        (r["video_id"], r["question_id"]): r for r in snapshot["source_rows"]
    }
    expected = {(v, q) for v, questions in cohort.items() for q in questions}
    if set(rows) != expected or len(rows) != len(snapshot["source_rows"]):
        raise ValueError("Source question cohort incomplete or duplicated")
    plans, orders = {}, {}
    for item in videos:
        vid = item["video_id"]
        if item["split"] != "dev" or item["duration"] != "long":
            raise ValueError("Only the frozen dev Long cohort is permitted")
        plan = checkpoint(
            source / "plans" / (report_digest(vid) + ".json"),
            {"suite_plan_sha256": suite_hash, "video_id": vid},
        )
        plans[vid] = plan
        if plan["video_id"] != vid or set(plan["questions"]) != set(
            cohort[vid]
        ):
            raise ValueError("Source prepared question cohort mismatch")
        for annotation in item["questions"]:
            q = question_from(annotation)
            row = rows[(vid, q.question_id)]
            old = row["conditions"]["legacy_R_topk"]
            selection = plan["questions"][q.question_id]["A_R_topk"]
            if (
                row["question"] != q.text
                or row["options"] != list(q.options)
                or row["answer_index_evaluator_only"] != q.answer_index
                or row["candidate_sha256"] != plan["candidate_sha256"]
                or row["candidate_count"] != plan["candidate_count"]
            ):
                raise ValueError(
                    "Source annotation/candidate identity mismatch"
                )
            for key in (
                "events",
                "allocation",
                "selected_indices",
                "selected_timestamps",
            ):
                if selection[key] != old[key]:
                    raise ValueError(f"Frozen A baseline changed: {key}")
            strict_trials(old["trials"], q, config)
            order_list = [t["order"] for t in old["trials"]]
            if (
                len(order_list) != 6
                or len({tuple(x) for x in order_list}) != 6
            ):
                raise ValueError(
                    "Require all six distinct frozen option orders"
                )
            orders[f"{vid}/{q.question_id}"] = order_list
    return (
        baseline,
        plans,
        orders,
        {
            "suite_plan_sha256": suite_hash,
            "snapshot_sha256": suite["snapshot_sha256"],
            "handoff_sha256": report_digest(handoff),
        },
    )


def preflight_summary(plans):
    values = {name: [] for name in CONDITIONS}
    changed = {name: [] for name in CONDITIONS if name != "R_topk"}
    for plan in plans.values():
        for conditions in plan["questions"].values():
            original = set(conditions["R_topk"]["selected_indices"])
            for name, selected in conditions.items():
                values[name].append(selected.get("diagnostics", {}))
                if name != "R_topk":
                    changed[name].append(
                        len(original - set(selected["selected_indices"]))
                    )
    return {
        "questions": len(values["R_topk"]),
        "versus_R_topk": {
            name: {
                "changed_questions": sum(x > 0 for x in counts),
                "mean_replaced_frames": float(np.mean(counts)),
                "max_replaced_frames": max(counts),
            }
            for name, counts in changed.items()
        },
        "condition_diagnostics": values,
    }


def pixel_request(vid, plan, source_sha256, side):
    """Portable CPU decoder request; independent of the execution protocol."""
    return {
        "video_id": vid,
        "candidate_sha256": plan["candidate_sha256"],
        "source_sha256": source_sha256,
        "qa_side": side,
        "indices": sorted(
            {
                i
                for conditions in plan["questions"].values()
                for selection in conditions.values()
                for i in selection["selected_indices"]
            }
        ),
    }


def external_pixels(directory, vid, plan, source_sha256, side):
    """Pin a verified Linux-decoded cache when local codecs differ."""
    request = pixel_request(vid, plan, source_sha256, side)
    metadata = read_json(directory / f"{vid}.json")
    if (
        any(metadata.get(k) != v for k, v in request.items())
        or metadata.get("selection_request_sha256") != report_digest(request)
        or metadata["decode_info"]["candidate_sha256"]
        != plan["candidate_sha256"]
        or metadata["decode_info"]["candidate_count"]
        != plan["candidate_count"]
        or metadata["duration_seconds"] != plan["duration_seconds"]
    ):
        raise ValueError(
            "External pixels do not match the frozen decode request"
        )
    npz_path = directory / f"{vid}.npz"
    if file_identity(npz_path)["sha256"] != metadata["file_sha256"]:
        raise ValueError("External QA frame file hash mismatch")
    return {
        "npz_path": str(npz_path),
        "metadata_path": str(directory / f"{vid}.json"),
        "metadata": metadata,
        "metadata_sha256": report_digest(metadata),
    }


def prepare(args):
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if (
        source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be separate from the source experiment")
    if args.qa_side not in (224, 320):
        raise ValueError(
            "Only the predeclared QA sides 320 and 224 are allowed"
        )
    baseline, source_plans, orders, source_hashes = load_source(source)
    original = config_from(baseline["config"])
    target = replace(
        original,
        device="cuda:0",
        output_dir=str(output),
        model_cache_dir=str(args.model_cache_dir.resolve()),
        manifest_path=str(source / "source_snapshot.json"),
        vlm_min_pixels=args.qa_side**2,
        vlm_max_pixels=args.qa_side**2,
        local_files_only=True,
    )
    target.validate()
    inputs, feature_info, selections = {}, {}, {}
    for item in baseline["videos"]:
        vid = item["video_id"]
        path = args.video_root.resolve() / Path(item["path"]).name
        actual = file_identity(path)
        prior = baseline["inputs"][vid]
        if any(actual[k] != prior[k] for k in ("sha256", "size_bytes")):
            raise ValueError(f"Original video content changed: {vid}")
        inputs[vid] = actual
        feature_path = args.feature_dir.resolve() / (
            report_digest(vid) + ".npy"
        )
        features = np.load(feature_path, allow_pickle=False)
        old = source_plans[vid]
        if (
            array_digest(features) != old["features_sha256"]
            or list(features.shape) != old["feature_shape"]
            or str(features.dtype) != old["feature_dtype"]
        ):
            raise ValueError(f"Frozen feature content changed: {vid}")
        feature_info[vid] = {
            "path": str(feature_path),
            "array_sha256": old["features_sha256"],
        }
        times = np.asarray(old["candidate_timestamps"], dtype=float)
        questions = {}
        for qid in baseline["cohort"][vid]:
            base = old["questions"][qid]["A_R_topk"]
            conditions = make_conditions(base, features, times, budget=32)
            if set(conditions) != set(CONDITIONS):
                raise ValueError(
                    "Selection implementation omitted a condition"
                )
            for name, selection in conditions.items():
                if selection["events"] != base["events"]:
                    raise ValueError("Event boundaries must remain identical")
                indices = selection["selected_indices"]
                if (
                    len(indices) != 32
                    or sorted(set(indices)) != indices
                    or min(indices) < 0
                    or max(indices) >= len(times)
                    or selection["selected_timestamps"]
                    != times[indices].tolist()
                    or sum(selection["allocation"]) != 32
                ):
                    raise ValueError("Invalid fixed-budget selected frames")
                if name == "R_topk" and any(
                    selection[k] != base[k]
                    for k in (
                        "events",
                        "allocation",
                        "selected_indices",
                        "selected_timestamps",
                    )
                ):
                    raise ValueError(
                        "R_topk must replay the exact source selection"
                    )
            questions[qid] = conditions
        selections[vid] = {
            "video_id": vid,
            "candidate_sha256": old["candidate_sha256"],
            "candidate_count": old["candidate_count"],
            "candidate_timestamps": old["candidate_timestamps"],
            "duration_seconds": old["duration_seconds"],
            "questions": questions,
            "source_plan_sha256": report_digest(old),
        }
        print(f"Prepared selections: {vid} ({len(selections)}/20)", flush=True)
    stress = max(
        (
            (
                len(q["text"]) + sum(map(len, q["options"])),
                v["video_id"],
                str(q["question_id"]),
            )
            for v in baseline["videos"]
            for q in v["questions"]
        ),
    )
    smoke = args.command == "smoke"
    execution_cohort = (
        {stress[1]: [stress[2]]} if smoke else baseline["cohort"]
    )
    external = {}
    if args.pixel_cache_dir is not None:
        directory = args.pixel_cache_dir.resolve()
        external = {
            vid: external_pixels(
                directory,
                vid,
                selections[vid],
                inputs[vid]["sha256"],
                args.qa_side,
            )
            for vid in execution_cohort
        }
    protocol = {
        "version": 1,
        "experiment": EXPERIMENT,
        "scope": "smoke" if smoke else "full",
        "source_root": str(source),
        "source": source_hashes,
        "cohort": baseline["cohort"],
        "videos": baseline["videos"],
        "execution_cohort": execution_cohort,
        "orders_per_question": 1 if smoke else 6,
        "smoke_selection": (
            "maximum question+options character count; "
            "ties by video/question ID"
        ),
        "candidate_config": original.to_dict(),
        "qa_config": target.to_dict(),
        "qa_side": args.qa_side,
        "frame_budget": 32,
        "conditions": list(CONDITIONS),
        "algorithm": ALGORITHM_SPEC,
        "inputs": inputs,
        "features": feature_info,
        "external_pixels": external,
        "orders": orders,
        "prepared_sha256": {
            v: report_digest(p) for v, p in selections.items()
        },
        "code_sha256": code_hashes(),
        "runtime": runtime_metadata(),
        "comparisons": list(COMPARISONS),
        "bootstrap": {"unit": "video", "samples": 5000, "seed": 42},
        "reuse_policy": (
            "Only exact inputs within this local protocol/runtime; "
            "no historical QA reuse"
        ),
        "interpretation": (
            "Exploratory dev; six option orders are repeats within "
            "each question, not independent samples"
        ),
    }
    atomic_fixed(output / "protocol.json", protocol)
    for vid, prepared in selections.items():
        checkpoint(
            plan_path(output, vid),
            plan_identity(protocol, vid),
            lambda p=prepared: p,
        )
    atomic_fixed(
        output / "selection_preflight.json",
        {
            "protocol_sha256": report_digest(protocol),
            **preflight_summary(selections),
        },
    )
    save_json(
        output / "prepare_complete.json",
        {
            "status": "complete",
            "protocol_sha256": report_digest(protocol),
            "videos": 20,
            "questions": 60,
            "prepared_sha256": protocol["prepared_sha256"],
        },
    )
    return protocol


def read_prepared(output):
    protocol = read_json(output / "protocol.json")
    if (
        protocol["experiment"] != EXPERIMENT
        or protocol["code_sha256"] != code_hashes()
    ):
        raise ValueError("Frozen experiment implementation changed")
    if protocol["runtime"] != runtime_metadata():
        raise ValueError("Python/package/environment runtime changed")
    complete = read_json(output / "prepare_complete.json")
    if (
        complete["protocol_sha256"] != report_digest(protocol)
        or complete["prepared_sha256"] != protocol["prepared_sha256"]
    ):
        raise ValueError("Incomplete or changed preparation")
    plans = {}
    for vid in protocol["cohort"]:
        plan = checkpoint(plan_path(output, vid), plan_identity(protocol, vid))
        if report_digest(plan) != protocol["prepared_sha256"][vid]:
            raise ValueError("Prepared selection content changed")
        plans[vid] = plan
    expected_preflight = {
        "protocol_sha256": report_digest(protocol),
        **preflight_summary(plans),
    }
    if read_json(output / "selection_preflight.json") != expected_preflight:
        raise ValueError("Selection preflight diagnostics changed")
    return protocol, plans


def frame_cache(output, protocol, plan):
    """Verify the candidate stream; cache the union of selected inputs."""
    vid = plan["video_id"]
    indices = sorted(
        {
            i
            for conditions in plan["questions"].values()
            for selection in conditions.values()
            for i in selection["selected_indices"]
        }
    )
    original = config_from(protocol["candidate_config"])
    expected_file = protocol["inputs"][vid]
    actual = file_identity(expected_file["path"])
    if any(actual[k] != expected_file[k] for k in ("sha256", "size_bytes")):
        raise ValueError("Original source video changed after preparation")
    identity = {
        **plan_identity(protocol, vid),
        "plan_sha256": report_digest(plan),
        "source_sha256": expected_file["sha256"],
    }
    stem = output / "frames" / report_digest(vid)
    npz_path, meta_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
    external = protocol.get("external_pixels", {}).get(vid)
    if external is not None:
        metadata = external["metadata"]
        if (
            read_json(Path(external["metadata_path"])) != metadata
            or report_digest(metadata) != external["metadata_sha256"]
        ):
            raise ValueError("Frozen external pixel metadata changed")
        npz_path = Path(external["npz_path"])
    elif not meta_path.exists():
        _, pixels, times, duration, info = decode_selected(
            actual["path"],
            original,
            plan["candidate_count"],
            indices,
            protocol["qa_side"],
            plan["candidate_sha256"],
        )
        if times.tolist() != plan["candidate_timestamps"]:
            raise ValueError("Decoded candidate timestamps changed")
        frames = np.stack([pixels[i] for i in indices])
        selected_times = times[indices]
        wanted = np.asarray(indices, dtype=np.int64)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = npz_path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                indices=wanted,
                frames=frames,
                timestamps=selected_times,
            )
        os.replace(temporary, npz_path)
        checkpoint(
            meta_path,
            identity,
            lambda: {
                "array_sha256": array_digest(wanted, frames, selected_times),
                "file_sha256": file_identity(npz_path)["sha256"],
                "duration_seconds": duration,
                "decode_info": info,
            },
        )
    if external is None:
        metadata = checkpoint(meta_path, identity)
    if file_identity(npz_path)["sha256"] != metadata["file_sha256"]:
        raise ValueError("QA frame cache file changed")
    with np.load(npz_path, allow_pickle=False) as saved:
        wanted, frames, times = (
            saved["indices"],
            saved["frames"],
            saved["timestamps"],
        )
    if (
        wanted.tolist() != indices
        or frames.dtype != np.uint8
        or frames.shape
        != (len(indices), protocol["qa_side"], protocol["qa_side"], 3)
        or times.tolist() != [plan["candidate_timestamps"][i] for i in indices]
        or array_digest(wanted, frames, times) != metadata["array_sha256"]
    ):
        raise ValueError("QA frame cache content changed")
    return {i: j for j, i in enumerate(indices)}, frames, times, metadata


def validate_trial(trial, question, order, side):
    if trial["order"] != order or trial["presented_options"] != [
        question.options[i] for i in order
    ]:
        raise ValueError("QA trial question/order mismatch")
    scores = np.asarray(trial["scoring"]["option_scores"], dtype=float)
    predicted = trial["scoring"]["predicted_index"]
    if (
        scores.shape != (len(order),)
        or not np.isfinite(scores).all()
        or type(predicted) is not int
        or predicted != int(np.argmax(scores))
        or trial["scored_original_index"] != order[predicted]
        or trial["scoring_correct"]
        != (order[predicted] == question.answer_index)
        or not np.isfinite(trial["qa_wall_seconds"])
        or trial["qa_wall_seconds"] < 0
    ):
        raise ValueError("Invalid saved QA scores or correctness")
    validate_processor(deepcopy(trial), side, 32)


def trial_identity(
    protocol, runtime_hash, vid, qid, selection, pixel_hash, order
):
    return {
        "protocol_sha256": report_digest(protocol),
        "runtime_sha256": runtime_hash,
        "video_id": vid,
        "question_id": qid,
        "selected_indices": selection["selected_indices"],
        "qa_pixels_sha256": pixel_hash,
        "order": order,
    }


def trial_path(output, identity):
    return output / "trials" / (report_digest(identity) + ".json")


def counts(rows):
    provenance = [
        p
        for row in rows
        for condition in row["conditions"].values()
        for p in condition["trial_provenance"]
    ]
    return {
        "qa_records": len(provenance),
        "new_qa_trials": sum(p["kind"] == "new" for p in provenance),
        "reused_qa_trials": sum(
            p["kind"] == "current_condition" for p in provenance
        ),
        "historical_qa_trials_reused": 0,
    }


def model_runtime(model):
    """Bind resumed QA to the actual local accelerator, not just cuda:0."""
    result = {"runtime": runtime_metadata(), "backend": model.metadata()}
    device = getattr(model, "device", None)
    if device is not None and device.type == "cuda":
        props = model.torch.cuda.get_device_properties(device)
        result["accelerator"] = {
            "name": props.name,
            "total_memory": props.total_memory,
            "compute_capability": [props.major, props.minor],
            "cuda_runtime": model.torch.version.cuda,
            "uuid": str(getattr(props, "uuid", "unavailable")),
        }
    return result


def execute(output, protocol, plans, timeout_seconds, backend_factory=None):
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be finite and positive")
    started = perf_counter()
    if backend_factory is None:
        from .qa_only_backend import QwenOnlyBackend

        backend_factory = QwenOnlyBackend
    config = config_from(protocol["qa_config"])
    rows, model = [], None
    expected_questions = sum(map(len, protocol["execution_cohort"].values()))
    status = {
        "protocol_sha256": report_digest(protocol),
        "status": "running",
        "scope": protocol["scope"],
        "started_at": now(),
        "expected_questions": expected_questions,
        "pid": os.getpid(),
    }

    def progress(**extra):
        save_json(
            output / "progress.json",
            {
                **status,
                "updated_at": now(),
                "completed_questions": len(rows),
                "elapsed_seconds": perf_counter() - started,
                **counts(rows),
                **extra,
            },
        )

    def time_check():
        if perf_counter() - started >= timeout_seconds:
            raise TimeoutError(
                "Experiment wall-clock limit reached; no automatic restart"
            )

    progress()
    try:
        for item in protocol["videos"]:
            vid = item["video_id"]
            if vid not in protocol["execution_cohort"]:
                continue
            time_check()
            plan = plans[vid]
            positions, frames, times, meta = frame_cache(
                output, protocol, plan
            )
            time_check()
            if model is None:
                model = backend_factory(config)
                runtime = model_runtime(model)
                atomic_fixed(output / "runtime.json", runtime)
                runtime_hash = report_digest(runtime)
                time_check()
            for annotation in item["questions"]:
                q = question_from(annotation)
                if q.question_id not in protocol["execution_cohort"][vid]:
                    continue
                execution_order = list(CONDITIONS)
                np.random.default_rng(
                    config.seed
                    + int(report_digest([vid, q.question_id])[:8], 16)
                ).shuffle(execution_order)
                row = {
                    "video_id": vid,
                    "question_id": q.question_id,
                    "duration_group": "long",
                    "execution_order": execution_order,
                    "conditions": {},
                    "decode_info": meta["decode_info"],
                }
                orders = protocol["orders"][f"{vid}/{q.question_id}"][
                    : protocol["orders_per_question"]
                ]
                for name in execution_order:
                    selected = plan["questions"][q.question_id][name]
                    local_indices = [
                        positions[i] for i in selected["selected_indices"]
                    ]
                    selected_frames, selected_times = (
                        frames[local_indices],
                        times[local_indices],
                    )
                    pixel_hash = array_digest(selected_frames, selected_times)
                    video = Video(
                        vid,
                        "dev",
                        selected_frames,
                        selected_times,
                        meta["duration_seconds"],
                        (),
                        item["source_id"],
                    )
                    trials, provenance = [], []
                    for order in orders:
                        time_check()
                        identity = trial_identity(
                            protocol,
                            runtime_hash,
                            vid,
                            q.question_id,
                            selected,
                            pixel_hash,
                            order,
                        )
                        path = trial_path(output, identity)

                        def compute():
                            trial = score_with_memory(
                                model,
                                video,
                                q,
                                order,
                                config,
                                protocol["qa_side"],
                            )
                            validate_trial(
                                trial, q, order, protocol["qa_side"]
                            )
                            return {"trial": trial, "origin_condition": name}

                        record = checkpoint(path, identity, compute)
                        time_check()
                        validate_trial(
                            record["trial"], q, order, protocol["qa_side"]
                        )
                        origin = record["origin_condition"]
                        if (
                            origin not in CONDITIONS
                            or plan["questions"][q.question_id][origin][
                                "selected_indices"
                            ]
                            != selected["selected_indices"]
                        ):
                            raise ValueError(
                                "Reused trial lacks identical selected inputs"
                            )
                        trials.append(record["trial"])
                        provenance.append(
                            {
                                "kind": "new"
                                if origin == name
                                else "current_condition",
                                "origin_condition": origin,
                                "trial_key": report_digest(identity),
                                "trial_sha256": report_digest(record["trial"]),
                            }
                        )
                        progress(
                            current_video=vid,
                            current_question=q.question_id,
                            current_condition=name,
                            current_order=len(trials),
                        )
                    row["conditions"][name] = {
                        **selected,
                        "qa_pixels_sha256": pixel_hash,
                        "trials": trials,
                        "trial_provenance": provenance,
                    }
                identity = {
                    **plan_identity(protocol, vid),
                    "question_id": q.question_id,
                    "runtime_sha256": runtime_hash,
                }
                persisted = checkpoint(
                    question_path(output, vid, q.question_id),
                    identity,
                    lambda: row,
                )
                if persisted != row:
                    raise ValueError(
                        "Question checkpoint differs from verified inputs"
                    )
                rows.append(row)
                progress()
                print(
                    f"Completed {vid}/{q.question_id} "
                    f"({len(rows)}/{expected_questions})",
                    flush=True,
                )
        time_check()
        result = aggregate(output)
        progress(status=result["status"], finished_at=now())
        return result
    except Exception as error:
        progress(
            status="timeout" if isinstance(error, TimeoutError) else "failed",
            error=f"{type(error).__name__}: {error}",
            finished_at=now(),
        )
        raise


def aggregate(output):
    protocol, plans = read_prepared(output)
    runtime = read_json(output / "runtime.json")
    runtime_hash = report_digest(runtime)
    rows, metrics = [], []
    config = config_from(protocol["qa_config"])
    for item in protocol["videos"]:
        vid = item["video_id"]
        if vid not in protocol["execution_cohort"]:
            continue
        for annotation in item["questions"]:
            q = question_from(annotation)
            if q.question_id not in protocol["execution_cohort"][vid]:
                continue
            identity = {
                **plan_identity(protocol, vid),
                "question_id": q.question_id,
                "runtime_sha256": runtime_hash,
            }
            row = checkpoint(
                question_path(output, vid, q.question_id), identity
            )
            if (
                row["video_id"] != vid
                or row["question_id"] != q.question_id
                or set(row["conditions"]) != set(CONDITIONS)
            ):
                raise ValueError(
                    "Incomplete or mismatched question checkpoint"
                )
            metric = {"video_id": vid, "question_id": q.question_id}
            orders = protocol["orders"][f"{vid}/{q.question_id}"][
                : protocol["orders_per_question"]
            ]
            for name in CONDITIONS:
                condition = row["conditions"][name]
                selection = plans[vid]["questions"][q.question_id][name]
                if any(condition.get(k) != v for k, v in selection.items()):
                    raise ValueError(
                        "Recorded trial selection differs from frozen plan"
                    )
                if len(condition["trials"]) != len(orders) or len(
                    condition["trial_provenance"]
                ) != len(orders):
                    raise ValueError("Question lacks every option-order trial")
                if len(orders) == 6:
                    strict_trials(condition["trials"], q, config)
                for order, trial, provenance in zip(
                    orders, condition["trials"], condition["trial_provenance"]
                ):
                    validate_trial(trial, q, order, protocol["qa_side"])
                    key = trial_identity(
                        protocol,
                        runtime_hash,
                        vid,
                        q.question_id,
                        selection,
                        condition["qa_pixels_sha256"],
                        order,
                    )
                    record = checkpoint(trial_path(output, key), key)
                    origin = record["origin_condition"]
                    expected = {
                        "kind": "new"
                        if origin == name
                        else "current_condition",
                        "origin_condition": origin,
                        "trial_key": report_digest(key),
                        "trial_sha256": report_digest(trial),
                    }
                    if (
                        record["trial"] != trial
                        or provenance != expected
                        or origin not in CONDITIONS
                        or row["conditions"][origin]["selected_indices"]
                        != condition["selected_indices"]
                        or row["conditions"][origin]["qa_pixels_sha256"]
                        != condition["qa_pixels_sha256"]
                    ):
                        raise ValueError(
                            "Trial checkpoint/reuse provenance mismatch"
                        )
                metric[name] = float(
                    np.mean(
                        [t["scoring_correct"] for t in condition["trials"]]
                    )
                )
            r, g, d, gd = (metric[x] for x in CONDITIONS)
            metric.update(
                {
                    "G_topk_minus_R_topk": g - r,
                    "R_diverse_minus_R_topk": d - r,
                    "G_diverse_minus_R_topk": gd - r,
                    "interaction": (gd - g) - (d - r),
                    "allocation_main_effect": ((g - r) + (gd - d)) / 2,
                    "selection_main_effect": ((d - r) + (gd - g)) / 2,
                }
            )
            metrics.append(metric)
            rows.append(row)
    full = protocol["scope"] == "full"
    expected = 60 if full else 1
    if len(rows) != expected or (
        full and len({r["video_id"] for r in rows}) != 20
    ):
        raise ValueError(
            "Incomplete experiment; partial accuracy is not reported"
        )
    resources = {}
    for name in CONDITIONS:
        trials = [
            (t, p)
            for r in rows
            for t, p in zip(
                r["conditions"][name]["trials"],
                r["conditions"][name]["trial_provenance"],
            )
        ]
        new = [t for t, p in trials if p["kind"] == "new"]
        resources[name] = {
            "new_qa_trials": len(new),
            "reused_qa_trials": len(trials) - len(new),
            "new_qa_seconds": sum(t["qa_wall_seconds"] for t in new),
            "new_qa_median_seconds": float(
                np.median([t["qa_wall_seconds"] for t in new])
            )
            if new
            else None,
            "new_peak_allocated_bytes": max(
                (
                    t.get("cuda_memory", {}).get("peak_allocated_bytes", 0)
                    for t in new
                ),
                default=0,
            ),
            "new_peak_reserved_bytes": max(
                (
                    t.get("cuda_memory", {}).get("peak_reserved_bytes", 0)
                    for t in new
                ),
                default=0,
            ),
        }
    result = {
        "version": 1,
        "experiment": EXPERIMENT,
        "status": "complete" if full else "smoke_complete",
        "completed": full,
        "protocol_sha256": report_digest(protocol),
        "runtime_sha256": runtime_hash,
        "summary": clustered_means(
            metrics, [*CONDITIONS, *COMPARISONS], seed=42, samples=5000
        ),
        "qa_counts": counts(rows),
        "resources": resources,
        "selection_preflight": read_json(output / "selection_preflight.json"),
        "interpretation": protocol["interpretation"],
        "questions": rows,
    }
    save_json(output / "comparison.json", result)
    save_json(
        output / "handoff.json",
        {k: v for k, v in result.items() if k != "questions"},
    )
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command", choices=("prepare", "smoke", "run", "aggregate")
    )
    result.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "outputs/query-diversity-campaign-20261026/downloaded/a4500_query_diversity_abc"
        ),
    )
    result.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("outputs/rtx3060-fixed-events-20261027/features"),
    )
    result.add_argument(
        "--video-root", type=Path, default=Path("data/videomme/videos")
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument(
        "--model-cache-dir", type=Path, default=Path(".cache/huggingface")
    )
    result.add_argument("--pixel-cache-dir", type=Path)
    result.add_argument("--qa-side", type=int, default=320)
    result.add_argument("--timeout-seconds", type=float, default=4 * 60 * 60)
    return result


def main(argv=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    with output_lock(output):
        if args.command == "aggregate":
            result = aggregate(output)
        elif args.command == "prepare":
            prepare(args)
            result = {"status": "prepared", "output": str(output)}
        else:
            if not (output / "prepare_complete.json").exists():
                prepare(args)
            protocol, plans = read_prepared(output)
            expected_scope = "smoke" if args.command == "smoke" else "full"
            if (
                protocol["scope"] != expected_scope
                or protocol["qa_side"] != args.qa_side
            ):
                raise ValueError(
                    "Use separate outputs for smoke/full or other resolutions"
                )
            result = execute(output, protocol, plans, args.timeout_seconds)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("status", "qa_counts", "summary", "output")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
