"""Frozen long-dev comparison at 32 frames, with paired 16-frame references."""

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import event_selection_ablation_dev as prior
from . import resolution_dev
from .backends import build_backend
from .bottleneck_diagnostics import read, write_csv
from .cache import write_json
from .data import array_digest
from .event_factorial_dev import validate_indices
from .event_refinement_dev import checked_payload, question_from, save_payload
from .frame_budget_analysis import ARMS, make_conditions, summarize
from .frame_replacement_probe import strict_trials
from .order_experiments import seed_call
from .qa_diagnostics import report_digest
from .selector_comparison_dev import canonical, indexed
from .selector_comparison_holdout import output_lock
from .types import Video

BUDGET = 32
SIDE = 320


def question_identity(digest, vid, qid):
    return {
        "protocol_sha256": digest,
        "video_id": vid,
        "question_id": qid,
    }


def load_inputs(args):
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if (
        output == source
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be separate from the 16-frame source")
    if any(
        n is not None and n < 1 for n in (args.max_videos, args.max_questions)
    ):
        raise ValueError("Subset limits must be positive")
    saved = read(source / "protocol.json")
    if saved.get("experiment") != "event_selection_ablation_fixed_320":
        raise ValueError("Require the completed seven-arm 16-frame experiment")
    # The prior runner is read-only here. Its output-dependent resume check
    # must not inspect this different experiment's protocol.
    validation_output = output / ".source_validation_read_only"
    if validation_output.exists():
        raise ValueError("Reserved source-validation path must not exist")
    old_args = prior.parser().parse_args(
        [
            "--source-dir",
            saved["source_dir"],
            "--output-dir",
            str(validation_output),
        ]
    )
    frozen = prior.load_inputs(old_args)
    # Added standalone runners do not change the source implementation.
    # Existing modules must match, except the exact recorded JSON-only fix
    # already recognized by the source runner's strict compatibility check.
    runtime = deepcopy(frozen.protocol)
    runtime["code_sha256"] = {
        name: runtime["code_sha256"][name] for name in saved["code_sha256"]
    }
    if prior.resume_protocol(runtime, source) != saved:
        raise ValueError("Source protocol does not reproduce")
    digest = report_digest(saved)
    results = read(source / "results.json")
    if results["protocol_sha256"] != digest:
        raise ValueError("16-frame results/protocol mismatch")
    baseline_rows = indexed(results["questions"])
    if set(baseline_rows) != set(frozen.plans):
        raise ValueError("16-frame source cohort is incomplete")
    for item in frozen.videos:
        for annotation in item["questions"]:
            q = question_from(annotation)
            key = (item["video_id"], q.question_id)
            row = baseline_rows[key]
            prior.validate_result(
                row,
                frozen.plans[key],
                frozen.source_rows[key],
                q,
                frozen.config,
            )
            prior.validate_checkpoints(row, source, digest)
            qi = question_identity(digest, *key)
            if (
                checked_payload(
                    source / "questions" / (report_digest(qi) + ".json"), qi
                )
                != row
            ):
                raise ValueError("16-frame question checkpoint changed")
    summary, _ = prior.summarize(list(baseline_rows.values()), saved)
    if (
        summary != read(source / "handoff.json")
        or summary != results["summary"]
    ):
        raise ValueError("16-frame summary does not match raw results")
    if (
        frozen.original.frame_budget != 16
        or frozen.original.max_segments != 128
        or frozen.config.vlm_max_pixels != SIDE**2
    ):
        raise ValueError("Require frozen 16-frame, cap-128, 320 source")
    config = replace(frozen.config, frame_budget=BUDGET)
    config.validate()
    selector_protocol = read(Path(saved["source_dir"]) / "protocol.json")
    resolution_dir = Path(selector_protocol["source_dir"])
    resolution_results = read(resolution_dir / "results.json")
    if (
        report_digest(resolution_results)
        != selector_protocol["source_results_sha256"]
    ):
        raise ValueError("C_320 source changed")
    resolution_rows = indexed(resolution_results["questions"])
    cap_results = read(Path(saved["cap_source_dir"]) / "results.json")
    frame_results = read(
        Path(selector_protocol["factorial_dir"]) / "results.json"
    )
    if (
        report_digest(cap_results) != saved["cap_results_sha256"]
        or report_digest(frame_results) != saved["frame_scores_source_sha256"]
    ):
        raise ValueError("Frozen event plans or frame scores changed")
    cap_rows, frame_rows = (
        indexed(cap_results["questions"]),
        indexed(frame_results["questions"]),
    )
    videos = [
        {**v, "questions": v["questions"][: args.max_questions]}
        for v in frozen.videos[: args.max_videos]
    ]
    plans, selected_baselines = {}, {}
    for item in videos:
        if item["split"] != "dev" or item["duration"] != "long":
            raise ValueError("Require long development videos")
        for annotation in item["questions"]:
            q = question_from(annotation)
            key = (item["video_id"], q.question_id)
            cap, frames = cap_rows[key], frame_rows[key]
            if (
                cap["candidate_sha256"]
                != baseline_rows[key]["candidate_sha256"]
                or cap["candidate_sha256"] != frames["candidate_sha256"]
            ):
                raise ValueError("Candidate identity differs across sources")
            conditions = make_conditions(
                cap["conditions"]["D"],
                cap["conditions"]["C"],
                frames["conditions"]["frame_top"]["frame_scores"],
                BUDGET,
            )
            plans[key] = {
                "video_id": key[0],
                "question_id": key[1],
                "candidate_sha256": cap["candidate_sha256"],
                "candidate_count": cap["conditions"]["D"]["events"][-1][1],
                "conditions": conditions,
            }
            baseline = deepcopy(baseline_rows[key])
            c = deepcopy(resolution_rows[key]["conditions"]["C_320"])
            strict_trials(c["trials"], q, frozen.config)
            for t in c["trials"]:
                resolution_dev.validate_processor(t, SIDE, 16)
            if (
                c["selected_indices"]
                != cap["conditions"]["C"]["selected_indices"]
            ):
                raise ValueError("Saved C_320 selection differs")
            c["qa_origin"] = "source"
            baseline["conditions"]["C"] = c
            selected_baselines[key] = baseline
    protocol = canonical(
        {
            "version": 1,
            "experiment": "frozen_frame_budget_32_long_dev",
            "evaluation_role": "exploratory_dev"
            if args.max_videos is None and args.max_questions is None
            else "dev_smoke",
            "source_dir": str(source),
            "source_protocol_sha256": digest,
            "source_results_sha256": report_digest(results),
            "C_16_source_dir": str(resolution_dir),
            "C_16_source_results_sha256": report_digest(resolution_results),
            "source_compatibility": read(
                source / "checkpoint_compatibility.json"
            )
            if (source / "checkpoint_compatibility.json").exists()
            else None,
            "baseline_rows_sha256": report_digest(
                list(selected_baselines.values())
            ),
            "candidate_config": frozen.original.to_dict(),
            "qa_config": config.to_dict(),
            "qa_resolution": [SIDE, SIDE],
            "frame_budget": BUDGET,
            "reference_frame_budget": 16,
            "max_segments": 128,
            "arms": list(ARMS),
            "cohort": {
                v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
                for v in videos
            },
            "manifest_sha256": saved["manifest_sha256"],
            "backend_metadata_sha256": report_digest(frozen.metadata),
            "plans_sha256": report_digest(list(plans.values())),
            "primary_comparison": "D_minus_uniform",
            "partition_comparison": "D_minus_C",
            "budget_interaction": "D_uniform_advantage_change",
            "selectors": (
                "Frozen cap-128 D2/D0 boundaries and event/frame cosine "
                "scores. "
                "Recompute allocation and controls for 32 frames using the "
                "same rules. C matches D segment count; D/D_topk share "
                "allocations; C/C_topk share allocations. B_topk uses "
                "candidate-count proportional allocation. Ties choose earlier "
                "candidates; no extra spacing; chronological distinct images."
            ),
            "selection_inputs": (
                "Saved CLIP scores and event plans; no choices, gold, "
                "captions "
                "or evidence labels enter selection."
            ),
            "QA_reuse": (
                "All 32-frame QA is new. Exact identical 32-frame inputs in "
                "the same question/order share checkpoints. Historical "
                "16-frame QA is used ONLY for paired cross-budget analysis."
            ),
            "cost_scope": (
                "QA only. Decode, selection, CLIP and model loading excluded. "
                "Baseline 16 timings are historical. No end-to-end efficiency "
                "or paired speed claim."
            ),
            "interpretation": (
                "Exploratory long-dev budget study after prior results were "
                "inspected. 32 frames is a budget point, not a universal "
                "paper "
                "standard. Same model, resolution, scoring and six option "
                "orders. Equal question weight; paired video bootstrap 5000. "
                "Secondary intervals are marginal, not multiplicity-adjusted. "
                "No independent holdout or full-benchmark claim. Increasing "
                "budget recalculates policies rather than guaranteeing a "
                "nested frame set. Current D/C plans may still allocate one "
                "frame per selected event; this is not a fixed-event density "
                "intervention. D minus C estimates the partition policy "
                "effect including changed pooled features and allocation, "
                "not isolated boundary accuracy."
            ),
            "code_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
    )
    return SimpleNamespace(
        original=frozen.original,
        config=config,
        manifest=frozen.manifest,
        metadata=frozen.metadata,
        videos=videos,
        plans=plans,
        baseline_rows=selected_baselines,
        protocol=protocol,
    )


def preparation_report(inputs):
    return {
        "stage": "prepared_no_decode_no_QA",
        "protocol_sha256": report_digest(inputs.protocol),
        "evaluation_role": inputs.protocol["evaluation_role"],
        "n_videos": len(inputs.videos),
        "n_questions": len(inputs.plans),
        "qa_resolution": [SIDE, SIDE],
        "frame_budget": BUDGET,
        "arms": list(ARMS),
        "reference_frame_budget": 16,
        "max_new_QA_calls": len(inputs.plans) * len(ARMS) * 6,
        "planned_unique_new_QA_calls": sum(
            len(
                {
                    tuple(p["selected_indices"])
                    for p in plan["conditions"].values()
                }
            )
            * 6
            for plan in inputs.plans.values()
        ),
        "new_CLIP_calls": 0,
        "D_single_frame_per_selected_event_questions": sum(
            max(p["conditions"]["D"]["allocation"]) == 1
            for p in inputs.plans.values()
        ),
    }


def prepare(args, inputs):
    output = args.output_dir
    if not (output / "protocol.json").exists() and any(
        p.name != ".run.lock" for p in output.iterdir()
    ):
        raise ValueError("Output is not an empty experiment directory")
    artifacts = {
        "protocol.json": inputs.protocol,
        "selection_plans.json": list(inputs.plans.values()),
    }
    for name, value in artifacts.items():
        path = output / name
        if path.exists() and read(path) != value:
            raise ValueError(
                f"Resume {name} changed; use a new output directory"
            )
        if not path.exists() and args.phase == "summarize":
            raise ValueError("No prepared experiment to summarize")
    for name, value in artifacts.items():
        if not (output / name).exists():
            write_json(output / name, value)
    report = preparation_report(inputs)
    write_json(output / "prepare_complete.json", report)
    return report


def validate_result(row, plan, q, config):
    if any(
        row[k] != plan[k]
        for k in ("video_id", "question_id", "candidate_sha256")
    ) or set(row["conditions"]) != set(ARMS):
        raise ValueError("Result identity or conditions differ")
    if row["duration_group"] != "long":
        raise ValueError("Require long-video results")
    shared = {}
    for arm in ARMS:
        entry, choice = row["conditions"][arm], plan["conditions"][arm]
        if (
            entry["selected_indices"] != choice["selected_indices"]
            or entry["selection_plan"] != choice
        ):
            raise ValueError("Frozen selection changed")
        if entry["qa_origin"] != "this_experiment":
            raise ValueError("32-frame QA must not reuse 16-frame trials")
        validate_indices(
            entry["selected_indices"], plan["candidate_count"], BUDGET
        )
        times = np.asarray(entry["selected_timestamps"], dtype=float)
        if (
            times.shape != (BUDGET,)
            or not np.isfinite(times).all()
            or times[0] < 0
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Invalid selected timestamps")
        strict_trials(entry["trials"], q, config)
        for trial in entry["trials"]:
            resolution_dev.validate_processor(trial, SIDE, BUDGET)
            seconds = trial.get("qa_wall_seconds")
            if (
                not isinstance(seconds, (int, float))
                or not np.isfinite(seconds)
                or seconds < 0
            ):
                raise ValueError("Invalid QA timing")
        ids = tuple(entry["selected_indices"])
        if ids in shared and any(
            entry[k] != shared[ids][k] for k in prior.QA_FIELDS
        ):
            raise ValueError("Identical 32-frame inputs must share QA")
        shared[ids] = entry


def validate_checkpoints(row, output, digest):
    for entry in row["conditions"].values():
        keys = []
        for trial in entry["trials"]:
            ti = resolution_dev.trial_key(
                digest,
                row["video_id"],
                row["question_id"],
                entry["selected_indices"],
                trial["order"],
                SIDE,
                entry["qa_pixels_sha256"],
            )
            key = report_digest(ti)
            if checked_payload(
                output / "trials" / (key + ".json"), ti
            ) != canonical(trial):
                raise ValueError("Question and trial checkpoints disagree")
            keys.append(key)
        if keys != entry["trial_keys"]:
            raise ValueError("Trial identities changed")


def build_checked_backend(inputs, output):
    seed_call(inputs.config)
    model = build_backend(inputs.config)
    if model.metadata() != inputs.metadata:
        raise ValueError("QA backend differs from the 16-frame source")
    write_json(output / "backend.json", model.metadata())
    return model


def cached_trial(
    output,
    digest,
    vid,
    q,
    ids,
    frames,
    stamps,
    duration,
    item,
    order,
    inputs,
    backend,
):
    pixel_hash = array_digest(frames, stamps)
    ti = resolution_dev.trial_key(
        digest, vid, q.question_id, ids, order, SIDE, pixel_hash
    )
    key = report_digest(ti)
    path = output / "trials" / (key + ".json")
    fresh = not path.exists()
    if fresh:
        video = Video(
            vid, "dev", frames, stamps, duration, (q,), item["source_id"]
        )
        trial = canonical(
            resolution_dev.score_with_memory(
                backend(), video, q, order, inputs.config, SIDE
            )
        )
        resolution_dev.validate_processor(trial, SIDE, BUDGET)
        save_payload(path, ti, trial)
    else:
        trial = checked_payload(path, ti)
        resolution_dev.validate_processor(trial, SIDE, BUDGET)
    return trial, key, fresh, pixel_hash


def memory_check(args, inputs):
    """One real QA call with Uniform and the longest question/options text."""
    output, digest = args.output_dir, report_digest(inputs.protocol)
    candidates = []
    for item in inputs.videos:
        for annotation in item["questions"]:
            q = question_from(annotation)
            candidates.append(
                (len(q.text) + sum(map(len, q.options)), item, q)
            )
    _, item, q = max(
        candidates, key=lambda x: (x[0], x[1]["video_id"], x[2].question_id)
    )
    vid = item["video_id"]
    plan = inputs.plans[(vid, q.question_id)]
    ids = plan["conditions"]["uniform"]["selected_indices"]
    print(
        f"Memory check {vid}/{q.question_id}: 32 frames, one QA order",
        flush=True,
    )
    low, pixels, times, duration, audit = resolution_dev.decode_selected(
        (inputs.manifest.parent / item["path"]).resolve(),
        inputs.original,
        plan["candidate_count"],
        ids,
        SIDE,
        plan["candidate_sha256"],
    )
    del low
    frames = np.stack([pixels[i] for i in ids])
    reference = inputs.baseline_rows[(vid, q.question_id)]["conditions"]["D"][
        "trials"
    ][0]
    try:
        trial, key, fresh, _ = cached_trial(
            output,
            digest,
            vid,
            q,
            ids,
            frames,
            times[ids],
            duration,
            item,
            reference["order"],
            inputs,
            lambda: build_checked_backend(inputs, output),
        )
    except RuntimeError as exc:
        write_json(
            output / "memory_check.json",
            {
                "status": "failed",
                "protocol_sha256": digest,
                "video_id": vid,
                "question_id": q.question_id,
                "error": str(exc),
                "automatic_fallback": False,
            },
        )
        raise
    report = {
        "status": "passed_one_QA_call",
        "protocol_sha256": digest,
        "video_id": vid,
        "question_id": q.question_id,
        "condition": "uniform",
        "selection_reason": (
            "Longest question plus options by character count; not an "
            "outcome-based choice or guarantee of worst-case GPU usage."
        ),
        "frame_budget": BUDGET,
        "qa_resolution": [SIDE, SIDE],
        "trial_key": key,
        "new_QA_calls_this_invocation": int(fresh),
        "visual_tokens": trial["visual_tokens"],
        "qa_wall_seconds": trial["qa_wall_seconds"],
        "cuda_memory": trial.get("cuda_memory"),
        "processor_info": trial["processor_info"],
        "decode_seconds": audit.get("decode_seconds"),
        "note": (
            "This trial is reused by the full experiment. One successful "
            "input does not guarantee every subsequent input will fit. "
            "No resolution, precision or model fallback."
        ),
    }
    write_json(output / "memory_check.json", report)
    return report


def evaluate(args, inputs):
    output, digest = args.output_dir, report_digest(inputs.protocol)
    model, rows, calls = None, [], 0

    def backend():
        nonlocal model
        if model is None:
            model = build_checked_backend(inputs, output)
        return model

    for item in inputs.videos:
        vid, pending = item["video_id"], []
        for annotation in item["questions"]:
            q = question_from(annotation)
            key = (vid, q.question_id)
            qi = question_identity(digest, *key)
            path = output / "questions" / (report_digest(qi) + ".json")
            if path.exists():
                row = checked_payload(path, qi)
                validate_result(row, inputs.plans[key], q, inputs.config)
                validate_checkpoints(row, output, digest)
                rows.append(row)
            else:
                pending.append((q, qi, path))
        if not pending:
            continue
        if args.phase == "summarize":
            raise ValueError(
                "Comparison incomplete; finish QA before summarizing"
            )
        selected = sorted(
            {
                i
                for q, _, _ in pending
                for p in inputs.plans[(vid, q.question_id)][
                    "conditions"
                ].values()
                for i in p["selected_indices"]
            }
        )
        first = inputs.plans[(vid, pending[0][0].question_id)]
        print(
            f"Decoding {vid}: {len(selected)} original frames at 320",
            flush=True,
        )
        low, pixels, times, duration, audit = resolution_dev.decode_selected(
            (inputs.manifest.parent / item["path"]).resolve(),
            inputs.original,
            first["candidate_count"],
            selected,
            SIDE,
            first["candidate_sha256"],
        )
        del low
        write_json(output / "decode" / (report_digest(vid) + ".json"), audit)
        for q, qi, path in pending:
            key = (vid, q.question_id)
            plan = inputs.plans[key]
            conditions, reusable = {}, {}
            reference = inputs.baseline_rows[key]["conditions"]["D"]["trials"]
            for arm in ARMS:
                choice = plan["conditions"][arm]
                ids = choice["selected_indices"]
                frames = np.stack([pixels[i] for i in ids])
                pixel_hash = array_digest(frames, times[ids])
                if tuple(ids) in reusable:
                    entry = deepcopy(reusable[tuple(ids)])
                    if entry["qa_pixels_sha256"] != pixel_hash:
                        raise ValueError(
                            "Identical indices have different pixels"
                        )
                else:
                    trials, keys = [], []
                    for number, ref in enumerate(reference):
                        trial, tk, fresh, trial_hash = cached_trial(
                            output,
                            digest,
                            vid,
                            q,
                            ids,
                            frames,
                            times[ids],
                            duration,
                            item,
                            ref["order"],
                            inputs,
                            backend,
                        )
                        if trial_hash != pixel_hash:
                            raise ValueError("Trial pixel identity changed")
                        trials.append(trial)
                        keys.append(tk)
                        calls += int(fresh)
                        print(
                            f"{vid}/{q.question_id}/{arm}: "
                            f"order {number + 1}/6 saved",
                            flush=True,
                        )
                    strict_trials(trials, q, inputs.config)
                    entry = {
                        "selected_indices": ids,
                        "selected_timestamps": times[ids].tolist(),
                        "qa_pixels_sha256": pixel_hash,
                        "trials": trials,
                        "trial_keys": keys,
                        "qa_origin": "this_experiment",
                    }
                entry["selection_plan"] = deepcopy(choice)
                conditions[arm] = entry
                reusable[tuple(ids)] = entry
            row = {
                "video_id": vid,
                "question_id": q.question_id,
                "duration_group": "long",
                "candidate_sha256": plan["candidate_sha256"],
                "conditions": conditions,
            }
            validate_result(row, plan, q, inputs.config)
            validate_checkpoints(row, output, digest)
            save_payload(path, qi, row)
            rows.append(row)
            write_json(
                output / "progress.json",
                {
                    "status": "running",
                    "completed_questions": len(rows),
                    "total_questions": len(inputs.plans),
                    "new_QA_calls_this_invocation": calls,
                },
            )
    rows.sort(key=lambda r: (r["video_id"], r["question_id"]))
    summary, metrics = summarize(rows, inputs.baseline_rows, inputs.protocol)
    write_json(
        output / "results.json",
        {"protocol_sha256": digest, "questions": rows, "summary": summary},
    )
    write_json(output / "handoff.json", summary)
    write_csv(output / "question_metrics.csv", metrics)
    write_csv(
        output / "selected_frames.csv",
        [
            {
                "video_id": r["video_id"],
                "question_id": r["question_id"],
                "condition": a,
                "selected_indices": r["conditions"][a]["selected_indices"],
                "selected_timestamps": r["conditions"][a][
                    "selected_timestamps"
                ],
            }
            for r in rows
            for a in ARMS
        ],
    )
    write_json(
        output / "progress.json",
        {
            "status": "complete",
            "completed_questions": len(rows),
            "total_questions": len(inputs.plans),
            "new_QA_calls_this_invocation": calls,
            "unique_new_QA_calls": summary["unique_new_QA_calls"],
        },
    )
    return summary


def run(args):
    inputs = load_inputs(args)
    if args.check_inputs:
        return {
            **preparation_report(inputs),
            "stage": "validated_no_decode_no_QA",
        }
    if args.phase == "summarize" and not args.output_dir.is_dir():
        raise ValueError("No completed experiment to summarize")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output_dir):
        report = prepare(args, inputs)
        if args.phase == "prepare":
            return report
        if args.phase == "memory-check":
            return memory_check(args, inputs)
        return evaluate(args, inputs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("outputs/event_selection320_long"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/frame_budget32_long")
    )
    p.add_argument(
        "--phase",
        choices=("prepare", "memory-check", "evaluate", "summarize"),
        default="prepare",
    )
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument("--max-videos", type=int)
    p.add_argument("--max-questions", type=int)
    return p


def main():
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
