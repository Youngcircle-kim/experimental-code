"""Compare frozen D, Uniform, global Top-k and Temporal-bin at 320 pixels."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import resolution_dev
from .ablation import select_frame_control
from .backends import build_backend
from .bottleneck_diagnostics import read, write_csv
from .cache import write_json
from .data import array_digest
from .event_factorial_dev import validate_indices
from .event_refinement_dev import checked_payload, question_from, save_payload
from .frame_replacement_probe import FROZEN_CODE, strict_trials
from .order_experiments import seed_call
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .types import Video

CONTROLS = ("uniform", "frame_top", "temporal_bin")
ARMS = (*CONTROLS, "D")
COMPARISONS = {f"D_minus_{arm}": ("D", arm) for arm in CONTROLS}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def verify_code(protocol, names):
    root = Path(__file__).parent
    for name in names:
        if (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            != protocol["code_sha256"][name]
        ):
            raise ValueError(f"Frozen implementation changed: {name}")


def indexed(rows):
    lookup = {(r["video_id"], r["question_id"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate source questions")
    return lookup


def control_selections(conditions, count, budget=16):
    """Reproduce saved controls from the SAME frozen question/CLIP scores."""
    scores = np.asarray(conditions["frame_top"]["frame_scores"], dtype=float)
    other = np.asarray(conditions["temporal_bin"]["frame_scores"], dtype=float)
    if (
        scores.shape != (count,)
        or not np.isfinite(scores).all()
        or not np.array_equal(scores, other)
    ):
        raise ValueError(
            "Controls must share finite scores for every candidate"
        )
    plans = {}
    for arm in CONTROLS:
        ids = select_frame_control(arm, scores, budget).tolist()
        if ids != conditions[arm]["selected_indices"]:
            raise ValueError(f"Stored {arm} selection does not reproduce")
        validate_indices(ids, count, budget)
        plans[arm] = {
            "selected_indices": ids,
            "selected_timestamps": conditions[arm].get("selected_timestamps"),
        }
    return plans, report_digest(scores.tolist())


def load_inputs(args):
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if any(
        n is not None and n < 1 for n in (args.max_videos, args.max_questions)
    ):
        raise ValueError("Subset limits must be positive")
    prior, results = (
        read(source / "protocol.json"),
        read(source / "results.json"),
    )
    source_digest = report_digest(prior)
    if (
        prior.get("experiment") != "frozen_selection_QA_resolution"
        or prior["qa_side"] != 320
        or "D" not in prior["conditions"]
        or results["protocol_sha256"] != source_digest
    ):
        raise ValueError("Require completed 320-resolution D results")
    verify_code(
        prior,
        (
            *FROZEN_CODE,
            "resolution_dev.py",
            "ablation.py",
            "event_refinement_dev.py",
            "cache.py",
            "types.py",
            "paired_reanalysis.py",
            "qa_diagnostics.py",
        ),
    )
    cap_dir = Path(prior["source_dir"]).resolve()
    cap_protocol = read(cap_dir / "protocol.json")
    if cap_protocol.get("experiment") != "relaxed_segment_cap_C_D":
        raise ValueError(
            "D must come from the completed segment-cap experiment"
        )
    validation_args = resolution_dev.parser().parse_args(
        [
            "--source-dir",
            str(cap_dir),
            "--output-dir",
            str(output),
            "--side",
            "320",
            "--conditions",
            *prior["conditions"],
        ]
    )
    validation_args.rerun_control = prior["rerun_control"]
    original, config, manifest, videos, cap_rows, metadata, expected = (
        resolution_dev.load_inputs(validation_args)
    )
    expected = canonical(expected)
    for key in (
        "source_protocol_sha256",
        "source_results_sha256",
        "manifest_sha256",
        "source_backend_sha256",
        "candidate_config",
        "qa_config",
        "qa_side",
        "conditions",
        "rerun_control",
        "cohort",
    ):
        if prior[key] != expected[key]:
            raise ValueError(f"Resolution source provenance changed: {key}")
    qa_metadata = read(source / "backend_320.json")
    if qa_metadata != {**metadata, "vlm_max_pixels": 320**2}:
        raise ValueError("320 backend differs beyond image pixel limit")
    resolution_rows = indexed(results["questions"])
    if set(resolution_rows) != set(cap_rows):
        raise ValueError("Resolution source cohort is incomplete")
    factorial = Path(
        args.factorial_dir or cap_protocol["source_dir"]
    ).resolve()
    fp, fr = (
        read(factorial / "protocol.json"),
        read(factorial / "results.json"),
    )
    if (
        report_digest(fp) != cap_protocol["source_protocol_sha256"]
        or report_digest(fr) != cap_protocol["source_results_sha256"]
        or fr["protocol_sha256"] != report_digest(fp)
    ):
        raise ValueError("Frame controls do not match the original D lineage")
    verify_code(fp, (*FROZEN_CODE, "ablation.py", "selection_dev.py"))
    if read(factorial / "backend.json") != metadata:
        raise ValueError(
            "Frame controls used different model/runtime settings"
        )
    for key, value in prior["candidate_config"].items():
        if key != "max_segments" and fp["config"][key] != value:
            raise ValueError(f"Frame-control configuration differs: {key}")
    for protected in (source, cap_dir, factorial):
        if (
            output == protected
            or output in protected.parents
            or protected in output.parents
        ):
            raise ValueError("Output must be separate from source experiments")
    frame_rows = indexed(fr["questions"])
    if args.video_id:
        videos = [v for v in videos if v["video_id"] == args.video_id]
    videos = videos[: args.max_videos]
    if not videos:
        raise ValueError("No matching source videos")
    videos = [
        {**v, "questions": v["questions"][: args.max_questions]}
        for v in videos
    ]
    plans, d_rows = {}, {}
    for item in videos:
        for annotation in item["questions"]:
            q = question_from(annotation)
            key = (item["video_id"], q.question_id)
            row, old, controls = (
                resolution_rows[key],
                cap_rows[key],
                frame_rows[key],
            )
            ri = {
                "protocol_sha256": source_digest,
                "video_id": key[0],
                "question_id": key[1],
            }
            saved = checked_payload(
                source / "questions" / (report_digest(ri) + ".json"), ri
            )
            if saved != row:
                raise ValueError(
                    "Resolution source result/checkpoint mismatch"
                )
            resolution_dev.validate_row(row, old, q, validation_args, original)
            if (
                controls["candidate_sha256"] != row["candidate_sha256"]
                or controls["duration_group"] != row["duration_group"]
            ):
                raise ValueError(
                    "Controls and D have different candidates/cohort"
                )
            count = old["conditions"]["D"]["events"][-1][1]
            selected, scores_hash = control_selections(
                controls["conditions"], count
            )
            for arm in CONTROLS:
                strict_trials(
                    controls["conditions"][arm]["trials"], q, original
                )
                if [
                    t["order"] for t in controls["conditions"][arm]["trials"]
                ] != [
                    t["order"] for t in row["conditions"]["D_320"]["trials"]
                ]:
                    raise ValueError("Control and D option orders differ")
            d = row["conditions"]["D_320"]
            selected["D"] = {
                k: d[k]
                for k in (
                    "selected_indices",
                    "selected_timestamps",
                    "qa_pixels_sha256",
                )
            }
            plans[key] = {
                "video_id": key[0],
                "question_id": key[1],
                "candidate_sha256": row["candidate_sha256"],
                "candidate_count": count,
                "frame_scores_sha256": scores_hash,
                "conditions": selected,
            }
            d_rows[key] = row
    protocol = canonical(
        {
            "version": 1,
            "experiment": "D_vs_frame_controls_fixed_320",
            "source_dir": str(source),
            "source_protocol_sha256": source_digest,
            "source_results_sha256": report_digest(results),
            "factorial_dir": str(factorial),
            "factorial_protocol_sha256": report_digest(fp),
            "factorial_results_sha256": report_digest(fr),
            "manifest_sha256": prior["manifest_sha256"],
            "backend_metadata_sha256": report_digest(qa_metadata),
            "candidate_config": original.to_dict(),
            "qa_config": config.to_dict(),
            "qa_side": 320,
            "frame_budget": 16,
            "arms": list(ARMS),
            "primary_comparison": "D_minus_uniform",
            "secondary_comparisons": [
                "D_minus_frame_top",
                "D_minus_temporal_bin",
            ],
            "cohort": {
                v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
                for v in videos
            },
            "plans_sha256": report_digest(list(plans.values())),
            "selectors": {
                "uniform": "16 global evenly spaced indices, with endpoints",
                "frame_top": "Top 16 CLIP scores; earlier candidates win ties",
                "temporal_bin": "16 equal-count temporal bins; "
                "one score maximum per bin",
                "D": "Saved cap-128 D2 partition, query allocation, "
                "within-event uniform selection",
            },
            "selection_inputs": "Frozen 224 candidate/CLIP scores; "
            "question text only, no choices, gold, captions or annotations.",
            "qa": "Same 320 original-frame decoding, 16 chronological images, "
            "weights, BF16, SDPA, prompt, scoring and orders as saved D.",
            "reuse": "Saved D_320 trials; 224 control QA is NEVER reused. "
            "Within a question, exact same 320 pixels/indices/order share QA.",
            "interpretation": "Exploratory dev comparison after tuning D. "
            "Video-cluster bootstrap after within-question order means. "
            "Secondary CIs are marginal, not multiplicity-adjusted. "
            "No evidence recall or external-framework claim.",
            "cost_scope": "QA cost only. Selection/CLIP extraction replayed, "
            "not timed; no end-to-end speed claim. D timing is historical.",
            "code_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
    )
    return SimpleNamespace(
        original=original,
        config=config,
        manifest=manifest,
        videos=videos,
        plans=plans,
        d_rows=d_rows,
        metadata=qa_metadata,
        protocol=protocol,
    )


def validate_result(row, plan, d_source, q, config):
    if (
        row["video_id"] != plan["video_id"]
        or row["question_id"] != plan["question_id"]
        or row["candidate_sha256"] != plan["candidate_sha256"]
        or set(row["conditions"]) != set(ARMS)
    ):
        raise ValueError("Result identity/conditions differ")
    d = d_source["conditions"]["D_320"]
    for arm in ARMS:
        entry, selected = row["conditions"][arm], plan["conditions"][arm]
        if entry["selected_indices"] != selected["selected_indices"]:
            raise ValueError("Frozen selection changed")
        expected_times = selected.get("selected_timestamps")
        if (
            expected_times is not None
            and entry["selected_timestamps"] != expected_times
        ):
            raise ValueError("Frozen timestamps changed")
        stamps = np.asarray(entry["selected_timestamps"], dtype=float)
        if (
            stamps.shape != (16,)
            or not np.isfinite(stamps).all()
            or np.any(np.diff(stamps) <= 0)
        ):
            raise ValueError("Invalid selected timestamps")
        strict_trials(entry["trials"], q, config)
        for trial in entry["trials"]:
            resolution_dev.validate_processor(trial, 320, 16)
    if any(
        row["conditions"]["D"][k] != d[k]
        for k in (
            "trials",
            "selected_indices",
            "selected_timestamps",
            "qa_pixels_sha256",
        )
    ):
        raise ValueError("Saved D QA was modified")
    for arm in CONTROLS:
        c = row["conditions"][arm]
        if (
            c["selected_indices"] == d["selected_indices"]
            and c["trials"] != d["trials"]
        ):
            raise ValueError("Identical inputs must share the saved D QA")


def summarize(rows, protocol):
    means = []
    for row in rows:
        values = {
            a: float(
                np.mean(
                    [
                        t["scoring_correct"]
                        for t in row["conditions"][a]["trials"]
                    ]
                )
            )
            for a in ARMS
        }
        means.append(
            {
                "video_id": row["video_id"],
                "question_id": row["question_id"],
                "duration_group": row["duration_group"],
                **values,
                **{
                    k: values[a] - values[b]
                    for k, (a, b) in COMPARISONS.items()
                },
            }
        )
    keys = [*ARMS, *COMPARISONS]
    resources = {}
    for arm in ARMS:
        entries = [r["conditions"][arm] for r in rows]
        trials = [t for e in entries for t in e["trials"]]
        seconds = [
            t["qa_wall_seconds"] for t in trials if "qa_wall_seconds" in t
        ]
        memory = [
            t["cuda_memory"]["peak_allocated_bytes"]
            for t in trials
            if "cuda_memory" in t
        ]
        resources[arm] = {
            "qa_seconds_median": float(np.median(seconds))
            if seconds
            else None,
            "max_peak_allocated_GiB": max(memory) / 1024**3
            if memory
            else None,
            "visual_tokens_per_question": 1600,
            "timing_origins": sorted({e["timing_origin"] for e in entries}),
            "selection_and_feature_seconds": None,
            "requires_CLIP_features": arm != "uniform",
        }
    return {
        "protocol_sha256": report_digest(protocol),
        "qa_resolution": [320, 320],
        "frame_budget": 16,
        "primary_comparison": "D_minus_uniform",
        "secondary_comparisons": list(COMPARISONS)[1:],
        "overall": clustered_means(means, keys),
        "by_duration": {
            d: clustered_means(
                [r for r in means if r["duration_group"] == d], keys
            )
            for d in sorted({r["duration_group"] for r in means})
        },
        "paired_question_changes": {
            k: {
                "improved": sum(r[k] > 1e-12 for r in means),
                "degraded": sum(r[k] < -1e-12 for r in means),
                "unchanged": sum(abs(r[k]) <= 1e-12 for r in means),
            }
            for k in COMPARISONS
        },
        "same_input_as_D_questions": {
            a: sum(
                r["conditions"][a]["selected_indices"]
                == r["conditions"]["D"]["selected_indices"]
                for r in rows
            )
            for a in CONTROLS
        },
        "resources": resources,
        "cost_scope": protocol["cost_scope"],
        "interpretation": protocol["interpretation"],
    }, means


def run(args):
    inputs = load_inputs(args)
    protocol = canonical(inputs.protocol)
    total = sum(len(v["questions"]) for v in inputs.videos)
    maximum = sum(
        len(inputs.d_rows[k]["conditions"]["D_320"]["trials"]) * len(CONTROLS)
        for k in inputs.plans
    )
    if args.check_inputs:
        return {
            "stage": "validated_no_decode_no_QA",
            "n_videos": len(inputs.videos),
            "n_questions": total,
            "qa_side": 320,
            "frame_budget": 16,
            "arms": list(ARMS),
            "D_QA": "verified_saved_320_results",
            "max_new_QA_calls": maximum,
            "new_CLIP_calls": 0,
        }
    output, digest = args.output_dir, report_digest(protocol)
    if (output / "protocol.json").exists():
        if read(output / "protocol.json") != protocol:
            raise ValueError(
                "Resume protocol changed; use a new output directory"
            )
    else:
        if args.summarize_only:
            raise ValueError("No completed comparison to summarize")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", protocol)
        write_json(
            output / "selection_plans.json", list(inputs.plans.values())
        )
    model, rows, calls = None, [], 0

    def backend():
        nonlocal model
        if model is None:
            seed_call(inputs.config)
            model = build_backend(inputs.config)
            if model.metadata() != inputs.metadata:
                raise ValueError("QA backend differs from saved D_320")
            write_json(output / "backend.json", model.metadata())
        return model

    for item in inputs.videos:
        vid, pending = item["video_id"], []
        for annotation in item["questions"]:
            q = question_from(annotation)
            key = (vid, q.question_id)
            qi = {
                "protocol_sha256": digest,
                "video_id": vid,
                "question_id": q.question_id,
            }
            path = output / "questions" / (report_digest(qi) + ".json")
            if path.exists():
                row = checked_payload(path, qi)
                validate_result(
                    row,
                    inputs.plans[key],
                    inputs.d_rows[key],
                    q,
                    inputs.config,
                )
                rows.append(row)
            else:
                pending.append((q, qi, path))
        if not pending:
            continue
        if args.summarize_only:
            raise ValueError(
                "Comparison incomplete; finish QA before summarizing"
            )
        selected = sorted(
            {
                i
                for q, _, _ in pending
                for a in ARMS
                for i in inputs.plans[(vid, q.question_id)]["conditions"][a][
                    "selected_indices"
                ]
            }
        )
        first = inputs.plans[(vid, pending[0][0].question_id)]
        print(
            f"Decoding {vid}: extract {len(selected)} frames at 320",
            flush=True,
        )
        low, pixels, times, duration, audit = resolution_dev.decode_selected(
            (inputs.manifest.parent / item["path"]).resolve(),
            inputs.original,
            first["candidate_count"],
            selected,
            320,
            first["candidate_sha256"],
        )
        del low
        write_json(output / "decode" / f"{vid}.json", audit)
        for q, qi, path in pending:
            key = (vid, q.question_id)
            plan, source = inputs.plans[key], inputs.d_rows[key]
            d = deepcopy(source["conditions"]["D_320"])
            ids = d["selected_indices"]
            d_pixels = np.stack([pixels[i] for i in ids])
            if (
                not np.array_equal(times[ids], d["selected_timestamps"])
                or array_digest(d_pixels, times[ids]) != d["qa_pixels_sha256"]
            ):
                raise ValueError(
                    "Saved D_320 pixels/timestamps do not reproduce"
                )
            d["timing_origin"] = "historical_D_320"
            conditions = {"D": d}
            reusable = {tuple(ids): d}
            for arm in CONTROLS:
                choice = plan["conditions"][arm]
                ids = choice["selected_indices"]
                expected_times = choice["selected_timestamps"]
                if expected_times is not None and not np.array_equal(
                    times[ids], expected_times
                ):
                    raise ValueError(
                        "Control timestamps differ from saved selection"
                    )
                frames = np.stack([pixels[i] for i in ids])
                qa_hash = array_digest(frames, times[ids])
                if tuple(ids) in reusable:
                    entry = deepcopy(reusable[tuple(ids)])
                    if entry["qa_pixels_sha256"] != qa_hash:
                        raise ValueError(
                            "Identical index selection has different pixels"
                        )
                    entry["qa_reused_identical_input"] = True
                    conditions[arm] = entry
                    continue
                video = Video(
                    vid,
                    "dev",
                    frames,
                    times[ids],
                    duration,
                    (q,),
                    item["source_id"],
                )
                trials = []
                for number, reference in enumerate(d["trials"]):
                    ti = resolution_dev.trial_key(
                        digest,
                        vid,
                        q.question_id,
                        ids,
                        reference["order"],
                        320,
                        qa_hash,
                    )
                    tp = output / "trials" / (report_digest(ti) + ".json")
                    if tp.exists():
                        trial = checked_payload(tp, ti)
                    else:
                        trial = resolution_dev.score_with_memory(
                            backend(),
                            video,
                            q,
                            reference["order"],
                            inputs.config,
                            320,
                        )
                        save_payload(tp, ti, trial)
                        calls += 1
                    resolution_dev.validate_processor(trial, 320, 16)
                    trials.append(trial)
                    print(
                        f"{vid}/{q.question_id}/{arm}: order {number + 1}/6 "
                        f"correct={trial['scoring_correct']}",
                        flush=True,
                    )
                strict_trials(trials, q, inputs.config)
                entry = {
                    "selected_indices": ids,
                    "selected_timestamps": times[ids].tolist(),
                    "qa_pixels_sha256": qa_hash,
                    "trials": trials,
                    "timing_origin": "this_comparison",
                    "qa_reused_identical_input": False,
                }
                reusable[tuple(ids)] = entry
                conditions[arm] = entry
            row = {
                "video_id": vid,
                "question_id": q.question_id,
                "duration_group": source["duration_group"],
                "candidate_sha256": plan["candidate_sha256"],
                "conditions": conditions,
            }
            validate_result(row, plan, source, q, inputs.config)
            save_payload(path, qi, row)
            rows.append(row)
            write_json(
                output / "progress.json",
                {
                    "status": "running",
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "new_QA_calls_this_invocation": calls,
                },
            )
    rows.sort(key=lambda r: (r["video_id"], r["question_id"]))
    summary, metrics = summarize(rows, protocol)
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
            "completed_questions": total,
            "total_questions": total,
            "new_QA_calls_this_invocation": calls,
        },
    )
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("outputs/resolution320_cap128_long"),
    )
    p.add_argument(
        "--factorial-dir",
        type=Path,
        help="Defaults to source experiment lineage",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/selector_compare320_long"),
    )
    p.add_argument("--video-id")
    p.add_argument("--max-videos", type=int)
    p.add_argument(
        "--max-questions",
        type=int,
        help="First N questions per video for smoke checks",
    )
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def main():
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
