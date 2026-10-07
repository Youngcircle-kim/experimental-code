"""Isolate within-event retrieval, partition and allocation at fixed 320/16."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import resolution_dev, selector_comparison_dev
from .algorithms import allocate_frames, segment
from .backends import build_backend
from .bottleneck_diagnostics import read, validate_plan, write_csv
from .cache import write_json
from .data import array_digest
from .event_factorial_dev import validate_indices
from .event_refinement import select_inside
from .event_refinement_dev import checked_payload, question_from, save_payload
from .frame_replacement_probe import strict_trials
from .order_experiments import seed_call
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .selector_comparison_holdout import load_frozen_source, output_lock
from .types import Video

OLD_ARMS = ("uniform", "frame_top", "temporal_bin", "D")
NEW_ARMS = ("D_topk", "B_topk", "C_topk")
ARMS = (*OLD_ARMS, *NEW_ARMS)
COMPARISONS = {
    "D_topk_minus_D": ("D_topk", "D"),
    "D_topk_minus_C_topk": ("D_topk", "C_topk"),
    "D_topk_minus_B_topk": ("D_topk", "B_topk"),
    "D_topk_minus_uniform": ("D_topk", "uniform"),
    "D_topk_minus_frame_top": ("D_topk", "frame_top"),
    "D_topk_minus_temporal_bin": ("D_topk", "temporal_bin"),
    "D_minus_uniform": ("D", "uniform"),
}
QA_FIELDS = (
    "selected_indices",
    "selected_timestamps",
    "qa_pixels_sha256",
    "trials",
)
canonical = selector_comparison_dev.canonical
LEGACY_CHECKPOINT_RUNNER_SHA256 = (
    "cf58728f39abb069f2120c0d70a61a8450c9e08f22cd722632c0430986a560ea"
)


def resume_protocol(current, output):
    """Retain trial identities for the explicitly recorded JSON-only fix.

    The compatibility record binds both complete protocol hashes. It cannot
    admit later code edits, changed inputs, or changes to another module.
    The original protocol and existing trial files remain untouched.
    """
    path = output / "protocol.json"
    if not path.exists():
        return current
    saved = read(path)
    if saved == current:
        return saved
    name = Path(__file__).name
    before, after = (
        saved.get("code_sha256", {}),
        current.get("code_sha256", {}),
    )
    changed = {
        k
        for k in before.keys() | after.keys()
        if before.get(k) != after.get(k)
    }
    record_path = output / "checkpoint_compatibility.json"
    expected_record = {
        "version": 1,
        "fix": "json_trial_container_normalization_v1",
        "original_protocol_sha256": report_digest(saved),
        "runtime_protocol_sha256": report_digest(current),
        "runner_sha256_before": before.get(name),
        "runner_sha256_after": after.get(name),
    }
    if (
        {k: v for k, v in saved.items() if k != "code_sha256"}
        != {k: v for k, v in current.items() if k != "code_sha256"}
        or changed != {name}
        or before.get(name) != LEGACY_CHECKPOINT_RUNNER_SHA256
        or not record_path.is_file()
        or read(record_path) != expected_record
    ):
        raise ValueError(
            "Resume protocol.json changed; no matching checkpoint fix record"
        )
    return saved


def make_conditions(d_plan, c_plan, scores, budget=16):
    """Keep pooled event scores; use frame scores only for within top-k."""
    for plan, detector in ((d_plan, "D2"), (c_plan, "D0")):
        validate_plan(plan, budget, "question_relevance")
        if plan["detector"] != detector:
            raise ValueError("Wrong source partition detector")
        if plan["allocation_policy"] != "question_relevance":
            raise ValueError("Require saved question-relevance allocation")
    count = d_plan["events"][-1][1]
    if c_plan["events"][-1][1] != count or len(c_plan["events"]) != len(
        d_plan["events"]
    ):
        raise ValueError("C and D must have matching candidate/segment counts")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ValueError("Require finite scores for every candidate")
    result = {}
    for arm in NEW_ARMS:
        source = c_plan if arm == "C_topk" else d_plan
        capacities = np.asarray(source["capacities"], dtype=int)
        allocation = (
            allocate_frames(
                np.log(capacities), capacities, budget, 1.0
            ).tolist()
            if arm == "B_topk"
            else list(source["allocation"])
        )
        # No temporal spacing: the dummy times cannot affect min_gap=0 top-k.
        indices, fallback = select_inside(
            scores, np.arange(count), source["events"], allocation, 0.0
        )
        if fallback:
            raise ValueError("Unexpected fallback in pure within-event top-k")
        validate_indices(indices, count, budget)
        result[arm] = {
            "detector": source["detector"],
            "events": deepcopy(source["events"]),
            "capacities": capacities.tolist(),
            "allocation": allocation,
            "allocation_policy": (
                "candidate_count" if arm == "B_topk" else "question_relevance"
            ),
            "within_event": "question_cosine_topk",
            "min_gap_seconds": 0.0,
            "selected_indices": indices,
            "selected_timestamps": None,
            "diagnostics": {
                "zero_allocation_events": sum(n == 0 for n in allocation),
                "multi_frame_events": sum(n > 1 for n in allocation),
                "max_event_allocation": max(allocation),
                "selected_mean_question_cosine": float(scores[indices].mean()),
                "overlap_with_D": len(
                    set(indices) & set(d_plan["selected_indices"])
                )
                / budget,
            },
        }
    return result


def load_inputs(args):
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if (
        output == source
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be separate from the source directory")
    if any(
        n is not None and n < 1 for n in (args.max_videos, args.max_questions)
    ):
        raise ValueError("Subset limits must be positive")
    frozen = load_frozen_source(args)
    config = frozen.original
    if (
        config.max_segments != 128
        or config.frame_budget != 16
        or config.detector != "D2"
        or frozen.config.vlm_max_pixels != 320**2
    ):
        raise ValueError("Require frozen cap-128 D2, 16 frames and 320 QA")
    resolution_source = Path(frozen.protocol["source_dir"])
    rp = read(resolution_source / "protocol.json")
    if set(rp["conditions"]) != {"C", "D"}:
        raise ValueError("Require verified cap-128 C and D source plans")
    cap_dir = Path(rp["source_dir"])
    cap_results = read(cap_dir / "results.json")
    if report_digest(cap_results) != rp["source_results_sha256"]:
        raise ValueError("Cap-128 source results changed")
    cap_rows = selector_comparison_dev.indexed(cap_results["questions"])
    frame_results = read(
        Path(frozen.protocol["factorial_dir"]) / "results.json"
    )
    if (
        report_digest(frame_results)
        != frozen.protocol["factorial_results_sha256"]
    ):
        raise ValueError("Frozen frame score source changed")
    frame_rows = selector_comparison_dev.indexed(frame_results["questions"])
    source_results = read(source / "results.json")
    source_rows = selector_comparison_dev.indexed(source_results["questions"])
    source_digest = report_digest(frozen.protocol)
    # Also verify every source question checkpoint against the result rows.
    for key, row in source_rows.items():
        identity = {
            "protocol_sha256": source_digest,
            "video_id": key[0],
            "question_id": key[1],
        }
        if (
            checked_payload(
                source / "questions" / (report_digest(identity) + ".json"),
                identity,
            )
            != row
        ):
            raise ValueError(
                "Source question checkpoint disagrees with results"
            )
    videos = frozen.videos[: args.max_videos]
    videos = [
        {**v, "questions": v["questions"][: args.max_questions]}
        for v in videos
    ]
    if not videos or any(
        v["split"] != "dev" or v["duration"] != "long" for v in videos
    ):
        raise ValueError("Require nonempty long-video development cohort")
    plans, selected_sources = {}, {}
    for item in videos:
        for annotation in item["questions"]:
            key = (item["video_id"], str(annotation["question_id"]))
            old, cap, frames = source_rows[key], cap_rows[key], frame_rows[key]
            if not (
                old["candidate_sha256"]
                == cap["candidate_sha256"]
                == frames["candidate_sha256"]
            ):
                raise ValueError("Sources have different candidates")
            d, c = cap["conditions"]["D"], cap["conditions"]["C"]
            count = d["events"][-1][1]
            expected_c = segment(
                np.ones((count, 1)),
                config,
                "D0",
                segment_count=len(d["events"]),
            )
            if c["events"] != [[e.start, e.stop] for e in expected_c]:
                raise ValueError(
                    "C is not the count-matched uniform partition"
                )
            if (
                d["selected_indices"]
                != old["conditions"]["D"]["selected_indices"]
            ):
                raise ValueError("D plan differs from saved 320 D")
            _, scores_hash = selector_comparison_dev.control_selections(
                frames["conditions"], count
            )
            if scores_hash != frozen.plans[key]["frame_scores_sha256"]:
                raise ValueError("Frame scores changed")
            new = make_conditions(
                d, c, frames["conditions"]["frame_top"]["frame_scores"]
            )
            conditions = {
                a: {
                    k: old["conditions"][a][k]
                    for k in QA_FIELDS
                    if k != "trials"
                }
                for a in OLD_ARMS
            }
            conditions.update(new)
            plans[key] = {
                "video_id": key[0],
                "question_id": key[1],
                "candidate_count": count,
                "candidate_sha256": old["candidate_sha256"],
                "frame_scores_sha256": scores_hash,
                "conditions": conditions,
            }
            selected_sources[key] = old
    protocol = canonical(
        {
            "version": 1,
            "experiment": "event_selection_ablation_fixed_320",
            "evaluation_role": "exploratory_dev"
            if args.max_videos is None and args.max_questions is None
            else "dev_smoke",
            "source_dir": str(source),
            "source_protocol_sha256": source_digest,
            "source_results_sha256": report_digest(source_results),
            "cap_source_dir": str(cap_dir),
            "cap_results_sha256": report_digest(cap_results),
            "frame_scores_source_sha256": report_digest(frame_results),
            "manifest_sha256": frozen.protocol["manifest_sha256"],
            "backend_metadata_sha256": report_digest(frozen.metadata),
            "candidate_config": config.to_dict(),
            "qa_config": frozen.config.to_dict(),
            "qa_resolution": [320, 320],
            "frame_budget": 16,
            "max_segments": 128,
            "arms": list(ARMS),
            "new_arms": list(NEW_ARMS),
            "reused_arms": list(OLD_ARMS),
            "cohort": {
                v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
                for v in videos
            },
            "plans_sha256": report_digest(list(plans.values())),
            "primary_comparison": "D_topk_minus_D",
            "mechanism_comparisons": [
                "D_topk_minus_C_topk",
                "D_topk_minus_B_topk",
            ],
            "practical_comparison": "D_topk_minus_uniform",
            "comparisons": COMPARISONS,
            "selectors": {
                "D_topk": "Saved D2 cap-128 boundaries AND allocation; "
                "per-event question-cosine top-k.",
                "B_topk": "Same D2 boundaries; candidate-count proportional "
                "allocation using log capacity and temperature 1; "
                "same within-event top-k.",
                "C_topk": "Saved count-matched D0 uniform boundaries AND C "
                "question allocation; same within-event top-k.",
                "within_rule": "No minimum gap. Earlier candidate wins ties. "
                "16 distinct frames, chronological order.",
            },
            "selection_inputs": "Frozen candidate CLIP frame-question scores "
            "and saved event plans; no choices, gold, captions or evidence "
            "labels enter selection.",
            "cost_scope": "QA only. Decode, CLIP feature extraction, "
            "selection and model loading excluded. Baseline timings "
            "historical; not paired speed comparisons.",
            "interpretation": "Exploratory development after prior dev and "
            "holdout inspection. Within-question six-order means; equal "
            "question weights; paired video bootstrap 5000. Secondary CIs "
            "marginal, not multiplicity-adjusted. D_topk-C_topk is the "
            "partition-policy effect including changes in pooled "
            "representations and allocations. B_topk still uses the question "
            "within events. Conditional ablations, not a full factorial "
            "decomposition or proof of independent contributions. CLIP "
            "cosine/overlap are proxies, not evidence recall.",
            "code_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
    )
    return SimpleNamespace(
        original=config,
        config=frozen.config,
        metadata=frozen.metadata,
        manifest=frozen.manifest,
        videos=videos,
        plans=plans,
        source_rows=selected_sources,
        protocol=resume_protocol(protocol, output),
    )


def preparation_report(inputs):
    unique = 0
    for key, plan in inputs.plans.items():
        old = {
            tuple(plan["conditions"][a]["selected_indices"]) for a in OLD_ARMS
        }
        new = {
            tuple(plan["conditions"][a]["selected_indices"]) for a in NEW_ARMS
        }
        unique += len(new - old) * len(
            inputs.source_rows[key]["conditions"]["D"]["trials"]
        )
    return {
        "stage": "prepared_no_decode_no_QA",
        "protocol_sha256": report_digest(inputs.protocol),
        "evaluation_role": inputs.protocol["evaluation_role"],
        "n_videos": len(inputs.videos),
        "n_questions": len(inputs.plans),
        "qa_resolution": [320, 320],
        "frame_budget": 16,
        "max_segments": 128,
        "new_arms": list(NEW_ARMS),
        "reused_arms": list(OLD_ARMS),
        "max_new_QA_calls": len(inputs.plans) * 6 * len(NEW_ARMS),
        "planned_unique_new_QA_calls": unique,
        "new_CLIP_calls": 0,
    }


def prepare(args, inputs):
    output = args.output_dir
    if not (output / "protocol.json").exists() and any(
        p.name != ".run.lock" for p in output.iterdir()
    ):
        raise ValueError("Output is not an empty experiment directory")
    expected = list(inputs.plans.values())
    paths = {
        "protocol.json": inputs.protocol,
        "selection_plans.json": expected,
    }
    for name, value in paths.items():
        path = output / name
        if path.exists() and read(path) != value:
            raise ValueError(
                f"Resume {name} changed; use a new output directory"
            )
    if args.phase == "summarize" and any(
        not (output / n).exists() for n in paths
    ):
        raise ValueError("No completed experiment to summarize")
    for name, value in paths.items():
        if not (output / name).exists():
            write_json(output / name, value)
    report = preparation_report(inputs)
    write_json(output / "prepare_complete.json", report)
    return report


def validate_result(row, plan, sources, q, config):
    if any(
        row[k] != plan[k]
        for k in ("video_id", "question_id", "candidate_sha256")
    ) or set(row["conditions"]) != set(ARMS):
        raise ValueError("Result identity or conditions differ")
    if row["duration_group"] != sources["duration_group"]:
        raise ValueError("Result duration group differs")
    shared = {}
    for arm in ARMS:
        entry, selected = row["conditions"][arm], plan["conditions"][arm]
        if entry["selected_indices"] != selected["selected_indices"]:
            raise ValueError("Frozen selection changed")
        validate_indices(
            entry["selected_indices"], plan["candidate_count"], 16
        )
        times = np.asarray(entry["selected_timestamps"], dtype=float)
        if (
            times.shape != (16,)
            or not np.isfinite(times).all()
            or times[0] < 0
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Invalid selected timestamps")
        if (
            selected.get("selected_timestamps") is not None
            and entry["selected_timestamps"] != selected["selected_timestamps"]
        ):
            raise ValueError("Frozen timestamps changed")
        if arm in NEW_ARMS and entry["selection_plan"] != selected:
            raise ValueError("New arm selection plan changed")
        strict_trials(entry["trials"], q, config)
        for t in entry["trials"]:
            resolution_dev.validate_processor(t, 320, 16)
            seconds = t.get("qa_wall_seconds")
            if (
                not isinstance(seconds, (int, float))
                or not np.isfinite(seconds)
                or seconds < 0
            ):
                raise ValueError("Invalid QA timing")
        if arm in OLD_ARMS and any(
            entry[k] != sources["conditions"][arm][k] for k in QA_FIELDS
        ):
            raise ValueError("Historical 320 QA was modified")
        if entry["qa_origin"] == "source":
            source_arm = entry["source_arm"]
            if source_arm not in OLD_ARMS or any(
                entry[k] != sources["conditions"][source_arm][k]
                for k in QA_FIELDS
            ):
                raise ValueError("Reused source QA does not match")
        elif entry["qa_origin"] != "this_experiment" or arm in OLD_ARMS:
            raise ValueError("Invalid QA provenance")
        ids = tuple(entry["selected_indices"])
        if ids in shared and any(
            entry[k] != shared[ids][k] for k in QA_FIELDS
        ):
            raise ValueError("Identical inputs must share QA")
        shared[ids] = entry


def validate_checkpoints(row, output, digest):
    for entry in row["conditions"].values():
        if entry["qa_origin"] == "source":
            continue
        expected = []
        for trial in entry["trials"]:
            ti = resolution_dev.trial_key(
                digest,
                row["video_id"],
                row["question_id"],
                entry["selected_indices"],
                trial["order"],
                320,
                entry["qa_pixels_sha256"],
            )
            key = report_digest(ti)
            if checked_payload(
                output / "trials" / (key + ".json"), ti
            ) != canonical(trial):
                raise ValueError("Question and trial checkpoints disagree")
            expected.append(key)
        if entry["trial_keys"] != expected:
            raise ValueError("Trial identities changed")


def summarize(rows, protocol):
    if not rows:
        raise ValueError("No completed questions")
    means, unique, resources = [], {}, {}
    for row in rows:
        if set(row["conditions"]) != set(ARMS):
            raise ValueError("Incomplete conditions")
        orders = [t["order"] for t in row["conditions"]["D"]["trials"]]
        if len(orders) != 6 or len({tuple(o) for o in orders}) != 6:
            raise ValueError("Require six distinct paired orders")
        values = {}
        for arm in ARMS:
            entry = row["conditions"][arm]
            trials = entry["trials"]
            if [t["order"] for t in trials] != orders or any(
                type(t["scoring_correct"]) is not bool for t in trials
            ):
                raise ValueError("Mismatched paired trials")
            values[arm] = float(
                np.mean([t["scoring_correct"] for t in trials])
            )
            if entry["qa_origin"] == "this_experiment":
                for trial in trials:
                    key = (
                        row["video_id"],
                        row["question_id"],
                        entry["qa_pixels_sha256"],
                        tuple(entry["selected_indices"]),
                        tuple(trial["order"]),
                    )
                    unique[key] = trial
        means.append(
            {
                **{
                    k: row[k]
                    for k in ("video_id", "question_id", "duration_group")
                },
                **values,
                **{
                    k: values[a] - values[b]
                    for k, (a, b) in COMPARISONS.items()
                },
            }
        )
    for arm in ARMS:
        entries = [r["conditions"][arm] for r in rows]
        trials = [t for e in entries for t in e["trials"]]
        memory = [
            t["cuda_memory"]["peak_allocated_bytes"]
            for t in trials
            if "cuda_memory" in t
        ]
        resources[arm] = {
            "qa_seconds_median": float(
                np.median([t["qa_wall_seconds"] for t in trials])
            ),
            "max_peak_allocated_GiB": max(memory) / 1024**3
            if memory
            else None,
            "visual_tokens_per_QA_call": 1600,
            "option_order_repeats_per_question": 6,
            "qa_origins": sorted({e["qa_origin"] for e in entries}),
            "selection_and_feature_seconds": None,
        }
    return {
        "stage": "complete",
        "protocol_sha256": report_digest(protocol),
        "evaluation_role": protocol["evaluation_role"],
        "qa_resolution": [320, 320],
        "frame_budget": 16,
        "max_segments": 128,
        "primary_comparison": "D_topk_minus_D",
        "mechanism_comparisons": [
            "D_topk_minus_C_topk",
            "D_topk_minus_B_topk",
        ],
        "practical_comparison": "D_topk_minus_uniform",
        "overall": clustered_means(means, [*ARMS, *COMPARISONS]),
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
            for a in NEW_ARMS
        },
        "resources": resources,
        "unique_new_QA_calls": len(unique),
        "unique_new_QA_seconds": sum(
            t["qa_wall_seconds"] for t in unique.values()
        ),
        "cost_scope": protocol["cost_scope"],
        "interpretation": protocol["interpretation"],
    }, means


def evaluate(args, inputs):
    output, digest = args.output_dir, report_digest(inputs.protocol)
    model, rows, calls = None, [], 0

    def backend():
        nonlocal model
        if model is None:
            seed_call(inputs.config)
            model = build_backend(inputs.config)
            if model.metadata() != inputs.metadata:
                raise ValueError(
                    "QA backend differs from saved 320 experiment"
                )
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
                    inputs.source_rows[key],
                    q,
                    inputs.config,
                )
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
                for a in ARMS
                for i in inputs.plans[(vid, q.question_id)]["conditions"][a][
                    "selected_indices"
                ]
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
            320,
            first["candidate_sha256"],
        )
        del low
        write_json(output / "decode" / (report_digest(vid) + ".json"), audit)
        for q, qi, path in pending:
            key = (vid, q.question_id)
            plan, source = inputs.plans[key], inputs.source_rows[key]
            conditions, reusable = {}, {}
            for arm in ARMS:
                selection = plan["conditions"][arm]
                ids = selection["selected_indices"]
                frames = np.stack([pixels[i] for i in ids])
                qa_hash = array_digest(frames, times[ids])
                if arm in OLD_ARMS:
                    saved = source["conditions"][arm]
                    if qa_hash != saved[
                        "qa_pixels_sha256"
                    ] or not np.array_equal(
                        times[ids], saved["selected_timestamps"]
                    ):
                        raise ValueError(
                            "Historical 320 pixels/times do not reproduce"
                        )
                    entry = {k: deepcopy(saved[k]) for k in QA_FIELDS}
                    entry.update(
                        qa_origin="source", source_arm=arm, trial_keys=[]
                    )
                elif tuple(ids) in reusable:
                    entry = deepcopy(reusable[tuple(ids)])
                    if entry["qa_pixels_sha256"] != qa_hash:
                        raise ValueError(
                            "Identical indices have different pixels"
                        )
                else:
                    video = Video(
                        vid,
                        "dev",
                        frames,
                        times[ids],
                        duration,
                        (q,),
                        item["source_id"],
                    )
                    trials, keys = [], []
                    for number, reference in enumerate(
                        source["conditions"]["D"]["trials"]
                    ):
                        ti = resolution_dev.trial_key(
                            digest,
                            vid,
                            q.question_id,
                            ids,
                            reference["order"],
                            320,
                            qa_hash,
                        )
                        tk = report_digest(ti)
                        tp = output / "trials" / (tk + ".json")
                        if tp.exists():
                            trial = checked_payload(tp, ti)
                        else:
                            trial = canonical(
                                resolution_dev.score_with_memory(
                                    backend(),
                                    video,
                                    q,
                                    reference["order"],
                                    inputs.config,
                                    320,
                                )
                            )
                            resolution_dev.validate_processor(trial, 320, 16)
                            save_payload(tp, ti, trial)
                            calls += 1
                        resolution_dev.validate_processor(trial, 320, 16)
                        trials.append(trial)
                        keys.append(tk)
                        print(
                            f"{vid}/{q.question_id}/{arm}: "
                            f"order {number + 1}/6 saved",
                            flush=True,
                        )
                    strict_trials(trials, q, inputs.config)
                    entry = {
                        "selected_indices": ids,
                        "selected_timestamps": times[ids].tolist(),
                        "qa_pixels_sha256": qa_hash,
                        "trials": trials,
                        "trial_keys": keys,
                        "qa_origin": "this_experiment",
                        "source_arm": None,
                    }
                if arm in NEW_ARMS:
                    # Sharing QA must never overwrite the new arm's own plan.
                    entry["selection_plan"] = deepcopy(selection)
                conditions[arm] = entry
                reusable[tuple(ids)] = entry
            row = {
                "video_id": vid,
                "question_id": q.question_id,
                "duration_group": source["duration_group"],
                "candidate_sha256": plan["candidate_sha256"],
                "conditions": conditions,
            }
            validate_result(row, plan, source, q, inputs.config)
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
    summary, metrics = summarize(rows, inputs.protocol)
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
        return evaluate(args, inputs)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("outputs/selector_compare320_long"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/event_selection320_long"),
    )
    p.add_argument(
        "--phase",
        choices=("prepare", "evaluate", "summarize"),
        default="prepare",
    )
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument(
        "--max-videos", type=int, help="Explicitly labelled dev smoke subset"
    )
    p.add_argument(
        "--max-questions",
        type=int,
        help="First N questions per video for dev smoke",
    )
    return p


def main():
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
