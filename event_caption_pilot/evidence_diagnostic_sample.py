"""Freeze a reproducible 20-question evidence-diagnostic review sample.

This is outcome-stratified exploratory sampling, not a population estimate.
No videos, images, encoders or QA models are opened. Gold/outcomes and arm
identities are written only to the private manifest. Public review files use
independently shuffled anonymous set labels for each question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("R_topk", "G_topk", "R_diverse", "G_diverse")
VERSION = "fixed-event-evidence-diagnostic20-v1"
SAMPLE_POLICY = {
    "design": "One question per source video; 20 distinct videos",
    "changed_definition": "Four arm correct-counts across six orders differ",
    "changed_priority": (
        "If any question in a video changed, maximize the arm correct-count "
        "range, then sum absolute changes from R_topk, then seeded SHA256 tie"
    ),
    "stable_balance": (
        "For remaining videos minimize stable-partial count, then minimize "
        "absolute all-zero minus all-six-correct count, then seeded SHA256 tie"
    ),
    "tie_and_review_order": "SHA256 of seed, purpose and stable IDs",
    "freeze": "Freeze the sample and anonymous set mapping before images",
    "scope": (
        "Outcome-stratified exploratory diagnosis. Not representative "
        "accuracy, evidence-recall prevalence or a held-out evaluation"
    ),
    "probe_baseline_arm": "G_topk",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_checkpoint(path, identity):
    value = read(path)
    require(value["identity"] == identity, f"Checkpoint identity: {path}")
    require(
        value["payload_sha256"] == digest(value["payload"]),
        f"Checkpoint payload: {path}",
    )
    return value["payload"]


def seeded_rank(seed, purpose, *values):
    return digest([seed, purpose, *values])


def stratum(counts):
    require(
        len(counts) == 4
        and all(type(n) is int and 0 <= n <= 6 for n in counts),
        "Invalid arm counts",
    )
    if min(counts) != max(counts):
        return "changed"
    if counts[0] == 0:
        return "all_zero"
    if counts[0] == 6:
        return "stable_success"
    return "stable_partial"


def choose_sample(candidates, seed=42, expected_videos=20):
    """Select using outcomes only; annotation content never affects choice."""
    require(type(seed) is int, "Seed must be an integer")
    by_video = defaultdict(list)
    seen = set()
    for entry in candidates:
        key = (entry["video_id"], entry["question_id"])
        require(key not in seen, "Duplicate sample candidate")
        seen.add(key)
        value = {
            "video_id": key[0],
            "question_id": key[1],
            "correct_counts": list(entry["correct_counts"]),
        }
        value["stratum"] = stratum(value["correct_counts"])
        by_video[key[0]].append(value)
    require(len(by_video) == expected_videos, "Unexpected video count")
    chosen, stable = [], []
    for vid in sorted(by_video):
        changed = [x for x in by_video[vid] if x["stratum"] == "changed"]
        if changed:

            def priority(value):
                counts = value["correct_counts"]
                return (
                    -(max(counts) - min(counts)),
                    -sum(abs(n - counts[0]) for n in counts[1:]),
                    seeded_rank(
                        seed, "changed_question", vid, value["question_id"]
                    ),
                )

            chosen.append(min(changed, key=priority))
        else:
            stable.append(vid)
    # Dynamic programming finds a balanced feasible assignment without
    # exponential enumeration or dependence on input record ordering.
    states = {(0, 0, 0): ((), [])}
    kinds = {"all_zero": 0, "stable_success": 1, "stable_partial": 2}
    for vid in stable:
        updated = {}
        for counts, (tie, selected) in states.items():
            for value in by_video[vid]:
                next_counts = list(counts)
                next_counts[kinds[value["stratum"]]] += 1
                next_counts = tuple(next_counts)
                next_tie = tie + (
                    seeded_rank(
                        seed, "stable_question", vid, value["question_id"]
                    ),
                )
                candidate = (next_tie, selected + [value])
                if (
                    next_counts not in updated
                    or next_tie < updated[next_counts][0]
                ):
                    updated[next_counts] = candidate
        states = updated
    _, (_, selected) = min(
        states.items(),
        key=lambda item: (
            item[0][2],
            abs(item[0][0] - item[0][1]),
            item[1][0],
        ),
    )
    chosen.extend(selected)
    require(
        len(chosen) == expected_videos
        and len({x["video_id"] for x in chosen}) == expected_videos,
        "Sampling did not retain one question per video",
    )
    return sorted(
        chosen,
        key=lambda x: seeded_rank(
            seed, "review_order", x["video_id"], x["question_id"]
        ),
    )


def validate_trial(trial, annotation, order):
    require(
        sorted(order) == [0, 1, 2, 3]
        and trial["order"] == order
        and trial["presented_options"]
        == [annotation["options"][i] for i in order],
        "Trial order or options differ",
    )
    scores = trial["scoring"]["option_scores"]
    require(
        len(scores) == 4 and all(math.isfinite(x) for x in scores),
        "Invalid option scores",
    )
    predicted = max(range(4), key=lambda i: scores[i])
    require(
        type(trial["scoring"]["predicted_index"]) is int
        and trial["scoring"]["predicted_index"] == predicted
        and trial["scored_original_index"] == order[predicted]
        and type(trial["scoring_correct"]) is bool
        and trial["scoring_correct"]
        == (order[predicted] == annotation["answer_index"]),
        "Trial argmax, remapping or correctness differs",
    )


def load_source(root):
    """Bind completed source, all 60 questions and all 1,440 raw records."""
    protocol, comparison = (
        read(root / "protocol.json"),
        read(root / "comparison.json"),
    )
    runtime, audit = read(root / "runtime.json"), read(root / "audit.json")
    protocol_hash, runtime_hash = digest(protocol), digest(runtime)
    require(
        protocol["scope"] == "full"
        and protocol["qa_side"] == 224
        and protocol["frame_budget"] == 32
        and protocol["orders_per_question"] == 6
        and protocol["conditions"] == list(ARMS),
        "Require completed full224 design",
    )
    require(
        comparison["status"] == "complete"
        and comparison["completed"] is True
        and comparison["protocol_sha256"] == protocol_hash
        and comparison["runtime_sha256"] == runtime_hash,
        "Source result identity differs",
    )
    require(
        audit["status"] == "verified"
        and audit["scope"] == "full"
        and audit["comparison_sha256"] == digest(comparison)
        and audit["protocol_sha256"] == protocol_hash
        and audit["runtime_sha256"] == runtime_hash,
        "Require a matching independently verified source",
    )
    require(
        read(root / "handoff.json")
        == {k: v for k, v in comparison.items() if k != "questions"},
        "Source handoff differs",
    )
    videos = {v["video_id"]: v for v in protocol["videos"]}
    annotations = {
        (vid, str(q["question_id"])): q
        for vid, item in videos.items()
        for q in item["questions"]
    }
    rows = {
        (r["video_id"], r["question_id"]): r for r in comparison["questions"]
    }
    require(
        len(videos) == 20
        and len(annotations) == len(rows) == len(comparison["questions"]) == 60
        and set(rows) == set(annotations),
        "Source cohort differs from 20 videos / 60 questions",
    )
    plans = {}
    for vid in videos:
        plan = checked_checkpoint(
            root / "plans" / (digest(vid) + ".json"),
            {"protocol_sha256": protocol_hash, "video_id": vid},
        )
        require(
            digest(plan) == protocol["prepared_sha256"][vid],
            "Source plan differs",
        )
        plans[vid] = plan
    bindings, candidates, n_trials = {}, [], 0
    for key, row in rows.items():
        vid, qid = key
        question_identity = {
            "protocol_sha256": protocol_hash,
            "video_id": vid,
            "question_id": qid,
            "runtime_sha256": runtime_hash,
        }
        persisted = checked_checkpoint(
            root / "questions" / (digest([vid, qid]) + ".json"),
            question_identity,
        )
        require(
            persisted == row and set(row["conditions"]) == set(ARMS),
            "Raw question or arms differ from comparison",
        )
        orders = protocol["orders"][f"{vid}/{qid}"]
        require(
            len(orders) == 6 and len({tuple(o) for o in orders}) == 6,
            "Require six distinct source option orders",
        )
        bound_arms, counts = {}, []
        for name in ARMS:
            arm = row["conditions"][name]
            selection = plans[vid]["questions"][qid][name]
            require(
                all(arm[k] == value for k, value in selection.items())
                and len(arm["trials"]) == len(arm["trial_provenance"]) == 6,
                "Raw arm selection or trial count differs",
            )
            trials = []
            for order, trial, provenance in zip(
                orders, arm["trials"], arm["trial_provenance"]
            ):
                validate_trial(trial, annotations[key], order)
                identity = {
                    "protocol_sha256": protocol_hash,
                    "runtime_sha256": runtime_hash,
                    "video_id": vid,
                    "question_id": qid,
                    "selected_indices": arm["selected_indices"],
                    "qa_pixels_sha256": arm["qa_pixels_sha256"],
                    "order": order,
                }
                trial_key = digest(identity)
                record = checked_checkpoint(
                    root / "trials" / (trial_key + ".json"), identity
                )
                origin = record["origin_condition"]
                require(
                    record["trial"] == trial
                    and origin in ARMS
                    and provenance
                    == {
                        "kind": "new"
                        if origin == name
                        else "current_condition",
                        "origin_condition": origin,
                        "trial_key": trial_key,
                        "trial_sha256": digest(trial),
                    },
                    "Raw trial provenance differs",
                )
                trials.append(
                    {
                        "order": order,
                        "trial_key": trial_key,
                        "trial_sha256": digest(trial),
                        "trial_payload_sha256": digest(record),
                        "predicted_original_index": trial[
                            "scored_original_index"
                        ],
                        "correct": trial["scoring_correct"],
                    }
                )
                n_trials += 1
            count = sum(t["correct"] for t in trials)
            counts.append(count)
            bound_arms[name] = {
                "selected_indices": arm["selected_indices"],
                "selected_timestamps": arm["selected_timestamps"],
                "allocation": arm["allocation"],
                "qa_pixels_sha256": arm["qa_pixels_sha256"],
                "correct_count": count,
                "trial_bindings": trials,
            }
        bindings[key] = {
            "source_question_sha256": digest(row),
            "source_question_identity": question_identity,
            "arms": bound_arms,
        }
        candidates.append(
            {"video_id": vid, "question_id": qid, "correct_counts": counts}
        )
    require(n_trials == 1440, "Require all 1,440 source trial records")
    return (
        protocol,
        comparison,
        audit,
        annotations,
        plans,
        bindings,
        candidates,
    )


def make_documents(root, seed=42):
    protocol, comparison, audit, annotations, plans, bindings, candidates = (
        load_source(root)
    )
    sample = choose_sample(candidates, seed)
    cases = []
    for number, entry in enumerate(sample, 1):
        vid, qid = entry["video_id"], entry["question_id"]
        key = (vid, qid)
        annotation, plan = annotations[key], plans[vid]
        arm_order = sorted(
            ARMS,
            key=lambda name: seeded_rank(seed, "blind_sets", vid, qid, name),
        )
        external = protocol["external_pixels"][vid]
        base = plan["questions"][qid]["G_topk"]
        cases.append(
            {
                "case_id": f"C{number:02d}",
                "video_id": vid,
                "question_id": qid,
                "question": {
                    k: deepcopy(annotation[k])
                    for k in ("text", "options", "answer_index")
                },
                "task_type": annotation.get("task_type"),
                "stratum": entry["stratum"],
                "arm_correct_counts": dict(zip(ARMS, entry["correct_counts"])),
                "blind_set_to_arm": {
                    f"S{i}": arm for i, arm in enumerate(arm_order, 1)
                },
                "source_video": deepcopy(protocol["inputs"][vid]),
                "duration_seconds": plan["duration_seconds"],
                "candidate_count": plan["candidate_count"],
                "candidate_sha256": plan["candidate_sha256"],
                "candidate_timestamps": plan["candidate_timestamps"],
                "candidate_question_cosines": base["frame_scores_raw_cosine"],
                "qa_side": protocol["qa_side"],
                "qa_pixel_cache": {
                    "npz_path": external["npz_path"],
                    "metadata_path": external["metadata_path"],
                    "file_sha256": external["metadata"]["file_sha256"],
                    "array_sha256": external["metadata"]["array_sha256"],
                },
                "probe_baseline_arm": "G_topk",
                **bindings[key],
            }
        )
    private = {
        "version": 1,
        "experiment": VERSION,
        "seed": seed,
        "source_root": str(root.resolve()),
        "source_protocol_sha256": digest(protocol),
        "source_runtime_sha256": comparison["runtime_sha256"],
        "source_comparison_sha256": digest(comparison),
        "source_audit_sha256": digest(audit),
        "candidate_config": deepcopy(protocol["candidate_config"]),
        "sample_policy": deepcopy(SAMPLE_POLICY),
        "sample_size": 20,
        "unique_videos": 20,
        "stratum_counts": dict(Counter(x["stratum"] for x in sample)),
        "sample_frozen_before_visual_review": True,
        "cases": cases,
    }
    blinded = make_blinded(private)
    template = make_review_template(blinded)
    return private, blinded, template


def make_blinded(private):
    """Whitelist public fields, excluding gold, outcomes and true arm names."""
    cases = []
    allowed = (
        "case_id",
        "video_id",
        "question_id",
        "source_video",
        "duration_seconds",
        "candidate_count",
        "candidate_sha256",
        "candidate_timestamps",
        "candidate_question_cosines",
        "qa_side",
        "qa_pixel_cache",
    )
    for entry in private["cases"]:
        public = {key: deepcopy(entry[key]) for key in allowed}
        public["question"] = {
            key: deepcopy(entry["question"][key])
            for key in ("text", "options")
        }
        public["sets"] = {
            label: {
                key: deepcopy(entry["arms"][arm][key])
                for key in (
                    "selected_indices",
                    "selected_timestamps",
                    "qa_pixels_sha256",
                )
            }
            for label, arm in entry["blind_set_to_arm"].items()
        }
        cases.append(public)
    return {
        "version": 1,
        "experiment": VERSION,
        "private_manifest_sha256": digest(private),
        "sample_size": len(cases),
        "reviewer_type": "ai_visual_review",
        "candidate_config": deepcopy(private["candidate_config"]),
        "blinding": (
            "Gold, true arm names, outcomes and sampling strata are omitted. "
            "Prior knowledge must be recorded."
        ),
        "cases": cases,
    }


def make_review_template(blinded):
    cases = []
    for entry in blinded["cases"]:
        cases.append(
            {
                "case_id": entry["case_id"],
                "video_id": entry["video_id"],
                "question_id": entry["question_id"],
                "status": "pending",
                "reviewer_type": "ai_visual_review",
                "reviewer_prior_outcome_knowledge": None,
                "gold_revealed": False,
                "question_video_mapping": "unknown",
                "reviewed_indices": [],
                "reviewed_time_intervals_seconds": [],
                "reviewed_coverage": {
                    "candidate_count": entry["candidate_count"],
                    "overview_indices": [],
                    "native_inspected_indices": [],
                    "all_candidates_reviewed": False,
                    "sparse_overview_can_prove_absence": False,
                },
                "candidate_evidence_status": "unknown",
                "required_evidence_groups": [],
                "observed_facts": [],
                "selection_coverage": {
                    label: {
                        "status": "unknown",
                        "evidence_group_hits": [],
                        "reason": None,
                    }
                    for label in entry["sets"]
                },
                "readability_224": {
                    "status": "unknown",
                    "supporting_indices": [],
                    "reason": None,
                },
                "native_comparison": {
                    "status": "not_reviewed",
                    "native_inspected_indices": [],
                    "improved_legibility": None,
                    "reason": None,
                },
                "question_answer_before_gold": {
                    "predicted_option_index": None,
                    "reasoning": None,
                    "confidence": "unknown",
                },
                "confidence": "unknown",
                "unresolved_reason": None,
            }
        )
    return {
        "version": 1,
        "experiment": VERSION,
        "blinded_review_sha256": digest(blinded),
        "private_manifest_sha256": blinded["private_manifest_sha256"],
        "reviewer_type": "ai_visual_review",
        "instructions": [
            "Review images first and record your own answer before revealing "
            "gold or true arm identities.",
            "Unknown is valid. Never infer evidence absence from sparse "
            "overview frames.",
            "Record inspected candidate indices and intervals; distinguish "
            "224 input from native resolution.",
            "Each evidence group is one required fact; its indices/intervals "
            "are alternative observations of that fact.",
            "Use only observed facts, and report unavailable audio/subtitles "
            "or unreviewed intervals explicitly.",
            "Do not present AI visual review as human annotation or an "
            "exhaustive ground-truth evidence label.",
        ],
        "evidence_group_fields": [
            "group_id",
            "description",
            "alternative_candidate_indices",
            "alternative_time_intervals_seconds",
            "modality",
            "confidence",
        ],
        "observed_fact_fields": [
            "fact",
            "supporting_candidate_indices",
            "supporting_intervals_seconds",
            "resolution",
        ],
        "cases": cases,
    }


def write_frozen(path, value):
    if path.exists():
        require(
            read(path) == value,
            f"Refusing to overwrite frozen artifact: {path}",
        )
        return
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare(root, output, seed=42):
    root, output = root.resolve(), output.resolve()
    require(
        root != output
        and root not in output.parents
        and output not in root.parents,
        "Output must be separate from source",
    )
    private, blinded, template = make_documents(root, seed)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "private_manifest.json": private,
        "blinded_review.json": blinded,
        "review_template.json": template,
    }
    # Validate every existing file before writing any new one.
    for name, value in artifacts.items():
        if (output / name).exists():
            require(
                read(output / name) == value, f"Frozen output differs: {name}"
            )
    for name, value in artifacts.items():
        write_frozen(output / name, value)
    marker = {
        "status": "frozen",
        "experiment": VERSION,
        "artifact_sha256": {
            name: digest(value) for name, value in artifacts.items()
        },
        "source_root": str(root),
        "sample_size": 20,
        "seed": seed,
        "images_opened": False,
        "gpu_inference_performed": False,
    }
    path = output / "sample_frozen.json"
    if path.exists():
        prior = read(path)
        require(
            all(prior.get(k) == v for k, v in marker.items()),
            "Freeze marker differs",
        )
    else:
        write_frozen(
            path,
            {**marker, "frozen_at": datetime.now(timezone.utc).isoformat()},
        )
    return private, blinded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    _, blinded = prepare(args.source_root, args.output_root, args.seed)
    print(
        json.dumps(
            {
                "status": "frozen",
                "sample_size": blinded["sample_size"],
                "cases": [
                    {
                        k: entry[k]
                        for k in ("case_id", "video_id", "question_id")
                    }
                    for entry in blinded["cases"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
