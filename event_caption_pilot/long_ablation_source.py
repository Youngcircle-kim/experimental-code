"""Freeze validated dev-Long B32 baselines without changing prior outputs."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from .aggregate_budget_shards import (
    expected_protocol,
    job_directory,
    rooted,
    validate_plan,
)
from .compare_budgets import (
    load_complete,
    normalized_backend,
    normalized_runtime,
    validate_backend,
)
from .config import Config
from .encoders import A4500_ENCODERS
from .event_refinement_dev import question_from
from .order_experiments import option_orders
from .portable_budget import (
    checkpoint,
    read_json,
    validate_completed_row,
)
from .qa_diagnostics import report_digest

NEW_CONDITIONS = ["D_topk", "B", "B_topk"]
BASELINES = ["uniform", "frame_top", "temporal_bin", "C", "D"]


def code_hashes():
    return {
        path.name: hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def verify_source_code(recorded):
    """New modules are allowed; every originally recorded module is frozen."""
    current = code_hashes()
    changed = [
        name for name, sha in recorded.items() if current.get(name) != sha
    ]
    if not recorded or changed:
        raise ValueError(
            "Source experiment code changed: "
            + ", ".join(changed)
            + ". Restore the original code before reusing its baselines."
        )


def build_snapshot(source_root, expected_videos=20, expected_questions=60):
    """Read only the two completed dev-B32 shards, never run baseline QA."""
    root = Path(source_root).resolve()
    source_plan = read_json(rooted(root, "suite_plan.json"))
    validate_plan(source_plan)
    if "dev" not in source_plan["splits"] or 32 not in source_plan["budgets"]:
        raise ValueError("Source suite must contain dev at frame budget 32")
    full = source_plan["splits"]["dev"]["protocol"]
    verify_source_code(full["code_sha256"])
    manifest = read_json(Path(source_plan["manifest_path"]))
    settings = read_json(Path(source_plan["config_path"]))
    if (
        report_digest(manifest) != full["manifest_sha256"]
        or report_digest(settings) != full["config_source_sha256"]
    ):
        raise ValueError("Source manifest/config changed since baseline QA")
    videos = [
        deepcopy(v)
        for v in manifest["videos"]
        if v["split"] == "dev" and v["duration"] == "long"
    ]
    for video in videos:
        for question in video["questions"]:
            question["question_id"] = str(question["question_id"])
    cohort = {
        v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
        for v in videos
    }
    if (
        len(cohort) != expected_videos
        or sum(map(len, cohort.values())) != expected_questions
    ):
        raise ValueError(
            f"Require all {expected_videos} dev Long videos / "
            f"{expected_questions} questions; found {len(cohort)} / "
            f"{sum(map(len, cohort.values()))}. No videos were truncated."
        )
    if len(videos) != len(cohort) or any(
        full["cohort"].get(vid) != questions
        for vid, questions in cohort.items()
    ):
        raise ValueError("Source B32 must cover every dev Long question")
    manifest_videos = {v["video_id"]: v for v in manifest["videos"]}
    rows, plans, protocols, shards = {}, {}, {}, []
    reference, config = None, None
    seen_candidates, seen_sources = {}, {}
    for shard in source_plan["splits"]["dev"]["shards"]:
        index = shard["index"]
        directory = job_directory(root, "dev", 32, index)
        print(
            f"Validating saved dev B32 shard {index}: {directory}", flush=True
        )
        for name in ("protocol", "results", "progress", "backend", "runtime"):
            rooted(root, directory, name + ".json")
        child = load_complete(directory, 32)
        protocol = expected_protocol(source_plan, "dev", 32, index)
        if child["protocol"] != protocol:
            raise ValueError("Source B32 child differs from its suite plan")
        config = Config(**protocol["config"])
        digest = report_digest(protocol)
        backend = child["backend"]
        if (
            child["runtime"].get("environment", {}).get("CUDA_VISIBLE_DEVICES")
            != str(shard["gpu"])
            or backend.get("device") != "cuda:0"
            or any(
                backend.get(key, {}).get("devices") != ["cuda:0"]
                for key in ("encoder_parameters", "vlm_parameters")
            )
        ):
            raise ValueError("Source B32 GPU assignment is inconsistent")
        if (
            checkpoint(
                rooted(root, directory, "backend_identity.json"),
                {"protocol_sha256": digest},
            )
            != backend
        ):
            raise ValueError("Source backend identity mismatch")
        identity = (
            normalized_backend(backend),
            normalized_runtime(child["runtime"]),
        )
        if reference is not None and reference != identity:
            raise ValueError("Source B32 backend/runtime differs by shard")
        reference = identity
        # Validate the raw checkpoint chain, including the non-Long rows in
        # each complete shard; aggregate JSON alone is not sufficient.
        for folder in ("plans", "trials", "questions"):
            for path in rooted(root, directory, folder).glob("*.json"):
                rooted(root, path)
        for key, row in child["rows"].items():
            vid, qid = key
            item = manifest_videos[vid]
            matches = [
                q for q in item["questions"] if str(q["question_id"]) == qid
            ]
            if len(matches) != 1:
                raise ValueError("Source question differs from its manifest")
            question = question_from(matches[0])
            qidentity = {
                "protocol_sha256": digest,
                "video_id": vid,
                "question_id": qid,
            }
            if (
                checkpoint(
                    rooted(
                        root,
                        directory,
                        "questions",
                        report_digest([vid, qid]) + ".json",
                    ),
                    qidentity,
                )
                != row
            ):
                raise ValueError("Source result/question checkpoint mismatch")
            validate_completed_row(
                row, item, question, config, digest, directory
            )
            for lookup, content in (
                (seen_candidates, row["candidate_sha256"]),
                (seen_sources, row["source_id"]),
            ):
                if content in lookup and lookup[content] != vid:
                    raise ValueError(
                        "Duplicate source/candidate across videos"
                    )
                lookup[content] = vid
            if vid not in cohort:
                continue
            if key in rows:
                raise ValueError("Duplicate baseline question across shards")
            if len(option_orders(len(question.options), config.seed)) != 6:
                raise ValueError("This experiment requires six option orders")
            rows[key] = row
            saved_plan = checkpoint(
                rooted(root, directory, "plans", report_digest(vid) + ".json"),
                {
                    "protocol_sha256": digest,
                    "video_id": vid,
                    "candidate_sha256": row["candidate_sha256"],
                },
            )
            if vid in plans and plans[vid] != saved_plan:
                raise ValueError("Baseline plan differs within a video")
            plans[vid] = saved_plan
        protocols[str(index)] = protocol
        selected = [vid for vid in shard["video_ids"] if vid in cohort]
        if not selected:
            raise ValueError("Both source GPU shards need dev Long videos")
        shards.append(
            {"index": index, "gpu": shard["gpu"], "video_ids": selected}
        )
    expected = {(v, q) for v, questions in cohort.items() for q in questions}
    if set(rows) != expected or len(shards) != 2:
        raise ValueError("Require complete Long baselines across two shards")
    snapshot = {
        "version": 1,
        "experiment": "a4500_long_ablation_source_v1",
        "source_root": str(root),
        "source_suite_plan_sha256": report_digest(source_plan),
        "source_protocols": protocols,
        "config": config.to_dict(),
        "cohort": cohort,
        "inputs": {vid: full["inputs"][vid] for vid in cohort},
        "videos": videos,
        "baseline_rows": [rows[key] for key in sorted(rows)],
        "baseline_plans": plans,
        "backend": reference[0],
        "runtime": reference[1],
        "source_shards": shards,
    }
    # JSON normalization also makes Config's tuple-valued fields resumable.
    snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
    validate_snapshot(snapshot)
    return snapshot


def validate_snapshot(snapshot):
    if (
        snapshot.get("version") != 1
        or snapshot.get("experiment") != "a4500_long_ablation_source_v1"
    ):
        raise ValueError("Invalid baseline snapshot version")
    config = Config(**snapshot["config"])
    config.validate()
    if (
        config.frame_budget != 32
        or config.device != "cuda:0"
        or config.mode != "real"
        or config.backend != "transformers"
        or config.pilot != "both"
        or config.encoder_model not in A4500_ENCODERS
        or config.vlm_model != "Qwen/Qwen3.5-4B"
        or any(
            not re.fullmatch(r"[0-9a-f]{40}", revision)
            for revision in (config.encoder_revision, config.vlm_revision)
        )
        or config.candidate_fps != 2.0
        or (config.frame_height, config.frame_width) != (224, 224)
        or config.vlm_min_pixels != 320**2
        or config.vlm_max_pixels != 320**2
        or config.model_dtype != "bfloat16"
        or config.attention_implementation != "sdpa"
        or not config.deterministic
        or not config.qa_length_normalize
        or config.bootstrap_samples != 5000
        or config.confidence_level != 0.95
    ):
        raise ValueError(
            "Require frozen B32 / 224 candidates / QA320 settings"
        )
    validate_backend(snapshot["backend"], config)
    # A new encoder requires newly generated baseline plans and QA. Merely
    # changing the snapshot config/backend must not relabel old CLIP results.
    protocols = snapshot.get("source_protocols", {})
    if not protocols or any(
        protocol.get("config") != snapshot["config"]
        for protocol in protocols.values()
    ):
        raise ValueError(
            "Baseline source protocol config/model differs from snapshot"
        )
    cohort = snapshot["cohort"]
    expected = {(v, q) for v, questions in cohort.items() for q in questions}
    if (
        not expected
        or sum(map(len, cohort.values())) != len(expected)
        or set(snapshot["inputs"]) != set(cohort)
        or set(snapshot["baseline_plans"]) != set(cohort)
    ):
        raise ValueError("Invalid baseline snapshot cohort")
    rows = snapshot["baseline_rows"]
    if (
        len(rows) != len(expected)
        or {(r["video_id"], r["question_id"]) for r in rows} != expected
        or any(
            r["duration_group"] != "long"
            or set(r["conditions"]) != set(BASELINES)
            for r in rows
        )
    ):
        raise ValueError("Invalid baseline snapshot rows")
    seen = set()
    for item in snapshot["videos"]:
        vid = item["video_id"]
        if (
            vid in seen
            or item["duration"] != "long"
            or item.get("split", "dev") != "dev"
            or [str(q["question_id"]) for q in item["questions"]]
            != cohort.get(vid)
        ):
            raise ValueError("Invalid baseline snapshot video metadata")
        seen.add(vid)
    if seen != set(cohort):
        raise ValueError("Missing baseline snapshot video metadata")


def validate_frozen(plan, snapshot):
    validate_snapshot(snapshot)
    if (
        plan.get("version") != 1
        or plan.get("experiment") != "a4500_long_ablation_v1"
        or plan.get("snapshot_sha256") != report_digest(snapshot)
        or plan.get("source_root") != snapshot["source_root"]
        or plan.get("conditions") != NEW_CONDITIONS
        or plan.get("qa_side") != 320
        or plan.get("frame_budget") != 32
        or plan.get("split") != "dev"
        or plan.get("duration") != "long"
        or plan.get("shards") != snapshot["source_shards"]
        or plan.get("n_videos") != len(snapshot["cohort"])
        or plan.get("n_questions") != len(snapshot["baseline_rows"])
        or plan.get("expected_new_qa_trials")
        != len(snapshot["baseline_rows"]) * 18
    ):
        raise ValueError("Frozen ablation plan/snapshot mismatch")
    shards = plan["shards"]
    gpus = plan["gpu_ids"]
    covered = []
    if (
        len(shards) != 2
        or len(gpus) != 2
        or len(set(gpus)) != 2
        or any(type(g) is not int or g < 0 for g in gpus)
    ):
        raise ValueError("Require two distinct nonnegative GPU IDs")
    for i, shard in enumerate(shards):
        if (
            shard["index"] != i
            or shard["gpu"] != gpus[i]
            or not shard["video_ids"]
            or shard["video_ids"] != sorted(set(shard["video_ids"]))
        ):
            raise ValueError("Invalid frozen GPU shard assignment")
        covered.extend(shard["video_ids"])
    if len(covered) != len(set(covered)) or set(covered) != set(
        snapshot["cohort"]
    ):
        raise ValueError("GPU shards must cover all videos exactly once")


def read_frozen(plan_path, expected_digest=None, verify_code=True):
    path = Path(plan_path).resolve()
    plan = read_json(path)
    if expected_digest is not None and report_digest(plan) != expected_digest:
        raise ValueError("Frozen plan digest differs from expected plan")
    snapshot = read_json(rooted(path.parent, "source_snapshot.json"))
    validate_frozen(plan, snapshot)
    if verify_code:
        if plan["code_sha256"] != code_hashes():
            raise ValueError(
                "Code changed after planning; restore frozen code"
            )
        for protocol in snapshot["source_protocols"].values():
            verify_source_code(protocol["code_sha256"])
    return deepcopy(plan), deepcopy(snapshot)
