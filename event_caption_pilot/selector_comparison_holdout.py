"""Prepare and evaluate frozen four-selector QA on video-disjoint holdout."""

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import resolution_dev, selector_comparison_dev
from .ablation import select_frame_control
from .algorithms import normalize_rows
from .backends import build_backend
from .bottleneck_diagnostics import read, replay, validate_plan, write_csv
from .cache import write_json
from .data import array_digest, validate_video
from .event_factorial import make_partitions, select_plans
from .event_factorial_dev import validate_indices
from .event_refinement_dev import checked_payload, question_from, save_payload
from .frame_replacement_probe import strict_trials
from .frozen_dev import check_manifest
from .midpoint_review import load_candidates
from .order_experiments import option_orders, seed_call
from .qa_diagnostics import report_digest
from .segment_cap_dev import validate_metadata
from .types import Video

ARMS = selector_comparison_dev.ARMS
CONTROLS = selector_comparison_dev.CONTROLS
COMPARISONS = selector_comparison_dev.COMPARISONS
canonical = selector_comparison_dev.canonical


def load_frozen_source(args):
    """Verify the completed dev lineage, without decoding or inference."""
    source = args.source_dir.resolve()
    saved = read(source / "protocol.json")
    summary = read(source / "handoff.json")
    if (
        saved.get("experiment") != "D_vs_frame_controls_fixed_320"
        or summary["protocol_sha256"] != report_digest(saved)
        or saved["arms"] != list(ARMS)
    ):
        raise ValueError("Require completed frozen four-selector dev source")
    selector_comparison_dev.verify_code(saved, saved["code_sha256"])
    source_args = selector_comparison_dev.parser().parse_args(
        [
            "--source-dir",
            saved["source_dir"],
            "--factorial-dir",
            saved["factorial_dir"],
            "--output-dir",
            str(args.output_dir),
        ]
    )
    frozen = selector_comparison_dev.load_inputs(source_args)
    # New standalone runners may be added; existing implementations may not
    # change. Their hashes were checked above against the completed dev run.
    actual = canonical(frozen.protocol)
    if {k: v for k, v in actual.items() if k != "code_sha256"} != {
        k: v for k, v in saved.items() if k != "code_sha256"
    }:
        raise ValueError("Frozen source protocol no longer reproduces")
    results = read(source / "results.json")
    if results["protocol_sha256"] != report_digest(saved):
        raise ValueError("Source results/protocol mismatch")
    rows = selector_comparison_dev.indexed(results["questions"])
    if set(rows) != set(frozen.plans):
        raise ValueError("Source comparison is incomplete")
    for item in frozen.videos:
        for annotation in item["questions"]:
            q = question_from(annotation)
            key = (item["video_id"], q.question_id)
            selector_comparison_dev.validate_result(
                rows[key],
                frozen.plans[key],
                frozen.d_rows[key],
                q,
                frozen.config,
            )
    expected_summary, _ = selector_comparison_dev.summarize(
        list(rows.values()), saved
    )
    if expected_summary != summary:
        raise ValueError("Source handoff does not match source results")
    factorial = read(Path(saved["factorial_dir"]) / "results.json")
    frozen.dev_candidate_hashes = {
        r["candidate_sha256"] for r in factorial["questions"]
    }
    frozen.protocol = saved
    return frozen


def load_inputs(args):
    if args.split == "eval" and (
        args.max_videos is not None or args.max_questions is not None
    ):
        raise ValueError(
            "Holdout cohort is fixed; subsets require --split dev"
        )
    if args.split == "dev" and args.max_videos is None:
        raise ValueError("Dev smoke requires an explicit --max-videos")
    if any(
        n is not None and n < 1 for n in (args.max_videos, args.max_questions)
    ):
        raise ValueError("Subset limits must be positive")
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if (
        output == source
        or output in source.parents
        or source in output.parents
    ):
        raise ValueError("Output must be separate from the source directory")
    frozen = load_frozen_source(args)
    original, config, manifest = (
        frozen.original,
        frozen.config,
        frozen.manifest,
    )
    original.validate()
    config.validate()
    if (
        original.frame_budget != 16
        or original.max_segments != 128
        or original.detector != "D2"
        or (original.frame_height, original.frame_width) != (224, 224)
        or original.mode != "real"
        or original.backend != "transformers"
        or original.pilot != "both"
        or original.vlm_revision == "main"
        or original.encoder_revision == "main"
    ):
        raise ValueError("Require frozen real cap-128 D2 / 224 candidates")
    if config != replace(original, vlm_max_pixels=320**2):
        raise ValueError(
            "QA config may differ only by the frozen 320 settings"
        )
    manifest_data = read(manifest)
    if report_digest(manifest_data) != frozen.protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    check_manifest(manifest_data)  # Unique video AND original source IDs.
    for item in manifest_data["videos"]:
        if not item.get("video_id") or not item.get("source_id"):
            raise ValueError("Each video needs nonempty video_id/source_id")
    videos = [
        v
        for v in manifest_data["videos"]
        if v["split"] == args.split and v["duration"] == "long"
    ]
    if args.split == "dev":
        # Smoke only on videos already included in the frozen dev comparison.
        known = {v["video_id"] for v in frozen.videos}
        videos = [v for v in videos if v["video_id"] in known]
    videos = videos[: args.max_videos]
    videos = [
        {**v, "questions": v["questions"][: args.max_questions]}
        for v in videos
    ]
    if not videos:
        raise ValueError("No matching long videos")
    for item in videos:
        if (
            item.get("timestamp_mode") != "constant_fps"
            or Path(item["path"]).suffix.lower() != ".mp4"
        ):
            raise ValueError("Require original constant_fps MP4 videos")
        seen = set()
        for annotation in item["questions"]:
            q = question_from(annotation)
            if (
                not q.question_id
                or q.question_id in seen
                or not q.text.strip()
                or len(q.options) != 4
                or any(not text.strip() for text in q.options)
                or type(q.answer_index) is not int
                or not 0 <= q.answer_index < 4
            ):
                raise ValueError("Require unique, valid four-option questions")
            seen.add(q.question_id)
        if not seen:
            raise ValueError("Video has no questions")
    missing = [
        {
            "video_id": v["video_id"],
            "path": str((manifest.parent / v["path"]).resolve()),
        }
        for v in videos
        if not (manifest.parent / v["path"]).is_file()
    ]
    protocol = canonical(
        {
            "version": 1,
            "experiment": "frozen_selector_holdout_320",
            "source_dir": str(source),
            "source_protocol_sha256": report_digest(frozen.protocol),
            "manifest_sha256": report_digest(manifest_data),
            "candidate_config": original.to_dict(),
            "qa_config": config.to_dict(),
            "backend_metadata_sha256": report_digest(frozen.metadata),
            "split": args.split,
            "evaluation_role": "holdout"
            if args.split == "eval"
            else "dev_smoke",
            "duration_filter": "long",
            "cohort": {
                v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
                for v in videos
            },
            "source_ids": {v["video_id"]: v["source_id"] for v in videos},
            "qa_side": 320,
            "frame_budget": 16,
            "arms": list(ARMS),
            "primary_comparison": "D_minus_uniform",
            "secondary_comparisons": list(COMPARISONS)[1:],
            "orders": [list(o) for o in option_orders(4, config.seed)],
            "selectors": {
                **frozen.protocol["selectors"],
                "D": "cap-128 D2 partition, query allocation, "
                "within-event uniform selection; recomputed on this cohort",
            },
            "selection_inputs": "Question text and 224 CLIP features only; "
            "no answer choices, gold answers, captions or evidence "
            "annotations.",
            "D_QA": "New QA on this cohort; no development trial reuse.",
            "cost_scope": "QA cost only. Decode, CLIP extraction, selection "
            "and model loading excluded; no end-to-end speed claim. All QA "
            "timings originate in this experiment; identical-input calls "
            "may be shared.",
            "interpretation": "Frozen-method holdout comparison on long "
            "videos (or explicitly labelled dev smoke). Within-question "
            "order means, equal question weight, paired video bootstrap "
            "5000. Secondary CIs "
            "are marginal, not multiplicity-adjusted. No evidence recall or "
            "external-framework claim. Prior holdout use cannot be proven "
            "by "
            "split metadata; audit before interpreting as confirmatory.",
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
        metadata=frozen.metadata,
        frozen=frozen,
        protocol=protocol,
        missing_videos=missing,
        dev_candidate_hashes=frozen.dev_candidate_hashes,
    )


def plan_path(output, video_id):
    return output / "plans" / (report_digest(video_id) + ".json")


def video_identity(item, digest):
    return {
        "protocol_sha256": digest,
        "video_id": item["video_id"],
        "source_id": item["source_id"],
    }


def select_conditions(features, query, timestamps, config):
    """Reproduce D and controls using visual features and question text."""
    normalized = normalize_rows(features, config.normalization_epsilon)
    q = normalize_rows(
        np.asarray(query)[None, :], config.normalization_epsilon
    )[0]
    scores = normalized @ q
    conditions = {}
    for arm in CONTROLS:
        ids = select_frame_control(arm, scores, config.frame_budget).tolist()
        conditions[arm] = {
            "selected_indices": ids,
            "selected_timestamps": np.asarray(timestamps)[ids].tolist(),
        }
    conditions["D"] = select_plans(
        features, query, timestamps, make_partitions(features, config), config
    )["D"]
    return {"frame_scores": scores.tolist(), "conditions": conditions}


def validate_prepared(prepared, item, config):
    n, times = prepared["candidate_count"], np.asarray(prepared["timestamps"])
    if (
        prepared["video_id"] != item["video_id"]
        or prepared["source_id"] != item["source_id"]
        or prepared["duration_group"] != item["duration"]
        or type(n) is not int
        or not 16 <= n <= config.max_candidate_frames
        or times.shape != (n,)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(prepared["duration_seconds"])
        or not 0 <= times[0] <= times[-1] < prepared["duration_seconds"]
        or set(prepared["questions"])
        != {str(q["question_id"]) for q in item["questions"]}
    ):
        raise ValueError(
            "Invalid prepared video identity/candidates/questions"
        )
    boundaries = None
    for choices in prepared["questions"].values():
        plans = choices["conditions"]
        scores = np.asarray(choices["frame_scores"], dtype=float)
        if set(plans) != set(ARMS) or scores.shape != (n,):
            raise ValueError("Incomplete prepared conditions/scores")
        for arm in ARMS:
            p = plans[arm]
            ids = p["selected_indices"]
            validate_indices(ids, n, 16)
            if not np.array_equal(times[ids], p["selected_timestamps"]):
                raise ValueError("Prepared selection timestamps changed")
            if (
                arm in CONTROLS
                and ids != select_frame_control(arm, scores, 16).tolist()
            ):
                raise ValueError("Prepared control does not reproduce")
        d = plans["D"]
        validate_plan(d, 16, "question_relevance")
        allocation, ids = replay(d, config.allocation_temperature)
        if (
            d["detector"] != "D2"
            or d["events"][-1][1] != n
            or len(d["events"]) > config.max_segments
            or d["temperature"] != config.allocation_temperature
            or allocation != d["allocation"]
            or ids != d["selected_indices"]
            or (boundaries is not None and boundaries != d["events"])
        ):
            raise ValueError("Prepared frozen D policy does not reproduce")
        boundaries = d["events"]


def prepare(args, inputs):
    digest, output = report_digest(inputs.protocol), args.output_dir
    encoder, plans, content_seen = None, {}, set()
    for item in inputs.videos:
        vid = item["video_id"]
        identity, path = video_identity(item, digest), plan_path(output, vid)
        if path.exists():
            prepared = checked_payload(path, identity)
        else:
            print(
                f"Preparing {vid}: original candidates + CLIP, no QA",
                flush=True,
            )
            frames, times, duration = load_candidates(
                item, inputs.manifest, inputs.original
            )
            video = Video(
                vid,
                args.split,
                frames,
                times,
                duration,
                tuple(question_from(q) for q in item["questions"]),
                item["source_id"],
            )
            validate_video(video, inputs.original)
            candidate_hash = array_digest(frames, times)
            if (
                args.split == "eval"
                and candidate_hash in inputs.dev_candidate_hashes
            ):
                raise ValueError(
                    "Holdout candidates duplicate development content"
                )
            if candidate_hash in content_seen:
                raise ValueError("Duplicate candidate content within cohort")
            if encoder is None:
                settings = replace(inputs.original, pilot="a")
                seed_call(settings)
                encoder = build_backend(settings)
                validate_metadata(
                    encoder.metadata(),
                    {
                        **inputs.metadata,
                        "vlm_max_pixels": inputs.original.vlm_max_pixels,
                    },
                    encoder_only=True,
                )
                write_json(output / "encoder_backend.json", encoder.metadata())
            seed_call(inputs.original)
            features = encoder.encode_frames(frames)
            questions = {
                q.question_id: select_conditions(
                    features,
                    encoder.encode_visual_question(q.text),
                    times,
                    inputs.original,
                )
                for q in video.questions
            }
            prepared = {
                "video_id": vid,
                "source_id": item["source_id"],
                "candidate_sha256": candidate_hash,
                "candidate_count": len(times),
                "timestamps": times.tolist(),
                "duration_seconds": duration,
                "duration_group": item["duration"],
                "features_sha256": array_digest(features),
                "questions": questions,
            }
            del features, frames, video
            validate_prepared(prepared, item, inputs.original)
            if args.split == "dev":
                # A smoke run proves the new path reproduces saved selection.
                for qid, selection in questions.items():
                    reference = inputs.frozen.plans[(vid, qid)]
                    if candidate_hash != reference["candidate_sha256"] or any(
                        selection["conditions"][a]["selected_indices"]
                        != reference["conditions"][a]["selected_indices"]
                        for a in ARMS
                    ):
                        raise ValueError(
                            "Dev smoke selections differ from frozen source"
                        )
                prepared["dev_selections_reproduced"] = True
            save_payload(path, identity, prepared)
        validate_prepared(prepared, item, inputs.original)
        content = prepared["candidate_sha256"]
        if content in content_seen or (
            args.split == "eval" and content in inputs.dev_candidate_hashes
        ):
            raise ValueError(
                "Duplicate candidate content across videos/splits"
            )
        content_seen.add(content)
        plans[vid] = report_digest(prepared)
        write_json(
            output / "prepare_progress.json",
            {
                "completed_videos": len(plans),
                "total_videos": len(inputs.videos),
                "status": "running",
                "evaluation_role": inputs.protocol["evaluation_role"],
            },
        )
    complete = {"protocol_sha256": digest, "plan_sha256": plans}
    write_json(output / "prepare_complete.json", complete)
    summary = {
        "stage": "prepared_no_QA",
        "protocol_sha256": digest,
        "evaluation_role": inputs.protocol["evaluation_role"],
        "n_videos": len(plans),
        "n_questions": sum(len(v["questions"]) for v in inputs.videos),
        "max_new_QA_calls": sum(len(v["questions"]) for v in inputs.videos)
        * 24,
    }
    write_json(output / "prepare_summary.json", summary)
    write_json(
        output / "prepare_progress.json",
        {
            "completed_videos": len(plans),
            "total_videos": len(plans),
            "status": "complete_no_QA",
            "evaluation_role": inputs.protocol["evaluation_role"],
        },
    )
    return summary


def load_prepared(args, inputs):
    digest = report_digest(inputs.protocol)
    marker = args.output_dir / "prepare_complete.json"
    if not marker.is_file():
        raise ValueError("Complete preparation required before QA")
    complete = read(marker)
    if complete["protocol_sha256"] != digest or set(
        complete["plan_sha256"]
    ) != {v["video_id"] for v in inputs.videos}:
        raise ValueError("Complete preparation required for the fixed cohort")
    result, seen = {}, set()
    for item in inputs.videos:
        vid = item["video_id"]
        p = checked_payload(
            plan_path(args.output_dir, vid), video_identity(item, digest)
        )
        if report_digest(p) != complete["plan_sha256"][vid]:
            raise ValueError("Prepared plan changed after completion")
        validate_prepared(p, item, inputs.original)
        content = p["candidate_sha256"]
        if content in seen or (
            args.split == "eval" and content in inputs.dev_candidate_hashes
        ):
            raise ValueError(
                "Duplicate candidate content across videos/splits"
            )
        seen.add(content)
        result[vid] = p
    return result


def validate_result(row, prepared, item, q, config, digest):
    if (
        row["video_id"] != item["video_id"]
        or row["question_id"] != q.question_id
        or row["duration_group"] != item["duration"]
        or row["candidate_sha256"] != prepared["candidate_sha256"]
        or row["prepared_sha256"] != report_digest(prepared)
        or set(row["conditions"]) != set(ARMS)
    ):
        raise ValueError("Result identity/prepared selection mismatch")
    shared = {}
    for arm, entry in row["conditions"].items():
        plan = prepared["questions"][q.question_id]["conditions"][arm]
        if any(
            entry[k] != plan[k]
            for k in ("selected_indices", "selected_timestamps")
        ):
            raise ValueError("QA selection differs from preparation")
        strict_trials(entry["trials"], q, config)
        for trial in entry["trials"]:
            resolution_dev.validate_processor(trial, 320, 16)
            if (
                not np.isfinite(trial["qa_wall_seconds"])
                or trial["qa_wall_seconds"] < 0
            ):
                raise ValueError("Invalid QA time")
        if entry["timing_origin"] != "this_holdout_experiment":
            raise ValueError(
                "Holdout must not reuse historical development QA"
            )
        ids = tuple(entry["selected_indices"])
        if ids in shared and any(
            entry[k] != shared[ids][k] for k in ("qa_pixels_sha256", "trials")
        ):
            raise ValueError("Identical inputs must share the same QA")
        shared[ids] = entry
        keys = [
            resolution_dev.trial_key(
                digest,
                item["video_id"],
                q.question_id,
                list(ids),
                t["order"],
                320,
                entry["qa_pixels_sha256"],
            )
            for t in entry["trials"]
        ]
        if entry["trial_keys"] != [report_digest(k) for k in keys]:
            raise ValueError("QA trial identities changed")


def evaluate(args, inputs):
    output, digest = args.output_dir, report_digest(inputs.protocol)
    prepared = load_prepared(args, inputs)  # All plans checked before any QA.
    model, calls, rows = None, 0, []

    def backend():
        nonlocal model
        if model is None:
            seed_call(inputs.config)
            model = build_backend(inputs.config)
            validate_metadata(model.metadata(), inputs.metadata)
            write_json(output / "backend.json", model.metadata())
        return model

    for item in inputs.videos:
        vid, pending = item["video_id"], []
        p = prepared[vid]
        for annotation in item["questions"]:
            q = question_from(annotation)
            qi = {
                "protocol_sha256": digest,
                "video_id": vid,
                "question_id": q.question_id,
            }
            path = output / "questions" / (report_digest(qi) + ".json")
            if path.exists():
                row = checked_payload(path, qi)
                validate_result(row, p, item, q, inputs.config, digest)
                # Also verify the per-order records referenced by this row.
                for entry in row["conditions"].values():
                    for key, trial in zip(
                        entry["trial_keys"], entry["trials"]
                    ):
                        ti = resolution_dev.trial_key(
                            digest,
                            vid,
                            q.question_id,
                            entry["selected_indices"],
                            trial["order"],
                            320,
                            entry["qa_pixels_sha256"],
                        )
                        if (
                            checked_payload(
                                output / "trials" / (key + ".json"), ti
                            )
                            != trial
                        ):
                            raise ValueError(
                                "Question and trial checkpoints disagree"
                            )
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
                for i in p["questions"][q.question_id]["conditions"][a][
                    "selected_indices"
                ]
            }
        )
        print(
            f"Decoding {vid}: {len(selected)} original frames at 320",
            flush=True,
        )
        low, pixels, times, duration, audit = resolution_dev.decode_selected(
            (inputs.manifest.parent / item["path"]).resolve(),
            inputs.original,
            p["candidate_count"],
            selected,
            320,
            p["candidate_sha256"],
        )
        del low
        if (
            not np.array_equal(times, p["timestamps"])
            or duration != p["duration_seconds"]
        ):
            raise ValueError("Prepared candidate timestamps/duration changed")
        write_json(output / "decode" / (report_digest(vid) + ".json"), audit)
        for q, qi, path in pending:
            conditions, reusable = {}, {}
            for arm in ARMS:
                plan = p["questions"][q.question_id]["conditions"][arm]
                ids = plan["selected_indices"]
                frames = np.stack([pixels[i] for i in ids])
                pixel_hash = array_digest(frames, times[ids])
                if tuple(ids) in reusable:
                    entry = deepcopy(reusable[tuple(ids)])
                    if entry["qa_pixels_sha256"] != pixel_hash:
                        raise ValueError(
                            "Identical indices have different pixels"
                        )
                    entry["qa_reused_identical_input"] = True
                    conditions[arm] = entry
                    continue
                video = Video(
                    vid,
                    args.split,
                    frames,
                    times[ids],
                    duration,
                    (q,),
                    item["source_id"],
                )
                trials, keys = [], []
                for number, order in enumerate(
                    option_orders(4, inputs.config.seed)
                ):
                    order = list(order)
                    ti = resolution_dev.trial_key(
                        digest, vid, q.question_id, ids, order, 320, pixel_hash
                    )
                    key = report_digest(ti)
                    tp = output / "trials" / (key + ".json")
                    if tp.exists():
                        trial = checked_payload(tp, ti)
                    else:
                        trial = resolution_dev.score_with_memory(
                            backend(), video, q, order, inputs.config, 320
                        )
                        resolution_dev.validate_processor(trial, 320, 16)
                        save_payload(tp, ti, trial)
                        calls += 1
                    resolution_dev.validate_processor(trial, 320, 16)
                    trials.append(trial)
                    keys.append(key)
                    # No interim correctness printing on holdout.
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
                    "timing_origin": "this_holdout_experiment",
                    "qa_reused_identical_input": False,
                }
                reusable[tuple(ids)] = entry
                conditions[arm] = entry
            row = {
                "video_id": vid,
                "question_id": q.question_id,
                "duration_group": item["duration"],
                "candidate_sha256": p["candidate_sha256"],
                "prepared_sha256": report_digest(p),
                "conditions": conditions,
            }
            validate_result(row, p, item, q, inputs.config, digest)
            save_payload(path, qi, row)
            rows.append(row)
            write_json(
                output / "progress.json",
                {
                    "status": "running",
                    "completed_questions": len(rows),
                    "total_questions": sum(
                        len(v["questions"]) for v in inputs.videos
                    ),
                    "new_QA_calls_this_invocation": calls,
                },
            )
    rows.sort(key=lambda r: (r["video_id"], r["question_id"]))
    summary, metrics = selector_comparison_dev.summarize(rows, inputs.protocol)
    unique = {
        k: t
        for r in rows
        for e in r["conditions"].values()
        for k, t in zip(e["trial_keys"], e["trials"])
    }
    summary.update(
        {
            "evaluation_role": inputs.protocol["evaluation_role"],
            "split": args.split,
            "D_QA": "new_on_this_cohort",
            "unique_QA_calls": len(unique),
            "max_new_QA_calls": len(rows) * 24,
            "total_unique_QA_seconds": sum(
                t["qa_wall_seconds"] for t in unique.values()
            ),
            "dev_source_protocol_sha256": inputs.protocol[
                "source_protocol_sha256"
            ],
        }
    )
    if args.split == "dev":
        summary["interpretation"] = (
            "Development smoke only; not holdout evidence. "
            + summary["interpretation"]
        )
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
                "selected_indices": e["selected_indices"],
                "selected_timestamps": e["selected_timestamps"],
            }
            for r in rows
            for a, e in r["conditions"].items()
        ],
    )
    write_json(
        output / "progress.json",
        {
            "status": "complete",
            "completed_questions": len(rows),
            "total_questions": len(rows),
            "new_QA_calls_this_invocation": calls,
            "unique_QA_calls": len(unique),
        },
    )
    return summary


@contextmanager
def output_lock(output):
    """OS-released lock: interrupted processes cannot leave a stale lock."""
    with (output / ".run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError(
                    "Another process is using this output directory"
                ) from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError(
                    "Another process is using this output directory"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(args):
    inputs = load_inputs(args)
    if args.check_inputs:
        return {
            "stage": "validated_no_decode_no_QA",
            "ready": not inputs.missing_videos,
            "evaluation_role": inputs.protocol["evaluation_role"],
            "split": args.split,
            "n_videos": len(inputs.videos),
            "n_questions": sum(len(v["questions"]) for v in inputs.videos),
            "max_new_QA_calls": sum(len(v["questions"]) for v in inputs.videos)
            * 24,
            "qa_side": 320,
            "frame_budget": 16,
            "arms": list(ARMS),
            "missing_videos": inputs.missing_videos,
            "prior_holdout_use": "Not inferable from metadata; "
            "audit separately.",
        }
    if inputs.missing_videos and args.phase != "summarize":
        raise ValueError(
            f"Missing {len(inputs.missing_videos)} cohort videos; "
            "run --check-inputs for paths"
        )
    output, protocol = args.output_dir, canonical(inputs.protocol)
    if args.phase != "prepare" and not (output / "protocol.json").is_file():
        raise ValueError("Complete preparation required before QA or summary")
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        path = output / "protocol.json"
        if path.exists():
            if read(path) != protocol:
                raise ValueError(
                    "Resume protocol changed; use a new output directory"
                )
        else:
            if any(p.name != ".run.lock" for p in output.iterdir()):
                raise ValueError("Output without protocol must be empty")
            write_json(path, protocol)
        return (
            prepare(args, inputs)
            if args.phase == "prepare"
            else evaluate(args, inputs)
        )


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
        default=Path("outputs/selector_compare320_holdout_long"),
    )
    p.add_argument(
        "--phase",
        choices=("prepare", "evaluate", "summarize"),
        default="prepare",
    )
    p.add_argument("--split", choices=("eval", "dev"), default="eval")
    p.add_argument("--max-videos", type=int, help="Development smoke only")
    p.add_argument("--max-questions", type=int, help="Development smoke only")
    p.add_argument("--check-inputs", action="store_true")
    return p


def main():
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
