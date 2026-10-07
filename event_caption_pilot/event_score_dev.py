"""Frozen-boundary event-score study: prepare/review, evidence audit, QA."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .backends import build_backend
from .bottleneck_diagnostics import read, validate_plan, write_csv
from .cache import write_json
from .config import Config
from .event_factorial_dev import cached_trial, measured_score, validate_indices
from .event_refinement_dev import checked_payload, question_from, save_payload
from .event_score_probe import (
    ARMS,
    longest_intervals,
    make_plans,
    review_html,
    write_review_index,
)
from .frame_replacement_probe import FROZEN_CODE, strict_trials
from .order_experiments import seed_call
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .segment_cap_dev import decode_item, validate_metadata

COMPARISONS = {
    "topn_minus_pooled": ("frame_topn_mean", "pooled_mean"),
    "max_minus_pooled": ("frame_max", "pooled_mean"),
    "topn_minus_max": ("frame_topn_mean", "frame_max"),
}


def load_inputs(args):
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if (
        source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Output must be separate from source")
    if (
        args.top_n < 1
        or args.review_limit < 0
        or args.preview_count < 2
        or not np.isfinite(args.long_seconds)
        or args.long_seconds <= 0
    ):
        raise ValueError("Invalid score/review settings")
    prior, results = (
        read(source / "protocol.json"),
        read(source / "results.json"),
    )
    source_digest = report_digest(prior)
    if results["protocol_sha256"] != source_digest:
        raise ValueError("Source protocol/results mismatch")
    root = Path(__file__).parent
    for name in (
        *FROZEN_CODE,
        "event_factorial.py",
        "segment_cap_dev.py",
        "event_refinement_dev.py",
        "cache.py",
        "types.py",
        "qa_diagnostics.py",
        "midpoint_review.py",
    ):
        if (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            != prior["code_sha256"][name]
        ):
            raise ValueError(f"Source implementation changed: {name}")
    settings = dict(prior["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    if (
        config.mode != "real"
        or config.backend != "transformers"
        or config.frame_budget != 16
        or config.pilot != "both"
        or config.encoder_revision == "main"
        or config.vlm_revision == "main"
    ):
        raise ValueError("Require pinned real configuration and 16 frames")
    manifest_path = Path(config.manifest_path).resolve()
    manifest, metadata = read(manifest_path), read(source / "backend.json")
    if (
        report_digest(manifest) != prior["manifest_sha256"]
        or report_digest(metadata) != prior["backend_metadata_sha256"]
    ):
        raise ValueError("Source manifest/backend changed")
    rows = {(r["video_id"], r["question_id"]): r for r in results["questions"]}
    if len(rows) != len(results["questions"]):
        raise ValueError("Duplicate source questions")
    dev = [
        v for v in manifest["videos"] if v["video_id"] in prior["video_ids"]
    ]
    if len(dev) != len(prior["video_ids"]) or set(rows) != {
        (v["video_id"], str(q["question_id"]))
        for v in dev
        for q in v["questions"]
    }:
        raise ValueError("Incomplete source cohort")
    if args.video_id:
        dev = [v for v in dev if v["video_id"] == args.video_id]
    if not dev:
        raise ValueError("No matching source video")
    complete = read(source / "prepare_complete.json")
    if complete["protocol_sha256"] != source_digest:
        raise ValueError("Source preparation identity mismatch")
    source_plans = {}
    for item in dev:
        if (
            item["split"] != "dev"
            or not (manifest_path.parent / item["path"]).is_file()
        ):
            raise ValueError("Require available source development videos")
        first = rows[
            (item["video_id"], str(item["questions"][0]["question_id"]))
        ]
        identity = {
            "protocol_sha256": source_digest,
            "video_id": item["video_id"],
            "candidate_sha256": first["candidate_sha256"],
        }
        payload = checked_payload(
            source / "plans" / (report_digest(item["video_id"]) + ".json"),
            identity,
        )
        if report_digest(payload) != complete["plan_sha256"][item["video_id"]]:
            raise ValueError("Source preparation changed")
        source_plans[item["video_id"]] = payload
        for annotation in item["questions"]:
            q = question_from(annotation)
            row = rows[(item["video_id"], q.question_id)]
            old = row["conditions"]["D"]
            if row["candidate_sha256"] != identity["candidate_sha256"] or any(
                old[k] != v
                for k, v in payload["questions"][q.question_id]["D"].items()
            ):
                raise ValueError("Source question/plan mismatch")
            validate_plan(old, config.frame_budget, "question_relevance")
            strict_trials(old["trials"], q, config)
    protocol = {
        "version": 1,
        "experiment": "event_score_only_fixed_D_boundaries",
        "source_dir": str(source),
        "source_protocol_sha256": source_digest,
        "source_results_sha256": report_digest(results),
        "manifest_sha256": report_digest(manifest),
        "backend_metadata_sha256": report_digest(metadata),
        "config": config.to_dict(),
        "top_n": args.top_n,
        "video_ids": [v["video_id"] for v in dev],
        "arms": list(ARMS),
        "primary_comparison": "topn_minus_pooled",
        "review": {
            "long_seconds": args.long_seconds,
            "limit": args.review_limit,
            "preview_count": args.preview_count,
        },
        "selection": "Fixed source D2 events; event score changes only. "
        "Original softmax temperature, allocator and uniform sampler; "
        "16 frames. "
        "Question text only; no choices, gold, captions or evidence labels.",
        "interpretation": "Exploratory dev. "
        "Max/top-n may favor longer events. "
        "Raw-score dispersion changes allocation; no per-arm calibration. "
        "Evidence annotations are used only by a separate post-hoc audit.",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))
        },
    }
    return config, manifest_path, dev, rows, source_plans, metadata, protocol


def identity_for(item, sources, digest):
    first = str(item["questions"][0]["question_id"])
    return {
        "protocol_sha256": digest,
        "video_id": item["video_id"],
        "candidate_sha256": sources[(item["video_id"], first)][
            "candidate_sha256"
        ],
    }


def path_for(output, identity):
    return output / "plans" / (report_digest(identity["video_id"]) + ".json")


def validate_plans(prepared, item, sources, config):
    if set(prepared["questions"]) != {
        str(q["question_id"]) for q in item["questions"]
    }:
        raise ValueError("Incomplete prepared questions")
    for qid, entry in prepared["questions"].items():
        old = sources[(item["video_id"], qid)]["conditions"]["D"]
        if set(entry["conditions"]) != set(ARMS):
            raise ValueError("Incomplete score conditions")
        for arm, plan in entry["conditions"].items():
            if (
                plan["events"] != old["events"]
                or plan["capacities"] != old["capacities"]
            ):
                raise ValueError("Frozen event partition changed")
            validate_indices(
                plan["selected_indices"], old["events"][-1][1], 16
            )
            if plan["temperature"] != config.allocation_temperature:
                raise ValueError("Allocation temperature changed")
            counts = [
                sum(a <= i < b for i in plan["selected_indices"])
                for a, b in old["events"]
            ]
            if counts != plan["allocation"]:
                raise ValueError("Allocation/selection mismatch")
            if arm == "pooled_mean" and (
                counts != old["allocation"]
                or plan["selected_indices"] != old["selected_indices"]
            ):
                raise ValueError("Baseline does not reproduce")


def prepare(args, inputs):
    config, manifest, dev, sources, source_plans, metadata, protocol = inputs
    digest = report_digest(protocol)
    reviews = longest_intervals(
        source_plans, args.long_seconds, args.review_limit
    )
    model = None
    hashes, files, rank_rows = {}, [], []
    for item in dev:
        identity = identity_for(item, sources, digest)
        path = path_for(args.output_dir, identity)
        if path.exists():
            prepared = checked_payload(path, identity)
        else:
            print(
                f"Preparing {item['video_id']}: decode + CLIP, no QA",
                flush=True,
            )
            video = decode_item(item, manifest, config, identity)
            if model is None:
                cc = replace(config, pilot="a")
                seed_call(cc)
                model = build_backend(cc)
                validate_metadata(model.metadata(), metadata, True)
                write_json(
                    args.output_dir / "encoder_backend.json", model.metadata()
                )
            seed_call(config)
            features = model.encode_frames(video.frames)
            questions = {}
            for q in video.questions:
                questions[q.question_id] = make_plans(
                    features,
                    model.encode_visual_question(q.text),
                    video.timestamps,
                    sources[(video.video_id, q.question_id)]["conditions"][
                        "D"
                    ],
                    config,
                    args.top_n,
                )
            local_reviews = [
                r for r in reviews if r["video_id"] == video.video_id
            ]
            review_files = {}
            if local_reviews:
                name = "review_" + report_digest(video.video_id)[:16] + ".html"
                page = review_html(
                    video, local_reviews, features, config, args.preview_count
                )
                (args.output_dir / name).write_text(page, encoding="utf-8")
                review_files[name] = hashlib.sha256(
                    (args.output_dir / name).read_bytes()
                ).hexdigest()
            prepared = {"questions": questions, "review_files": review_files}
            validate_plans(prepared, item, sources, config)
            save_payload(path, identity, prepared)
            del video, features
        validate_plans(prepared, item, sources, config)
        for name, expected in prepared["review_files"].items():
            # Names are generated locally, but verify cached paths as well.
            file = (args.output_dir / name).resolve()
            if (
                file.parent != args.output_dir.resolve()
                or hashlib.sha256(file.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Review file changed/missing")
            files.append(name)
        hashes[item["video_id"]] = report_digest(prepared)
        for qid, entry in prepared["questions"].items():
            for arm, p in entry["conditions"].items():
                for i, (a, b) in enumerate(p["events"]):
                    rank_rows.append(
                        {
                            "video_id": item["video_id"],
                            "question_id": qid,
                            "condition": arm,
                            "event_index_zero_based": i,
                            "candidate_start": a,
                            "candidate_stop_exclusive": b,
                            "candidate_count": b - a,
                            "event_score": p["event_scores"][i],
                            "rank_1based": p["event_ranks_1based"][i],
                            "allocated_frames": p["allocation"][i],
                            "selected_indices": [
                                j for j in p["selected_indices"] if a <= j < b
                            ],
                            "highest_score_candidate": entry[
                                "highest_score_candidate_per_event"
                            ][i],
                        }
                    )
        write_json(
            args.output_dir / "prepare_progress.json",
            {
                "completed_videos": len(hashes),
                "total_videos": len(dev),
                "status": "running",
            },
        )
        print(f"Prepared {len(hashes)}/{len(dev)}", flush=True)
    write_csv(args.output_dir / "event_ranks.csv", rank_rows)
    write_csv(args.output_dir / "long_intervals.csv", reviews)
    write_review_index(args.output_dir, files)
    summary = {
        "stage": "prepared_no_QA",
        "n_videos": len(dev),
        "n_questions": sum(len(v["questions"]) for v in dev),
        "top_n": args.top_n,
        "reviewed_intervals": len(reviews),
        "conditions": {
            arm: {
                "mean_zero_allocation_events_per_question": sum(
                    r["allocated_frames"] == 0
                    for r in rank_rows
                    if r["condition"] == arm
                )
                / sum(len(v["questions"]) for v in dev),
            }
            for arm in ARMS
        },
        "note": "Ranking/selection only; no QA or evidence recall measured.",
    }
    write_json(args.output_dir / "prepare_summary.json", summary)
    write_json(
        args.output_dir / "prepare_complete.json",
        {"protocol_sha256": digest, "plan_sha256": hashes},
    )
    write_json(
        args.output_dir / "prepare_progress.json",
        {
            "completed_videos": len(dev),
            "total_videos": len(dev),
            "status": "complete_no_QA",
        },
    )
    print(json.dumps(summary, indent=2))


def read_prepared(args, inputs):
    config, _, dev, sources, _, _, protocol = inputs
    digest = report_digest(protocol)
    complete = read(args.output_dir / "prepare_complete.json")
    if complete["protocol_sha256"] != digest or set(
        complete["plan_sha256"]
    ) != {v["video_id"] for v in dev}:
        raise ValueError("Complete preparation required")
    prepared = {}
    for item in dev:
        identity = identity_for(item, sources, digest)
        p = checked_payload(path_for(args.output_dir, identity), identity)
        if report_digest(p) != complete["plan_sha256"][item["video_id"]]:
            raise ValueError("Preparation changed")
        validate_plans(p, item, sources, config)
        prepared[item["video_id"]] = p
    return prepared


def summarize(rows, seed):
    means = []
    for row in rows:
        values = {
            arm: float(
                np.mean(
                    [
                        t["scoring_correct"]
                        for t in row["conditions"][arm]["trials"]
                    ]
                )
            )
            for arm in ARMS
        }
        values.update(
            {k: values[a] - values[b] for k, (a, b) in COMPARISONS.items()}
        )
        means.append(
            {
                **{
                    k: row[k]
                    for k in ("video_id", "question_id", "duration_group")
                },
                **values,
            }
        )
    keys = [*ARMS, *COMPARISONS]
    return {
        "primary_comparison": "topn_minus_pooled",
        "overall": clustered_means(means, keys, seed=seed),
        "by_duration": {
            g: clustered_means(
                [r for r in means if r["duration_group"] == g], keys, seed=seed
            )
            for g in sorted({r["duration_group"] for r in means})
        },
        "paired_question_changes": {
            k: {
                "improved": sum(r[k] > 1e-12 for r in means),
                "degraded": sum(r[k] < -1e-12 for r in means),
                "unchanged": sum(abs(r[k]) <= 1e-12 for r in means),
            }
            for k in COMPARISONS
        },
    }, means


def validate_result(row, entry, source, question, config):
    if (
        row["video_id"] != source["video_id"]
        or row["question_id"] != question.question_id
        or row["candidate_sha256"] != source["candidate_sha256"]
        or set(row["conditions"]) != set(ARMS)
    ):
        raise ValueError("Result identity mismatch")
    for arm, c in row["conditions"].items():
        if any(c[k] != v for k, v in entry["conditions"][arm].items()):
            raise ValueError("QA selection differs from preparation")
        strict_trials(c["trials"], question, config)
    if (
        row["conditions"]["pooled_mean"]["trials"]
        != source["conditions"]["D"]["trials"]
    ):
        raise ValueError("Original control QA changed")


def evaluate(args, inputs):
    config, manifest, dev, sources, _, metadata, protocol = inputs
    prepared = read_prepared(args, inputs)
    digest, model = report_digest(protocol), None

    def backend():
        nonlocal model
        if model is None:
            seed_call(config)
            model = build_backend(config)
            validate_metadata(model.metadata(), metadata)
            write_json(args.output_dir / "backend.json", model.metadata())
        return model

    rows = []
    total = sum(len(v["questions"]) for v in dev)
    for item in dev:
        identity = identity_for(item, sources, digest)
        pending = []
        for annotation in item["questions"]:
            q = question_from(annotation)
            source = sources[(item["video_id"], q.question_id)]
            entry = prepared[item["video_id"]]["questions"][q.question_id]
            qi = {**identity, "question_id": q.question_id}
            path = (
                args.output_dir / "questions" / (report_digest(qi) + ".json")
            )
            if path.exists():
                row = checked_payload(path, qi)
                validate_result(row, entry, source, q, config)
                rows.append(row)
            else:
                pending.append((q, source, entry, qi, path))
        if not pending:
            print(f"Reused completed video {item['video_id']}", flush=True)
            continue
        video = decode_item(item, manifest, config, identity)
        for q, source, entry, qi, path in pending:
            old = source["conditions"]["D"]
            reusable = {tuple(old["selected_indices"]): old["trials"]}
            conditions = {}
            for arm, plan in entry["conditions"].items():
                ids = plan["selected_indices"]
                trials = reusable.get(tuple(ids))
                if trials is None:
                    trials = []
                    for i, old_trial in enumerate(old["trials"]):
                        order = old_trial["order"]
                        ti = {**qi, "indices": ids, "order": order}
                        trial, reused = cached_trial(
                            args.output_dir
                            / "trials"
                            / (report_digest(ti) + ".json"),
                            ti,
                            lambda video=video: measured_score(
                                backend(), video, q, ids, order, config
                            ),
                        )
                        trials.append(trial)
                        print(
                            f"{video.video_id}/{q.question_id}/{arm} "
                            f"order {i + 1}/6 cached={reused}",
                            flush=True,
                        )
                strict_trials(trials, q, config)
                reusable[tuple(ids)] = trials
                conditions[arm] = {**plan, "trials": trials}
            row = {
                "video_id": item["video_id"],
                "question_id": q.question_id,
                "duration_group": item["duration"],
                "candidate_sha256": identity["candidate_sha256"],
                "conditions": conditions,
            }
            validate_result(row, entry, source, q, config)
            save_payload(path, qi, row)
            rows.append(row)
            write_json(
                args.output_dir / "qa_progress.json",
                {
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "status": "running",
                },
            )
        del video
    rows.sort(key=lambda r: (r["video_id"], r["question_id"]))
    summary, means = summarize(rows, config.seed)
    write_json(
        args.output_dir / "results.json",
        {"protocol_sha256": digest, "questions": rows, "summary": summary},
    )
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "question_metrics.csv", means)
    write_json(
        args.output_dir / "handoff.json",
        {
            "protocol": protocol,
            "summary": summary,
            "question_metrics": means,
            "preparation": read(args.output_dir / "prepare_summary.json"),
        },
    )
    write_json(
        args.output_dir / "qa_progress.json",
        {
            "completed_questions": total,
            "total_questions": total,
            "status": "complete",
        },
    )
    print(json.dumps(summary, indent=2))


def evidence_rows(annotations, prepared):
    rows, seen = [], set()
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Provide a nonempty annotations list")
    for note in annotations:
        vid, qid = note["video_id"], note["question_id"]
        if (vid, qid) in seen:
            raise ValueError("Duplicate annotated question")
        seen.add((vid, qid))
        entry = prepared[vid]["questions"][qid]
        indices = note["candidate_indices"]
        end = entry["conditions"]["pooled_mean"]["events"][-1][1]
        if (
            not indices
            or any(type(i) is not int or not 0 <= i < end for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Evidence indices must be unique valid integers")
        for index in indices:
            for arm, plan in entry["conditions"].items():
                e = next(
                    i
                    for i, (a, b) in enumerate(plan["events"])
                    if a <= index < b
                )
                a, b = plan["events"][e]
                rows.append(
                    {
                        "video_id": vid,
                        "question_id": qid,
                        "condition": arm,
                        "human_candidate_index": index,
                        "event_index_zero_based": e,
                        "event_rank_1based": plan["event_ranks_1based"][e],
                        "event_score": plan["event_scores"][e],
                        "event_allocated_frames": plan["allocation"][e],
                        "exact_candidate_selected": index
                        in plan["selected_indices"],
                        "selected_in_event": [
                            j for j in plan["selected_indices"] if a <= j < b
                        ],
                    }
                )
    return rows


def run(args):
    inputs = load_inputs(args)
    *_, protocol = inputs
    if args.check_inputs:
        count = sum(len(v["questions"]) for v in inputs[2])
        print(
            json.dumps(
                {
                    "status": "validated_no_inference",
                    "n_videos": len(inputs[2]),
                    "n_questions": count,
                    "top_n": args.top_n,
                    "maximum_new_QA_calls": count * 12,
                }
            )
        )
        return
    if args.phase != "prepare" or args.resume:
        if report_digest(
            read(args.output_dir / "protocol.json")
        ) != report_digest(protocol):
            raise ValueError(
                "Protocol changed; use a separate output directory"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "protocol.json", protocol)
    if args.phase == "prepare":
        prepare(args, inputs)
    elif args.phase == "evaluate":
        evaluate(args, inputs)
    else:
        if args.evidence_file is None:
            raise ValueError("evidence phase requires --evidence-file")
        prepared = read_prepared(args, inputs)
        labels = read(args.evidence_file)
        rows = evidence_rows(labels["annotations"], prepared)
        audit = {
            "protocol_sha256": report_digest(protocol),
            "annotations_sha256": report_digest(labels),
            "rows": rows,
            "note": "Post-hoc supplied candidate references only, "
            "not exhaustive evidence recall or proof of sufficiency. "
            "Never used for selection/QA.",
        }
        name = "evidence_audit_" + report_digest(labels)[:12]
        write_json(args.output_dir / (name + ".json"), audit)
        write_csv(args.output_dir / (name + ".csv"), rows)
        print(json.dumps(audit, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", choices=("prepare", "evaluate", "evidence"), required=True
    )
    p.add_argument(
        "--source-dir", type=Path, default=Path("outputs/segment_cap128_long")
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--long-seconds", type=float, default=60)
    p.add_argument("--review-limit", type=int, default=10)
    p.add_argument("--preview-count", type=int, default=12)
    p.add_argument("--video-id")
    p.add_argument("--evidence-file", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-inputs", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
