"""Change QA resolution while replaying frozen frame selections and orders."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .backends import build_backend
from .bottleneck_diagnostics import read, write_csv
from .cache import write_json
from .config import Config
from .data import array_digest
from .event_factorial_dev import measured_score, validate_indices
from .event_refinement_dev import checked_payload, question_from, save_payload
from .frame_replacement_probe import FROZEN_CODE, strict_trials
from .order_experiments import seed_call
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .types import Video


def load_inputs(args):
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if (
        source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("Use an output separate from the source experiment")
    if args.side <= 224 or args.side % 32:
        raise ValueError("side must exceed 224 and be a multiple of 32")
    for limit in (args.max_videos, args.max_questions):
        if limit is not None and limit < 1:
            raise ValueError("Subset limits must be positive")
    if not args.conditions or len(set(args.conditions)) != len(
        args.conditions
    ):
        raise ValueError("Require distinct source conditions")
    prior, results = (
        read(source / "protocol.json"),
        read(source / "results.json"),
    )
    kind = prior.get("experiment")
    if kind not in {
        "relaxed_segment_cap_C_D",
        "event_score_only_fixed_D_boundaries",
    }:
        raise ValueError(
            "Use a completed segment-cap or event-score experiment"
        )
    digest = report_digest(prior)
    if results["protocol_sha256"] != digest:
        raise ValueError("Source results/protocol mismatch")
    root = Path(__file__).parent
    for name in FROZEN_CODE:
        if (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            != prior["code_sha256"][name]
        ):
            raise ValueError(
                f"Frozen QA/decode implementation changed: {name}"
            )
    settings = dict(prior["config"])
    settings["observation_strides"] = tuple(settings["observation_strides"])
    config = Config(**settings)
    config.validate()
    if (
        config.mode != "real"
        or config.backend != "transformers"
        or config.frame_budget != 16
        or config.pilot != "both"
        or config.frame_height != 224
        or config.frame_width != 224
        or config.vlm_max_pixels != 224**2
        or config.vlm_model != "Qwen/Qwen3.5-4B"
        or config.encoder_revision == "main"
        or config.vlm_revision == "main"
    ):
        raise ValueError("Require pinned Qwen3.5-4B, 224 pixels, 16 frames")
    target = replace(config, vlm_max_pixels=args.side**2)
    target.validate()
    manifest_path = Path(config.manifest_path).resolve()
    manifest, metadata = read(manifest_path), read(source / "backend.json")
    if (
        report_digest(manifest) != prior["manifest_sha256"]
        or report_digest(metadata) != prior["backend_metadata_sha256"]
    ):
        raise ValueError("Source manifest/backend identity changed")
    rows = {(r["video_id"], r["question_id"]): r for r in results["questions"]}
    videos = [
        v for v in manifest["videos"] if v["video_id"] in prior["video_ids"]
    ]
    if (
        len(rows) != len(results["questions"])
        or len(videos) != len(set(prior["video_ids"]))
        or set(rows)
        != {
            (v["video_id"], str(q["question_id"]))
            for v in videos
            for q in v["questions"]
        }
    ):
        raise ValueError("Incomplete or duplicated source cohort")
    if args.video_id:
        videos = [v for v in videos if v["video_id"] == args.video_id]
    videos = videos[: args.max_videos]
    if not videos:
        raise ValueError("No matching source videos")
    complete = read(source / "prepare_complete.json")
    if complete["protocol_sha256"] != digest:
        raise ValueError("Source preparation/protocol mismatch")
    selected = []
    for item in videos:
        path = (manifest_path.parent / item["path"]).resolve()
        if (
            item["split"] != "dev"
            or not path.is_file()
            or path.suffix.lower() == ".npz"
            or item.get("timestamp_mode") != "constant_fps"
        ):
            raise ValueError("Require original constant-FPS dev video files")
        first = rows[
            (item["video_id"], str(item["questions"][0]["question_id"]))
        ]
        identity = {
            "protocol_sha256": digest,
            "video_id": item["video_id"],
            "candidate_sha256": first["candidate_sha256"],
        }
        plan = checked_payload(
            source / "plans" / (report_digest(item["video_id"]) + ".json"),
            identity,
        )
        if report_digest(plan) != complete["plan_sha256"][item["video_id"]]:
            raise ValueError("Source preparation payload changed")
        questions = item["questions"][: args.max_questions]
        for annotation in questions:
            q = question_from(annotation)
            row = rows[(item["video_id"], q.question_id)]
            if row["candidate_sha256"] != identity["candidate_sha256"]:
                raise ValueError("Source candidate identity mismatch")
            prepared = plan["questions"][q.question_id]
            if kind == "event_score_only_fixed_D_boundaries":
                prepared = prepared["conditions"]
            orders = None
            for arm in args.conditions:
                if arm not in prepared or arm not in row["conditions"]:
                    raise ValueError(f"Unavailable prepared condition: {arm}")
                old = row["conditions"][arm]
                if any(old[k] != v for k, v in prepared[arm].items()):
                    raise ValueError("Source selections/results mismatch")
                validate_indices(
                    old["selected_indices"], old["events"][-1][1], 16
                )
                strict_trials(old["trials"], q, config)
                current = [t["order"] for t in old["trials"]]
                if orders is not None and current != orders:
                    raise ValueError("Source conditions have different orders")
                orders = current
        selected.append({**item, "questions": questions})
    protocol = {
        "version": 1,
        "experiment": "frozen_selection_QA_resolution",
        "source_dir": str(source),
        "source_protocol_sha256": digest,
        "source_results_sha256": report_digest(results),
        "manifest_sha256": report_digest(manifest),
        "source_backend_sha256": report_digest(metadata),
        "candidate_config": config.to_dict(),
        "qa_config": target.to_dict(),
        "qa_side": args.side,
        "baseline_side": 224,
        "conditions": args.conditions,
        "rerun_control": args.rerun_control,
        "cohort": {
            v["video_id"]: [str(q["question_id"]) for q in v["questions"]]
            for v in selected
        },
        "selection": "Replay saved indices/timestamps/orders; no selection, "
        "CLIP scoring, event changes, captions or human evidence injection.",
        "pixels": "Resize original decoded RGB directly to each square size, "
        "using the baseline interpolation and aspect-ratio policy.",
        "primary_comparison": f"{args.conditions[-1]}_{args.side}_minus_224",
        "interpretation": "Exploratory dev, fixed algorithm selections. "
        "No evidence recall measured. Bootstrap video clusters after "
        "within-question option-order means. Saved 224 QA is the baseline "
        "unless rerun_control is true. Historical timing is not paired.",
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.py"))
        },
    }
    return config, target, manifest_path, selected, rows, metadata, protocol


def decode_selected(path, config, count, indices, side, expected_hash):
    """Hash all original candidates; retain only selected RGB frames."""
    import cv2

    validate_indices(indices, count, len(indices))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open source video: {path}")
    digest = hashlib.sha256()
    shape = (count, config.frame_height, config.frame_width, 3)
    digest.update(str((shape, np.dtype(np.uint8).str)).encode())
    times, pixels, low, raw_sizes = [], {}, {}, {}
    wanted = set(indices)
    frame_index, next_observation = 0, 0.0
    started = perf_counter()
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid source FPS")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stamp = frame_index / fps
            if stamp >= next_observation:
                index = len(times)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small = cv2.resize(
                    rgb, (config.frame_width, config.frame_height)
                )
                digest.update(memoryview(small).cast("B"))
                if index in wanted:
                    low[index] = small
                    pixels[index] = cv2.resize(rgb, (side, side))
                    raw_sizes[index] = [int(rgb.shape[1]), int(rgb.shape[0])]
                times.append(stamp)
                next_observation += 1.0 / min(config.candidate_fps, fps)
                if (
                    len(times) > count
                    or len(times) > config.max_candidate_frames
                ):
                    raise ValueError("Original candidate count changed")
            frame_index += 1
    finally:
        cap.release()
    timestamps = np.asarray(times, dtype=float)
    digest.update(str((timestamps.shape, timestamps.dtype.str)).encode())
    digest.update(memoryview(timestamps).cast("B"))
    if len(times) != count or digest.hexdigest() != expected_hash:
        raise ValueError(
            "Original candidate pixels/timestamps do not reproduce"
        )
    if set(pixels) != wanted:
        raise ValueError("Some selected frames were not decoded")
    return (
        low,
        pixels,
        timestamps,
        frame_index / fps,
        {
            "candidate_sha256": digest.hexdigest(),
            "candidate_count": count,
            "source_fps": fps,
            "decode_seconds": perf_counter() - started,
            "original_selected_sizes": raw_sizes,
            "selected_native_smaller_than_target": sum(
                min(size) < side for size in raw_sizes.values()
            ),
        },
    )


def validate_processor(trial, side, count):
    info = trial["processor_info"]
    expected = [[1, side // 16, side // 16]] * count
    if (
        info["n_frames"] != count
        or info["image_sizes_width_height"] != [[side, side]] * count
        or info["image_grid_thw"] != expected
    ):
        raise ValueError(
            "Actual QA image size/grid differs from requested size"
        )
    trial["visual_tokens"] = count * (side // 32) ** 2


def score_with_memory(model, video, q, order, config, side):
    cuda = model.torch.cuda if model.device.type == "cuda" else None
    if cuda:
        cuda.synchronize(model.device)
        cuda.reset_peak_memory_stats(model.device)
    trial = measured_score(
        model, video, q, list(range(len(video.frames))), order, config
    )
    if cuda:
        trial["cuda_memory"] = {
            "peak_allocated_bytes": int(
                cuda.max_memory_allocated(model.device)
            ),
            "peak_reserved_bytes": int(cuda.max_memory_reserved(model.device)),
            "scope": "PyTorch allocator; includes resident model parameters",
        }
    validate_processor(trial, side, len(video.frames))
    return trial


def trial_key(protocol_hash, video_id, qid, indices, order, side, pixels_hash):
    return {
        "protocol_sha256": protocol_hash,
        "video_id": video_id,
        "question_id": qid,
        "indices": indices,
        "order": order,
        "qa_side": side,
        "qa_pixels_sha256": pixels_hash,
    }


def question_metrics(row, q, arms, side):
    metrics = {
        "video_id": row["video_id"],
        "question_id": row["question_id"],
        "duration_group": row["duration_group"],
    }
    for arm in arms:
        for resolution in (224, side):
            name = f"{arm}_{resolution}"
            trials = row["conditions"][name]["trials"]
            metrics[name] = float(
                np.mean([t["scoring_correct"] for t in trials])
            )
            margins = []
            for t in trials:
                gold = t["order"].index(q.answer_index)
                scores = np.asarray(t["scoring"]["option_scores"])
                margins.append(scores[gold] - np.max(np.delete(scores, gold)))
            metrics[name + "_gold_margin"] = float(np.mean(margins))
        metrics[f"{arm}_{side}_minus_224"] = (
            metrics[f"{arm}_{side}"] - metrics[f"{arm}_224"]
        )
    if set(arms) == {"C", "D"}:
        for resolution in (224, side):
            metrics[f"D_minus_C_{resolution}"] = (
                metrics[f"D_{resolution}"] - metrics[f"C_{resolution}"]
            )
        metrics["change_in_D_minus_C"] = (
            metrics[f"D_minus_C_{side}"] - metrics["D_minus_C_224"]
        )
    return metrics


def summarize(rows, questions, protocol):
    arms, side = protocol["conditions"], protocol["qa_side"]
    metrics = [
        question_metrics(
            r, questions[(r["video_id"], r["question_id"])], arms, side
        )
        for r in rows
    ]
    keys = [
        k
        for k in metrics[0]
        if k not in {"video_id", "question_id", "duration_group"}
    ]
    comparisons = [k for k in keys if "minus" in k]
    resources = {}
    for arm in arms:
        for resolution in (224, side):
            name = f"{arm}_{resolution}"
            trials = [t for r in rows for t in r["conditions"][name]["trials"]]
            seconds = [
                t["qa_wall_seconds"] for t in trials if "qa_wall_seconds" in t
            ]
            memory = [
                t["cuda_memory"]["peak_allocated_bytes"]
                for t in trials
                if "cuda_memory" in t
            ]
            tokens = [
                t["processor_info"]["expanded_input_tokens"]
                for t in trials
                if "processor_info" in t
            ]
            resources[name] = {
                "timing_source": "current_run"
                if resolution == side or protocol["rerun_control"]
                else "historical_source_run",
                "qa_seconds_median": float(np.median(seconds))
                if seconds
                else None,
                "max_peak_allocated_GiB": max(memory) / 1024**3
                if memory
                else None,
                "expanded_tokens_median": float(np.median(tokens))
                if tokens
                else None,
            }
    result = {
        "protocol_sha256": report_digest(protocol),
        "primary_comparison": protocol["primary_comparison"],
        "qa_resolution": [224, side],
        "frame_budget": 16,
        "overall": clustered_means(metrics, keys),
        "by_duration": {
            d: clustered_means(
                [m for m in metrics if m["duration_group"] == d], keys
            )
            for d in sorted({m["duration_group"] for m in metrics})
        },
        "paired_question_changes": {
            k: {
                "improved": sum(m[k] > 1e-12 for m in metrics),
                "degraded": sum(m[k] < -1e-12 for m in metrics),
                "unchanged": sum(abs(m[k]) <= 1e-12 for m in metrics),
            }
            for k in comparisons
        },
        "resources": resources,
        "interpretation": protocol["interpretation"],
    }
    return result, metrics


def run(args):
    original, target, manifest, videos, sources, metadata, protocol = (
        load_inputs(args)
    )
    # Config contains tuples, whereas JSON checkpoints contain lists.
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    total = sum(len(v["questions"]) for v in videos)
    if args.check_inputs:
        return {
            "stage": "validated_no_decode_no_QA",
            "n_videos": len(videos),
            "n_questions": total,
            "qa_side": args.side,
            "conditions": args.conditions,
            "max_new_qa_calls": total
            * len(args.conditions)
            * 6
            * (2 if args.rerun_control else 1),
        }
    output, digest = args.output_dir, report_digest(protocol)
    if (output / "protocol.json").exists():
        if read(output / "protocol.json") != protocol:
            raise ValueError(
                "Resume protocol changed; use a new output directory"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", protocol)
    model, rows, questions = None, [], {}

    def backend(side):
        nonlocal model
        config = replace(original, vlm_max_pixels=side**2)
        if model is None:
            print("Loading frozen QA model", flush=True)
            seed_call(config)
            model = build_backend(config)
        model.config = config
        expected = {**metadata, "vlm_max_pixels": side**2}
        if model.metadata() != expected:
            raise ValueError("Runtime/model metadata changed beyond QA pixels")
        write_json(output / f"backend_{side}.json", model.metadata())
        return model, config

    for item in videos:
        vid, pending = item["video_id"], []
        for annotation in item["questions"]:
            q = question_from(annotation)
            questions[(vid, q.question_id)] = q
            identity = {
                "protocol_sha256": digest,
                "video_id": vid,
                "question_id": q.question_id,
            }
            path = output / "questions" / (report_digest(identity) + ".json")
            if path.exists():
                row = checked_payload(path, identity)
                validate_row(
                    row, sources[(vid, q.question_id)], q, args, original
                )
                rows.append(row)
            else:
                pending.append((q, identity, path))
        if not pending:
            continue
        if args.summarize_only:
            raise ValueError("Evaluation is incomplete; cannot summarize only")
        first = sources[(vid, pending[0][0].question_id)]
        count = first["conditions"][args.conditions[0]]["events"][-1][1]
        indices = sorted(
            {
                i
                for q, _, _ in pending
                for a in args.conditions
                for i in sources[(vid, q.question_id)]["conditions"][a][
                    "selected_indices"
                ]
            }
        )
        print(
            f"Decoding {vid}: verify candidates, keep {len(indices)} frames"
            f" at {args.side}",
            flush=True,
        )
        low, high, times, duration, audit = decode_selected(
            (manifest.parent / item["path"]).resolve(),
            original,
            count,
            indices,
            args.side,
            first["candidate_sha256"],
        )
        write_json(output / "decode" / f"{vid}.json", audit)
        for q, identity, path in pending:
            source, conditions = sources[(vid, q.question_id)], {}
            for arm in args.conditions:
                old = source["conditions"][arm]
                ids = old["selected_indices"]
                if not np.array_equal(times[ids], old["selected_timestamps"]):
                    raise ValueError("Selected timestamps changed")
                if not args.rerun_control:
                    conditions[f"{arm}_224"] = {
                        "selected_indices": ids,
                        "selected_timestamps": times[ids].tolist(),
                        "trials": old["trials"],
                        "origin": "saved_source",
                    }
                for side in (
                    [224, args.side] if args.rerun_control else [args.side]
                ):
                    pixels = np.stack(
                        [(low if side == 224 else high)[i] for i in ids]
                    )
                    qa_hash = array_digest(pixels, times[ids])
                    video = Video(
                        vid,
                        "dev",
                        pixels,
                        times[ids],
                        duration,
                        (q,),
                        item["source_id"],
                    )
                    trials = []
                    for n, old_trial in enumerate(old["trials"]):
                        ti = trial_key(
                            digest,
                            vid,
                            q.question_id,
                            ids,
                            old_trial["order"],
                            side,
                            qa_hash,
                        )
                        tp = output / "trials" / (report_digest(ti) + ".json")
                        if tp.exists():
                            trial = checked_payload(tp, ti)
                        else:
                            active, config = backend(side)
                            trial = score_with_memory(
                                active,
                                video,
                                q,
                                old_trial["order"],
                                config,
                                side,
                            )
                            save_payload(tp, ti, trial)
                        validate_processor(trial, side, 16)
                        trials.append(trial)
                        print(
                            f"{vid}/{q.question_id}/{arm}/{side} "
                            f"order {n + 1}/6 "
                            f"correct={trial['scoring_correct']}",
                            flush=True,
                        )
                    strict_trials(trials, q, target)
                    conditions[f"{arm}_{side}"] = {
                        "selected_indices": ids,
                        "selected_timestamps": times[ids].tolist(),
                        "qa_pixels_sha256": qa_hash,
                        "trials": trials,
                        "origin": "resolution_run",
                    }
            row = {
                "video_id": vid,
                "question_id": q.question_id,
                "candidate_sha256": source["candidate_sha256"],
                "duration_group": source["duration_group"],
                "conditions": conditions,
            }
            if args.rerun_control:
                row["control_reproduction"] = {
                    a: {
                        "same_predictions": all(
                            x["scored_original_index"]
                            == y["scored_original_index"]
                            for x, y in zip(
                                conditions[f"{a}_224"]["trials"],
                                source["conditions"][a]["trials"],
                            )
                        ),
                        "max_abs_score_difference": max(
                            float(
                                np.max(
                                    np.abs(
                                        np.asarray(
                                            x["scoring"]["option_scores"]
                                        )
                                        - y["scoring"]["option_scores"]
                                    )
                                )
                            )
                            for x, y in zip(
                                conditions[f"{a}_224"]["trials"],
                                source["conditions"][a]["trials"],
                            )
                        ),
                    }
                    for a in args.conditions
                }
            validate_row(row, source, q, args, original)
            save_payload(path, identity, row)
            rows.append(row)
            write_json(
                output / "progress.json",
                {
                    "completed_questions": len(rows),
                    "total_questions": total,
                    "status": "running",
                },
            )
    rows.sort(key=lambda r: (r["video_id"], r["question_id"]))
    summary, metrics = summarize(rows, questions, protocol)
    write_json(
        output / "results.json",
        {"protocol_sha256": digest, "questions": rows, "summary": summary},
    )
    write_json(output / "handoff.json", summary)
    write_csv(output / "question_metrics.csv", metrics)
    write_json(
        output / "progress.json",
        {
            "status": "complete",
            "completed_questions": total,
            "total_questions": total,
        },
    )
    return summary


def validate_row(row, source, q, args, config):
    if (
        row["video_id"] != source["video_id"]
        or row["question_id"] != q.question_id
        or row["candidate_sha256"] != source["candidate_sha256"]
        or set(row["conditions"])
        != {f"{a}_{s}" for a in args.conditions for s in (224, args.side)}
    ):
        raise ValueError("Saved resolution result identity changed")
    for arm in args.conditions:
        old = source["conditions"][arm]
        for side in (224, args.side):
            entry = row["conditions"][f"{arm}_{side}"]
            if any(
                entry[k] != old[k]
                for k in ("selected_indices", "selected_timestamps")
            ):
                raise ValueError("Frozen selected frames changed")
            strict_trials(entry["trials"], q, config)
            if side == 224 and not args.rerun_control:
                if entry["trials"] != old["trials"]:
                    raise ValueError("Saved baseline QA changed")
            else:
                for trial in entry["trials"]:
                    validate_processor(trial, side, 16)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir", type=Path, default=Path("outputs/segment_cap128_long")
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/resolution320_cap128_long"),
    )
    p.add_argument("--conditions", nargs="+", default=["C", "D"])
    p.add_argument("--side", type=int, default=320)
    p.add_argument("--video-id")
    p.add_argument("--max-videos", type=int)
    p.add_argument(
        "--max-questions",
        type=int,
        help="First N source questions per video, for smoke runs",
    )
    p.add_argument("--rerun-control", action="store_true")
    p.add_argument("--check-inputs", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def main():
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
