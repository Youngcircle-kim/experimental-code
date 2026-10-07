"""Boundary diagnostics and paired eight-condition VQA ablations."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np

from .ablation import (
    COMPARISONS,
    CONDITIONS,
    HYPOTHESES,
    select_frame_control,
)
from .algorithms import (
    ScoreCalibrator,
    allocate_frames,
    clustered_comparison,
    event_features,
    evidence_metrics,
    normalize_rows,
    sample_indices,
    segment,
)
from .backends import build_backend, make_demo_dataset
from .cache import event_caption, write_json
from .config import Config
from .data import array_digest, load_manifest, validate_dataset
from .reporting import export_artifacts
from .reproducibility import fix_seed
from .types import Backend, Event, Question, Video

LOGGER = logging.getLogger(__name__)
CONDITION_KEYS = CONDITIONS


@dataclass
class PreparedVideo:
    """Segmentation/candidate relationships shared across all conditions."""

    video: Video
    events: list[Event]
    visual_event_features: np.ndarray
    text_event_features: np.ndarray | None
    record: dict[str, Any]
    frame_features: np.ndarray
    matched: PreparedVideo | None = None


def describe_events(
    video: Video, events: list[Event], source_indices: np.ndarray
) -> dict[str, Any]:
    """Map sampled segments to original indices and continuous time spans.

    Args:
        video: Original pool and duration.
        events: Half-open intervals on the observed sub-pool.
        source_indices: Mapping from observed indices to the original pool.

    Returns:
        Complete event lengths and before/after observations at each boundary.
    """
    event_rows: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for event in events:
        start_seconds = (
            float(video.timestamps[source_indices[event.start]])
            if event.start
            else 0.0
        )
        end_seconds = (
            float(video.timestamps[source_indices[event.stop]])
            if event.stop < len(source_indices)
            else video.duration_seconds
        )
        event_rows.append(
            {
                "start_observation_index": event.start,
                "stop_observation_index": event.stop,
                "start_candidate_index": int(source_indices[event.start]),
                "stop_candidate_index": int(source_indices[event.stop])
                if event.stop < len(source_indices)
                else len(video.frames),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": end_seconds - start_seconds,
                "observation_count": event.stop - event.start,
            }
        )
        if event.start:
            boundaries.append(
                {
                    "before_index": int(source_indices[event.start - 1]),
                    "after_index": int(source_indices[event.start]),
                    "timestamp_seconds": start_seconds,
                }
            )
    return {"events": event_rows, "boundaries": boundaries}


def prepare_video(
    video: Video, backend: Backend, config: Config
) -> PreparedVideo:
    """Run query-independent features, diagnostics and fixed captions.

    Args:
        video: Validated input video.
        backend: Frozen models.
        config: Detector/caption settings frozen before evaluation.

    Returns:
        Shared per-event features and a complete provenance/timing record.
    """
    feature_started = perf_counter()
    features = np.asarray(backend.encode_frames(video.frames), dtype=float)
    if features.ndim != 2 or features.shape[0] != len(video.frames):
        raise ValueError(
            "Encoder features must have shape (candidate_count, feature_dim)"
        )
    features = normalize_rows(features, config.normalization_epsilon)
    assert (
        features.shape[0] == video.timestamps.size
        and np.isfinite(features).all()
    )
    feature_seconds = perf_counter() - feature_started
    segmentation_started = perf_counter()
    diagnostics: list[dict[str, Any]] = []
    for stride in config.observation_strides:
        source_indices = np.arange(0, len(features), stride)
        observed_features = features[source_indices]
        partitions = {
            name: segment(observed_features, config, name)
            for name in ("D0", "D1", "D2")
        }
        for name in ("D1", "D2"):
            partitions[f"D0_matched_{name}"] = segment(
                observed_features,
                config,
                "D0",
                segment_count=len(partitions[name]),
            )
        for name, events in partitions.items():
            diagnostics.append(
                {
                    "detector": name,
                    "observation_stride": stride,
                    "observed_candidate_count": len(source_indices),
                    **describe_events(video, events, source_indices),
                }
            )
    # B always uses the complete fixed candidate pool, independently of the
    # stride experiments used only for pilot A diagnostics.
    events = segment(features, config, config.detector)
    visual_features = event_features(
        features, events, config.normalization_epsilon
    )
    segmentation_seconds = perf_counter() - segmentation_started
    captions: list[dict[str, Any]] = []
    text_features: np.ndarray | None = None
    text_feature_seconds = 0.0
    if config.pilot == "both":
        for event in events:
            indices = sample_indices(
                event.start,
                event.stop,
                min(config.caption_max_frames, event.stop - event.start),
            )
            captions.append(
                event_caption(video, event, indices, backend, config)
            )
        text_started = perf_counter()
        text_features = normalize_rows(
            np.asarray(
                backend.encode_texts(
                    [caption["text"] for caption in captions]
                ),
                dtype=float,
            ),
            config.normalization_epsilon,
        )
        if text_features.shape[0] != len(events):
            raise ValueError(
                "Caption embedding count differs from fixed event count"
            )
        assert np.isfinite(text_features).all()
        text_feature_seconds = perf_counter() - text_started
    record = {
        "video_id": video.video_id,
        "source_id": video.source_id,
        "split": video.split,
        "candidate_count": len(video.frames),
        "candidate_timestamps": video.timestamps.tolist(),
        "candidate_content_sha256": array_digest(
            video.frames, video.timestamps
        ),
        "frame_shape": list(video.frames.shape),
        "duration_seconds": video.duration_seconds,
        "question_count": len(video.questions),
        "fixed_detector": config.detector,
        "fixed_segmentation": describe_events(
            video, events, np.arange(len(features))
        ),
        "detector_comparisons": diagnostics,
        "captions": captions,
        "timings": {
            "decode_seconds": video.decode_seconds,
            "feature_seconds": feature_seconds,
            "segmentation_diagnostics_seconds": segmentation_seconds,
            "caption_wall_seconds": sum(
                item["current_caption_wall_seconds"] for item in captions
            ),
            "caption_generation_seconds": sum(
                item["current_generation_seconds"] for item in captions
            ),
            "caption_text_feature_seconds": text_feature_seconds,
        },
        "caption_cost": {
            "cache_hits": sum(item["cache_hit"] for item in captions),
            "cache_misses": sum(not item["cache_hit"] for item in captions),
            "current_input_frame_count": sum(
                item["current_input_frame_count"] for item in captions
            ),
            "tokens": [
                {
                    "input": item["current_input_tokens"],
                    "output": item["current_output_tokens"],
                }
                for item in captions
            ],
            "historical_cold_generation_seconds": sum(
                item["generation_seconds"] for item in captions
            ),
        },
    }
    prepared = PreparedVideo(
        video, events, visual_features, text_features, record, features
    )
    if config.pilot == "both":
        matched_events = segment(
            features, config, "D0", segment_count=len(events)
        )
        matched_started = perf_counter()
        matched_captions = [
            event_caption(
                video, event,
                sample_indices(
                    event.start, event.stop,
                    min(config.caption_max_frames, event.stop - event.start),
                ),
                backend, config,
            )
            for event in matched_events
        ]
        matched_text = normalize_rows(
            np.asarray(backend.encode_texts(
                [item["text"] for item in matched_captions]
            ), dtype=float), config.normalization_epsilon,
        )
        if matched_text.shape[0] != len(matched_events):
            raise ValueError("Matched caption embedding count is invalid")
        matched_record = {
            "segmentation": describe_events(
                video, matched_events, np.arange(len(features))
            ),
            "captions": matched_captions,
            "preparation_seconds": perf_counter() - matched_started,
            "current_generation_seconds": sum(
                item["current_generation_seconds"]
                for item in matched_captions
            ),
            "historical_cold_generation_seconds": sum(
                item["generation_seconds"] for item in matched_captions
            ),
        }
        record["matched_uniform_partition"] = matched_record
        prepared.matched = PreparedVideo(
            video, matched_events,
            event_features(
                features, matched_events, config.normalization_epsilon
            ),
            matched_text, matched_record, features,
        )
    return prepared


def relevance_scores(
    prepared: PreparedVideo,
    question: Question,
    backend: Backend,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute separate cosine relevances without options or answer labels.

    Args:
        prepared: Fixed event embeddings.
        question: Only its text is passed into encoders.
        backend: Frozen visual/text encoders.
        config: Numerical tolerance.

    Returns:
        Visual and caption relevance vectors, one score per event.
    """
    visual_query = np.asarray(
        backend.encode_visual_question(question.text), dtype=float
    )
    if (
        visual_query.ndim != 1
        or visual_query.shape[0] != prepared.visual_event_features.shape[1]
    ):
        raise ValueError(
            "Question and visual embeddings must share the same feature space"
        )
    visual_query = normalize_rows(
        visual_query[None, :], config.normalization_epsilon
    )[0]
    text_query = normalize_rows(
        np.asarray(backend.encode_texts([question.text]), dtype=float),
        config.normalization_epsilon,
    )
    if prepared.text_event_features is None or text_query.shape != (
        1,
        prepared.text_event_features.shape[1],
    ):
        raise ValueError(
            "Caption and question text embeddings must share one feature space"
        )
    control_visual_scores = prepared.visual_event_features @ visual_query
    treatment_text_scores = prepared.text_event_features @ text_query[0]
    assert (
        control_visual_scores.shape
        == treatment_text_scores.shape
        == (len(prepared.events),)
    )
    assert (
        np.isfinite(control_visual_scores).all()
        and np.isfinite(treatment_text_scores).all()
    )
    return control_visual_scores, treatment_text_scores


def evaluate_question(
    prepared: PreparedVideo,
    question: Question,
    backend: Backend,
    calibrator: ScoreCalibrator,
    config: Config,
) -> dict[str, Any]:
    """Run equal-budget allocations and fixed original-frame QA.

    Args:
        prepared: Shared event and candidate pool state.
        question: Text/options plus evaluator-only annotations.
        backend: One frozen QA model for all conditions.
        calibrator: Previously fitted development-only normalization.
        config: Frozen fusion, budget and within-event selection settings.

    Returns:
        Paired predictions, allocations, timestamps, evidence hits and costs.
    """
    question_started = perf_counter()
    relevance_started = perf_counter()
    control_visual_scores, treatment_text_scores = relevance_scores(
        prepared, question, backend, config
    )
    control_normalized, treatment_normalized = calibrator.transform(
        control_visual_scores, treatment_text_scores
    )
    scores = {
        "control_v": control_normalized,
        "treatment_t": treatment_normalized,
        "treatment_vt": (1 - config.text_weight) * control_normalized
        + config.text_weight * treatment_normalized,
    }
    if prepared.matched is None:
        raise ValueError("QA requires a count-matched uniform partition")
    matched_v, matched_t = relevance_scores(
        prepared.matched, question, backend, config
    )
    matched_v, matched_t = calibrator.transform(matched_v, matched_t)
    scores["uniform_event_v"] = matched_v
    scores["uniform_event_vt"] = (
        (1 - config.text_weight) * matched_v + config.text_weight * matched_t
    )
    visual_query = np.asarray(
        backend.encode_visual_question(question.text), dtype=float
    )
    visual_query = normalize_rows(
        visual_query[None, :], config.normalization_epsilon
    )[0]
    frame_scores = prepared.frame_features @ visual_query
    for name in ("uniform", "frame_top", "temporal_bin"):
        scores[name] = frame_scores
    relevance_seconds = perf_counter() - relevance_started
    question_seed = int.from_bytes(
        hashlib.sha256(
            f"{config.seed}:{prepared.video.video_id}:{question.question_id}".encode(),
        ).digest()[:4],
        "big",
    )
    rng = np.random.default_rng(question_seed)
    condition_order = (
        rng.permutation(CONDITION_KEYS).tolist()
        if config.randomize_condition_order
        else list(CONDITION_KEYS)
    )
    condition_results: dict[str, Any] = {}
    for condition in condition_order:
        selection_started = perf_counter()
        if condition in ("uniform", "frame_top", "temporal_bin"):
            selected_indices = select_frame_control(
                condition, frame_scores, config.frame_budget
            )
            allocations = np.array([config.frame_budget])
            partition = "candidate_pool"
        else:
            use_matched = condition.startswith("uniform_event_")
            events = (
                prepared.matched.events if use_matched else prepared.events
            )
            partition = "matched_uniform" if use_matched else "detected"
            allocations = allocate_frames(
                scores[condition],
                np.array([event.stop - event.start for event in events]),
                config.frame_budget,
                config.allocation_temperature,
            )
            selected_indices = np.concatenate([
                sample_indices(event.start, event.stop, int(count))
                for event, count in zip(events, allocations)
            ])
        if (
            len(selected_indices) != config.frame_budget
            or len(np.unique(selected_indices)) != config.frame_budget
        ):
            raise ValueError(
                "Allocation failed the exact unique-frame budget constraint"
            )
        assert np.all(np.diff(selected_indices) > 0)
        frames = prepared.video.frames[selected_indices]
        timestamps = prepared.video.timestamps[selected_indices]
        assert frames.shape[0] == timestamps.size == config.frame_budget
        assert np.isfinite(frames).all() and np.isfinite(timestamps).all()
        selection_seconds = perf_counter() - selection_started
        # Re-seed paired conditions identically. No label/caption enters QA.
        fix_seed(
            question_seed,
            config.deterministic,
            config.seed_torch or config.backend == "transformers",
            config.seed_tensorflow,
            config.cublas_workspace_config,
        )
        qa_started = perf_counter()
        output = backend.answer(
            frames.copy(), timestamps.copy(), question.text, question.options
        )
        qa_seconds = perf_counter() - qa_started
        option_scores = np.asarray(output.option_scores, dtype=float)
        if (
            option_scores.shape != (len(question.options),)
            or not np.isfinite(option_scores).all()
        ):
            raise ValueError("QA returned invalid option scores")
        if type(
            output.predicted_index
        ) is not int or not 0 <= output.predicted_index < len(
            question.options
        ):
            raise ValueError("QA returned invalid predicted_index")
        condition_results[condition] = {
            "partition": partition,
            "event_scores": scores[condition].tolist()
            if partition != "candidate_pool" else None,
            "allocation": allocations.tolist(),
            "selected_indices": selected_indices.tolist(),
            "selected_timestamps": timestamps.tolist(),
            "predicted_index": output.predicted_index,
            "option_scores": option_scores.tolist(),
            "correct": output.predicted_index == question.answer_index,
            **evidence_metrics(timestamps, question.evidence_intervals),
            "selection_seconds": selection_seconds,
            "qa_seconds": qa_seconds,
        }
    control_correct = condition_results["control_v"]["correct"]
    treatment_correct = condition_results["treatment_vt"]["correct"]
    return {
        "video_id": prepared.video.video_id,
        "question_id": question.question_id,
        "question": question.text,
        "options": list(question.options),
        "answer_index": question.answer_index,
        "evidence_intervals": [
            list(interval) for interval in question.evidence_intervals
        ],
        "control_visual_raw_scores": control_visual_scores.tolist(),
        "treatment_text_raw_scores": treatment_text_scores.tolist(),
        "frame_visual_raw_scores": frame_scores.tolist(),
        **{
            f"{key}_correct": condition_results[key]["correct"]
            for key in CONDITION_KEYS
        },
        "vt_change": "improved"
        if treatment_correct and not control_correct
        else "degraded"
        if control_correct and not treatment_correct
        else "unchanged",
        "conditions": {key: condition_results[key] for key in CONDITION_KEYS},
        "condition_order": condition_order,
        "qa_seed": question_seed,
        "relevance_seconds": relevance_seconds,
        "question_wall_seconds": perf_counter() - question_started,
    }


def reproducible_result_digest(report: dict[str, Any]) -> str:
    """Hash outputs while excluding paths, clock times and cache state.

    Args:
        report: Full result artifact.

    Returns:
        SHA-256 digest for checking repeated runs in the same environment.
    """
    stable = {
        "metrics": report["metrics"],
        "calibration": report["calibration"],
        "videos": [
            {
                "video_id": video["video_id"],
                "content": video["candidate_content_sha256"],
                "fixed_segmentation": video["fixed_segmentation"],
                "diagnostics": video["detector_comparisons"],
                "captions": [item["text"] for item in video["captions"]],
                "matched_partition": {
                    "segmentation": video["matched_uniform_partition"][
                        "segmentation"
                    ],
                    "captions": [item["text"] for item in video[
                        "matched_uniform_partition"
                    ]["captions"]],
                } if "matched_uniform_partition" in video else None,
            }
            for video in report["videos"]
        ],
        "questions": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "conditions",
                    "relevance_seconds",
                    "question_wall_seconds",
                }
            }
            | {
                "conditions": {
                    name: {
                        key: value
                        for key, value in condition.items()
                        if not key.endswith("_seconds")
                    }
                    for name, condition in row["conditions"].items()
                }
            }
            for row in report["questions"]
        ],
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def run_experiment(config: Config) -> dict[str, Any]:
    """Execute the configured pilot and export a complete, unique run artifact.

    Args:
        config: Validated research settings, passed explicitly to all stages.

    Returns:
        Full result dictionary also exported to results.json.
    """
    config.validate()
    started = perf_counter()
    started_at = datetime.now(timezone.utc)
    seed_metadata = fix_seed(
        config.seed,
        config.deterministic,
        config.seed_torch or config.backend == "transformers",
        config.seed_tensorflow,
        config.cublas_workspace_config,
    )
    output_path = (
        Path(config.output_dir).resolve()
        / f"{started_at.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
    )
    output_path.mkdir(parents=True, exist_ok=False)
    LOGGER.info(
        "Starting %s pilot %s; output=%s",
        config.mode,
        config.pilot,
        output_path,
    )
    write_json(output_path / "config.json", config.to_dict())
    load_started = perf_counter()
    videos = (
        make_demo_dataset(config)
        if config.mode == "demo"
        else load_manifest(config)
    )
    validate_dataset(videos, config)
    data_load_seconds = perf_counter() - load_started
    backend_started = perf_counter()
    backend = build_backend(config)
    backend_metadata = backend.metadata()
    if config.mode == "real" and backend_metadata.get("synthetic", False):
        raise ValueError(
            "A synthetic backend cannot be reported as a real experiment"
        )
    backend_load_seconds = perf_counter() - backend_started
    prepared_videos: list[PreparedVideo] = []
    for video in videos:
        LOGGER.info(
            "Preparing %s [%s], %d candidates",
            video.video_id,
            video.split,
            len(video.frames),
        )
        prepared_videos.append(prepare_video(video, backend, config))
    calibration: dict[str, Any] | None = None
    calibration_seconds = 0.0
    rows: list[dict[str, Any]] = []
    if config.pilot == "both":
        calibration_started = perf_counter()
        control_dev_scores: list[np.ndarray] = []
        treatment_dev_scores: list[np.ndarray] = []
        for prepared in prepared_videos:
            if prepared.video.split != "dev":
                continue
            for question in prepared.video.questions:
                control_scores, treatment_scores = relevance_scores(
                    prepared, question, backend, config
                )
                control_dev_scores.append(control_scores)
                treatment_dev_scores.append(treatment_scores)
                assert prepared.matched is not None
                matched_v, matched_t = relevance_scores(
                    prepared.matched, question, backend, config
                )
                control_dev_scores.append(matched_v)
                treatment_dev_scores.append(matched_t)
        calibrator = ScoreCalibrator.fit(
            np.concatenate(control_dev_scores),
            np.concatenate(treatment_dev_scores),
            config.normalization_epsilon,
        )
        calibration = {
            "parameters": asdict(calibrator),
            "fit_video_ids": [
                item.video.video_id
                for item in prepared_videos
                if item.video.split == "dev"
            ],
            "fit_event_question_pairs": sum(
                len(values) for values in control_dev_scores
            ),
            "fusion_text_weight": config.text_weight,
            "fit_rule": (
                "global development-only z-score; equal event-question-pair "
                "weight across detected and count-matched uniform "
                "partitions; no QA labels"
            ),
        }
        calibration_seconds = perf_counter() - calibration_started
        for prepared in prepared_videos:
            if prepared.video.split == "eval":
                for question in prepared.video.questions:
                    rows.append(
                        evaluate_question(
                            prepared, question, backend, calibrator, config
                        )
                    )
    metric_started = perf_counter()
    metrics = (
        clustered_comparison(rows, config, CONDITION_KEYS, COMPARISONS)
        if rows
        else {"status": "pilot_a_only_no_qa"}
    )
    if rows:
        metrics["hypotheses"] = {
            name: {
                f"{treatment}_minus_{control}": metrics[
                    "paired_comparisons"
                ][f"{treatment}_minus_{control}"]
                for treatment, control in pairs
            }
            for name, pairs in HYPOTHESES.items()
        }
        metrics["vt_changes"] = {
            name: sum(row["vt_change"] == name for row in rows)
            for name in ("improved", "degraded", "unchanged")
        }
        metrics["evidence"] = {}
        for condition in CONDITION_KEYS:
            annotated = [
                row["conditions"][condition]
                for row in rows
                if row["evidence_intervals"]
            ]
            metrics["evidence"][condition] = {
                "annotated_questions": len(annotated),
                "mean_interval_hit": float(
                    np.mean([item["interval_hit"] for item in annotated])
                )
                if annotated
                else None,
                "all_interval_hit_rate": float(
                    np.mean([item["all_interval_hit"] for item in annotated])
                )
                if annotated
                else None,
            }
    metric_seconds = perf_counter() - metric_started
    for prepared in prepared_videos:
        video_rows = [
            row for row in rows if row["video_id"] == prepared.video.video_id
        ]
        if not video_rows:
            continue
        q_count = len(video_rows)
        caption_cost = prepared.record["caption_cost"]
        caption_cost["amortization"] = {
            "observed_question_count": q_count,
            "current_generation_seconds_per_question": prepared.record[
                "timings"
            ]["caption_generation_seconds"]
            / q_count,
            "historical_cold_generation_seconds_per_question": caption_cost[
                "historical_cold_generation_seconds"
            ]
            / q_count,
            "measured_shared_relevance_seconds_per_question": sum(
                row["relevance_seconds"] for row in video_rows
            )
            / q_count,
            "measured_selection_plus_qa_seconds_per_question": {
                key: sum(
                    row["conditions"][key]["selection_seconds"]
                    + row["conditions"][key]["qa_seconds"]
                    for row in video_rows
                )
                / q_count
                for key in CONDITION_KEYS
            },
            "note": (
                "C/q terms amortize observed/historical caption generation "
                "only. These are cost components, not separately benchmarked "
                "cold/warm end-to-end systems."
            ),
        }
    package_versions: dict[str, str] = {"numpy": np.__version__}
    for package in ("torch", "transformers", "opencv-python", "Pillow"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    report: dict[str, Any] = {
        "schema_version": 2,
        "conditions": list(CONDITION_KEYS),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(output_path),
        "config": config.to_dict(),
        "result_kind": "synthetic_smoke_test_not_research_evidence"
        if config.mode == "demo"
        else "real_pilot_measurement",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "reproducibility": seed_metadata,
        "backend": backend_metadata,
        "calibration": calibration,
        "metrics": metrics,
        "questions": rows,
        "videos": [prepared.record for prepared in prepared_videos],
        "timings": {
            "data_load_and_validation_seconds": data_load_seconds,
            "backend_load_seconds": backend_load_seconds,
            "development_calibration_seconds": calibration_seconds,
            "evaluation_question_seconds": sum(
                row["question_wall_seconds"] for row in rows
            ),
            "metrics_seconds": metric_seconds,
            "total_before_export_seconds": perf_counter() - started,
        },
        "limitations": [
            "Pilot A thresholds use sampled frozen features; these are "
            "minimal baselines, not reproduced learned detectors.",
            "No utility predictor is trained; relevance alone does not "
            "estimate the optimal per-event frame count.",
            "No caption is sent to final QA; the caption observation budget "
            "is separate from the exact QA frame budget.",
            "Bootstrap resamples original-video groups; small numbers of "
            "videos give unstable uncertainty estimates.",
            "Runtime and timestamps are not deterministic; matching "
            "scientific outputs requires matching software, hardware, "
            "model checkpoints and data.",
            "Timing includes actual shared computation. Captioner and QA "
            "may share a warmed model; no independent cold/warm speedup "
            "is claimed.",
            "No external API cost is incurred by built-in local backends. "
            "Third-party backend billing needs its own recorded "
            "cost metadata.",
        ],
    }
    report["reproducible_result_sha256"] = reproducible_result_digest(report)
    export_started = perf_counter()
    export_artifacts(report, videos, config, output_path)
    report["timings"]["artifact_export_seconds"] = (
        perf_counter() - export_started
    )
    report["timings"]["total_through_artifact_export_seconds"] = (
        perf_counter() - started
    )
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output_path / "results.json", report)
    LOGGER.info("Completed: %s", output_path / "results.json")
    LOGGER.info(
        "Metrics: %s", json.dumps(metrics, ensure_ascii=False, allow_nan=False)
    )
    return report
