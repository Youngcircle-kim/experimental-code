"""Compare complete matched B16/B32 runs with paired video uncertainty."""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from .cache import write_json
from .config import Config
from .frame_replacement_probe import strict_trials
from .order_experiments import option_orders
from .paired_reanalysis import clustered_means
from .qa_diagnostics import report_digest
from .resolution_dev import validate_processor
from .types import Question

ARMS = ("uniform", "frame_top", "temporal_bin", "C", "D")
CONFIG_BOOKKEEPING = {"frame_budget", "output_dir", "cache_dir", "device"}


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    json.dumps(value, allow_nan=False)
    return value


def normalized_protocol(protocol):
    """Permit budget and run-location changes, preserving model semantics."""
    result = deepcopy(protocol)
    result.pop("config_source_sha256", None)
    result["config"] = {
        key: value
        for key, value in result["config"].items()
        if key not in CONFIG_BOOKKEEPING
    }
    return result


def normalized_backend(metadata):
    result = deepcopy(metadata)
    result.pop("device", None)
    for name in ("encoder_parameters", "vlm_parameters"):
        if isinstance(result.get(name), dict):
            result[name].pop("devices", None)
    return result


def normalized_runtime(runtime):
    result = deepcopy(runtime)
    result.get("environment", {}).pop("CUDA_VISIBLE_DEVICES", None)
    return result


def validate_backend(backend, config):
    expected = {
        "backend": "transformers",
        "synthetic": False,
        "frozen": True,
        "vlm_loaded": True,
        "encoder_model": config.encoder_model,
        "vlm_model": config.vlm_model,
        "encoder_resolved_commit": config.encoder_revision,
        "vlm_resolved_commit": config.vlm_revision,
        "dtype": config.model_dtype,
        "attention_implementation": config.attention_implementation,
        "vlm_min_pixels": config.vlm_min_pixels,
        "vlm_max_pixels": config.vlm_max_pixels,
    }
    if any(backend.get(key) != value for key, value in expected.items()):
        raise ValueError("Actual backend differs from pinned run settings")
    if not backend.get("torch_version") or not backend.get(
        "transformers_version"
    ):
        raise ValueError("Backend software versions are missing")


def load_complete(directory, budget):
    directory = Path(directory).resolve()
    protocol, results = (
        read(directory / "protocol.json"),
        read(directory / "results.json"),
    )
    digest = report_digest(protocol)
    if (
        protocol.get("experiment") != "portable_frame_budget_v1"
        or protocol.get("version") != 1
        or results.get("protocol_sha256") != digest
        or results.get("completed") is not True
        or protocol.get("qa_side") != 320
        or protocol.get("conditions") != list(ARMS)
        or protocol.get("split") not in {"dev", "eval"}
    ):
        raise ValueError("Require complete portable B16/B32 QA320 results")
    config = Config(**protocol["config"])
    config.validate()
    if (
        config.frame_budget != budget
        or (config.frame_height, config.frame_width) != (224, 224)
        or config.vlm_max_pixels != 320**2
        or config.mode != "real"
        or config.backend != "transformers"
        or config.pilot != "both"
        or any(
            not re.fullmatch(r"[0-9a-f]{40}", revision)
            for revision in (config.encoder_revision, config.vlm_revision)
        )
    ):
        raise ValueError("Require pinned models, 224 selection and QA320")
    if results.get("config", protocol["config"]) != protocol["config"]:
        raise ValueError("Result config differs from its protocol")
    if not protocol.get("manifest_sha256") or not protocol.get("code_sha256"):
        raise ValueError("Missing manifest or source-code provenance")
    cohort = protocol["cohort"]
    expected = {(v, q) for v, questions in cohort.items() for q in questions}
    if (
        not expected
        or sum(map(len, cohort.values())) != len(expected)
        or set(protocol["inputs"]) != set(cohort)
    ):
        raise ValueError("Invalid or duplicate protocol cohort")
    rows = {}
    sources, videos, candidate_videos = {}, {}, {}
    for row in results["questions"]:
        key = (row["video_id"], row["question_id"])
        if key in rows:
            raise ValueError("Duplicate result question")
        rows[key] = row
        if not row.get("source_id") or not row.get("candidate_sha256"):
            raise ValueError("Missing original-source/candidate identity")
        candidate_count = row["candidate_count"]
        if type(candidate_count) is not int or candidate_count < budget:
            raise ValueError("Candidate pool cannot satisfy frame budget")
        if row["source_id"] in sources and sources[row["source_id"]] != key[0]:
            raise ValueError("Original source reused as multiple videos")
        sources[row["source_id"]] = key[0]
        identity = (row["source_id"], row["candidate_sha256"])
        if key[0] in videos and videos[key[0]] != identity:
            raise ValueError("Inconsistent original source within one video")
        videos[key[0]] = identity
        candidate = row["candidate_sha256"]
        if (
            candidate in candidate_videos
            and candidate_videos[candidate] != (key[0])
        ):
            raise ValueError("Duplicate candidate content across videos")
        candidate_videos[candidate] = key[0]
        question = Question(
            key[1],
            row["question"],
            tuple(row["options"]),
            row["answer_index_evaluator_only"],
        )
        if (
            not isinstance(question.text, str)
            or not question.text.strip()
            or not 2 <= len(question.options) <= 4
            or any(
                not isinstance(o, str) or not o.strip()
                for o in question.options
            )
            or type(question.answer_index) is not int
            or not 0 <= question.answer_index < len(question.options)
        ):
            raise ValueError("Invalid saved question or evaluation label")
        orders = [
            list(o) for o in option_orders(len(question.options), config.seed)
        ]
        if protocol["orders_by_option_count"][str(len(question.options))] != (
            orders
        ):
            raise ValueError("Protocol option orders changed")
        if set(row["conditions"]) != set(ARMS):
            raise ValueError("Missing or unexpected selector results")
        for entry in row["conditions"].values():
            indices = entry["selected_indices"]
            stamps = np.asarray(entry["selected_timestamps"], dtype=float)
            if (
                len(indices) != budget
                or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices))
                or indices[-1] >= candidate_count
                or stamps.shape != (budget,)
                or not np.isfinite(stamps).all()
                or stamps[0] < 0
                or np.any(np.diff(stamps) <= 0)
            ):
                raise ValueError("Invalid selection budget or chronology")
            strict_trials(entry["trials"], question, config)
            for trial in entry["trials"]:
                validate_processor(deepcopy(trial), 320, budget)
                if trial.get("visual_tokens", budget * 100) != budget * 100:
                    raise ValueError("Incorrect visual-token accounting")
    if set(rows) != expected:
        raise ValueError("Incomplete results or cohort mismatch")
    progress = read(directory / "progress.json")
    if (
        progress.get("status") != "complete"
        or progress.get("completed_questions") != len(expected)
        or progress.get("total_questions") != len(expected)
        or progress.get("protocol_sha256") != digest
    ):
        raise ValueError("Run completion record is missing or inconsistent")
    backend, runtime = (
        read(directory / "backend.json"),
        read(directory / "runtime.json"),
    )
    validate_backend(backend, config)
    if any(not runtime.get(k) for k in ("python", "platform", "packages")):
        raise ValueError("Runtime provenance is incomplete")
    if any(
        runtime["packages"].get(package) != backend[f"{package}_version"]
        for package in ("torch", "transformers")
    ):
        raise ValueError("Backend/runtime software versions disagree")
    return {
        "directory": directory,
        "protocol": protocol,
        "results": results,
        "rows": rows,
        "backend": backend,
        "runtime": runtime,
    }


def compare(baseline, larger):
    left, right = load_complete(baseline, 16), load_complete(larger, 32)
    for description, first, second in (
        (
            "protocol",
            normalized_protocol(left["protocol"]),
            normalized_protocol(right["protocol"]),
        ),
        (
            "backend",
            normalized_backend(left["backend"]),
            normalized_backend(right["backend"]),
        ),
        (
            "runtime",
            normalized_runtime(left["runtime"]),
            normalized_runtime(right["runtime"]),
        ),
    ):
        if first != second:
            raise ValueError(f"Matched B16/B32 {description} differs")
    if set(left["rows"]) != set(right["rows"]):
        raise ValueError("Question sets differ across budgets")
    metrics = []
    for key in sorted(left["rows"]):
        first, second = left["rows"][key], right["rows"][key]
        for name in (
            "video_id",
            "question_id",
            "source_id",
            "duration_group",
            "candidate_sha256",
            "candidate_count",
            "question",
            "options",
            "answer_index_evaluator_only",
        ):
            if first[name] != second[name]:
                raise ValueError(f"Question identity differs: {key}/{name}")
        row = {
            name: first[name]
            for name in (
                "video_id",
                "question_id",
                "source_id",
                "duration_group",
            )
        }
        count = len(first["conditions"]["uniform"]["trials"])
        numerators = {}
        for arm in ARMS:
            a, b = (r["conditions"][arm]["trials"] for r in (first, second))
            if [t["order"] for t in a] != [t["order"] for t in b]:
                raise ValueError("Option orders differ across budgets")
            na, nb = (
                sum(t["scoring_correct"] for t in trials) for trials in (a, b)
            )
            numerators[arm] = nb - na
            row.update(
                {
                    f"{arm}_B16_correct_orders": na,
                    f"{arm}_B32_correct_orders": nb,
                    f"{arm}_B16": na / count,
                    f"{arm}_B32": nb / count,
                    f"{arm}_32_minus_16_numerator": nb - na,
                    f"{arm}_32_minus_16": (nb - na) / count,
                }
            )
        numerator = numerators["D"] - numerators["uniform"]
        row["D_advantage_32_minus_16_numerator"] = numerator
        row["D_advantage_32_minus_16"] = numerator / count
        row["n_option_orders"] = count
        metrics.append(row)
    keys = [
        name
        for arm in ARMS
        for name in (
            f"{arm}_B16",
            f"{arm}_B32",
            f"{arm}_32_minus_16",
        )
    ] + ["D_advantage_32_minus_16"]
    seed = left["protocol"]["config"]["seed"]
    split = left["protocol"]["split"]
    interpretation = (
        "Exploratory development comparison; not held-out evidence. "
        if split == "dev"
        else "Comparison on the frozen evaluation cohort. Prior holdout use "
        "cannot be established from metadata; only interpret as confirmatory "
        "if methods were frozen before evaluation. "
    )
    return {
        "experiment": "paired_frame_budget_16_vs_32",
        "version": 1,
        "baseline": {
            "directory": str(left["directory"]),
            "frame_budget": 16,
            "protocol_sha256": report_digest(left["protocol"]),
            "results_sha256": report_digest(left["results"]),
        },
        "larger": {
            "directory": str(right["directory"]),
            "frame_budget": 32,
            "protocol_sha256": report_digest(right["protocol"]),
            "results_sha256": report_digest(right["results"]),
        },
        "split": split,
        "qa_side": 320,
        "conditions": list(ARMS),
        "metric_definitions": {
            "arm_32_minus_16": "Within-question B32 minus B16 accuracy",
            "D_advantage_32_minus_16": "(D32 - uniform32) - (D16 - uniform16)",
        },
        "overall": clustered_means(metrics, keys, seed, samples=5000),
        "by_duration": {
            duration: clustered_means(
                [r for r in metrics if r["duration_group"] == duration],
                keys,
                seed,
                samples=5000,
            )
            for duration in sorted({r["duration_group"] for r in metrics})
        },
        "question_metrics": metrics,
        "interpretation": interpretation
        + "Contrasts use integer correct-order counts before division, "
        "then paired video-cluster resampling after within-question "
        "order means. CIs are marginal, without multiplicity adjustment; "
        "they are not obtained by subtracting separate CI bounds. "
        "This is a custom subset experiment, not an official benchmark.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--larger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if any(
        output == source.resolve() or source.resolve() in output.parents
        for source in (args.baseline, args.larger)
    ):
        raise ValueError("Comparison output must not overwrite run artifacts")
    result = compare(args.baseline, args.larger)
    write_json(args.output, result)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
