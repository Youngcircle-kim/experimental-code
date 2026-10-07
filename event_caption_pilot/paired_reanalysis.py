"""Matched-order analysis with paired video-cluster uncertainty."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cache import write_json
from .qa_diagnostics import report_digest


def clustered_means(rows, keys, seed=42, samples=5000):
    """Rows are question means, NOT individual option-order replicates."""
    if not rows:
        return {"n_questions": 0, "n_videos": 0, "metrics": {}}
    clusters = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["video_id"], row["question_id"])
        if identity in seen:
            raise ValueError("Duplicate question")
        seen.add(identity)
        values = [row[k] for k in keys]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite question means")
        clusters[row["video_id"]].append(values)
    ids = sorted(clusters)
    sizes = np.array([len(clusters[i]) for i in ids])
    sums = np.array([np.sum(clusters[i], axis=0) for i in ids])
    means = sums.sum(axis=0) / sizes.sum()
    boot = None
    if len(ids) >= 2:
        draws = np.random.default_rng(seed).multinomial(
            len(ids), np.full(len(ids), 1 / len(ids)), size=samples
        )
        boot = (draws @ sums) / (draws @ sizes)[:, None]
    return {
        "n_questions": len(rows),
        "n_videos": len(ids),
        "metrics": {
            key: {
                "mean": float(means[i]),
                "ci95": None
                if boot is None
                else np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
            }
            for i, key in enumerate(keys)
        },
        "weighting": "equal question weight after within-question order mean",
        "bootstrap_unit": "video",
        "bootstrap_samples": samples,
    }


def analyze(source):
    matched, all_questions = [], []
    counts = {
        "all_pairs": 0,
        "parsed_pairs": 0,
        "generation_correct": 0,
        "scoring_correct_on_matched": 0,
        "disagreements": 0,
    }
    for question in source["questions"]:
        trials = question["trials"]
        parsed = [
            t for t in trials if t["generated_original_index"] is not None
        ]
        counts["all_pairs"] += len(trials)
        counts["parsed_pairs"] += len(parsed)
        g = sum(t["generation_correct"] for t in parsed)
        s = sum(t["scoring_correct"] for t in parsed)
        d = sum(not t["methods_agree"] for t in parsed)
        counts["generation_correct"] += g
        counts["scoring_correct_on_matched"] += s
        counts["disagreements"] += d
        base = {
            k: question[k]
            for k in (
                "video_id",
                "question_id",
                "question_type",
                "duration_group",
            )
        }
        all_questions.append(
            {
                **base,
                "parse_coverage": len(parsed) / len(trials),
                "generation_all_unparsed_wrong": g / len(trials),
                "generation_all_unparsed_right": (
                    g + len(trials) - len(parsed)
                )
                / len(trials),
                "scoring_all": sum(t["scoring_correct"] for t in trials)
                / len(trials),
            }
        )
        if parsed:
            matched.append(
                {
                    **base,
                    "parsed_orders": len(parsed),
                    "generation": g / len(parsed),
                    "scoring": s / len(parsed),
                    "generation_minus_scoring": (g - s) / len(parsed),
                    "disagreement": d / len(parsed),
                }
            )
    keys = (
        "generation",
        "scoring",
        "generation_minus_scoring",
        "disagreement",
    )
    return {
        "source_sha256": report_digest(source),
        "pair_counts": counts,
        "matched": clustered_means(matched, keys),
        "complete_questions": clustered_means(
            [
                m
                for m in matched
                if m["parsed_orders"]
                == next(
                    len(q["trials"])
                    for q in source["questions"]
                    if q["video_id"] == m["video_id"]
                    and q["question_id"] == m["question_id"]
                )
            ],
            keys,
        ),
        "all_question_sensitivity": clustered_means(
            all_questions,
            (
                "parse_coverage",
                "generation_all_unparsed_wrong",
                "generation_all_unparsed_right",
                "scoring_all",
            ),
        ),
        "by_duration": {
            value: clustered_means(
                [r for r in matched if r["duration_group"] == value], keys
            )
            for value in sorted({r["duration_group"] for r in matched})
        },
        "questions": matched,
        "limitations": "Matched subset is selected by parseability. Bounds "
        "are sensitivity bounds, not estimates. All CIs are exploratory "
        "video-cluster percentile bootstrap without multiple-test correction.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "results.json", result)
    compact = {k: v for k, v in result.items() if k != "questions"}
    (args.output_dir / "summary.md").write_text(
        "# Paired QA reanalysis\n\n```json\n"
        + json.dumps(compact, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
