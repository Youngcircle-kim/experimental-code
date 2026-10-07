"""Reparse existing order-experiment JSON without decoding or inference."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from .answer_parser import PARSER_VERSION, parse_answer
from .cache import write_json
from .order_experiments import question_summary, summarize_dev
from .qa_diagnostics import report_digest


def reparse_trial(trial, gold):
    row = copy.deepcopy(trial)
    options = row["presented_options"]
    order = row.get("order", row.get("option_order_original_indices"))
    if sorted(order) != list(range(len(options))):
        raise ValueError("Invalid option permutation")
    if type(gold) is not int or not 0 <= gold < len(options):
        raise ValueError("Missing or invalid evaluator answer index")
    if "scoring_correct" in row and row["scoring_correct"] != (
        row["scored_original_index"] == gold
    ):
        raise ValueError("Reference label conflicts with saved correctness")
    parsing = parse_answer(
        row["generation"]["text"],
        options,
        row["generation"].get("output_token_limit_reached", False),
    )
    index = parsing["presented_index"]
    original = None if index is None else order[index]
    row["previous_parse"] = {
        key: row.get(key)
        for key in (
            "generated_original_index",
            "generation_correct",
            "methods_agree",
        )
    }
    row.update(
        {
            "generation_parsing": parsing,
            "generated_original_index": original,
            "generation_correct": None if index is None else original == gold,
            "methods_agree": None
            if index is None
            else original == row["scored_original_index"],
            "answer_index_evaluator_only": gold,
        }
    )
    return row


def reaggregate(source, reference):
    result = copy.deepcopy(source)
    all_trials = []
    if source.get("stage") == "dev":
        for question in result["questions"]:
            key = (question["video_id"], question["question_id"])
            gold, options = reference[key]
            for trial in question["trials"]:
                order = trial["order"]
                if trial["presented_options"] != [options[i] for i in order]:
                    raise ValueError("Reference options differ from saved run")
            question["trials"] = [
                reparse_trial(t, gold) for t in question["trials"]
            ]
            question["summary"] = question_summary(question["trials"])
            all_trials.extend(question["trials"])
        result["summary"] = summarize_dev(result["questions"])
        result["by_question_type"] = {
            kind: summarize_dev(
                [q for q in result["questions"] if q["question_type"] == kind]
            )
            for kind in sorted(
                {q["question_type"] for q in result["questions"]}
            )
        }
    else:
        # For length rechecks, the reference is the original experiment and
        # the source hash identifies it; the recheck has only one question.
        if source.get("stage") != "length":
            raise ValueError("Supported inputs are dev or length results.json")
        for trial in result["trials"]:
            matches = []
            for key, (gold, options) in reference.items():
                order = trial["order"]
                if len(options) == len(order) and trial[
                    "presented_options"
                ] == [options[i] for i in order]:
                    # The exact Question: line prevents option-only matching.
                    if (
                        key in reference.questions
                        and (
                            "Question: "
                            + reference.questions[key]
                            + "\nOptions:"
                        )
                        in trial["generation"]["prompt"]
                    ):
                        matches.append(gold)
            if len(matches) != 1:
                raise ValueError("Cannot uniquely identify recheck question")
            all_trials.append(reparse_trial(trial, matches[0]))
        result["trials"] = all_trials
        result["summary"] = question_summary(all_trials)
    result["parsing_status_counts"] = dict(
        Counter(t["generation_parsing"]["status"] for t in all_trials)
    )
    result["reaggregation"] = {
        "parser_version": PARSER_VERSION,
        "source_sha256": report_digest(source),
        "inference_performed": False,
        "note": "Only validated final answers are parsed. Numeric-only "
        "answers and conflicts remain unavailable; report coverage alongside "
        "accuracy and disagreement. Gold labels never enter parsing.",
    }
    return result


class References(dict):
    def __init__(self):
        super().__init__()
        self.questions = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    ref = json.loads(args.reference_results.read_text(encoding="utf-8-sig"))
    references = References()
    if source.get("stage") == "length":
        if source["source_report_sha256"] != report_digest(ref):
            raise ValueError("Reference result hash mismatch")
        for q in ref["questions"]:
            key = (q["video_id"], q["question_id"])
            references[key] = (q["answer_index"], q["options"])
            references.questions[key] = q["question"]
    elif source.get("stage") == "dev":
        # Dev gold labels are in the original manifest, not eval report rows.
        if source["config"]["manifest_path"] != ref["config"]["manifest_path"]:
            raise ValueError("Dev/reference manifest paths differ")
        manifest = json.loads(
            Path(source["config"]["manifest_path"]).read_text(
                encoding="utf-8-sig"
            )
        )
        texts = {}
        for video in manifest["videos"]:
            if video["split"] == "dev":
                for q in video["questions"]:
                    key = (video["video_id"], q["question_id"])
                    references[key] = (q["answer_index"], q["options"])
                    texts[key] = q["text"]
        for q in source["questions"]:
            if texts[(q["video_id"], q["question_id"])] != q["question"]:
                raise ValueError("Dev question text changed")
    result = reaggregate(source, references)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "results.json", result)
    compact = {
        key: result[key]
        for key in ("summary", "parsing_status_counts", "reaggregation")
    }
    (args.output_dir / "summary.md").write_text(
        "# Reparsed answers\n\n```json\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
