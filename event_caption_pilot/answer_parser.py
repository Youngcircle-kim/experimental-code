"""Conservative final-answer parsing, independent of gold labels."""

import re

PARSER_VERSION = "validated-final-answer-v1"


def parse_answer(text, options, truncated=False):
    def result(status, index=None, final=None):
        return {
            "parser_version": PARSER_VERSION,
            "status": status,
            "presented_index": index,
            "final_line": final,
        }

    if truncated:
        return result("truncated_requires_review")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return result("empty")
    # Only the last nonempty line can supply an answer; never search prose.
    final = lines[-1]
    cleaned = final.replace("**", "").strip()
    label = re.match(
        r"^(?:final\s+answer|answer)\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE
    )
    value = label.group(1).strip() if label else cleaned
    numbered = re.fullmatch(r"(\d+)\s*[.)]\s*(\S.*)", value)
    normalized = [option.strip().casefold() for option in options]
    if numbered:
        number, content = int(numbered.group(1)), numbered.group(2).strip()
        if not 1 <= number <= len(options):
            return result("number_out_of_range", final=final)
        matches = [
            i
            for i, option in enumerate(normalized)
            if option == content.casefold()
        ]
        if len(matches) != 1:
            return result("unknown_or_duplicate_option_text", final=final)
        if matches[0] != number - 1:
            return result("number_text_conflict", final=final)
        return result("validated_number_and_text", matches[0], final)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return result("numeric_only_ambiguous", final=final)
    # A bare text option is accepted only as the whole response, or with an
    # explicit final Answer: label. Trailing prose is never guessed from.
    if len(lines) > 1 and not label:
        return result("no_explicit_final_answer", final=final)
    matches = [
        i for i, option in enumerate(normalized) if option == value.casefold()
    ]
    if len(matches) != 1:
        return result("unparsed_or_ambiguous", final=final)
    return result("exact_option_text", matches[0], final)
