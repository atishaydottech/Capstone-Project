# eligibility_correct: right output type, and for EligibilityAnswer, right
# yes/no/partial verdict.
#
# citation_grounded: is cited_clause actually in the real source doc for
# cited_source? Checked against the PDF text directly rather than a keyword
# match, since a keyword can show up in a totally unrelated chunk and still
# look "correct."
#
# hallucination_rate is just 1 - citation_accuracy.

import re


def _normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def eligibility_correct(output, case):
    # None = output type didn't match what the case expected (e.g. agent
    # asked for more info instead of answering) -- a separate kind of miss
    # from getting the verdict wrong.
    got_type = type(output).__name__
    if got_type != case["expected_output"]:
        return None
    if case["expected_output"] != "EligibilityAnswer":
        return True  # correctly returned NeedMoreInfo; nothing else to check
    return output.eligible == case["expected_eligible"]


def citation_grounded(output, source_texts, min_run=25):
    # source_texts is {filename: full text}, built fresh from the PDFs --
    # checking against the ground-truth corpus, not the run's own retrieval
    # (which could itself be wrong).
    if type(output).__name__ != "EligibilityAnswer":
        return None

    clause = _normalize(output.cited_clause)
    if not clause:
        return False

    # Loosely match cited_source (free text, e.g. "Cal Grant A" or the filename)
    # to a real program file by substring overlap either direction.
    cited = _normalize(output.cited_source)
    matched_texts = [
        text for filename, text in source_texts.items()
        if _normalize(filename).replace(".pdf", "").replace("_", " ") in cited
        or cited in _normalize(filename)
    ]
    # Citation didn't name a recognizable real source at all -> not grounded.
    # (Deliberately not falling back to searching every file: a citation that
    # can't be tied to a real, named source is itself a grounding failure.)
    if not matched_texts:
        return False

    window = clause[:min_run] if len(clause) >= min_run else clause
    return any(window in text for text in (_normalize(t) for t in matched_texts))


def evaluate_case(output, case, source_texts):
    return {
        "eligibility_correct": eligibility_correct(output, case),
        "citation_grounded": citation_grounded(output, source_texts),
    }


def aggregate(results):
    elig = [r["eligibility_correct"] for r in results if r["eligibility_correct"] is not None]
    cite = [r["citation_grounded"] for r in results if r["citation_grounded"] is not None]
    citation_accuracy = (sum(cite) / len(cite)) if cite else float("nan")
    return {
        "eligibility_accuracy": (sum(elig) / len(elig)) if elig else float("nan"),
        "citation_accuracy": citation_accuracy,
        "hallucination_rate": (1 - citation_accuracy) if cite else float("nan"),
    }
