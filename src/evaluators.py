"""
Reference-based evaluators for LLM answers about the World Publishing Houses
dataset. Each answer is graded, facet by facet, against a *verified* ground-truth
record. Pure standard library -- runs anywhere, no API key.

Failure modes checked (the ones that matter for a source-grounded product):
  - hallucinated_translator      : names a translator the record doesn't support
  - factual_error                : states a value that contradicts the record
  - language_code_confusion      : misreads an ISO code (e.g. 'uk' -> "UK English")
  - unverified_as_confirmed      : presents curated_needs_check data as fact
  - unsupported_availability      : availability claim contradicts the record
"""
import re

# Wrong-language traps: what a model tends to say when it misreads a code.
_CODE_CONFUSION = {
    "uk": ["uk english", "united kingdom", "british english", "english"],
}


def _mentions(text, value):
    return bool(value) and value.lower() in text.lower()


def _extract_translator(text):
    """Pull a translator name out of 'translated by X' / 'translation ... by X'."""
    m = re.search(r"translat(?:ed|ion)[^.]*?\bby\s+([A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){0,2})", text)
    return m.group(1).strip().rstrip(".") if m else None


def _claims_available(text):
    t = text.lower()
    if re.search(r"\b(not (yet )?available|no english|not been translated|not yet in english)\b", t):
        return "not_yet"
    if re.search(r"\b(is )?available in english|available in english|released in|translated by\b", t):
        return "available"
    return None


def check_language(record, text, codes):
    code = record.get("original_language")
    correct = (codes.get(code) or "").lower()
    facet = {"facet": "language", "status": "na", "severity": "info", "detail": ""}
    if not correct:
        return facet
    if correct and _mentions(text, correct):
        facet.update(status="grounded", detail=f"correctly identifies {correct.title()}")
        return facet
    for trap in _CODE_CONFUSION.get(code, []):
        if trap in text.lower():
            facet.update(status="factual_error", severity="high",
                         detail=f"language-code confusion: code '{code}' means {correct.title()}, "
                                f"answer says '{trap}'")
            return facet
    facet.update(status="unsupported", severity="low",
                 detail=f"does not state the original language ({correct.title()})")
    return facet


def check_translator(record, text):
    named = _extract_translator(text)
    truth = record.get("translator_en")
    facet = {"facet": "translator", "status": "na", "severity": "info", "detail": ""}
    if not named:
        return facet
    if truth is None:
        facet.update(status="hallucination", severity="high",
                     detail=f"names translator '{named}' but the record has no verified translator")
        return facet
    if named.lower() in truth.lower() or truth.lower() in named.lower():
        facet.update(status="grounded", detail=f"translator '{truth}' matches the record")
    else:
        facet.update(status="factual_error", severity="high",
                     detail=f"says translator is '{named}', record says '{truth}'")
    return facet


def check_availability(record, text):
    claimed = _claims_available(text)
    truth = record.get("english_availability")
    facet = {"facet": "availability", "status": "na", "severity": "info", "detail": ""}
    if claimed is None:
        return facet
    if claimed == truth:
        facet.update(status="grounded", detail=f"availability '{truth}' matches the record")
    else:
        facet.update(status="unsupported_availability", severity="high",
                     detail=f"claims '{claimed}', record says '{truth}'")
    return facet


def check_verification(record, text):
    """A curated_needs_check record must not be presented as confirmed fact."""
    facet = {"facet": "verification", "status": "na", "severity": "info", "detail": ""}
    if record.get("verification_status") != "curated_needs_check":
        return facet
    hedged = bool(re.search(r"\b(unconfirmed|not verified|needs? (review|checking)|may|might|reportedly|not yet)\b",
                            text, re.I))
    if _claims_available(text) == "available" and not hedged:
        facet.update(status="unverified_as_confirmed", severity="high",
                     detail="record is curated_needs_check but the answer presents it as confirmed")
    else:
        facet.update(status="grounded", detail="treats unverified record with appropriate caution")
    return facet


HIGH_SEVERITY = {"hallucination", "factual_error", "unsupported_availability", "unverified_as_confirmed"}


def evaluate(record, output, codes):
    """Grade one LLM output against its ground-truth record. Returns a result dict."""
    text = output["response"]
    facets = [
        check_language(record, text, codes),
        check_translator(record, text),
        check_availability(record, text),
        check_verification(record, text),
    ]
    issues = [f for f in facets if f["status"] in HIGH_SEVERITY]
    # Faithfulness is scored over claims the answer actually makes (grounded or
    # wrong) -- omissions ("unsupported") are a completeness concern, not a
    # faithfulness one, so they neither earn nor cost credit here.
    claims = [f for f in facets if f["status"] in ({"grounded"} | HIGH_SEVERITY)]
    grounded = [f for f in claims if f["status"] == "grounded"]
    faithfulness = round(len(grounded) / len(claims), 2) if claims else 1.0
    return {
        "id": output["id"],
        "record_id": output["record_id"],
        "prompt": output["prompt"],
        "response": text,
        "facets": facets,
        "issues": issues,
        "faithfulness": faithfulness,
        "verdict": "fail" if issues else "pass",
    }
