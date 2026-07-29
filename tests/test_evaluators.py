"""
These tests are the real signal: they assert the evaluator catches the planted
failure modes and does NOT false-flag the faithful answers. A green check here
means "the eval harness works", which is exactly what an LLM-eval role cares about.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluators import evaluate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load():
    with open(os.path.join(ROOT, "data", "wph_ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)
    with open(os.path.join(ROOT, "data", "llm_outputs.json"), encoding="utf-8") as f:
        outputs = json.load(f)["outputs"]
    return gt, {r["id"]: r for r in gt["records"]}, outputs


def _result(oid):
    gt, records, outputs = _load()
    o = next(x for x in outputs if x["id"] == oid)
    return evaluate(records[o["record_id"]], o, gt["language_codes"])


def test_faithful_answers_pass():
    for oid in ("out-001", "out-004"):
        assert _result(oid)["verdict"] == "pass", f"{oid} should pass"


def test_language_code_confusion_is_caught():
    r = _result("out-002")
    assert r["verdict"] == "fail"
    assert any(f["facet"] == "language" and f["status"] == "factual_error" for f in r["facets"])


def test_hallucinated_translator_is_caught():
    r = _result("out-003")
    assert r["verdict"] == "fail"
    assert any(f["status"] == "hallucination" for f in r["facets"])


def test_every_output_matches_its_expected_verdict():
    gt, records, outputs = _load()
    for o in outputs:
        r = evaluate(records[o["record_id"]], o, gt["language_codes"])
        assert r["verdict"] == o["expected_verdict"], f"{o['id']}: {r['verdict']} != {o['expected_verdict']}"
