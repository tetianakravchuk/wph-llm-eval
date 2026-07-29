"""
Judges turn a (prompt, response, record) into a verdict.

- RuleBasedJudge: deterministic, offline, no API key. Uses evaluators.py.
- LLMJudge (Anthropic): optional. Set ANTHROPIC_API_KEY and pass provider="anthropic"
  to have Claude grade faithfulness with the same rubric. This is the seam that
  turns the project from "reference checks" into "LLM-as-judge" evaluation.

The rubric is identical for both so the rule-based judge is a cheap, deterministic
proxy you can run in CI, and the LLM judge is the higher-fidelity version.
"""
import json
import os

from evaluators import evaluate

RUBRIC = """You are grading whether an ANSWER about a book is faithful to a VERIFIED record.
Rules:
1. Every factual claim (language, translator, publisher, year, English availability)
   must be supported by the record. Unsupported claims are hallucinations.
2. ISO language codes are not country codes: 'uk' = Ukrainian, not the United Kingdom.
3. If the record's verification_status is 'curated_needs_check', the answer must NOT
   present it as confirmed fact; it should hedge.
Return strict JSON: {"verdict":"pass|fail","faithfulness":0.0-1.0,"issues":["..."]}"""


class RuleBasedJudge:
    provider = "rule"

    def __init__(self, codes):
        self.codes = codes

    def score(self, record, output):
        return evaluate(record, output, self.codes)


class AnthropicJudge:
    """LLM-as-judge via the Claude Messages API. Requires `pip install anthropic`
    and ANTHROPIC_API_KEY. Kept import-light so the offline path never needs it."""
    provider = "anthropic"

    def __init__(self, codes, model="claude-sonnet-5"):
        from anthropic import Anthropic  # local import: optional dependency
        self.codes = codes
        self.model = model
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def score(self, record, output):
        content = (
            f"{RUBRIC}\n\nRECORD:\n{json.dumps(record, ensure_ascii=False)}\n\n"
            f"PROMPT: {output['prompt']}\nANSWER: {output['response']}"
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": content}],
        )
        raw = msg.content[0].text
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        return {
            "id": output["id"], "record_id": output["record_id"],
            "prompt": output["prompt"], "response": output["response"],
            "facets": [], "issues": [{"detail": i} for i in data.get("issues", [])],
            "faithfulness": float(data.get("faithfulness", 0.0)),
            "verdict": data.get("verdict", "fail"),
        }


def get_judge(codes, provider="rule"):
    if provider == "rule":
        return RuleBasedJudge(codes)
    if provider == "anthropic":
        return AnthropicJudge(codes)
    raise ValueError(f"unknown judge provider: {provider}")
