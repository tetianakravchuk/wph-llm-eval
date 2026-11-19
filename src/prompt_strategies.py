"""Prompt-strategy comparison helpers for the WPH source-cited evaluation set.

Strategies share the same question, model, evidence policy, and scoring rubric.
The ReAct variant can call a read-only lookup tool. Chain-of-thought is not requested;
the model returns concise, checkable claims with their source URLs.
"""
import json

STRATEGIES = ("direct", "few_shot", "react", "chain")

OUTPUT_SCHEMA = """Return only valid JSON with this shape:
{"answer":"concise answer","claims":[
  {"field":"original_language","value":"...","source_url":"exact URL from evidence"},
  {"field":"english_translators","value":"...","source_url":"exact URL from evidence"}
]}
Do not provide hidden chain-of-thought. Give only the answer and concise evidence-linked claims.
If evidence is insufficient, say so and do not guess."""

LOOKUP_TOOL = {
    "name": "lookup_record_evidence",
    "description": "Read the approved bibliographic evidence packet for one evaluation case.",
    "input_schema": {
        "type": "object",
        "properties": {"record_id": {"type": "string"}},
        "required": ["record_id"],
    },
}


def evidence_packet(record):
    return {
        "record_id": record["id"],
        "work_title": record["work_title"],
        "author": record["author"],
        "original_language": record["original_language"],
        "english_translators": record["english_translators"],
        "english_edition_publisher": record["english_edition_publisher"],
        "english_edition_publication_date": record["english_edition_publication_date"],
        "source_urls": record["source_urls"],
        "field_evidence": record["field_evidence"],
    }


def user_prompt(strategy, case, record, demonstration=None):
    question = case["question"]
    if strategy == "direct":
        return (
            f"Answer the question using only this source-cited evidence.\n"
            f"QUESTION: {question}\nEVIDENCE: "
            f"{json.dumps(evidence_packet(record), ensure_ascii=False)}\n{OUTPUT_SCHEMA}"
        )
    if strategy == "few_shot":
        example = ""
        if demonstration:
            example = (
                "Example of the required evidence-linked format (use only as a format example):\n"
                f"QUESTION: {demonstration['question']}\n"
                f"REFERENCE FACTS: {json.dumps(demonstration['expected_facts'], ensure_ascii=False)}\n"
                f"ANSWER: {json.dumps(demonstration['demonstration_answer'], ensure_ascii=False)}\n\n"
            )
        return (
            f"{example}Now answer using only the current evidence.\nQUESTION: {question}\n"
            f"EVIDENCE: {json.dumps(evidence_packet(record), ensure_ascii=False)}\n{OUTPUT_SCHEMA}"
        )
    if strategy == "react":
        return (
            "Use the lookup_record_evidence tool to retrieve the cited record before answering. "
            "You may call it again only if needed to check a field. Do not use outside knowledge.\n"
            f"CASE RECORD ID: {record['id']}\nQUESTION: {question}\n{OUTPUT_SCHEMA}"
        )
    if strategy == "chain":
        return (
            "Stage 1 of a two-stage prompt chain: extract only the fields needed to answer. "
            "Return JSON with keys claims and source_urls; each claim must include field, value, "
            "and source_url. Do not draft the user-facing answer yet.\n"
            f"QUESTION: {question}\nEVIDENCE: "
            f"{json.dumps(evidence_packet(record), ensure_ascii=False)}"
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def final_chain_prompt(case, extracted):
    return (
        "Stage 2 of a prompt chain. Draft a concise answer using only the extracted evidence. "
        "Preserve every source URL and do not add facts.\n"
        f"QUESTION: {case['question']}\nEXTRACTED EVIDENCE: "
        f"{json.dumps(extracted, ensure_ascii=False)}\n{OUTPUT_SCHEMA}"
    )


def demonstration_for(record, cases, records):
    """Select a different record as a few-shot example to avoid target leakage."""
    for case in cases:
        other = records.get(case["record_id"])
        if other and other["id"] != record["id"]:
            return {
                "question": case["question"],
                "expected_facts": case["expected_facts"],
                "demonstration_answer": {
                    "answer": "; ".join(
                        f"{key}: {', '.join(value) if isinstance(value, list) else value}"
                        for key, value in case["expected_facts"].items()
                    ),
                    "claims": [
                        {"field": key, "value": value, "source_url": other["source_urls"][0]}
                        for key, value in case["expected_facts"].items()
                    ],
                },
            }
    return None
