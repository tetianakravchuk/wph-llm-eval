"""Run a paired comparison of prompt strategies against cited WPH cases.

Requires the optional Anthropic SDK and ANTHROPIC_API_KEY. Candidate data is blocked
by default until the project owner changes the approval status in the dataset.
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from prompt_strategies import (  # noqa: E402
    LOOKUP_TOOL,
    STRATEGIES,
    demonstration_for,
    final_chain_prompt,
    user_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ORDER_SEED = 1701


def parse_json(text):
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"answer": str(parsed), "claims": []}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else {"answer": str(parsed), "claims": []}
            except json.JSONDecodeError:
                pass
    return {"answer": text, "claims": []}


def normalized(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def score_response(case, record, payload):
    if not isinstance(payload, dict):
        payload = {"answer": str(payload), "claims": []}
    answer = str(payload.get("answer", ""))
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    claim_values = " ".join(
        str(claim.get("value", "")) for claim in claims if isinstance(claim, dict)
    )
    combined = normalized(answer + " " + claim_values)
    expected = case["expected_facts"]
    correct_fields = 0
    for field, expected_value in expected.items():
        values = expected_value if isinstance(expected_value, list) else [expected_value]
        if all(normalized(value) and normalized(value) in combined for value in values):
            correct_fields += 1
    fact_accuracy = correct_fields / max(len(expected), 1)

    field_sources = record.get("field_sources", {})
    accurate_citations = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        field = claim.get("field")
        allowed_sources = field_sources.get(field, [])
        expected_value = expected.get(field)
        if expected_value is None:
            continue
        expected_values = expected_value if isinstance(expected_value, list) else [expected_value]
        claim_value = normalized(claim.get("value", ""))
        value_matches = any(
            normalized(value) and normalized(value) in claim_value
            for value in expected_values
        )
        if claim.get("source_url") in allowed_sources and value_matches:
            accurate_citations.append(claim)
    citation_coverage = len(accurate_citations) / max(len(claims), 1)
    unsupported_claims = max(0, len(claims) - len(accurate_citations))
    score = round(0.7 * fact_accuracy + 0.3 * citation_coverage, 4)
    return {
        "fact_accuracy": round(fact_accuracy, 4),
        "citation_coverage": round(citation_coverage, 4),
        "unsupported_citation_count": unsupported_claims,
        "quality_score": score,
    }


def _text_blocks(response):
    return "\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", "") == "text"
    )


def call_text(client, model, prompt):
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=900,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = (time.perf_counter() - start) * 1000
    return _text_blocks(response), {
        "calls": 1,
        "latency_ms": round(elapsed, 2),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def call_react(client, model, case, record):
    first_prompt = user_prompt("react", case, record)
    messages = [{"role": "user", "content": first_prompt}]
    total = {"calls": 0, "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0}
    for _ in range(4):
        start = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=900,
            temperature=0,
            tools=[LOOKUP_TOOL],
            tool_choice={"type": "auto"},
            messages=messages,
        )
        total["latency_ms"] += (time.perf_counter() - start) * 1000
        total["calls"] += 1
        total["input_tokens"] += response.usage.input_tokens
        total["output_tokens"] += response.usage.output_tokens
        tool_uses = [
            block for block in response.content
            if getattr(block, "type", "") == "tool_use"
        ]
        if not tool_uses:
            total["latency_ms"] = round(total["latency_ms"], 2)
            return _text_blocks(response), total, first_prompt
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            requested_id = block.input.get("record_id")
            result = (
                json.dumps(record, ensure_ascii=False)
                if requested_id == record["id"]
                else json.dumps({"error": "record_id is outside this evaluation case"})
            )
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"ReAct tool-call limit reached for {case['id']}")


def run_case(client, model, strategy, case, record, cases, records):
    if strategy == "react":
        raw, usage, prompt_material = call_react(client, model, case, record)
        payload = parse_json(raw)
    elif strategy == "chain":
        extraction_prompt = user_prompt("chain", case, record)
        extracted_raw, first_usage = call_text(client, model, extraction_prompt)
        extracted = parse_json(extracted_raw)
        final_prompt = final_chain_prompt(case, extracted)
        final_raw, second_usage = call_text(client, model, final_prompt)
        payload = parse_json(final_raw)
        raw = final_raw
        prompt_material = extraction_prompt + "\n---CHAIN STAGE 2---\n" + final_prompt
        usage = {
            "calls": first_usage["calls"] + second_usage["calls"],
            "latency_ms": round(first_usage["latency_ms"] + second_usage["latency_ms"], 2),
            "input_tokens": first_usage["input_tokens"] + second_usage["input_tokens"],
            "output_tokens": first_usage["output_tokens"] + second_usage["output_tokens"],
        }
    else:
        demo = demonstration_for(record, cases, records) if strategy == "few_shot" else None
        prompt_material = user_prompt(strategy, case, record, demo)
        raw, usage = call_text(client, model, prompt_material)
        payload = parse_json(raw)
    return raw, payload, usage, prompt_material


def bootstrap_interval(values, seed=17, rounds=2000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(rounds)]
    means.sort()
    return [
        round(means[int(0.025 * rounds)], 4),
        round(means[min(int(0.975 * rounds), rounds - 1)], 4),
    ]


def paired_bootstrap_interval(candidate, baseline, seed=17, rounds=2000):
    baseline_by_case = {row["case_id"]: row["quality_score"] for row in baseline}
    differences = [
        row["quality_score"] - baseline_by_case[row["case_id"]]
        for row in candidate if row["case_id"] in baseline_by_case
    ]
    return bootstrap_interval(differences, seed=seed, rounds=rounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data/prompt_strategy_cases.json"))
    parser.add_argument("--model", required=True, help="Anthropic model ID; same ID for every strategy")
    parser.add_argument("--allow-pending-review", action="store_true",
                        help="Run an explicitly exploratory comparison before owner approval")
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--output", default=str(ROOT / "report/prompt_strategy_comparison.json"))
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    metadata = dataset["dataset_metadata"]
    approval = metadata.get("approval_status", "missing")
    records = {record["id"]: record for record in dataset["records"]}
    pending = [
        record["id"] for record in dataset["records"]
        if record.get("approval_status") != "approved_for_evaluation"
    ]
    if (approval != "approved_for_evaluation" or pending) and not args.allow_pending_review:
        raise SystemExit(
            "Dataset is not approved. Review the cited records, set dataset and record "
            "approval_status to 'approved_for_evaluation', or use --allow-pending-review "
            "for an explicitly exploratory run."
        )

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise SystemExit("Install the optional anthropic package to run live comparisons.") from exc
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    cases = dataset["cases"]
    results = []
    order_rng = random.Random(STRATEGY_ORDER_SEED)
    run_sequence = 0
    for case in cases:
        case_strategy_order = list(STRATEGIES)
        order_rng.shuffle(case_strategy_order)
        for strategy in case_strategy_order:
            run_sequence += 1
            record = records[case["record_id"]]
            raw, payload, usage, prompt_material = run_case(
                client, args.model, strategy, case, record, cases, records
            )
            score = score_response(case, record, payload)
            cost = None
            if args.input_price_per_million is not None and args.output_price_per_million is not None:
                cost = round(
                    usage["input_tokens"] * args.input_price_per_million / 1_000_000
                    + usage["output_tokens"] * args.output_price_per_million / 1_000_000,
                    8,
                )
            results.append({
                "run_sequence": run_sequence,
                "case_id": case["id"],
                "record_id": record["id"],
                "strategy": strategy,
                "model": args.model,
                "prompt_sha256": hashlib.sha256(prompt_material.encode("utf-8")).hexdigest(),
                "raw_response": raw,
                "parsed_response": payload,
                **score,
                **usage,
                "estimated_cost_usd": cost,
            })

    baseline = [row for row in results if row["strategy"] == "direct"]
    baseline_by_case = {row["case_id"]: row["quality_score"] for row in baseline}
    summary = {}
    for strategy in STRATEGIES:
        rows = [row for row in results if row["strategy"] == strategy]
        scores = [row["quality_score"] for row in rows]
        differences = [
            row["quality_score"] - baseline_by_case[row["case_id"]]
            for row in rows if row["case_id"] in baseline_by_case
        ]
        summary[strategy] = {
            "cases": len(rows),
            "mean_quality_score": round(statistics.mean(scores), 4) if scores else None,
            "bootstrap_95_percent_interval": bootstrap_interval(scores),
            "paired_quality_difference_vs_direct": (
                0.0 if strategy == "direct" else round(statistics.mean(differences), 4)
            ),
            "paired_difference_95_percent_interval_vs_direct": (
                [0.0, 0.0] if strategy == "direct"
                else paired_bootstrap_interval(rows, baseline)
            ),
            "mean_fact_accuracy": round(statistics.mean(r["fact_accuracy"] for r in rows), 4),
            "mean_citation_coverage": round(statistics.mean(r["citation_coverage"] for r in rows), 4),
            "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in rows), 2),
            "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in rows), 2),
            "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows), 2),
            "total_cost_usd": (
                round(sum(r["estimated_cost_usd"] for r in rows), 8)
                if all(r["estimated_cost_usd"] is not None for r in rows) else None
            ),
        }

    report = {
        "experiment": {
            "dataset_id": metadata["dataset_id"],
            "dataset_version": metadata["version"],
            "dataset_approval_status": approval,
            "approval_override_used": bool(args.allow_pending_review and pending),
            "model": args.model,
            "temperature": 0,
            "strategy_order_seed": STRATEGY_ORDER_SEED,
            "strategies": list(STRATEGIES),
            "scoring": {
                "quality_score": "0.7 * field accuracy + 0.3 * citation coverage",
                "field_accuracy": "all expected values for a field must appear in answer or claims",
                "citation_coverage": "claims with a citation URL allowed for that exact field and a matching expected value / all claims",
                "interval": "paired percentile bootstrap over case-level score differences versus direct",
            },
            "limitations": [
                "Two source records are a smoke-test set, not enough for a reliable winner or broad model claim.",
                "Prompt variants are compared on one model at temperature 0; this does not establish generality across models.",
                "One run per case/strategy is exploratory; repeat stochastic generations before drawing conclusions.",
                "Estimated cost is null unless both per-million token prices are supplied explicitly.",
                "Approval-pending runs must be labeled exploratory and must not be reported as approved-set results."
            ],
        },
        "summary": summary,
        "runs": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
