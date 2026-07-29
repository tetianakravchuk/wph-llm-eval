# WPH LLM Evaluation

**Source-grounded faithfulness evaluation for LLM answers about the World Publishing Houses dataset.**

LLMs are happy to tell you who translated a book, what language it was written in, and whether it's available in English — whether or not any of it is true. This project treats the **verified** World Publishing Houses records as ground truth and grades LLM answers against them, catching the failure modes that actually break a source-backed product: **hallucinated attributions, factual errors, language-code confusion, and unverified data presented as fact.**

It runs offline with no API key (a deterministic rule-based judge) and has a drop-in seam for LLM-as-judge with Claude.

```bash
python src/run_eval.py            # offline judge, no key needed
pytest -q                         # the harness's own tests
python src/run_eval.py --judge anthropic   # LLM-as-judge (needs ANTHROPIC_API_KEY)
```

## Why this dataset is the right test bed

The WPH dataset already carries a `verification_status` on every record (`verified_public_source` vs `curated_needs_check`) and full provenance. That makes it a **golden set**: I can ask an LLM questions about a book, then check every claim in its answer against a record I trust — and specifically check that it never launders an unverified lead into a confirmed fact.

## What it evaluates

Each answer is graded facet-by-facet against its record:

| Check | Failure it catches |
|---|---|
| **Grounding / factual consistency** | claims that contradict the verified record (wrong translator, wrong year) |
| **Hallucinated attribution** | naming a translator the record doesn't support |
| **Language-code confusion** | reading ISO `uk` (Ukrainian) as "UK / English" — a real multilingual-metadata bug |
| **Verification awareness** | presenting a `curated_needs_check` record as a confirmed fact instead of hedging |

`faithfulness` is scored over the claims an answer actually makes (a hallucination costs credit; a *mere omission* does not — that's a separate completeness concern). Any high-severity issue makes the whole answer `fail`.

## Example output

```
# WPH LLM Faithfulness Report (rule judge)
- Outputs graded: 4
- Failed (contains a high-severity issue): 2
- Mean faithfulness: 0.5

| output   | verdict | faithfulness | issues                                                              |
|----------|---------|--------------|---------------------------------------------------------------------|
| out-001  | PASS    | 1.0          | -                                                                   |
| out-002  | FAIL    | 0.0          | language-code confusion: 'uk' means Ukrainian, answer says 'uk english' |
| out-003  | FAIL    | 0.0          | hallucinated translator 'Mark Peterson'; availability wrong; unverified presented as confirmed |
| out-004  | PASS    | 1.0          | -                                                                   |
```

The four sample answers in `data/llm_outputs.json` include two deliberately planted failures; the [`tests`](tests/) assert the evaluator catches them and doesn't false-flag the good ones. A green CI run therefore means *the eval harness works* — which is the whole point.

## The LLM-as-judge upgrade

`src/judge.py` ships two judges behind one interface:

- **`RuleBasedJudge`** (default) — deterministic, offline, cheap enough to gate CI.
- **`AnthropicJudge`** — sends the same rubric to Claude to grade faithfulness on free-form answers the rule checks can't fully parse. Set `ANTHROPIC_API_KEY`, `pip install anthropic`, and run with `--judge anthropic`.

Using a cheap deterministic judge as a regression gate and an LLM judge for fidelity is the same pattern I'd apply to a production eval suite.

## Layout

```
data/   wph_ground_truth.json   verified records (the golden set)
        llm_outputs.json        sample LLM answers, incl. planted failures
src/    evaluators.py           facet checks (grounding, hallucination, codes, verification)
        judge.py                rule-based + Anthropic judges behind one interface
        run_eval.py             CLI -> scored report + report/results.md
tests/  test_evaluators.py      proves the harness catches the planted failures
```

## Roadmap

- Have the LLM *generate* the answers too (currently the answers are fixtures), so the loop is generate → evaluate end to end.
- Add retrieval so the model must ground answers in provided WPH records (RAG faithfulness).
- Expand the golden set from the full WPH dataset and track faithfulness over time.

---

Part of the [World Publishing Houses](https://tetianakravchuk.com/pages/world-publishing-houses.html) work by **Tetiana Kravchuk** — AI QA · LLM Evaluation · Data Quality.
