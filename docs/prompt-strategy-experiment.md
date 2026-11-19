# WPH prompt-strategy experiment

## Status and data approval

The checked-in source set is a **candidate**, not yet an approved WPH golden set. It contains factual bibliographic metadata only, with per-field source URLs and evidence notes. No publisher descriptions or book text are copied.

- [The Summer Book](https://www.nyrb.com/products/the-summer-book): NYRB's product page supports the title, author, Swedish source language, Thomas Teal translation, and the listed 2008 edition date.
- [The Orphanage](https://yalebooks.yale.edu/book/9780300243017/the-orphanage/): Yale University Press supports the title, author, translators, publisher, and 2021 edition date.
- [Yale University Press article](https://yalebooks.yale.edu/2021/03/19/the-orphanage/): supports Ukrainian as the source language and names both translators.

Before using the set as approved data, the WPH project owner should review each record's `field_evidence` and `field_sources` in `data/prompt_strategy_cases.json`. After approval, set the dataset and each record's `approval_status` to `approved_for_evaluation`. The runner blocks by default while any record is pending. `--allow-pending-review` is an exploratory override; reports record that override and must not be described as approved-set results.

## Strategies

The runner uses the same cases, model ID, temperature, expected facts, and scoring rules for each strategy:

1. **Direct** — asks for a concise answer from the supplied source packet.
2. **Few-shot** — adds one other case as an example of the desired evidence-linked format.
3. **ReAct-style tool use** — gives the model a read-only `lookup_record_evidence` tool restricted to the current case.
4. **Prompt chain** — first extracts cited claims, then drafts the answer from that extraction.

The prompts request concise evidence-linked claims. They do not ask the model to reveal hidden chain-of-thought.

## Run

Install the optional SDK and set your API key:

```bash
python -m pip install anthropic
export ANTHROPIC_API_KEY="..."
```

After approving the dataset:

```bash
python src/run_prompt_comparison.py --model YOUR_ANTHROPIC_MODEL_ID
```

This writes `report/prompt_strategy_comparison.json`, including every response, strategy, case ID, prompt SHA-256, factual score, field-level citation score, tool/model call count, token counts, latency, and optional cost. Supply current per-million prices explicitly to calculate cost:

```bash
python src/run_prompt_comparison.py \
  --model YOUR_ANTHROPIC_MODEL_ID \
  --input-price-per-million INPUT_PRICE \
  --output-price-per-million OUTPUT_PRICE
```

An exploratory run against the pending candidate records can be made with `--allow-pending-review`; this still requires the API key and is clearly labeled in the report.

## Scoring and interpretation

- **Field accuracy:** each requested field receives credit only if every expected value appears in the response.
- **Citation coverage:** a claim counts only when its value matches the expected field and its URL is listed as evidence for that specific field.
- **Quality score:** `0.7 × field accuracy + 0.3 × citation coverage`.
- **Paired comparison:** each prompt strategy uses the same cases; the report includes a bootstrap interval for score differences versus the direct baseline.
- **Efficiency:** reports model calls, input/output tokens, latency, and optional user-supplied cost.

This is a smoke-test harness with only two cases. It can show whether the implementation runs and surface case-level differences; it cannot support a reliable claim that one strategy is generally better. Expand the reviewed set, define the decision threshold and sample-size/power approach in advance, repeat runs where generation is stochastic, and evaluate across more than one model before making comparative claims. The implementation is a custom evaluator; it does not use RAGAS, LangSmith, or TruLens.
