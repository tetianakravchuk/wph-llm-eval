# WPH LLM Evaluation

**A small evaluation harness for source-grounded World Publishing Houses answers.**

The repository keeps two datasets distinct:

- The original four-record harness uses **synthetic fixtures** to demonstrate evaluator behavior. They are not verified WPH records or model-quality results.
- A two-record, source-cited **prompt-strategy candidate set** is in `data/prompt_strategy_cases.json`. Its fields map to publisher-catalog URLs. It is marked pending project-owner review and is blocked by the live experiment runner until approved.

## Offline fixture harness

```bash
python src/run_eval.py
pytest -q
python src/run_eval.py --judge anthropic  # optional LLM judge; needs ANTHROPIC_API_KEY
```

The fixture dataset is in [`data/wph_ground_truth.json`](data/wph_ground_truth.json). Each row says `synthetic_fixture`; its empty source URLs are intentional. The four handwritten responses in [`data/llm_outputs.json`](data/llm_outputs.json) include planted failures. Their scores are evaluator smoke-test results, not model-quality metrics.

## Prompt-strategy experiment

The candidate set contains factual metadata from publisher pages:

- [The Summer Book — New York Review Books](https://www.nyrb.com/products/the-summer-book)
- [The Orphanage — Yale University Press](https://yalebooks.yale.edu/book/9780300243017/the-orphanage/)
- [Yale University Press source-language and translator note](https://yalebooks.yale.edu/2021/03/19/the-orphanage/)

Each field has both a source URL and a short evidence note. Only bibliographic facts are stored; publisher descriptions and book text are not copied. The candidate set needs project-owner approval before it is represented as an approved golden set. Review `field_sources` and `field_evidence`, then set the dataset and each record's `approval_status` to `approved_for_evaluation`. The runner blocks by default until then; `--allow-pending-review` creates an explicitly exploratory run.

The live runner compares four prompt patterns using the same cases and model:

- Direct evidence-context prompt
- Few-shot example prompt
- ReAct-style prompt with a read-only evidence lookup tool
- Two-stage extraction and answer prompt chain

It scores requested facts and field-level citations, then records model calls, tokens, latency, and optional cost. It reports paired score differences and bootstrap intervals versus the direct baseline. It requests concise evidence-linked claims; it does not ask for hidden chain-of-thought.

After approval, install the optional Anthropic SDK and set `ANTHROPIC_API_KEY`:

```bash
python -m pip install anthropic
python src/run_prompt_comparison.py --model YOUR_ANTHROPIC_MODEL_ID
```

See [the experiment protocol](docs/prompt-strategy-experiment.md) for data approval, scoring details, limitations, and optional cost inputs.

**This is a smoke-test set of two records.** It can exercise the comparison pipeline and reveal case-level differences, but cannot establish a generally better strategy. Expand the reviewed set and define the sample-size/power approach before making broad comparative claims. The implementation uses custom evaluation code; it does not use RAGAS, LangSmith, or TruLens.

## Original fixture checks

The deterministic rule evaluator checks:

| Check | Example failure |
|---|---|
| Grounding | Claim contradicts the fixture record |
| Hallucinated attribution | Names a translator the fixture does not support |
| Language-code confusion | Misreads ISO `uk` as “UK / English” |
| Verification awareness | Presents a simulated unverified record as confirmed |

The [Library of Congress ISO 639 code list](https://www.loc.gov/standards/iso639-2/php/code_list-txt.php) lists `uk` as Ukrainian.

## Layout

```
data/   wph_ground_truth.json          synthetic evaluator fixtures
        llm_outputs.json               handwritten fixture answers
        prompt_strategy_cases.json      source-cited candidate records and cases
src/    evaluators.py                  deterministic facet checks
        judge.py                       rule-based + optional Anthropic judges
        prompt_strategies.py           prompt variants and read-only lookup tool
        run_eval.py                    offline fixture evaluation
        run_prompt_comparison.py       live paired strategy experiment
docs/   prompt-strategy-experiment.md  protocol and limitations
tests/  test_evaluators.py              fixture behavior checks
```

---

Part of the [World Publishing Houses](https://tetianakravchuk.com/pages/world-publishing-houses.html) work by **Tetiana Kravchuk** — AI evaluation · data quality.
