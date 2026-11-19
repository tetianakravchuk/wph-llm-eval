# WPH LLM Evaluation

**A small, facet-based evaluation harness demonstrated with synthetic fixture data.**

The evaluator checks whether a known set of rules catches selected failures in answers about language, translators, availability, and record verification status. The checked-in examples are **fictional fixtures**. They are not verified World Publishing Houses records, real model responses, or evidence of a model's production quality.

## Run it

```bash
python src/run_eval.py            # offline rule-based judge, no key needed
pytest -q                         # run the harness tests
python src/run_eval.py --judge anthropic   # optional LLM judge (needs ANTHROPIC_API_KEY)
```

## Dataset provenance and scope

The fixture dataset is in [`data/wph_ground_truth.json`](data/wph_ground_truth.json). Its metadata identifies the version, data kind, source policy, and the ISO language-code reference. Each record has a `provenance` object:

- `kind: synthetic_fixture` means the row is fictional.
- `source_urls` and `field_evidence` are empty because the bibliographic fields are invented for tests.
- `verification_status_is_simulated: true` means the status is a test condition, not a claim that a real record was verified.

The four responses in [`data/llm_outputs.json`](data/llm_outputs.json) are handwritten fixtures. Two contain planted failures. The evaluation report measures how the evaluator handles those fixtures; its scores are **not** model-quality metrics and do not establish facts about real WPH records.

The language-code map follows ISO 639-1. The [Library of Congress ISO 639 code list](https://www.loc.gov/standards/iso639-2/php/code_list-txt.php) lists `uk` as Ukrainian.

## What the evaluator checks

Each fixture answer is graded facet by facet:

| Check | Failure it catches |
|---|---|
| **Grounding / factual consistency** | Claims that contradict the fixture record |
| **Hallucinated attribution** | Naming a translator when the fixture has no verified translator |
| **Language-code confusion** | Misreading ISO `uk` (Ukrainian) as “UK / English” |
| **Verification awareness** | Presenting a simulated `curated_needs_check` record as confirmed |

`faithfulness` is scored over claims the answer actually makes: an incorrect claim costs credit; omission is treated as a separate completeness concern. Any high-severity issue makes the fixture answer fail.

## Example output

```
# WPH LLM Faithfulness Report (rule judge)
- Dataset: wph-llm-eval-synthetic-fixtures (1.0.0)
- Data kind: synthetic_test_fixtures
- Scope: evaluator behavior over handwritten fixtures; not a model-quality score
- Outputs graded: 4
- Failed: 2
- Mean fixture faithfulness: 0.5
```

The mean is only a summary of these four hand-authored cases. It should not be compared across models or reported as a production success rate.

## Rule-based and LLM judges

`src/judge.py` ships two judges behind one interface:

- **`RuleBasedJudge`** — deterministic, offline, and suitable for demonstrating a CI gate.
- **`AnthropicJudge`** — optional Claude-based scoring for free-form answers.

The rule judge tests explicit patterns; it does not independently establish truth. The LLM judge is exercised on the same handwritten fixtures and should also be treated as a harness demonstration, not a model benchmark.

## Layout

```
data/   wph_ground_truth.json   synthetic records and provenance metadata
        llm_outputs.json        handwritten answers, including planted failures
src/    evaluators.py           facet checks
        judge.py                rule-based + Anthropic judges
        run_eval.py             CLI -> scored report + report/results.md
tests/  test_evaluators.py      checks fixture verdicts and planted failures
```

## Next step for real WPH evaluation

Before treating any catalog data as a real golden set, replace or supplement these fixtures with a small, shareable, source-checked dataset. Include record-level source URLs and field-level evidence for every evaluated claim, the verification date, and the dataset/license provenance. Then connect the evaluator to actual generated answers and retrieval context.

---

Part of the [World Publishing Houses](https://tetianakravchuk.com/pages/world-publishing-houses.html) work by **Tetiana Kravchuk** — AI evaluation · data quality.
