# WPH LLM Faithfulness Report (rule judge)

- Outputs graded: **4**
- Failed (contains a high-severity issue): **2**
- Mean faithfulness: **0.5**

| output | verdict | faithfulness | issues |
|---|---|---|---|
| `out-001` | PASS | 1.0 | - |
| `out-002` | FAIL | 0.0 | language-code confusion: code 'uk' means Ukrainian, answer says 'uk english' |
| `out-003` | FAIL | 0.0 | names translator 'Mark Peterson' but the record has no verified translator; claims 'available', record says 'not_yet'; record is curated_needs_check but the answer presents it as confirmed |
| `out-004` | PASS | 1.0 | - |
