"""
Run the WPH LLM-faithfulness evaluation.

    python src/run_eval.py                 # offline rule-based judge (no key)
    python src/run_eval.py --judge anthropic   # LLM-as-judge (needs ANTHROPIC_API_KEY)

Prints a scored report and writes report/results.md.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from judge import get_judge  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(__file__))


def load(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="rule", choices=["rule", "anthropic"])
    args = ap.parse_args()

    gt = load("wph_ground_truth.json")
    outputs = load("llm_outputs.json")["outputs"]
    records = {r["id"]: r for r in gt["records"]}
    judge = get_judge(gt["language_codes"], args.judge)

    results = [judge.score(records[o["record_id"]], o) for o in outputs]
    failed = [r for r in results if r["verdict"] == "fail"]
    mean_faith = round(sum(r["faithfulness"] for r in results) / len(results), 2)

    lines = [f"# WPH LLM Faithfulness Report ({judge.provider} judge)\n",
             f"- Outputs graded: **{len(results)}**",
             f"- Failed (contains a high-severity issue): **{len(failed)}**",
             f"- Mean faithfulness: **{mean_faith}**\n",
             "| output | verdict | faithfulness | issues |",
             "|---|---|---|---|"]
    for r in results:
        issue = "; ".join(i["detail"] for i in r["issues"]) or "-"
        mark = "PASS" if r["verdict"] == "pass" else "FAIL"
        lines.append(f"| `{r['id']}` | {mark} | {r['faithfulness']} | {issue} |")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    with open(os.path.join(ROOT, "report", "results.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Wrote report/results.md  |  caught {len(failed)}/{len(outputs)} planted issues")


if __name__ == "__main__":
    main()
