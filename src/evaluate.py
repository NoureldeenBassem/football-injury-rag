"""Run the eval set through rag() and save the results.

Checks per question: the predicted topic matches, and recovery-time questions cite the
recovery-time statistics records. Each question uses 2 Gemini requests.

Run:  python -m src.evaluate
Writes outputs/eval_results.csv and outputs/eval_answers.md.
"""
import csv
import sys
from datetime import datetime

from src.config import EVAL_SET, LLM, OUTPUTS_DIR, STATS_SOURCE
from src.generate import DailyQuotaExceeded, rag, source_list


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(EVAL_SET, encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    rows, answers = [], [f"# Eval answers ({LLM}, {datetime.now():%Y-%m-%d %H:%M})\n"]
    for case in cases:
        q = case["question"]
        print(f"> {q}", flush=True)
        try:
            r = rag(q)
        except DailyQuotaExceeded as e:
            print(f"\nStopped: {e}")
            break
        hits = r["hits"]
        stats = sum(h["meta"]["source"] == STATS_SOURCE for h in hits)
        articles = len({h["meta"].get("url") or h["meta"]["title"] for h in hits})
        expects_stats = case["expects_stats"].strip().lower() == "yes"
        row = {
            "question": q,
            "expected_topic": case["expected_topic"],
            "predicted_topic": r["rewrite"]["topic"] or "none",
            "topic_ok": r["rewrite"]["topic"] == case["expected_topic"],
            "language": r["rewrite"].get("language", ""),
            "search_query": r["rewrite"]["query"],
            "sources": len(hits),
            "distinct_articles": articles,
            "stats_chunks": stats,
            "stats_ok": (stats > 0) == expects_stats,
        }
        rows.append(row)
        answers.append(f"## {q}\n\n*{row['search_query']} · topic: {row['predicted_topic']}*\n\n"
                       f"{r['answer']}\n\n**Sources**\n\n```\n{source_list(hits)}\n```\n")
        print(f"  topic {row['predicted_topic']} ({'OK' if row['topic_ok'] else 'WRONG'}) | "
              f"{articles} articles | {stats} stats chunks "
              f"({'OK' if row['stats_ok'] else 'WRONG'})", flush=True)

    if not rows:
        return
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUTPUTS_DIR / "eval_answers.md").write_text("\n".join(answers), encoding="utf-8")

    passed = sum(r["topic_ok"] and r["stats_ok"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed ({len(cases)} in the eval set). Saved outputs/eval_results.csv and outputs/eval_answers.md")


if __name__ == "__main__":
    main()
