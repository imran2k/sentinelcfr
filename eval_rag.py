import json
import re
from pathlib import Path
from app import rag

ROOT_DIR = Path(__file__).resolve().parent
QA_FILE = ROOT_DIR / "data" / "eval_qa.json"
RESULTS_FILE = ROOT_DIR / "data" / "eval_results.json"

def evaluate():
    if not QA_FILE.exists():
        raise FileNotFoundError(f"Benchmark file {QA_FILE} not found.")

    with open(QA_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 1. Retrieval Precision Test (all 10 questions)
    retrieval_results = []
    hits_count = 0

    print(f"Running retrieval test on {len(eval_data)} queries...")
    for item in eval_data:
        q = item["query"]
        expected = item["expected_standard"]
        retrieved = rag.retrieve(q, k=4)
        sources = [m.get("source", "") for _, m in retrieved]
        is_hit = any(expected in src for src in sources)
        if is_hit:
            hits_count += 1
        retrieval_results.append({
            "query": q,
            "expected_standard": expected,
            "retrieved_sources": sources,
            "hit": is_hit
        })

    retrieval_precision = round(hits_count / len(eval_data), 4)

    # 2. Citation & Answer Test (first 2 queries)
    citation_results = []
    citation_hits = 0
    evaluated_queries = eval_data[:2]

    print(f"Running citation & answer test on {len(evaluated_queries)} queries against local Ollama...")
    for item in evaluated_queries:
        q = item["query"]
        ans_obj = rag.ask(q)
        ans_text = ans_obj.get("answer", "") if isinstance(ans_obj, dict) else str(ans_obj)
        has_citation = bool(re.search(r"\[[^\]]*(?:p\.\s*\d+|29_CFR|OSHA|CFR)[^\]]*\]", ans_text, re.IGNORECASE)) or bool(re.search(r"\[.+?\]", ans_text))
        if has_citation:
            citation_hits += 1
        citation_results.append({
            "query": q,
            "answer": ans_text,
            "has_citation": has_citation
        })

    citation_rate = round(citation_hits / len(evaluated_queries), 4) if evaluated_queries else 0.0

    summary = {
        "total_queries": len(eval_data),
        "retrieval_hits": hits_count,
        "retrieval_precision": retrieval_precision,
        "retrieval_precision_pct": round(retrieval_precision * 100, 2),
        "citation_evaluated_count": len(evaluated_queries),
        "citation_hits": citation_hits,
        "citation_rate": citation_rate,
        "citation_rate_pct": round(citation_rate * 100, 2),
        "retrieval_details": retrieval_results,
        "citation_details": citation_results
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Results Summary ===")
    print(f"Retrieval Precision: {summary['retrieval_precision_pct']}% ({hits_count}/{len(eval_data)})")
    print(f"Citation Rate: {summary['citation_rate_pct']}% ({citation_hits}/{len(evaluated_queries)})")
    print(f"Saved evaluation metrics to {RESULTS_FILE}")
    return summary

if __name__ == "__main__":
    evaluate()
