import json
from pathlib import Path

RESULTS_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"

def test_eval_results_exists():
    assert RESULTS_FILE.exists(), f"Evaluation results file not found at {RESULTS_FILE}"

def test_retrieval_precision_threshold():
    assert RESULTS_FILE.exists(), f"Evaluation results file not found at {RESULTS_FILE}"
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    precision = data.get("retrieval_precision")
    assert precision is not None, "Missing 'retrieval_precision' key in evaluation results"

    precision_pct = precision * 100.0 if precision <= 1.0 else float(precision)
    assert precision_pct >= 70.0, (
        f"Retrieval Precision {precision_pct:.1f}% failed to meet the >= 70.0% threshold"
    )
