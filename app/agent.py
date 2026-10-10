"""
SentinelCFR — Air-Gapped Regulatory Compliance Agent.

Zero Runtime Egress Architecture:
- Tool calls (`search_rules`) query only the local ChromaDB vector index via `app.rag.retrieve()`.
- LLM inference targets strictly the local loopback Ollama endpoint (`http://localhost:11434/v1`).
- Document ingestion (`python -m app.rag`) is a maintenance-phase step decoupled from runtime inference.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

from app.rag import retrieve, llm

MODEL = os.getenv("LLM_MODEL", "safety-qwen")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_rules",
        "description": "Search the local indexed safety documents for regulatory rules about a topic.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}]

PPE_REGULATION_MAPPING = {
    "goggles": "29 CFR 1910.133 eye face protection",
    "goggle": "29 CFR 1910.133 eye face protection",
    "eye": "29 CFR 1910.133 eye and face protection",
    "helmet": "29 CFR 1910.135 1926.100 head protection hard hat",
    "head": "29 CFR 1910.135 1926.100 head protection",
    "footwear": "29 CFR 1910.136 foot protection safety shoes boots",
    "shoes": "29 CFR 1910.136 foot protection safety shoes",
    "boots": "29 CFR 1910.136 foot protection",
    "gloves": "29 CFR 1910.138 hand protection chemical resistant gloves",
    "glove": "29 CFR 1910.138 hand protection",
    "hand": "29 CFR 1910.138 hand protection",
    "mask": "29 CFR 1910.134 respiratory protection",
    "vest": "high visibility safety apparel 1926.201",
}


def _search(query):
    """Query local ChromaDB vector index only. Never executes external network requests."""
    q_lower = query.lower()
    expanded_terms = []
    for kw, std in PPE_REGULATION_MAPPING.items():
        if kw in q_lower and std not in expanded_terms:
            expanded_terms.append(std)
    search_q = f"{query} {' '.join(expanded_terms)}".strip() if expanded_terms else query
    return [{"text": d[:500], "source": m} for d, m in retrieve(search_q, 3)]


def write_report(violations, max_steps=5):
    """Run local multi-step compliance evaluation loop against loopback Ollama."""
    system_prompt = (
        "You are SentinelCFR, an Air-Gapped Industrial Safety & Regulatory Compliance Agent. "
        "For each reported violation or incident, call search_rules to retrieve the relevant rule from the local index, "
        "then write a structured incident evaluation report in markdown.\n\n"
        "Core Operating Directives:\n"
        "1. Regulatory Precision: Quote exact standard numbers (e.g., 29 CFR 1910.135, 29 CFR 1910.133, "
        "29 CFR 1910.136, 29 CFR 1910.138) and cite exact document sources like [file p.N].\n"
        "2. Concept Explanation: Provide a clear, practical explanation of why the violation is hazardous to the worker "
        "before or alongside the legal reference and corrective action.\n"
        "3. Grounding: Maintain strict hallucination guardrails—never invent standard numbers or citations not present "
        "in the retrieved local context. If a topic is not covered by the indexed documents, state clearly that the "
        "documents are not present in the local index.\n\n"
        "Format the output for each violation with the following sections:\n"
        "- **Violation**\n"
        "- **Applicable Regulation** (quote exact 29 CFR standard from context)\n"
        "- **Source Citation** (exact [file p.N] reference)\n"
        "- **Hazard Explanation & Mandatory Corrective Action** (practical safety explanation + required action)\n\n"
        "If there are no violations, state that no violations were detected."
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Violations detected: " + json.dumps(violations)}
    ]
    for _ in range(max_steps):
        try:
            m = llm.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS).choices[0].message
        except Exception as e:
            if "not found" in str(e).lower():
                fallback_model = "safety-qwen" if MODEL != "safety-qwen" else "qwen2.5:7b"
                m = llm.chat.completions.create(model=fallback_model, messages=msgs, tools=TOOLS).choices[0].message
            else:
                raise
        msgs.append(m)
        if not m.tool_calls:
            return m.content or "No violations detected."
        for t in m.tool_calls:
            args = json.loads(t.function.arguments)
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(_search(args["query"]))})
    return "Stopped: step limit reached."