"""
SentinelCFR — Air-Gapped Gradio Operator Interface.

All user queries and incident evaluations execute strictly against the local ChromaDB index
(`data/chroma`) and loopback Ollama server (`http://localhost:11434/v1`).
Document ingestion (`python -m app.rag`) is a separate maintenance-phase operation.
"""
import json
import gradio as gr
from app import rag, agent


def ask(q):
    if not q or not q.strip():
        return "Please enter a safety or regulatory question."
    r = rag.ask(q)
    src = "\n".join(f"- `{m['source']}` (p.{m['page']})" for m in r["sources"])
    return f"{r['answer']}\n\n**Sources retrieved:**\n{src}"


def report(text):
    try:
        payload = json.loads(text)
        return agent.write_report(payload)
    except json.JSONDecodeError:
        return 'Paste valid JSON, e.g. `[{"person": 0, "missing": "helmet"}]`'


with gr.Blocks(title="SentinelCFR") as demo:
    gr.Markdown(
        "# SentinelCFR: Air-Gapped Regulatory Compliance Assistant\n"
        "100% On-Premise Regulatory Question Answering and Incident Compliance Evaluation."
    )
    with gr.Tab("Safety Rulebook & Assistant"):
        q = gr.Textbox(
            label="Regulatory Question or Hazard Clarification",
            placeholder="e.g., What are the OSHA requirements for head protection under 1910.135, and why is working without a hard hat hazardous?",
            lines=2
        )
        ask_btn = gr.Button("Ask Safety Assistant", variant="primary")
        a = gr.Markdown(label="Compliance Answer & Citations")
        q.submit(ask, q, a)
        ask_btn.click(ask, q, a)
        gr.Examples(
            examples=[
                ["What are the OSHA requirements for industrial protective helmets under 29 CFR 1910.135?"],
                ["Why is working in a permit-required confined space hazardous, and what does 29 CFR 1910.146 require?"],
                ["Explain simply when eye and face protection is required under 29 CFR 1910.133."],
                ["What procedures govern lockout/tagout energy control under 29 CFR 1910.147?"],
            ],
            inputs=q
        )

    with gr.Tab("Incident Evaluation"):
        v = gr.Textbox(
            label="Structured Safety Violation Payload (JSON)",
            value='[{"person": 0, "missing": "helmet"}, {"person": 1, "missing": "goggles"}]',
            lines=4
        )
        b = gr.Button("Evaluate Incident & Generate Report", variant="primary")
        r = gr.Markdown(label="Regulatory Incident Evaluation Report")
        b.click(report, v, r)
        gr.Examples(
            examples=[
                ['[{"person": 0, "missing": "helmet"}]'],
                ['[{"person": 0, "missing": "goggles"}]'],
                ['[{"person": 0, "missing": "footwear"}, {"person": 1, "missing": "gloves"}]'],
            ],
            inputs=v
        )

if __name__ == "__main__":
    demo.launch()