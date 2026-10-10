"""
SentinelCFR — FastAPI REST Interface.

All endpoints execute 100% locally against `data/chroma` and loopback Ollama (`localhost:11434`).
Document ingestion (`python -m app.rag`) is a maintenance-phase operation decoupled from API inference.
"""
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app import rag, agent

app = FastAPI(
    title="SentinelCFR: Air-Gapped Industrial Safety Regulatory Engine",
    description="Zero-Egress On-Premise Regulatory RAG & Compliance Evaluation API",
    version="1.0.0"
)


class ViolationItem(BaseModel):
    person: int = Field(default=0, description="Worker or incident subject index")
    missing: str = Field(default="helmet", description="Missing safety equipment or compliance item")
    box: Optional[list[float]] = Field(default=[10.0, 10.0, 50.0, 100.0], description="Optional bounding box [x1, y1, x2, y2]")
    confidence: Optional[float] = Field(default=0.9, description="Optional confidence score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "person": 0,
                "missing": "helmet",
                "box": [10.0, 10.0, 50.0, 100.0],
                "confidence": 0.9
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask(q: str = "What are the OSHA requirements for head protection under 29 CFR 1910.135, and why is it hazardous to work without a helmet?"):
    return rag.ask(q)


@app.post("/report")
def report(violations: list[ViolationItem]):
    viols_data = [v.model_dump() for v in violations]
    return {"report": agent.write_report(viols_data)}