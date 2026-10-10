"""
SentinelCFR — Air-Gapped Regulatory Retrieval-Augmented Generation (RAG) Engine.

Zero Runtime Egress Architecture:
- Dense Embeddings: Loaded strictly from local disk (`models/all-MiniLM-L6-v2`) with
  `local_files_only=True` and offline Hugging Face flags enforced.
- Vector Persistence: Queries only local on-disk ChromaDB (`data/chroma`) with telemetry disabled.
- LLM Inference: Connects strictly to local loopback Ollama daemon (`http://localhost:11434/v1`).
- Maintenance vs. Runtime Decoupling: Document ingestion (`ingest()` / `python -m app.rag`) is a
  dedicated maintenance-phase operation and is completely decoupled from runtime user inference.
"""
import os
import glob
from pathlib import Path
from dotenv import load_dotenv

# Enforce offline execution flags before importing Hugging Face / Chroma libraries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1. Local Dense Embedding Engine (Strictly local_files_only=True, zero external downloads)
EMB_PATH = ROOT_DIR / "models" / "all-MiniLM-L6-v2"
if not EMB_PATH.exists():
    EMB_PATH = Path("models/all-MiniLM-L6-v2")
if not EMB_PATH.exists():
    raise FileNotFoundError(
        f"Local embedding model not found at {EMB_PATH}. "
        "Zero-egress constraint active: remote downloads are disabled."
    )

EMB = SentenceTransformer(str(EMB_PATH), local_files_only=True)

# 2. Local On-Disk Vector Store (data/chroma) with anonymized telemetry disabled
CHROMA_PATH = ROOT_DIR / "data" / "chroma"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
COL = chromadb.PersistentClient(
    path=str(CHROMA_PATH),
    settings=Settings(anonymized_telemetry=False),
).get_or_create_collection("safety_docs")

# 3. Local Loopback LLM Client (Ollama on localhost:11434)
llm = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
)
MODEL = os.getenv("LLM_MODEL", "safety-qwen")


def _chunks(text, size=800, overlap=150):
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def _upsert_batch(ids, docs, metas, batch_size=64):
    if not docs:
        return
    try:
        embeddings = EMB.encode(docs, batch_size=batch_size, show_progress_bar=False).tolist()
        COL.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    except Exception as e:
        print(f"Error encoding or upserting batch of size {len(docs)} to Chroma: {e}")
        raise


def ingest(folder="docs", batch_size=64):
    """
    Maintenance-Phase Document Ingestion (`python -m app.rag`).
    Completely decoupled from runtime user inference. Reads local PDFs from `docs/`,
    chunks text, encodes embeddings locally via MiniLM-L6-v2, and upserts into `data/chroma`.
    """
    target_folder = folder
    if not glob.glob(f"{target_folder}/*.pdf") and (ROOT_DIR / folder).exists():
        target_folder = str(ROOT_DIR / folder)
    pdf_files = glob.glob(f"{target_folder}/*.pdf")
    if not pdf_files:
        print(f"No PDF files found in '{folder}'.")
        return 0

    batch_ids = []
    batch_docs = []
    batch_metas = []
    total_ingested = 0

    for path in pdf_files:
        name = os.path.basename(path)
        try:
            reader = PdfReader(path)
            for p, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                for j, c in enumerate(_chunks(text)):
                    c_clean = c.strip()
                    if not c_clean:
                        continue
                    batch_ids.append(f"{name}-{p}-{j}")
                    batch_docs.append(c)
                    batch_metas.append({"source": name, "page": p + 1})

                    if len(batch_docs) >= batch_size:
                        _upsert_batch(batch_ids, batch_docs, batch_metas, batch_size)
                        total_ingested += len(batch_docs)
                        batch_ids.clear()
                        batch_docs.clear()
                        batch_metas.clear()
        except Exception as e:
            print(f"Error processing PDF '{path}': {e}")

    if batch_docs:
        _upsert_batch(batch_ids, batch_docs, batch_metas, batch_size)
        total_ingested += len(batch_docs)
        batch_ids.clear()
        batch_docs.clear()
        batch_metas.clear()

    return total_ingested


def retrieve(q, k=4):
    """Query local on-disk ChromaDB collection (`data/chroma`). Never performs external HTTP calls."""
    total = COL.count()
    if total == 0:
        return []
    n_results = min(k, total)
    r = COL.query(query_embeddings=[EMB.encode(q).tolist()], n_results=n_results)
    if not r["documents"] or not r["documents"][0]:
        return []
    return list(zip(r["documents"][0], r["metadatas"][0]))


SYSTEM_PROMPT = (
    "You are SentinelCFR, an Air-Gapped Industrial Safety & Regulatory Compliance Assistant.\n"
    "Answer strictly from the provided local safety documentation context while addressing dual user needs:\n"
    "1. Regulatory Precision: Always quote exact OSHA standard numbers (e.g., 29 CFR 1910.135) present in the context "
    "and cite exact document sources in the format [file p.N].\n"
    "2. Concept Explanation: If a worker asks for clarification or an explanation (e.g., 'why is this hazardous?', "
    "'explain simply'), provide a clear, practical workplace safety explanation grounded in the context, followed by "
    "the formal legal reference and [file p.N] citation.\n"
    "3. Grounding: Maintain strict hallucination guardrails—never invent standard numbers or citations not present in "
    "the context. If a user asks a question not covered by the indexed documents in docs/, state clearly that the "
    "required documents are not present in the local index."
)


def ask(q):
    """Runtime inference against local ChromaDB and loopback Ollama LLM. Zero external network egress."""
    hits = retrieve(q)
    if not hits:
        return {"answer": "No safety documentation is currently indexed in the local knowledge base.", "sources": []}
    ctx = "\n\n".join(f"[{m['source']} p.{m['page']}] {d}" for d, m in hits)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}
    ]
    try:
        out = llm.chat.completions.create(model=MODEL, messages=msgs, temperature=0)
    except Exception as e:
        if "not found" in str(e).lower():
            fallback_model = "safety-qwen" if MODEL != "safety-qwen" else "qwen2.5:7b"
            out = llm.chat.completions.create(model=fallback_model, messages=msgs, temperature=0)
        else:
            raise
    return {"answer": out.choices[0].message.content, "sources": [m for _, m in hits]}


if __name__ == "__main__":
    print(f"Starting maintenance-phase PDF ingestion from docs/ (current collection count: {COL.count()})...")
    count = ingest()
    print(f"Ingestion complete. Total indexed chunks: {COL.count()} (ingested in this run: {count}).")