import pytest
from app.rag import retrieve, ask

def test_retrieve_head_protection():
    hits = retrieve("head protection", k=5)
    assert len(hits) > 0, "Should retrieve at least one result for head protection"
    sources = [meta["source"] for _, meta in hits]
    assert any("1910.135" in s or "1926.100" in s for s in sources), (
        f"Expected references to 1910.135 or 1926.100, got: {sources}"
    )

def test_retrieve_confined_spaces():
    hits = retrieve("permit-required confined spaces", k=5)
    assert len(hits) > 0, "Should retrieve at least one result for confined spaces"
    sources = [meta["source"] for _, meta in hits]
    assert any("1910.146" in s for s in sources), (
        f"Expected references to 1910.146, got: {sources}"
    )

def test_ask_against_local_ollama():
    res = ask("What standard covers head protection?")
    assert isinstance(res, dict)
    assert "answer" in res and res["answer"]
    assert "sources" in res and len(res["sources"]) > 0
    assert len(res["answer"].strip()) > 0
