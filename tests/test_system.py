import json

import pytest

import multi_agent_system as mas
from config import WORKFLOW_SETTINGS
from multi_agent_system import JSONOutputParser, extract_score, to_content
from offline import HashingEmbeddings, ScriptedLLM


def test_to_content_is_valid_json_for_dicts():
    # str(dict) would give single quotes, which json.loads (used by the UI) rejects
    assert json.loads(to_content({"a": "b", "n": 1})) == {"a": "b", "n": 1}


def test_parser_extracts_json_from_chatty_reply():
    assert JSONOutputParser().parse('Sure!\n{"score": 7}\nHope that helps') == {"score": 7}


def test_parser_falls_back_to_content_when_not_json():
    assert JSONOutputParser().parse("just text") == {"content": "just text"}
    assert JSONOutputParser().parse("{broken json}") == {"content": "{broken json}"}


@pytest.mark.parametrize("raw,expected", [
    ('{"score": 7}', 7.0), ('{"score": "8.5"}', 8.5), ('{"x": 1}', None), ("not json", None),
])
def test_extract_score(raw, expected):
    assert extract_score(raw) == expected


def test_embeddings_are_deterministic_and_normalised():
    e = HashingEmbeddings()
    a, b = e.embed_query("privacy of student data"), e.embed_query("privacy of student data")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-5


def test_rag_retrieves_the_relevant_chunk(system):
    docs = system.rag_agent.retrieve_relevant_docs("student privacy and data security", k=1)
    assert docs[0].metadata["source"] == "challenges"
    docs = system.rag_agent.retrieve_relevant_docs("volcanic eruption magma", k=1)
    assert docs[0].metadata["source"] == "volcano"


def test_rag_agent_returns_json_and_sources(system):
    r = system.run_single_agent("rag", "student privacy")
    assert "answer" in json.loads(r.content)
    assert r.metadata["retrieved_docs_count"] == 3
    assert "challenges" in r.metadata["retrieved_sources"]


def test_each_agent_returns_json(system):
    for name in ("rag", "research", "writer", "critic"):
        r = system.run_single_agent(name, "AI in education", "some context. " * 5)
        assert isinstance(json.loads(r.content), dict), name


def test_unknown_agent_rejected(system):
    with pytest.raises(ValueError):
        system.run_single_agent("nope", "x")


def test_rag_status_reports_embedding_class(system):
    st = system.get_rag_status()
    assert st["has_vectorstore"] and st["embeddings_model"] == "HashingEmbeddings"


def test_workflow_revises_until_threshold(system):
    result = system.run_workflow("AI in education")
    meta = result["review"].metadata
    assert meta["scores_by_round"] == [6.5, 8.5]
    assert meta["met_threshold"] is True
    assert "revision_note" in json.loads(result["content"].content)
    assert result["workflow_status"] == "completed"
    assert {"rag", "research", "content", "review"} <= set(result)


def test_workflow_respects_disabled_revision(system, monkeypatch):
    monkeypatch.setitem(WORKFLOW_SETTINGS, "enable_iterative_improvement", False)
    meta = system.run_workflow("AI in education")["review"].metadata
    assert meta["revision_rounds"] == 1 and meta["met_threshold"] is False


def test_workflow_stops_at_max_iterations(system, monkeypatch):
    monkeypatch.setitem(WORKFLOW_SETTINGS, "quality_threshold", 10.0)
    monkeypatch.setitem(WORKFLOW_SETTINGS, "max_iterations", 2)
    meta = system.run_workflow("AI in education")["review"].metadata
    assert meta["revision_rounds"] == 2 and meta["met_threshold"] is False


def test_workflow_without_documents_skips_rag():
    s = mas.MultiAgentSystem(llm=ScriptedLLM(), embeddings=HashingEmbeddings())
    assert s.run_workflow("anything")["rag"] is None


def test_create_system_offline_switch(monkeypatch):
    monkeypatch.setenv("MULTI_AGENT_OFFLINE", "1")
    assert isinstance(mas.create_system().coordinator.llm, ScriptedLLM)
