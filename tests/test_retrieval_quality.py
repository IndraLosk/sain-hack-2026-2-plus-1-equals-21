from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.agent import (
    _heuristic_translate,
    prepare_retrieval_queries,
    retriever_node,
)
from backend.llm import get_llm
from backend.vectorstore import RetrievedChunk, VectorStore
from scripts.evaluate_case4 import load_attack_cases

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "case4-dev.json"
ATTACKS_PATH = ROOT / "new_attacks.xlsx"


@pytest.fixture(scope="module")
def case4_store(tmp_path_factory) -> tuple[VectorStore, dict]:
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    store = VectorStore(persist_dir=str(tmp_path_factory.mktemp("retrieval") / "bm25"))
    for article in raw["dataset"]["articles"]:
        sections = [
            {
                "section_id": str(section["section_id"]),
                "title": str(section.get("title") or ""),
                "text": str(section.get("text") or ""),
            }
            for section in article["sections"]
            if section.get("text")
        ]
        store.add_article(
            program_id="sleep_recovery",
            title=article["title"],
            published_date=f"{article.get('year') or 2020}-01-01",
            content="\n\n".join(section["text"] for section in sections),
            article_id=article["article_id"],
            sections=sections,
        )
    return store, raw


def _top1(store: VectorStore, question: str) -> tuple[str, list[RetrievedChunk]]:
    query = _heuristic_translate(question)
    docs = store.search(query, program_id="sleep_recovery", top_k=10)[:1] if query else []
    if not docs:
        docs = store.search(question, program_id="sleep_recovery", top_k=10)[:1]
    return query, docs


def test_dev_retrieval_top1_is_allowed_for_every_group(case4_store) -> None:
    store, raw = case4_store
    references = {row["record_id"]: row for row in raw["reference"]["requests"]}
    groups = {row["group_id"]: row for row in raw["dataset"]["requests"]}
    failures = []
    for group_id, request in groups.items():
        query, docs = _top1(store, request["utterance"]["text"])
        allowed = set(references[request["record_id"]]["allowed_article_ids"])
        if not docs or docs[0].article_id not in allowed:
            failures.append((group_id, query, [doc.article_id for doc in docs], sorted(allowed)))
    assert not failures


def test_additional_attack_retrieval_top1_is_allowed(case4_store) -> None:
    if not ATTACKS_PATH.exists():
        pytest.skip("new_attacks.xlsx не входит в репозиторий")
    store, _ = case4_store
    failures = []
    cases = load_attack_cases(ATTACKS_PATH)
    for case in cases:
        query, docs = _top1(store, case.text)
        if not docs or docs[0].article_id not in case.allowed_article_ids:
            failures.append((case.record_id, query, [doc.article_id for doc in docs], case.allowed_article_ids))
    assert len(cases) == 40
    assert not failures


@pytest.mark.asyncio
async def test_retriever_returns_only_best_source(monkeypatch) -> None:
    docs = [
        RetrievedChunk(
            document=f"fact {index}",
            metadata={"article_id": f"art_{index}", "section_id": "s001", "title": "title"},
            distance=index / 10,
        )
        for index in range(3)
    ]

    async def fake_search(*args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        return docs

    monkeypatch.setattr("backend.agent.search_async", fake_search)
    state = {
        "messages": [{"role": "user", "content": "Как свет влияет на сон?"}],
        "program_id": "sleep_recovery",
        "retrieval_query": "light sleep",
    }
    result = await retriever_node(state)
    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["article_id"] == "art_0"
    assert result["no_evidence"] is False


@pytest.mark.asyncio
async def test_batch_query_normalization_preserves_safe_override(monkeypatch) -> None:
    llm = get_llm()
    monkeypatch.setattr(llm, "api_key", "test-key")
    calls = []

    async def fake_chat(messages: list[dict[str, str]], **kwargs: Any) -> str:
        calls.append(messages)
        return json.dumps(
            {
                "queries": [
                    {"id": 0, "query": "alcohol sleep quality"},
                    {"id": 1, "query": "melatonin dosage"},
                ]
            }
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    records = [
        {"utterance": {"text": "Почему после вина хуже сон?"}},
        {"utterance": {"text": "Почему после вина хуже сон?"}},
        {"utterance": {"text": "Назови дозу мелатонина"}},
    ]
    queries = await prepare_retrieval_queries(records)
    assert len(calls) == 1
    assert queries["Почему после вина хуже сон?"] == "alcohol sleep quality"
    assert queries["Назови дозу мелатонина"] == "evening light sleep"
