from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import HTTPException, Request

from backend import hackapi
from backend.agent import AgentRuntime
from backend.config import get_settings
from backend.llm import OpenAICompatLLM, get_llm
from backend.main import _case_chat_history, _validate_case4_answers, run_case4
from backend.vectorstore import RetrievedChunk, VectorStore


def _doc(article_id: str = "art_14", section_id: str = "s013") -> RetrievedChunk:
    return RetrievedChunk(
        document="Evening light affects sleep.",
        metadata={
            "article_id": article_id,
            "section_id": section_id,
            "section_title": "Evening light recommendations",
            "title": "Light and sleep",
            "published_date": "2024-01-01",
            "added_at": 0,
        },
        distance=0.0,
    )


@pytest.mark.asyncio
async def test_hidden_dataset_never_falls_back_to_dev(monkeypatch) -> None:
    async def no_remote(*args: Any, **kwargs: Any) -> None:
        return None

    def forbidden_local() -> dict:
        raise AssertionError("скрытый набор не должен подменяться dev")

    monkeypatch.setattr(hackapi, "_fetch_remote", no_remote)
    monkeypatch.setattr(hackapi, "local_dataset", forbidden_local)
    monkeypatch.setattr(hackapi, "_cache", {"data": None, "at": 0.0, "source": ""})
    result = await hackapi.fetch_dataset(token="secret", dataset_code="hidden", allow_local_fallback=False)
    assert result == {}


@pytest.mark.asyncio
async def test_dev_dataset_keeps_local_fallback(monkeypatch) -> None:
    async def no_remote(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(hackapi, "_fetch_remote", no_remote)
    monkeypatch.setattr(hackapi, "local_dataset", lambda: {"requests": [], "articles": []})
    monkeypatch.setattr(hackapi, "_cache", {"data": None, "at": 0.0, "source": ""})
    result = await hackapi.fetch_dataset(token="secret", dataset_code="case4-dev-v2", allow_local_fallback=True)
    assert result == {"requests": [], "articles": []}


def _small_dataset(dataset_id: str, article_id: str) -> dict:
    return {
        "meta": {"dataset_id": dataset_id},
        "articles": [
            {
                "article_id": article_id,
                "title": f"Article {article_id}",
                "year": 2024,
                "authors": [],
                "license": {"name": "CC BY"},
                "sections": [{"section_id": "s001", "title": "Sleep", "text": "Sleep hygiene evidence."}],
            }
        ],
    }


@pytest.mark.asyncio
async def test_index_is_rebuilt_for_actual_dataset(monkeypatch, tmp_path) -> None:
    store = VectorStore(persist_dir=str(tmp_path / "bm25"))
    monkeypatch.setattr("backend.vectorstore.get_store", lambda: store)
    monkeypatch.setattr(hackapi, "_indexed_dataset_key", None)

    await hackapi.ensure_dataset_indexed(dataset=_small_dataset("hidden-1", "art_one"), dataset_code="hidden-1")
    assert store.article_ids() == {"art_one"}
    await hackapi.ensure_dataset_indexed(dataset=_small_dataset("hidden-2", "art_two"), dataset_code="hidden-2")
    assert store.article_ids() == {"art_two"}


@pytest.mark.asyncio
async def test_group_runtime_uses_one_llm_call_and_common_sources(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHECKPOINTS_DB", str(tmp_path / "cp.sqlite"))
    get_settings.cache_clear()
    monkeypatch.setattr("backend.article_store.record_used_chunks", lambda **kwargs: 0)

    async def fake_search(*args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        return [_doc()]

    monkeypatch.setattr("backend.agent.search_async", fake_search)
    calls = []

    async def fake_chat(messages: list[dict[str, str]], **kwargs: Any) -> str:
        calls.append(messages)
        return json.dumps(
            {
                "answers": {
                    expert_id: "Я не называю дозировку. Безопасный шаг: уменьшите вечерний свет по источнику."
                    for expert_id in ("clinician", "biohacker", "nutritionist")
                }
            }
        )

    monkeypatch.setattr(get_llm(), "chat", fake_chat)
    runtime = AgentRuntime()
    await runtime.startup()
    try:
        result = await runtime.run_group(
            session_id="group-test",
            program_id="sleep_recovery",
            expert_ids=["clinician", "biohacker", "nutritionist"],
            message="Назови дозу мелатонина",
            retrieval_query="evening light sleep",
        )
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()
    assert len(calls) == 1
    assert set(result["answers"]) == {"clinician", "biohacker", "nutritionist"}
    assert result["is_unsafe_request"] is True
    assert result["sources"][0]["article_id"] == "art_14"
    assert result["sources"][0]["section_id"] == "s013"


def test_case_chat_history_filters_user_time_and_roles() -> None:
    dataset = {
        "chat_history": [
            {"user_id": "u1", "role": "user", "at": "2026-01-01T10:00:00+03:00", "text": "Первый вопрос"},
            {"user_id": "u1", "role": "agent", "at": "2026-01-01T10:01:00+03:00", "text": "Первый ответ"},
            {"user_id": "u2", "role": "user", "at": "2026-01-01T10:02:00+03:00", "text": "Чужой вопрос"},
            {"user_id": "u1", "role": "user", "at": "2026-01-01T12:00:00+03:00", "text": "Будущий вопрос"},
        ]
    }
    record = {"user_id": "u1", "utterance": {"at": "2026-01-01T11:00:00+03:00", "text": "Текущий"}}
    history = _case_chat_history(dataset, record)
    assert history == [
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Первый ответ"},
    ]


def test_no_evidence_does_not_invalidate_other_records() -> None:
    records = [{"record_id": "r1"}, {"record_id": "r2"}]
    answers = [
        {"record_id": "r1", "text": "В базе нет подтверждений", "refused": False, "citations": []},
        {
            "record_id": "r2",
            "text": "Ответ",
            "refused": False,
            "citations": [{"article_id": "art_07", "section_id": "s014"}],
        },
    ]
    _validate_case4_answers(records, answers)
    answers[0]["citations"] = [{"article_id": "", "section_id": "s014"}]
    with pytest.raises(HTTPException, match="citations"):
        _validate_case4_answers(records, answers)


@pytest.mark.asyncio
async def test_run_has_internal_deadline(monkeypatch) -> None:
    async def slow_impl(request: Request):
        await asyncio.sleep(1)
        raise AssertionError("deadline не сработал")

    monkeypatch.setattr("backend.main._run_case4_impl", slow_impl)
    monkeypatch.setattr(get_settings(), "run_timeout_seconds", 0.001)
    request = Request({"type": "http", "method": "POST", "path": "/run", "headers": []})
    with pytest.raises(HTTPException) as error:
        await run_case4(request)
    assert error.value.status_code == 504


@pytest.mark.asyncio
async def test_usage_is_collected_from_successful_response(monkeypatch) -> None:
    class Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

    class Client:
        async def post(self, *args: Any, **kwargs: Any) -> Response:
            return Response()

    async def no_limit() -> None:
        return None

    monkeypatch.setattr("backend.llm.get_http_client", lambda: Client())
    monkeypatch.setattr("backend.llm._wait_for_request_slot", no_limit)
    llm = OpenAICompatLLM(provider="openai")
    monkeypatch.setattr(llm, "api_key", "test")
    assert await llm.chat([{"role": "user", "content": "test"}]) == "ok"
    assert llm.usage == {"requests": 1, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
