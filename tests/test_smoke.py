"""Smoke-тесты на восстановленный код (ideas/09 п. 16).

Проверяем то, что нельзя проверить глазами: каталог, health, стрим с заглушкой LLM,
мульти-стрим и правила «источники общие, различается только стиль».

Модель НЕ вызывается: `get_llm().stream_chat` подменяется заглушкой,
поэтому тесты не тратят токены.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, AsyncIterator, Dict, List

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from backend.agent import (  # noqa: E402
    LLM_FAILED_PHRASE,
    STYLE_VARIANTS,
    AgentRuntime,
    _fallback_answer,
    style_variant_line,
)
from backend.config import get_catalog, get_settings  # noqa: E402
from backend.llm import get_llm, heuristic_guardrail  # noqa: E402


# --------------------------------------------------------------------------- #
# Каталог и конфигурация
# --------------------------------------------------------------------------- #
def test_catalog_is_valid() -> None:
    catalog = get_catalog()
    assert catalog.programs, "каталог должен содержать программы"
    assert catalog.experts, "каталог должен содержать экспертов"
    assert len({e.id for e in catalog.experts}) == len(catalog.experts)
    # доступ по id и фолбэк на первый элемент
    assert catalog.expert(catalog.experts[0].id).id == catalog.experts[0].id
    assert catalog.expert("нет-такого").id == catalog.experts[0].id


def test_settings_provider_switch() -> None:
    settings = get_settings()
    assert settings.provider in {"deepseek", "openai"}
    assert settings.active_model
    assert settings.active_base_url and not settings.active_base_url.endswith("/")
    assert settings.retrieval_top_k_multi >= settings.retrieval_top_k


# --------------------------------------------------------------------------- #
# Guardrail без LLM
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,unsafe",
    [
        ("плохо засыпаю, что делать", False),
        ("сколько мг мелатонина выпить", True),
        ("поставь диагноз по описанию", True),
        ("пропиши препарат от бессонницы", True),
    ],
)
def test_heuristic_guardrail(text: str, unsafe: bool) -> None:
    assert heuristic_guardrail(text).is_unsafe_request is unsafe


# --------------------------------------------------------------------------- #
# Fallback-ответ: только русский, без обрыва и дубля цитаты
# --------------------------------------------------------------------------- #
def test_fallback_is_russian_and_not_truncated() -> None:
    docs = [
        {
            "title": "Dose and timing effects of caffeine on subsequent sleep",
            "published_date": "2025-03-01",
            "document": "lity when consumed 4 hours prior to bedtime, but not when consumed 8 hours...",
        }
    ]
    answer = _fallback_answer(docs, unsafe=False)
    assert answer.startswith(LLM_FAILED_PHRASE)
    assert "lity when consumed" not in answer, "английский сниппет не должен попадать в ответ"
    assert answer.count("Dose and timing effects") == 1, "цитата не должна дублироваться"
    assert "Безопасный шаг" in answer


def test_fallback_without_docs() -> None:
    answer = _fallback_answer([], unsafe=False)
    assert LLM_FAILED_PHRASE in answer
    assert "Безопасный шаг" in answer


def test_style_variants_are_distinct() -> None:
    lines = {style_variant_line(i) for i in range(len(STYLE_VARIANTS))}
    assert len(lines) == len(STYLE_VARIANTS)
    assert style_variant_line(None) == ""


# --------------------------------------------------------------------------- #
# Runtime: стрим с заглушкой LLM (без вызова модели)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def runtime(tmp_path, monkeypatch) -> AgentRuntime:
    """Runtime с временными БД и заглушкой LLM."""
    monkeypatch.setenv("CHECKPOINTS_DB", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "app.sqlite"))
    get_settings.cache_clear()

    rt = AgentRuntime()
    asyncio.get_event_loop_policy().new_event_loop()
    return rt


async def _fake_stream(messages: List[Dict[str, str]], **_: Any) -> AsyncIterator[str]:
    yield "Ответ "
    yield "эксперта."


def _stub_retrieval(monkeypatch) -> None:
    from backend.vectorstore import RetrievedChunk

    doc = RetrievedChunk(
        document="Evening light exposure can affect sleep and circadian timing.",
        metadata={
            "article_id": "art_test",
            "section_id": "s001",
            "title": "Light and sleep",
            "published_date": "2024-01-01",
            "added_at": 0,
        },
        distance=0.0,
    )

    async def fake_search(*args: Any, **kwargs: Any) -> List[RetrievedChunk]:
        return [doc]

    monkeypatch.setattr("backend.agent.search_async", fake_search)


async def _collect(runtime: AgentRuntime, coro_factory) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    async for event in coro_factory():
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_single_stream_with_stub_llm(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHECKPOINTS_DB", str(tmp_path / "cp.sqlite"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "app.sqlite"))
    get_settings.cache_clear()

    from backend.storage import init_db

    await init_db()
    rt = AgentRuntime()
    await rt.startup()
    _stub_retrieval(monkeypatch)
    monkeypatch.setattr(get_llm(), "stream_chat", _fake_stream)
    try:
        events = await _collect(
            rt,
            lambda: rt.stream(
                session_id="t-single",
                program_id="sleep_recovery",
                expert_id="clinician",
                message="плохо засыпаю, яркий свет вечером",
            ),
        )
    finally:
        await rt.shutdown()

    types = [e["type"] for e in events]
    assert types[0] == "safety_flag"
    assert "sources" in types and "meta" in types
    assert types[-1] == "done"
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text  # ответ не пустой


@pytest.mark.asyncio
async def test_multi_stream_shares_sources_and_varies_style(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHECKPOINTS_DB", str(tmp_path / "cp.sqlite"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "app.sqlite"))
    get_settings.cache_clear()

    from backend.storage import init_db

    await init_db()
    rt = AgentRuntime()
    await rt.startup()
    _stub_retrieval(monkeypatch)

    captured: List[str] = []

    async def capture_stream(messages: List[Dict[str, str]], **_: Any) -> AsyncIterator[str]:
        captured.append(messages[0]["content"])
        yield "ok"

    monkeypatch.setattr(get_llm(), "stream_chat", capture_stream)
    expert_ids = [e.id for e in get_catalog().experts]
    try:
        events = await _collect(
            rt,
            lambda: rt.stream_multi(
                session_id="t-multi",
                program_id="sleep_recovery",
                expert_ids=expert_ids,
                message="плохо засыпаю, яркий свет вечером",
            ),
        )
    finally:
        await rt.shutdown()

    types = [e["type"] for e in events]
    # источники приходят ровно один раз — они общие для всех экспертов
    assert types.count("sources") == 1
    assert types.count("expert_start") == len(expert_ids)
    assert types.count("expert_done") == len(expert_ids)
    assert types[-1] == "done"

    # каждый эксперт получил свой промпт: факты одинаковые, персона разная
    assert len(captured) == len(expert_ids)
    import re

    facts = {re.search(r"\[FACTS_ONLY\](.*?)\[FACTS_ONLY_END\]", p, re.S).group(1).strip() for p in captured}
    personas = {re.search(r"\[PERSONA_STYLE\](.*?)\[/PERSONA_STYLE\]", p, re.S).group(1).strip() for p in captured}
    assert len(facts) == 1, "блок [FACTS_ONLY] должен быть одинаковым у всех"
    assert len(personas) == len(expert_ids), "стиль должен различаться у всех"


@pytest.mark.asyncio
async def test_multi_stream_survives_llm_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHECKPOINTS_DB", str(tmp_path / "cp.sqlite"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "app.sqlite"))
    get_settings.cache_clear()

    from backend.storage import init_db

    await init_db()
    rt = AgentRuntime()
    await rt.startup()
    _stub_retrieval(monkeypatch)

    async def broken_stream(messages: List[Dict[str, str]], **_: Any) -> AsyncIterator[str]:
        raise RuntimeError("LLM down")
        yield ""  # pragma: no cover

    monkeypatch.setattr(get_llm(), "stream_chat", broken_stream)
    try:
        events = await _collect(
            rt,
            lambda: rt.stream_multi(
                session_id="t-multi-fail",
                program_id="sleep_recovery",
                expert_ids=["clinician", "biohacker"],
                message="плохо засыпаю, яркий свет вечером",
            ),
        )
    finally:
        await rt.shutdown()

    warnings = [e for e in events if e["type"] == "warning"]
    assert warnings, "при падении LLM должно приходить событие warning"
    assert all(w["message"] == LLM_FAILED_PHRASE for w in warnings)
    texts = "".join(e["content"] for e in events if e["type"] == "token")
    assert LLM_FAILED_PHRASE in texts
    assert [e["type"] for e in events][-1] == "done"


# --------------------------------------------------------------------------- #
# API: каталог и health
# --------------------------------------------------------------------------- #
def test_api_catalog_and_health() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        catalog = client.get("/api/v1/catalog")
        assert catalog.status_code == 200
        body = catalog.json()
        assert body["experts"] and body["programs"]

        health = client.get("/api/v1/health")
        assert health.status_code == 200
        payload = health.json()
        assert "llm" in payload and "vector_store" in payload and "guardrail" in payload
        assert payload["llm"]["model"]
        assert "degraded" in payload["guardrail"] and "attempts" in payload["guardrail"]

        config = client.get("/api/v1/config")
        assert config.status_code == 200
        assert config.json()["max_message_length"] > 0


def test_index_is_not_cached() -> None:
    """index.html не должен кэшироваться — иначе после сборки виден старый UI."""
    from fastapi.testclient import TestClient

    from backend.main import FRONTEND_DIST, app

    if not FRONTEND_DIST.exists():
        pytest.skip("frontend/dist не собран")
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "no-cache" in resp.headers.get("cache-control", "")


# --------------------------------------------------------------------------- #
# Переключение LLM на лету (п. 2 из задачи)
# --------------------------------------------------------------------------- #
def test_llm_provider_switch() -> None:
    from backend.llm import AVAILABLE_PROVIDERS, get_llm, provider_info, set_provider

    info = provider_info()
    assert info["provider"] in AVAILABLE_PROVIDERS
    assert {o["id"] for o in info["options"]} == set(AVAILABLE_PROVIDERS)

    original = info["provider"]
    try:
        switched = set_provider("openai")
        assert switched["provider"] == "openai"
        assert get_llm().provider == "openai"
        assert get_llm().model == get_settings().llm_model

        back = set_provider("deepseek")
        assert back["provider"] == "deepseek"
        assert get_llm().model == get_settings().deepseek_model
    finally:
        set_provider(original)

    with pytest.raises(ValueError):
        set_provider("нет-такого")


def test_llm_api_endpoints() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        info = client.get("/api/v1/llm")
        assert info.status_code == 200
        body = info.json()
        assert "provider" in body and "options" in body

        bad = client.post("/api/v1/llm", json={"provider": "unknown"})
        assert bad.status_code == 400

        current = body["provider"]
        other = next(o["id"] for o in body["options"] if o["id"] != current)
        switched = client.post("/api/v1/llm", json={"provider": other})
        assert switched.status_code == 200
        assert switched.json()["provider"] == other

        # возвращаем как было
        client.post("/api/v1/llm", json={"provider": current})
