"""FastAPI-приложение «Эксперт как плейлист» (ассистент Сэйн).

Раздаёт собранный фронтенд (frontend/dist) и API:

    GET    /api/v1/catalog                 каталог программ и экспертов
    GET    /api/v1/config                  публичные настройки (модель, дисклеймер)
    POST   /api/v1/chat/stream             SSE-стрим ответа одного эксперта
    POST   /api/v1/chat/stream-multi       SSE-стрим: один вопрос — все эксперты
    POST   /api/v1/chat/stop               остановить генерацию
    GET    /api/v1/chat/history            история сессии
    DELETE /api/v1/chat/history            «начать чат заново»
    GET    /api/v1/memory                  факты о пользователе
    POST   /api/v1/memory                  сохранить факт
    DELETE /api/v1/memory                  очистить память
    POST   /api/v1/articles                добавить статью (индексация в фоне)
    GET    /api/v1/articles                список статей
    DELETE /api/v1/articles/{article_id}   удалить статью
    POST   /api/v1/articles/recommend      «Порекомендовать статью»: ссылка → SQLite → векторная БД
    GET    /api/v1/articles/used-chunks    фрагменты, которые агент реально использовал
    GET    /api/v1/articles/{article_id}   статья + использованные фрагменты
    POST   /api/v1/documents/upload        загрузить файл/документ в векторную БД
    GET    /api/v1/library                 всё для экрана «Список статей»
    GET    /api/v1/knowledge-gaps          пробелы знаний
    PATCH  /api/v1/knowledge-gaps/{gap_id} сменить статус пробела
    GET    /api/v1/health                  состояние LLM / векторной БД / БД
    POST   /run                            демо-стенд кейса 4
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import NO_EVIDENCE_PHRASE, get_runtime
from .article_store import (
    delete_article as delete_article_row,
    dispose_article_db,
    get_article as get_article_row,
    init_article_db,
    list_articles as list_article_rows,
    list_used_chunks,
    record_used_chunks,
    update_article_status,
    upsert_article,
)
from .ingest import file_to_text, index_text, ingest_link
from .config import BASE_DIR, get_catalog, get_settings
from .hackapi import dataset_info, ensure_dataset_indexed, fetch_dataset
from .llm import close_http_client, get_llm, provider_info, set_provider
from .schemas import (
    ArticleAccepted,
    ArticleDetailOut,
    ArticleIn,
    ArticleOut,
    ArticleRecommendIn,
    RecommendedArticleOut,
    UploadAccepted,
    UsedChunkOut,
    CatalogOut,
    ChatRequest,
    ExpertOut,
    HistoryOut,
    KnowledgeGapOut,
    MemoryIn,
    MemoryOut,
    MultiChatRequest,
    ProgramOut,
)
from .storage import (
    delete_memory,
    dispose_db,
    get_memory,
    init_db,
    list_knowledge_gaps,
    set_gap_status,
    stats,
    touch_session,
    upsert_memory,
)
from .vectorstore import add_article_async, get_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    await asyncio.to_thread(init_article_db)
    await get_runtime().startup()
    await ensure_dataset_indexed()
    logger.info("app started (provider=%s)", get_settings().provider)
    try:
        yield
    finally:
        await get_runtime().shutdown()
        await close_http_client()
        await dispose_db()
        await asyncio.to_thread(dispose_article_db)
        logger.info("app stopped")


app = FastAPI(title="Сэйн · ассистент по научным статьям", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Каталог и публичная конфигурация
# --------------------------------------------------------------------------- #
@app.get("/api/v1/catalog", response_model=CatalogOut, tags=["catalog"])
async def catalog() -> CatalogOut:
    cat = get_catalog()
    return CatalogOut(
        version=cat.version,
        programs=[ProgramOut(**p.model_dump()) for p in cat.programs],
        experts=[ExpertOut(**e.model_dump()) for e in cat.experts],
    )


@app.get("/api/v1/config", tags=["catalog"])
async def public_config() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "llm_model": settings.active_model,
        "llm_provider": settings.provider,
        "embedding_model": "bm25",
        "disclaimer": settings.disclaimer,
        "freshness_days": settings.freshness_days,
        "max_message_length": 4000,
    }


# --------------------------------------------------------------------------- #
# Переключение LLM на лету: deepseek ↔ gpt-6-luna
# --------------------------------------------------------------------------- #
@app.get("/api/v1/llm", tags=["catalog"])
async def read_llm() -> Dict[str, Any]:
    """Текущий провайдер и список доступных для переключателя."""
    return provider_info()


@app.post("/api/v1/llm", tags=["catalog"])
async def switch_llm(payload: Dict[str, str]) -> Dict[str, Any]:
    """Переключает LLM. Тело: {"provider": "deepseek" | "openai"}."""
    provider = (payload or {}).get("provider", "")
    try:
        return set_provider(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --------------------------------------------------------------------------- #
# Чат (SSE)
# --------------------------------------------------------------------------- #
def _validate_ids(program_id: str, expert_id: str) -> None:
    catalog_ = get_catalog()
    if program_id not in catalog_.program_map():
        raise HTTPException(status_code=404, detail=f"unknown program_id: {program_id}")
    if expert_id not in catalog_.expert_map():
        raise HTTPException(status_code=404, detail=f"unknown expert_id: {expert_id}")


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


@app.post("/api/v1/chat/stream", tags=["chat"])
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    _validate_ids(payload.program_id, payload.expert_id)
    runtime = get_runtime()

    async def event_generator() -> AsyncIterator[str]:
        try:
            await touch_session(
                session_id=payload.session_id,
                program_id=payload.program_id,
                expert_id=payload.expert_id,
                user_id=payload.user_id,
            )
            async for event in runtime.stream(
                session_id=payload.session_id,
                program_id=payload.program_id,
                expert_id=payload.expert_id,
                message=payload.message,
                user_id=payload.user_id,
            ):
                yield _sse(event)
        except asyncio.CancelledError:  # клиент закрыл соединение / нажал «Остановить»
            runtime.cancel(payload.session_id)
            logger.info("client disconnected: %s", payload.session_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream failed")
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/api/v1/chat/stream-multi", tags=["chat"])
async def chat_stream_multi(payload: MultiChatRequest) -> StreamingResponse:
    """Один вопрос — все специалисты. Поиск один раз, поэтому источники у всех одинаковые."""
    catalog_ = get_catalog()
    expert_ids = [e for e in dict.fromkeys(payload.expert_ids) if e]
    if not expert_ids:
        expert_ids = [e.id for e in catalog_.experts]
    for expert_id in expert_ids:
        _validate_ids(payload.program_id, expert_id)
    runtime = get_runtime()

    async def event_generator() -> AsyncIterator[str]:
        try:
            await touch_session(
                session_id=payload.session_id,
                program_id=payload.program_id,
                expert_id=expert_ids[0],
                user_id=payload.user_id,
            )
            async for event in runtime.stream_multi(
                session_id=payload.session_id,
                program_id=payload.program_id,
                expert_ids=expert_ids,
                message=payload.message,
                user_id=payload.user_id,
            ):
                yield _sse(event)
        except asyncio.CancelledError:
            runtime.cancel(payload.session_id)
            logger.info("client disconnected (multi): %s", payload.session_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream-multi failed")
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_SSE_HEADERS)


@app.post("/api/v1/chat/stop", tags=["chat"])
async def chat_stop(session_id: str = Query(min_length=1)) -> Dict[str, str]:
    get_runtime().cancel(session_id)
    return {"status": "stopping", "session_id": session_id}


@app.get("/api/v1/chat/history", response_model=HistoryOut, tags=["chat"])
async def chat_history(session_id: str = Query(min_length=1)) -> HistoryOut:
    return HistoryOut(session_id=session_id, messages=await get_runtime().get_history(session_id))


@app.delete("/api/v1/chat/history", tags=["chat"])
async def clear_history(session_id: str = Query(min_length=1)) -> Dict[str, str]:
    """«Начать чат заново»."""
    await get_runtime().reset(session_id)
    return {"status": "cleared", "session_id": session_id}


# --------------------------------------------------------------------------- #
# Персонализация: память о пользователе
# --------------------------------------------------------------------------- #
@app.get("/api/v1/memory", response_model=MemoryOut, tags=["memory"])
async def read_memory(user_id: str = Query(min_length=1)) -> MemoryOut:
    return MemoryOut(user_id=user_id, items=await get_memory(user_id, limit=50))


@app.post("/api/v1/memory", response_model=MemoryOut, tags=["memory"])
async def write_memory(payload: MemoryIn) -> MemoryOut:
    await upsert_memory(payload.user_id, payload.key, payload.value, confidence=payload.confidence, source="user")
    return MemoryOut(user_id=payload.user_id, items=await get_memory(payload.user_id, limit=50))


@app.delete("/api/v1/memory", tags=["memory"])
async def clear_memory(user_id: str = Query(min_length=1)) -> Dict[str, Any]:
    return {"removed": await delete_memory(user_id), "user_id": user_id}


# --------------------------------------------------------------------------- #
# Статьи: добавление и индексация
# --------------------------------------------------------------------------- #
@app.post("/api/v1/articles", status_code=202, response_model=ArticleAccepted, tags=["articles"])
async def add_article(payload: ArticleIn, background: BackgroundTasks) -> ArticleAccepted:
    catalog_ = get_catalog()
    if payload.program_id not in catalog_.program_map():
        raise HTTPException(status_code=404, detail=f"unknown program_id: {payload.program_id}")

    sections = [s.model_dump() for s in payload.sections] if payload.sections else None
    extra = {"doi": payload.doi, "year": payload.year}
    article_id = payload.article_id or f"art_{datetime.now(timezone.utc).timestamp():.0f}"

    async def index_task() -> None:
        try:
            await add_article_async(
                program_id=payload.program_id,
                title=payload.title,
                published_date=payload.published_date,
                content=payload.content,
                article_id=article_id,
                sections=sections,
                extra_metadata={k: v for k, v in extra.items() if v is not None},
            )
            logger.info("article indexed: %s", article_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("article indexing failed: %s", exc)

    background.add_task(index_task)
    return ArticleAccepted(
        article_id=article_id,
        program_id=payload.program_id,
        chunks_planned=len(sections) if sections else 1,
        message="Статья принята: индексация идёт в фоне, эксперты учтут её сразу после завершения.",
    )


@app.get("/api/v1/articles", response_model=List[ArticleOut], tags=["articles"])
async def list_articles(program_id: Optional[str] = None) -> List[ArticleOut]:
    return [ArticleOut(**row) for row in await asyncio.to_thread(get_store().list_articles, program_id)]


# --------------------------------------------------------------------------- #
# «Порекомендовать статью»: ссылка → SQLite → векторная БД
# --------------------------------------------------------------------------- #
def _article_id_for_url(url: str) -> str:
    return f"art_{hashlib.sha1(url.strip().lower().encode('utf-8')).hexdigest()[:12]}"


@app.post(
    "/api/v1/articles/recommend",
    status_code=202,
    response_model=RecommendedArticleOut,
    tags=["articles"],
)
async def recommend_article(payload: ArticleRecommendIn, background: BackgroundTasks) -> RecommendedArticleOut:
    """Пользователь присылает ссылку на статью: сохраняем в SQLite и индексируем в фоне."""
    catalog_ = get_catalog()
    if payload.program_id not in catalog_.program_map():
        raise HTTPException(status_code=404, detail=f"unknown program_id: {payload.program_id}")

    url = payload.url.strip()
    article_id = _article_id_for_url(url)
    row = await asyncio.to_thread(
        upsert_article,
        url=url,
        title=payload.title or url,
        program_id=payload.program_id,
        user_id=payload.user_id,
        comment=payload.comment,
        source_kind="link",
        status="pending",
        article_id=article_id,
    )

    async def index_task() -> None:
        try:
            await ingest_link(
                article_id=article_id,
                program_id=payload.program_id,
                url=url,
                title_hint=payload.title,
                fallback_text=" ".join(filter(None, [payload.title, payload.comment])),
            )
        except Exception as exc:  # noqa: BLE001 — статус уже записан в SQLite
            logger.warning("recommended article not indexed: %s (%s)", url, exc)

    background.add_task(index_task)
    return RecommendedArticleOut(**row)


# --------------------------------------------------------------------------- #
# «Загрузить файл/документ в векторную базу»
# --------------------------------------------------------------------------- #
@app.post("/api/v1/documents/upload", status_code=202, response_model=UploadAccepted, tags=["articles"])
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    program_id: str = Form(default="sleep_recovery"),
    title: str = Form(default=""),
    user_id: str = Form(default=""),
) -> UploadAccepted:
    """Принимает .pdf/.txt/.md и ставит в очередь индексации (чанки + bge-m3)."""
    catalog_ = get_catalog()
    if program_id not in catalog_.program_map():
        raise HTTPException(status_code=404, detail=f"unknown program_id: {program_id}")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="файл пустой")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="файл больше 20 МБ")

    filename = file.filename or "document.txt"
    try:
        extracted_title, text = await asyncio.to_thread(file_to_text, filename, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if len(text) < 80:
        raise HTTPException(status_code=400, detail="в документе слишком мало текста")

    final_title = (title or extracted_title or filename)[:500]
    article_id = f"doc_{hashlib.sha1((filename + str(len(raw)) + final_title).encode('utf-8')).hexdigest()[:12]}"
    await asyncio.to_thread(
        upsert_article,
        url=f"file://{filename}",
        title=final_title,
        program_id=program_id,
        user_id=user_id or None,
        comment=f"загружен файл {filename}",
        source_kind="file",
        status="pending",
        article_id=article_id,
    )

    async def index_task() -> None:
        try:
            await index_text(
                article_id=article_id,
                program_id=program_id,
                title=final_title,
                text=text,
                extra_metadata={"source_file": filename},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("uploaded document not indexed: %s (%s)", filename, exc)

    background.add_task(index_task)
    return UploadAccepted(
        article_id=article_id,
        title=final_title,
        message=f"«{final_title}» принят: индексация в векторную базу идёт в фоне.",
    )


# --------------------------------------------------------------------------- #
# «Список статей»: что есть в базе и какие куски агент реально использовал
# --------------------------------------------------------------------------- #
@app.get("/api/v1/articles/used-chunks", response_model=List[UsedChunkOut], tags=["articles"])
async def used_chunks(
    article_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=1000),
) -> List[UsedChunkOut]:
    rows = await asyncio.to_thread(list_used_chunks, article_id, session_id, limit)
    return [UsedChunkOut(**row) for row in rows]


@app.get("/api/v1/library", tags=["articles"])
async def library(program_id: Optional[str] = None) -> Dict[str, Any]:
    """Единый ответ для экрана «Список статей»: база + рекомендованные + использованные куски."""
    store = get_store()
    vector_articles = await asyncio.to_thread(store.list_articles, program_id)
    recommended = await asyncio.to_thread(list_article_rows, program_id)
    used = await asyncio.to_thread(list_used_chunks, None, None, 500)

    used_by_article: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in used:
        used_by_article.setdefault(chunk["article_id"], []).append(chunk)

    articles: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for art in vector_articles:
        aid = str(art["article_id"])
        seen.add(aid)
        rec = next((r for r in recommended if r["id"] == aid), None)
        articles.append(
            {
                "article_id": aid,
                "title": art["title"],
                "url": (rec or {}).get("url", ""),
                "program_id": art["program_id"],
                "status": "indexed",
                "chunks": art["chunks"],
                "source_kind": (rec or {}).get("source_kind", "vector"),
                "added_at": art["added_at"],
                "used_chunks": used_by_article.get(aid, []),
            }
        )
    for rec in recommended:
        if rec["id"] in seen:
            continue
        articles.append(
            {
                "article_id": rec["id"],
                "title": rec["title"],
                "url": rec["url"],
                "program_id": rec["program_id"],
                "status": rec["status"],
                "chunks": rec["chunks"],
                "source_kind": rec["source_kind"],
                "added_at": rec["created_at"],
                "used_chunks": used_by_article.get(rec["id"], []),
            }
        )
    articles.sort(key=lambda a: a.get("added_at") or 0, reverse=True)
    return {
        "articles": articles,
        "total": len(articles),
        "used_chunks_total": len(used),
        "vector_chunks": store.count(),
    }


@app.get("/api/v1/articles/{article_id}", response_model=ArticleDetailOut, tags=["articles"])
async def article_detail(article_id: str) -> ArticleDetailOut:
    """Одна статья: только те фрагменты, которые агент подставил в ответ."""
    store = get_store()
    vector = await asyncio.to_thread(store.list_articles)
    meta = next((a for a in vector if str(a["article_id"]) == article_id), None)
    rec = await asyncio.to_thread(get_article_row, article_id)
    if meta is None and rec is None:
        raise HTTPException(status_code=404, detail=f"unknown article_id: {article_id}")
    used = await asyncio.to_thread(list_used_chunks, article_id, None, 200)
    return ArticleDetailOut(
        article_id=article_id,
        title=(meta or {}).get("title") or (rec or {}).get("title", ""),
        url=(rec or {}).get("url", ""),
        program_id=(meta or {}).get("program_id") or (rec or {}).get("program_id", ""),
        status="indexed" if meta else (rec or {}).get("status", "pending"),
        chunks=int((meta or {}).get("chunks") or (rec or {}).get("chunks") or 0),
        source_kind=(rec or {}).get("source_kind", "vector"),
        added_at=float((meta or {}).get("added_at") or (rec or {}).get("created_at") or 0),
        used_chunks=[UsedChunkOut(**row) for row in used],
    )


@app.delete("/api/v1/articles/{article_id}", tags=["articles"])
async def delete_article(article_id: str) -> Dict[str, str]:
    await asyncio.to_thread(get_store().delete_article, article_id)
    await asyncio.to_thread(delete_article_row, article_id)
    return {"status": "deleted", "article_id": article_id}


# --------------------------------------------------------------------------- #
# Пробелы знаний
# --------------------------------------------------------------------------- #
@app.get("/api/v1/knowledge-gaps", response_model=List[KnowledgeGapOut], tags=["curator"])
async def knowledge_gaps(
    status: Optional[str] = Query(default=None, pattern=r"^(open|indexed|rejected)$"),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[KnowledgeGapOut]:
    return [KnowledgeGapOut(**row) for row in await list_knowledge_gaps(status=status, limit=limit)]


@app.patch("/api/v1/knowledge-gaps/{gap_id}", tags=["curator"])
async def patch_knowledge_gap(gap_id: str, status: str = Query(pattern=r"^(open|indexed|rejected)$")) -> Dict[str, Any]:
    if not await set_gap_status(gap_id, status):
        raise HTTPException(status_code=404, detail=f"unknown gap_id: {gap_id}")
    return {"status": status, "gap_id": gap_id}


# --------------------------------------------------------------------------- #
# Состояние системы
# --------------------------------------------------------------------------- #
@app.get("/api/v1/health", tags=["service"])
async def health() -> Dict[str, Any]:
    llm_health = await get_llm().health()
    store = get_store()
    try:
        db_stats: Dict[str, Any] = await stats()
    except Exception as exc:  # noqa: BLE001
        db_stats = {"error": str(exc)}
    try:
        articles_count = len(await asyncio.to_thread(store.list_articles))
    except Exception as exc:  # noqa: BLE001
        logger.warning("articles count failed: %s", exc)
        articles_count = 0
    return {
        "status": "ok" if llm_health.get("ok") else "degraded",
        "llm": {"model": get_llm().model, "provider": get_llm().provider, **llm_health},
        "vector_store": {
            "collection": store.collection_name,
            "chunks": store.count(),
            "articles": articles_count,
            "retriever": "bm25",
        },
        "database": db_stats,
        "case4_dataset": dataset_info(),
        "time": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Демо-стенд (кейс 4): POST /run → Case4Answers
# --------------------------------------------------------------------------- #
def _case4_answer(result: Dict[str, Any], record_id: str) -> Dict[str, Any]:
    citations, seen = [], set()
    for src in result.get("sources") or []:
        key = (src.get("article_id"), src.get("section_id") or "sec_main")
        if key not in seen:
            seen.add(key)
            citations.append({"article_id": key[0], "section_id": key[1]})
    return {
        "record_id": record_id,
        "text": result.get("answer") or NO_EVIDENCE_PHRASE,
        "refused": bool(result.get("is_unsafe_request")),
        "citations": citations,
    }


def _authorization_bearer(request: Request) -> str:
    """Токен из заголовка Authorization: Bearer <токен>. Пустой, если не задан."""
    value = request.headers.get("authorization", "")
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value.strip()


def _case_utterance(record: Dict[str, Any]) -> str:
    """Текст вопроса из записи набора кейса 4 (поле `utterance.text`)."""
    utterance = record.get("utterance")
    if isinstance(utterance, dict):
        return str(utterance.get("text") or "").strip()
    if isinstance(utterance, str):
        return utterance.strip()
    return str(record.get("question") or record.get("query") or "").strip()


@app.post("/run", tags=["case4"])
async def run_case4(request: Request) -> JSONResponse:
    """Демо-стенд кейса 4.

    Стенд шлёт `{"case": 4, "dataset_code": "..."}` с токеном в `Authorization`.
    Решение само получает набор через `GET /case/4/getData` (этим токеном),
    обрабатывает весь набор и возвращает контейнер `{"requests": [...]}`.
    """
    body: Any = None
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="body must be valid JSON")

    if isinstance(body, dict):
        dataset_code = str(body.get("dataset_code") or "").strip() or None
    else:
        dataset_code = None

    token = _authorization_bearer(request)

    dataset = await fetch_dataset(token=token or None, dataset_code=dataset_code)
    records = [r for r in (dataset or {}).get("requests") or [] if isinstance(r, dict)]
    if not records:
        # фолбэк для локальной отладки: тело может прийти списком записей
        if isinstance(body, list):
            records = [r for r in body if isinstance(r, dict)]
        elif isinstance(body, dict):
            records = [r for r in (body.get("requests") or body.get("records") or []) if isinstance(r, dict)]

    if not records:
        raise HTTPException(status_code=400, detail="no requests in payload")

    # база может быть пустой, если приложение успело ответить на GET / до индексации
    await ensure_dataset_indexed(token=token or None)

    runtime = get_runtime()
    answers: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        record_id = str(record.get("record_id") or record.get("id") or index)
        question = _case_utterance(record)
        if not question:
            answers.append({"record_id": record_id, "text": NO_EVIDENCE_PHRASE, "refused": False, "citations": []})
            continue
        result = await runtime.run(
            session_id=f"case4-{record_id}",
            program_id=str(record.get("program_id") or "sleep_recovery"),
            expert_id=str(record.get("expert_id") or "clinician"),
            message=question,
            user_id=record.get("user_id"),
        )
        answers.append(_case4_answer(result, record_id))

    return JSONResponse({"requests": answers})


# --------------------------------------------------------------------------- #
# Статика: собранный фронтенд
# --------------------------------------------------------------------------- #
class ImmutableStaticFiles(StaticFiles):
    """Файлы в /assets имеют хеш в имени — их можно кэшировать навсегда."""

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


# index.html не кэшируем: после пересборки браузер должен взять новый бандл
_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}


def _build_id() -> str:
    """Версия сборки фронтенда: меняется при каждой пересборке."""
    assets = FRONTEND_DIST / "assets"
    if not assets.exists():
        return "0"
    newest = max((p.stat().st_mtime for p in assets.iterdir() if p.is_file()), default=0)
    return str(int(newest))


def _index_html() -> str:
    """index.html с ?v=<build> у ссылок на ассеты: старый кэш не отдаст прежний бандл."""
    html = (FRONTEND_DIST / "index.html").read_text(encoding="utf-8")
    build = _build_id()
    return re.sub(r'(/assets/[^"?\s]+)', r'\1?v=' + build, html)


if FRONTEND_DIST.exists():
    app.mount("/assets", ImmutableStaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(_index_html(), headers=_NO_CACHE)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> Response:
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate, headers=_NO_CACHE)
        return HTMLResponse(_index_html(), headers=_NO_CACHE)
else:
    @app.get("/", include_in_schema=False)
    async def no_frontend() -> JSONResponse:
        return JSONResponse({"detail": "frontend/dist не собран — запустите ./run.sh с REBUILD_UI=1"})


__all__ = ["app"]
