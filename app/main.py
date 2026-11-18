"""FastAPI-приложение «Эксперт как плейлист»: каталог, SSE-чат, куратор статей, /run."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import NO_EVIDENCE_PHRASE, get_runtime
from .config import BASE_DIR, get_catalog, get_settings
from .llm import get_llm
from .schemas import (
    ArticleAccepted,
    ArticleIn,
    ArticleOut,
    CatalogOut,
    ChatRequest,
    ExpertOut,
    HistoryOut,
    ProgramOut,
)
from .vectorstore import _seed_added_at, add_article_async, get_store, parse_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


# --------------------------------------------------------------------------- #
# Lifespan: каталог + сидинг базы + граф
# --------------------------------------------------------------------------- #
def _load_dataset_articles() -> List[Dict[str, Any]]:
    """Читает статьи из data/case4-dev.json (единственный источник доказательной базы)."""
    path = Path(get_settings().dataset_path)
    if not path.exists():
        logger.warning("dataset not found: %s", path)
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = raw.get("dataset", raw)  # поддерживаем и обёртку {dataset, reference}
    return dataset.get("articles") or []


def _dataset_article_kwargs(article: Dict[str, Any], program_id: str) -> Dict[str, Any]:
    year = int(article.get("year") or 2020)
    sections = [
        {
            "section_id": str(s.get("section_id")),
            "title": str(s.get("title") or ""),
            "text": str(s.get("text") or ""),
        }
        for s in article.get("sections") or []
        if str(s.get("text") or "").strip()
    ]
    license_info = article.get("license") or {}
    return {
        "program_id": program_id,
        "title": str(article.get("title") or article.get("article_id")),
        "published_date": f"{year:04d}-01-01",
        "content": "\n\n".join(s["text"] for s in sections),
        "article_id": str(article.get("article_id")),
        "sections": sections,
        "added_at": None,  # время индексации → статьи датасета считаются свежими (бейдж NEW)
        "extra_metadata": {
            "authors": ", ".join(article.get("authors") or []),
            "doi": article.get("doi"),
            "year": year,
            "license": f"{license_info.get('name', '')} {license_info.get('version', '')}".strip(),
            "source": "case4-dev",
        },
    }


async def seed_if_empty() -> None:
    """Наполнение Chroma статьями из data/case4-dev.json (только если база пуста)."""
    store = get_store()
    if store.count() > 0:
        logger.info("vector store already has %d chunks — skip seeding", store.count())
        return
    articles = _load_dataset_articles()
    if not articles:
        logger.warning("no articles to seed")
        return
    program_ids = [p.id for p in get_catalog().programs]
    program_id = "sleep_recovery" if "sleep_recovery" in program_ids else program_ids[0]
    for article in articles:
        await add_article_async(**_dataset_article_kwargs(article, program_id))
    logger.info("seeded %d articles from %s", len(articles), get_settings().dataset_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.checkpoints_db).parent.mkdir(parents=True, exist_ok=True)
    catalog = get_catalog()  # валидация конфига при старте
    logger.info("catalog loaded: %d programs, %d experts", len(catalog.programs), len(catalog.experts))
    await asyncio.to_thread(get_store)  # инициализация Chroma
    # временно отключено: seed грузит torch/sentence-transformers (bge-m3),
    # что на локальной/стендовой машине вешает старт (см. get_embedder)
    # await seed_if_empty()
    await get_runtime().startup()
    yield
    await get_runtime().shutdown()


app = FastAPI(
    title="Эксперт как плейлист",
    description="AI-платформа научно обоснованных программ здоровья (LangGraph + Chroma + OpenAI-совместимый LLM)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Каталог
# --------------------------------------------------------------------------- #
@app.get("/api/v1/catalog", response_model=CatalogOut, tags=["catalog"])
async def get_catalog_endpoint() -> CatalogOut:
    catalog = get_catalog()
    return CatalogOut(
        version=catalog.version,
        programs=[ProgramOut(**p.model_dump()) for p in catalog.programs],
        experts=[ExpertOut(**e.model_dump()) for e in catalog.experts],
    )


# --------------------------------------------------------------------------- #
# Чат (SSE)
# --------------------------------------------------------------------------- #
def _validate_ids(program_id: str, expert_id: str) -> None:
    catalog = get_catalog()
    if program_id not in catalog.program_map():
        raise HTTPException(status_code=404, detail=f"unknown program_id: {program_id}")
    if expert_id not in catalog.expert_map():
        raise HTTPException(status_code=404, detail=f"unknown expert_id: {expert_id}")


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/v1/chat/stream", tags=["chat"])
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    _validate_ids(payload.program_id, payload.expert_id)
    runtime = get_runtime()

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in runtime.stream(
                session_id=payload.session_id,
                program_id=payload.program_id,
                expert_id=payload.expert_id,
                message=payload.message,
            ):
                yield _sse(event)
        except asyncio.CancelledError:  # клиент закрыл соединение
            logger.info("client disconnected: %s", payload.session_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream failed")
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/chat/history", response_model=HistoryOut, tags=["chat"])
async def chat_history(session_id: str = Query(min_length=1)) -> HistoryOut:
    messages = await get_runtime().get_history(session_id)
    return HistoryOut(session_id=session_id, messages=messages)


@app.delete("/api/v1/chat/history", tags=["chat"])
async def clear_history(session_id: str = Query(min_length=1)) -> Dict[str, str]:
    """Сброс диалога: удаляем тред из чекпойнтера."""
    runtime = get_runtime()
    if runtime._saver is not None:  # noqa: SLF001 — намеренно, нужен доступ к хранилищу
        try:
            await runtime._saver.adelete_thread(session_id)  # noqa: SLF001
        except Exception as exc:  # pragma: no cover
            logger.warning("delete thread failed: %s", exc)
    return {"status": "deleted", "session_id": session_id}


# --------------------------------------------------------------------------- #
# Статьи (куратор)
# --------------------------------------------------------------------------- #
@app.post("/api/v1/articles", status_code=202, response_model=ArticleAccepted, tags=["articles"])
async def add_article(payload: ArticleIn, background: BackgroundTasks) -> ArticleAccepted:
    catalog = get_catalog()
    if payload.program_id not in catalog.program_map():
        raise HTTPException(status_code=404, detail=f"unknown program_id: {payload.program_id}")
    if parse_date(payload.published_date) is None:
        raise HTTPException(status_code=422, detail="published_date must be YYYY-MM-DD")

    article_id = payload.article_id or f"art_{uuid.uuid4().hex[:12]}"
    sections = [s.model_dump() for s in payload.sections] if payload.sections else None
    content = payload.content
    from .vectorstore import chunk_text

    chunks_planned = (
        sum(len(chunk_text(s["text"])) for s in sections) if sections else len(chunk_text(content))
    )

    async def index_task() -> None:
        try:
            await add_article_async(
                program_id=payload.program_id,
                title=payload.title,
                published_date=payload.published_date,
                content=content,
                article_id=article_id,
                sections=sections,
                added_at=int(time.time()),
                extra_metadata={
                    "authors": ", ".join(payload.authors or []),
                    "doi": payload.doi,
                },
            )
            logger.info("article %s indexed in background", article_id)
        except Exception:  # pragma: no cover
            logger.exception("failed to index article %s", article_id)

    background.add_task(index_task)
    return ArticleAccepted(
        article_id=article_id,
        program_id=payload.program_id,
        chunks_planned=chunks_planned,
        message="Статья принята в обработку: чанки и эмбеддинги будут добавлены в ChromaDB.",
    )


@app.get("/api/v1/articles", response_model=List[ArticleOut], tags=["articles"])
async def list_articles(program_id: Optional[str] = None) -> List[ArticleOut]:
    articles = await asyncio.to_thread(get_store().list_articles, program_id)
    return [ArticleOut(**a) for a in articles]


@app.delete("/api/v1/articles/{article_id}", tags=["articles"])
async def delete_article(article_id: str) -> Dict[str, str]:
    await asyncio.to_thread(get_store().delete_article, article_id)
    return {"status": "deleted", "article_id": article_id}


# --------------------------------------------------------------------------- #
# Служебное
# --------------------------------------------------------------------------- #
@app.get("/api/v1/health", tags=["service"])
async def health() -> Dict[str, Any]:
    llm_health = await get_llm().health()
    store = get_store()
    return {
        "status": "ok" if llm_health.get("ok") else "degraded",
        "llm": {"model": get_settings().llm_model, **llm_health},
        "vector_store": {
            "collection": store.collection_name,
            "chunks": store.count(),
            "embedding_model": get_settings().embedding_model,
        },
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/admin/reseed", tags=["service"])
async def reseed() -> Dict[str, Any]:
    """Пересобрать демо-базу знаний с нуля (для отладки)."""
    store = get_store()
    await asyncio.to_thread(store.reset)
    await seed_if_empty()
    return {"status": "reseeded", "chunks": store.count()}


# --------------------------------------------------------------------------- #
# Демо-стенд (кейс 4): POST /run — ответы в формате Case4Answers
# --------------------------------------------------------------------------- #
def _case4_answer(runtime_result: Dict[str, Any], record_id: str) -> Dict[str, Any]:
    citations: List[Dict[str, str]] = []
    seen = set()
    for src in runtime_result.get("sources") or []:
        key = (src.get("article_id"), src.get("section_id") or "sec_main")
        if key in seen:
            continue
        seen.add(key)
        citations.append({"article_id": key[0], "section_id": key[1]})
    return {
        "record_id": record_id,
        "text": runtime_result.get("answer") or NO_EVIDENCE_PHRASE,
        "refused": bool(runtime_result.get("is_unsafe_request")),
        "citations": citations,
    }


@app.post("/run", tags=["case4"])
async def run_case4(request: Request) -> JSONResponse:
    """Контракт стенда: {"case": 4, "dataset_code": "..."} → Case4Answers.

    Если в теле передан сам датасет (поле dataset) — он используется напрямую,
    иначе датасет запрашивается у стенда через getData по токену.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    case = int(body.get("case", 4) or 4)
    if case != 4:
        raise HTTPException(status_code=400, detail="this service implements case 4 only")

    dataset = body.get("dataset")
    if not dataset:
        dataset = await _fetch_dataset(request, body.get("dataset_code"))
    if not dataset:
        raise HTTPException(status_code=400, detail="dataset is required (pass `dataset` or a valid dataset_code)")

    runtime = get_runtime()
    answers = []
    for idx, req in enumerate(dataset.get("requests") or []):
        record_id = str(req.get("record_id") or f"rec_{idx}")
        user_id = str(req.get("user_id") or "user")
        expert_id = str(req.get("expert_id") or "clinician")
        utterance = req.get("utterance") or {}
        text = str(utterance.get("text") or "")
        session_id = f"case4:{user_id}:{record_id}"
        result = await runtime.run(
            session_id=session_id,
            program_id=_infer_program(dataset, req),
            expert_id=_map_expert(expert_id),
            message=text,
        )
        answers.append(_case4_answer(result, record_id))
    return JSONResponse({"requests": answers})


def _map_expert(expert_id: str) -> str:
    catalog = get_catalog()
    if expert_id in catalog.expert_map():
        return expert_id
    mapping = {"clinician": "clinician", "biohacker": "biohacker", "nutritionist": "nutritionist"}
    candidate = mapping.get(expert_id, "clinician")
    return candidate if candidate in catalog.expert_map() else catalog.experts[0].id


def _infer_program(dataset: Dict[str, Any], req: Dict[str, Any]) -> str:
    catalog = get_catalog()
    known = catalog.program_map()
    program_id = req.get("program_id")
    if isinstance(program_id, str) and program_id in known:
        return program_id
    return catalog.programs[0].id


async def _fetch_dataset(request: Request, dataset_code: Optional[str]) -> Optional[Dict[str, Any]]:
    """getData у стенда хакатона (если задан SAIN_BASE_URL и есть токен)."""
    import os

    base = os.getenv("SAIN_BASE_URL")
    if not base or not dataset_code:
        return None
    token = request.headers.get("authorization") or os.getenv("SAIN_TOKEN")
    if not token:
        return None
    import httpx

    url = f"{base.rstrip('/')}/case/4/getData"
    headers = {"Authorization": token if token.lower().startswith("bearer") else f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url, params={"dataset_code": dataset_code}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("dataset")
    except Exception as exc:  # pragma: no cover
        logger.warning("getData failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Статика фронтенда (если собран; иначе — встроенный демо-UI)
# --------------------------------------------------------------------------- #
_dist = BASE_DIR / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")
else:

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        page = BASE_DIR / "web" / "index.html"
        if not page.exists():
            return JSONResponse({"service": "expert-as-playlist", "docs": "/docs"})
        from fastapi.responses import HTMLResponse

        return HTMLResponse(page.read_text(encoding="utf-8"))
