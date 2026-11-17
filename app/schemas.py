"""Pydantic-схемы API и внутренние схемы графа."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Каталог
# --------------------------------------------------------------------------- #
class ProgramOut(BaseModel):
    id: str
    title: str
    description: str
    cover_color: str
    default_expert_id: str


class ExpertOut(BaseModel):
    id: str
    name: str
    role_badge: str
    avatar_url: str = ""
    # стиль отдаём фронту, чтобы можно было показать «чем отличается персона»
    system_style_prompt: str


class CatalogOut(BaseModel):
    version: str
    programs: List[ProgramOut]
    experts: List[ExpertOut]


# --------------------------------------------------------------------------- #
# Чат
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    program_id: str = Field(default="sleep_recovery", min_length=1)
    expert_id: str = Field(default="clinician", min_length=1)
    message: str = Field(min_length=1, max_length=4000)
    user_id: Optional[str] = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


class SourceItem(BaseModel):
    article_id: str
    section_id: Optional[str] = None
    title: str
    date: Optional[str] = None
    is_new: bool = False
    snippet: str = ""


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    expert_id: Optional[str] = None
    safety_flag: bool = False
    sources: List[SourceItem] = []


class HistoryOut(BaseModel):
    session_id: str
    messages: List[HistoryMessage]


# --------------------------------------------------------------------------- #
# Статьи
# --------------------------------------------------------------------------- #
class SectionIn(BaseModel):
    section_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ArticleIn(BaseModel):
    program_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    published_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    content: str = Field(min_length=1)
    article_id: Optional[str] = Field(default=None, max_length=128)
    sections: Optional[List[SectionIn]] = None
    authors: Optional[List[str]] = None
    doi: Optional[str] = None
    year: Optional[int] = None


class ArticleAccepted(BaseModel):
    status: Literal["accepted"] = "accepted"
    article_id: str
    program_id: str
    chunks_planned: int
    message: str


class ArticleOut(BaseModel):
    article_id: str
    program_id: str
    title: str
    published_date: str
    added_at: int
    chunks: int
    is_new: bool


# --------------------------------------------------------------------------- #
# Guardrail (structured output)
# --------------------------------------------------------------------------- #
class SafetyCheckSchema(BaseModel):
    """Структурированный ответ LLM-guardrail'а."""

    is_unsafe_request: bool = Field(
        description="True, если пользователь просит дозировку препарата/БАД, схему приёма, отмену назначений врача или диагноз."
    )
    category: Literal["dosage", "regimen", "diagnosis", "override_prescription", "none"] = Field(
        default="none", description="Категория нарушения"
    )
    reason: str = Field(default="", description="Короткое обоснование на русском")


class GuardrailResult(BaseModel):
    is_unsafe_request: bool
    category: str = "none"
    reason: str = ""
    source: Literal["llm", "heuristic", "none"] = "none"


# --------------------------------------------------------------------------- #
# События SSE
# --------------------------------------------------------------------------- #
class SafetyFlagEvent(BaseModel):
    type: Literal["safety_flag"] = "safety_flag"
    triggered: bool


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    items: List[SourceItem]


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str


class MetaEvent(BaseModel):
    type: Literal["meta"] = "meta"
    session_id: str
    expert_id: str
    program_id: str
    is_new_knowledge: bool = False


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


def sse_payload(event: BaseModel | Dict[str, Any]) -> str:
    """Сериализация события в строку SSE-потока (data: {...})."""
    if isinstance(event, BaseModel):
        return event.model_dump_json()
    return BaseModel.model_dump_json(event)  # pragma: no cover
