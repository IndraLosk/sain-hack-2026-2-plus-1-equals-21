"""Pydantic-схемы API и внутренние схемы графа."""
from __future__ import annotations

import re
from typing import List, Literal, Optional

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


class MultiChatRequest(BaseModel):
    """Один вопрос — всем экспертам. Источники общие, различается только стиль ответа."""

    session_id: str = Field(min_length=1, max_length=128)
    program_id: str = Field(default="sleep_recovery", min_length=1)
    expert_ids: List[str] = Field(default_factory=list, max_length=8)
    message: str = Field(min_length=1, max_length=4000)
    user_id: Optional[str] = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def _strip_multi(cls, v: str) -> str:
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
# Guardrail
# --------------------------------------------------------------------------- #
class SafetyCheckSchema(BaseModel):
    is_unsafe_request: bool
    category: Literal["dosage", "regimen", "diagnosis", "override_prescription", "none"] = "none"
    reason: str = ""


class GuardrailResult(BaseModel):
    is_unsafe_request: bool
    category: str = "none"
    reason: str = ""
    source: Literal["llm", "heuristic", "none"] = "none"


# --------------------------------------------------------------------------- #
# Память и пробелы знаний
# --------------------------------------------------------------------------- #
class MemoryItem(BaseModel):
    key: str
    value: str
    confidence: float = 0.6
    updated_at: Optional[float] = None


class MemoryOut(BaseModel):
    user_id: str
    items: List[MemoryItem] = []


class MemoryIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    key: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=500)


class KnowledgeGapOut(BaseModel):
    id: str
    query: str
    program_id: str
    expert_id: str
    session_id: str
    occurrences: int = 1
    top_similarity: float = 0.0
    status: str = "open"
    created_at: float
    updated_at: float


# --------------------------------------------------------------------------- #
# Рекомендация статьи (ссылка) и список статей
# --------------------------------------------------------------------------- #
class ArticleRecommendIn(BaseModel):
    """Кнопка «Порекомендовать статью»: пользователь присылает ссылку."""

    url: str = Field(min_length=8, max_length=1024)
    title: str = Field(default="", max_length=500)
    program_id: str = Field(default="sleep_recovery", min_length=1)
    user_id: Optional[str] = Field(default=None, max_length=128)
    comment: str = Field(default="", max_length=2000)

    @field_validator("url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^https?://[^\s]+\.[^\s]+", v, re.I):
            raise ValueError("нужна ссылка вида https://example.com/article")
        return v


class RecommendedArticleOut(BaseModel):
    id: str
    url: str
    title: str
    program_id: str
    user_id: Optional[str] = None
    comment: str = ""
    status: Literal["pending", "indexed", "failed"] = "pending"
    chunks: int = 0
    error: str = ""
    source_kind: str = "link"
    created_at: float
    updated_at: float


class UploadAccepted(BaseModel):
    status: Literal["accepted"] = "accepted"
    article_id: str
    title: str
    message: str


class UsedChunkOut(BaseModel):
    id: str
    article_id: str
    article_title: str = ""
    section_id: str = ""
    section_title: str = ""
    text: str
    query: str = ""
    session_id: Optional[str] = None
    expert_id: Optional[str] = None
    similarity: float = 0.0
    used_at: float


class ArticleDetailOut(BaseModel):
    """Статья + только те фрагменты, которые агент реально подставил в ответ."""

    article_id: str
    title: str
    url: str = ""
    program_id: str = ""
    status: str = "unknown"
    chunks: int = 0
    source_kind: str = "vector"
    added_at: float = 0.0
    used_chunks: List[UsedChunkOut] = []
