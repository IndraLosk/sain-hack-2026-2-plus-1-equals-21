"""Конфигурация приложения + валидируемый каталог программ и экспертов.

Все настройки читаются из окружения / .env (см. env.example).
Дефолты рассчитаны на стенд хакатона: LLM — OpenAI-совместимый шлюз,
поиск — лёгкий BM25 (чистый Python), хранилища — SQLite с лёгким переходом на PostgreSQL.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # корень проекта


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # --- LLM: OpenAI-совместимый API. Провайдер выбирается LLM_PROVIDER ---
    # provider=openai (по умолчанию) → gpt-6-luna, LLM_BASE_URL + LLM_API_KEY
    # provider=deepseek             → deepseek-chat, DEEPSEEK_BASE_URL + DEEPSEEK_API_KEY
    # Переключается из UI: POST /api/v1/llm {"provider": "openai"|"deepseek"}
    llm_provider: str = "openai"
    llm_model: str = "gpt-6-luna"
    llm_base_url: str = "https://bifrost.be-healthy.tech/v1"
    llm_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1200
    llm_request_timeout: float = 60.0
    llm_max_retries: int = 2
    history_messages: int = 8

    @property
    def provider(self) -> str:
        return (self.llm_provider or "openai").strip().lower()

    @property
    def active_model(self) -> str:
        return self.deepseek_model if self.provider == "deepseek" else self.llm_model

    @property
    def active_base_url(self) -> str:
        url = self.deepseek_base_url if self.provider == "deepseek" else self.llm_base_url
        return (url or "").rstrip("/")

    @property
    def active_api_key(self) -> str:
        return self.deepseek_api_key if self.provider == "deepseek" else self.llm_api_key

    # --- поиск: лёгкий BM25 (чистый Python, без моделей и векторов) ---
    chroma_dir: str = str(BASE_DIR / "data" / "bm25")  # каталог persist-копии индекса
    chroma_collection: str = "scientific_articles"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 3
    retrieval_top_k_multi: int = 6  # мульти-режим: больше контекста, чтобы ответы различались
    retrieval_min_similarity: float = 0.15  # ниже порога — «в базе нет подтверждённых исследований»

    # --- хранилища: SQLite по умолчанию, PostgreSQL при заданном DATABASE_URL ---
    database_url: str = ""
    sqlite_path: str = str(BASE_DIR / "data" / "app.sqlite")
    checkpoints_db: str = str(BASE_DIR / "data" / "checkpoints.sqlite")

    # --- каталог программ/экспертов и свежесть знаний ---
    catalog_path: str = str(BASE_DIR / "config" / "catalog.json")
    freshness_days: int = 7

    # --- API хакатона (данные кейса 4) ---
    dataset_path: str = str(BASE_DIR / "data" / "case4-dev.json")
    hack_api_url: str = "https://hack-api.be-healthy.tech"
    hack_team_token: str = ""
    dataset_code: str = "case4-dev-v2"
    dataset_cache_ttl: int = 3600

    # --- прочее ---
    cors_origins: str = "*"
    disclaimer: str = "Рекомендации ассистента Сэйн не заменяют консультацию врача"

    @property
    def cors_origin_list(self) -> List[str]:
        raw = (self.cors_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def hack_api_configured(self) -> bool:
        return bool((self.hack_api_url or "").strip() and (self.hack_team_token or "").strip())

    @property
    def effective_database_url(self) -> str:
        return (self.database_url or "").strip() or f"sqlite+aiosqlite:///{self.sqlite_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# Каталог: программы здоровья и эксперты-персоны
# --------------------------------------------------------------------------- #
class Program(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    cover_color: str = "#4F46E5"
    default_expert_id: Optional[str] = None


class Expert(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role_badge: str = ""
    avatar_url: str = ""
    system_style_prompt: str = Field(min_length=1)


class Catalog(BaseModel):
    version: str = "1.0.0"
    programs: List[Program] = []
    experts: List[Expert] = []

    @field_validator("experts")
    @classmethod
    def _unique_experts(cls, v: List[Expert]) -> List[Expert]:
        seen = set()
        return [e for e in v if not (e.id in seen or seen.add(e.id))]

    @field_validator("programs")
    @classmethod
    def _unique_programs(cls, v: List[Program]) -> List[Program]:
        seen = set()
        return [p for p in v if not (p.id in seen or seen.add(p.id))]

    def program_map(self) -> Dict[str, Program]:
        return {p.id: p for p in self.programs}

    def expert_map(self) -> Dict[str, Expert]:
        return {e.id: e for e in self.experts}

    def program(self, program_id: str) -> Program:
        programs = self.program_map()
        if program_id in programs:
            return programs[program_id]
        if self.programs:
            return self.programs[0]
        raise KeyError("catalog has no programs")

    def expert(self, expert_id: str) -> Expert:
        experts = self.expert_map()
        if expert_id in experts:
            return experts[expert_id]
        if self.experts:
            return self.experts[0]
        raise KeyError("catalog has no experts")


def load_catalog(path: Optional[str] = None) -> Catalog:
    """Читает и валидирует config/catalog.json."""
    catalog_path = Path(path or get_settings().catalog_path)
    with catalog_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Catalog.model_validate(data)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return load_catalog()


__all__ = [
    "BASE_DIR",
    "Settings",
    "get_settings",
    "Program",
    "Expert",
    "Catalog",
    "load_catalog",
    "get_catalog",
]
