"""Конфигурация приложения и валидируемая модель каталога (config/catalog.json)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # корень проекта


class Settings(BaseSettings):
    """Настройки читаются из окружения или из .env в корне проекта."""

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # --- LLM (OpenAI-совместимый прокси; на стенде задаётся организаторами) ---
    llm_base_url: str = "https://bifrost.be-healthy.tech/v1"
    llm_api_key: str = ""  # заполняется из env LLM_API_KEY или файла llm.key (см. ниже)
    llm_model: str = "gpt-6-luna"
    llm_temperature: float = 0.3
    llm_num_ctx: int = 8192
    llm_request_timeout: float = 300.0
    # модель для guardrail (можно оставить той же)
    guard_model: str = "gpt-6-luna"

    # Файл ключа LLM (локальная разработка) — по аналогии с api.token
    llm_key_file: str = str(BASE_DIR / "llm.key")

    # --- эмбеддинги / chroma ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "mps"
    chroma_dir: str = str(BASE_DIR / "data" / "chroma")
    chroma_collection: str = "scientific_articles"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 3

    # --- память диалогов ---
    checkpoints_db: str = str(BASE_DIR / "data" / "checkpoints.sqlite")

    # --- контент ---
    catalog_path: str = str(BASE_DIR / "config" / "catalog.json")
    freshness_days: int = 7
    # единственный источник научных статей — датасет кейса 4
    dataset_path: str = str(BASE_DIR / "data" / "case4-dev.json")

    # --- прочее ---
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> List[str]:
        raw = (self.cors_origins or "*").strip()
        return ["*"] if raw == "*" else [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def resolved_llm_api_key(self) -> str:
        """Ключ LLM: env LLM_API_KEY → файл llm.key → пусто (guardrail уйдёт в эвристику)."""
        if self.llm_api_key:
            return self.llm_api_key
        key = os.environ.get("LLM_API_KEY", "").strip()
        if key:
            return key
        try:
            if os.path.exists(self.llm_key_file):
                with open(self.llm_key_file, encoding="utf-8") as fh:
                    k = fh.read().strip()
                if k:
                    return k
        except OSError:
            pass
        return ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# Каталог программ и экспертов
# --------------------------------------------------------------------------- #
class Program(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cover_color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    default_expert_id: str = "clinician"


class Expert(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    name: str = Field(min_length=1)
    role_badge: str = Field(min_length=1)
    system_style_prompt: str = Field(min_length=20)
    avatar_url: str = ""


class Catalog(BaseModel):
    version: str = "1.0.0"
    programs: List[Program] = Field(min_length=1)
    experts: List[Expert] = Field(min_length=1)

    @field_validator("experts")
    @classmethod
    def _unique_experts(cls, v: List[Expert]) -> List[Expert]:
        ids = [e.id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("expert ids must be unique")
        return v

    @field_validator("programs")
    @classmethod
    def _unique_programs(cls, v: List[Program]) -> List[Program]:
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("program ids must be unique")
        return v

    def program_map(self) -> Dict[str, Program]:
        return {p.id: p for p in self.programs}

    def expert_map(self) -> Dict[str, Expert]:
        return {e.id: e for e in self.experts}

    def program(self, program_id: str) -> Program:
        programs = self.program_map()
        if program_id not in programs:
            raise KeyError(f"unknown program_id: {program_id}")
        return programs[program_id]

    def expert(self, expert_id: str) -> Expert:
        experts = self.expert_map()
        if expert_id not in experts:
            raise KeyError(f"unknown expert_id: {expert_id}")
        return experts[expert_id]


def load_catalog(path: str | None = None) -> Catalog:
    """Читает и валидирует каталог. Падает при старте, если конфиг битый."""
    catalog_path = Path(path or get_settings().catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"catalog not found: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    catalog = Catalog.model_validate(raw)
    known = set(catalog.expert_map())
    for program in catalog.programs:
        if program.default_expert_id not in known:
            raise ValueError(
                f"program {program.id} references unknown default_expert_id {program.default_expert_id}"
            )
    return catalog


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
    "os",
]
