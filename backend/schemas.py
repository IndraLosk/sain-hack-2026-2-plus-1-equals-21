"""Pydantic-модели контрактов кейса 4 — по реальным схемам.

POST /run приложения команды (см. run-openapi.json):
  вход  — тело {"case": int, "dataset_code": str} + Authorization: Bearer <token>.
          Датасет приложение получает САМО через GET /case/4/getData по
          dataset_code + токену; в теле /run датасета нет.
  выход — Case4Answers.schema.json → {"requests": [...]}; у каждого ответа
          обязательный citations (minItems: 1, uniqueItems: true).

Модели отражают схемы организаторов. Держим их в одном месте.
Link: https://sainorg.github.io/sain-hackathon-docs/schemas/
"""

from typing import List

from pydantic import BaseModel, Field


# --- Вход POST /run ─────────────────────────────────────────────────────

class RunBody(BaseModel):
    """Тело запроса /run: какие данные взять. Сам набор сюда НЕ передаётся."""
    case: int = Field(4, ge=1, le=4, description="номер кейса (у нас 4)")
    dataset_code: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
        description="код набора; по нему приложение запросит данные через getData",
    )


# --- Выход Case4Answers ─────────────────────────────────────────────────

class Citation(BaseModel):
    article_id: str
    section_id: str


class RunAnswer(BaseModel):
    record_id: str
    text: str
    refused: bool
    # обязателен и ≥1 по схеме
    citations: List[Citation] = Field(min_items=1)


class RunResult(BaseModel):
    requests: List[RunAnswer]