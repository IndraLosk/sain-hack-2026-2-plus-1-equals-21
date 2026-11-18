"""Обёртки над удалённой LLM (OpenAI-совместимый прокси) и guardrail-логика.

LLM доступна только через прокси сервиса (OpenAI-совместимый API): адрес, ключ
и модель в env LLM_BASE_URL / LLM_API_KEY / LLM_MODEL. Пакет `openai` объявлен
в requirements.txt, но НЕ устанавливается автоматически: без него приложение
работает в режиме эвристического guardrail (см. llm_guardrail).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from .config import get_settings
from .schemas import GuardrailResult, SafetyCheckSchema

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# OpenAI-совместимый клиент
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """Асинхронный клиент OpenAI-совместимого прокси: chat/stream + JSON-режим.

    Пакет `openai` подключается лениво: если он не установлен, health() вернёт
    ok=False с пояснением, а guardrail продолжит работать на эвристике.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = settings.resolved_llm_api_key
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_num_ctx or 8192
        self.timeout = settings.llm_request_timeout

    # Инициализация настоящего клиента — лениво, чтобы не падал импорт.
    def _client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Библиотека 'openai' не установлена. Установите зависимости: "
                "pip install openai  (см. requirements.txt)"
            ) from exc
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _max_or_none(self, num_predict: int | None) -> int | None:
        # открытые прокси часто ограничены; по умолчанию используем llm_num_ctx
        return num_predict if num_predict is not None else self.max_tokens

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        num_predict: int | None = None,
    ) -> str:
        client = self._client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self._max_or_none(num_predict),
            "stream": False,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await client.chat.completions.create(**kwargs)
        finally:
            await client.close()
        if not resp.choices:
            return ""
        return str((resp.choices[0].message or {}).content or "")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> AsyncIterator[str]:
        client = self._client()
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self._max_or_none(num_predict),
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                piece = (chunk.choices[0].delta or {}).content
                if piece:
                    yield piece
        finally:
            await client.close()

    async def health(self) -> Dict[str, Any]:
        try:
            client = self._client()
            # не-деструктивно проверяем простым запросом моделей (прокси OpenAI)
            try:
                models = await client.models.list()
                names = [m.id for m in models.data]
                ok = True
            except Exception as exc:  # прокси может не отдавать /models
                logger.debug("models.list недоступен: %s", exc)
                names = []
                ok = True  # сам прокси доступен, если не бросил на подключение
            finally:
                await client.close()
            return {"ok": ok, "models": names, "model_available": self.model in names}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": str(exc), "model_available": False}


_llm: Optional[OpenAIClient] = None


def get_llm() -> OpenAIClient:
    global _llm
    if _llm is None:
        _llm = OpenAIClient()
    return _llm


# --------------------------------------------------------------------------- #
# Guardrail
# --------------------------------------------------------------------------- #
_DOSAGE_PATTERNS = [
    r"\b\d+([.,]\d+)?\s*(мг|мкг|г|гр|мл|капл\w*|таблет\w*|доз\w*|mg|mcg|ml|iu|ме)\b",
    r"\bсколько\b.{0,40}\b(мг|мкг|мл|капель|таблеток|доз\w*)",
    r"\b(дозировк\w*|дозу|доза)\b",
]
_REGIMEN_PATTERNS = [
    r"\b(схем\w*\s+при[её]м\w*|как\s+принимать|сколько\s+раз\s+в\s+день|курс\w*\s+при[её]м\w*)",
    r"\b(пропиши|назначь|выпиши)\b.{0,40}\b(препарат|лекарств|таблетк|бад|мелатонин|добавк)",
]
_DIAGNOSIS_PATTERNS = [
    r"\b(поставь\s+диагноз|какой\s+у\s+меня\s+диагноз|что\s+у\s+меня\s+за\s+болезнь|диагноз)\b",
]
_OVERRIDE_PATTERNS = [
    r"\b(отмен\w*|бросить|перестать\s+принимать)\b.{0,40}\b(назначен\w*|врач\w*|препарат\w*|таблетк\w*)",
    r"\bвместо\s+(того,?\s*)?что\s+назначил\s+врач\b",
]
_SUBSTANCE_WORDS = (
    "мелатонин",
    "магни",
    "мелатонина",
    "бад",
    "добавк",
    "препарат",
    "таблетк",
    "капл",
    "снотворн",
    "феназепам",
    "адаптол",
    "глицин",
    "витамин",
    "омега",
    "мелатонин",
    "пустырник",
    "валерьян",
)

_COMPILED = {
    "dosage": [re.compile(p, re.IGNORECASE) for p in _DOSAGE_PATTERNS],
    "regimen": [re.compile(p, re.IGNORECASE) for p in _REGIMEN_PATTERNS],
    "diagnosis": [re.compile(p, re.IGNORECASE) for p in _DIAGNOSIS_PATTERNS],
    "override_prescription": [re.compile(p, re.IGNORECASE) for p in _OVERRIDE_PATTERNS],
}


def heuristic_guardrail(text: str) -> GuardrailResult:
    """Быстрая детерминированная проверка — работает даже без LLM."""
    lowered = (text or "").lower()
    hits: List[str] = []
    for category, patterns in _COMPILED.items():
        for pattern in patterns:
            if pattern.search(lowered):
                hits.append(category)
                break
    # «сколько миллиграмм мелатонина» — дозировка без единиц измерения
    if not hits and re.search(r"\bсколько\b", lowered) and any(w in lowered for w in _SUBSTANCE_WORDS):
        hits.append("dosage")
    if hits:
        return GuardrailResult(
            is_unsafe_request=True,
            category=hits[0],
            reason="Сработал эвристический фильтр: запрос дозировки/схемы/диагноза.",
            source="heuristic",
        )
    return GuardrailResult(is_unsafe_request=False, category="none", reason="", source="none")


_GUARD_SYSTEM = (
    "Ты — медицинский фильтр безопасности. Твоя задача: определить, просит ли пользователь то, "
    "что нельзя давать в чате самопомощи.\n"
    "Небезопасно (is_unsafe_request=true):\n"
    "1) дозировка лекарств, БАДов, гормонов, витаминов (мг, мкг, капли, таблетки, «сколько пить»);\n"
    "2) схема/курс приёма препаратов, «как принимать», «сколько раз в день»;\n"
    "3) отмена или замена назначений врача;\n"
    "4) постановка диагноза по симптомам.\n"
    "Безопасно (is_unsafe_request=false): вопросы о поведении, режиме сна, свете, температуре, "
    "кофеине, питании, движении, стрессе, общие вопросы «как работает мелатонин естественно».\n"
    "Отвечай строго JSON-объектом с ключами: is_unsafe_request (boolean), category "
    "(dosage|regimen|diagnosis|override_prescription|none), reason (строка, максимум 20 слов, на русском)."
)


def _parse_safety_json(raw: str) -> Optional[SafetyCheckSchema]:
    raw = (raw or "").strip()
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(data.get("is_unsafe_request"), str):
        data["is_unsafe_request"] = data["is_unsafe_request"].strip().lower() in {"true", "1", "yes", "да"}
    if data.get("category") not in {"dosage", "regimen", "diagnosis", "override_prescription", "none"}:
        data["category"] = "none"
    try:
        return SafetyCheckSchema.model_validate(data)
    except Exception:
        return None


async def llm_guardrail(text: str, llm: OpenAIClient | None = None) -> GuardrailResult:
    """LLM-проверка с json-режимом + fallback на эвристику."""
    heuristic = heuristic_guardrail(text)
    llm = llm or get_llm()
    messages = [
        {"role": "system", "content": _GUARD_SYSTEM},
        {
            "role": "user",
            "content": f"Запрос пользователя:\n<<<\n{text}\n>>>\nВерни только JSON.",
        },
    ]
    try:
        raw = await llm.chat(messages, temperature=0.0, json_mode=True, num_predict=160)
        parsed = _parse_safety_json(raw)
    except Exception as exc:  # LLM недоступна — работаем на эвристике
        logger.warning("guardrail LLM failed: %s", exc)
        parsed = None

    if parsed is None:
        return heuristic

    # объединяем: любая положительная проверка = небезопасно
    if heuristic.is_unsafe_request and not parsed.is_unsafe_request:
        return GuardrailResult(
            is_unsafe_request=True,
            category=heuristic.category,
            reason=heuristic.reason,
            source="heuristic",
        )
    return GuardrailResult(
        is_unsafe_request=parsed.is_unsafe_request,
        category=parsed.category,
        reason=parsed.reason,
        source="llm",
    )


async def llm_guardrail_async(text: str) -> GuardrailResult:
    return await asyncio.wait_for(llm_guardrail(text), timeout=get_settings().llm_request_timeout)


__all__ = ["OpenAIClient", "get_llm", "heuristic_guardrail", "llm_guardrail", "llm_guardrail_async"]