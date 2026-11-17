"""Обёртки над локальными LLM (Ollama, qwen2.5:7b) и guardrail-логика."""
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
# Ollama
# --------------------------------------------------------------------------- #
class OllamaLLM:
    """Минимальный асинхронный клиент Ollama: chat (stream / non-stream) + JSON-режим."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.temperature = settings.llm_temperature
        self.num_ctx = settings.llm_num_ctx
        self.timeout = settings.llm_request_timeout

    # ------------------------------------------------------------------ #
    def _payload(
        self,
        messages: List[Dict[str, str]],
        *,
        stream: bool,
        temperature: float | None = None,
        json_mode: bool = False,
        num_predict: int | None = None,
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        num_predict: int | None = None,
    ) -> str:
        payload = self._payload(
            messages, stream=False, temperature=temperature, json_mode=json_mode, num_predict=num_predict
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return str((data.get("message") or {}).get("content") or "")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True, temperature=temperature, num_predict=num_predict)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    piece = (chunk.get("message") or {}).get("content") or ""
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break

    async def health(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                names = [m.get("name") for m in resp.json().get("models", [])]
            return {"ok": True, "models": names, "model_available": self.model in names}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": str(exc), "model_available": False}


_llm: Optional[OllamaLLM] = None


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
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


async def llm_guardrail(text: str, llm: OllamaLLM | None = None) -> GuardrailResult:
    """LLM-проверка с with_structured_output(json_mode) + fallback на эвристику."""
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


__all__ = ["OllamaLLM", "get_llm", "heuristic_guardrail", "llm_guardrail", "llm_guardrail_async"]
