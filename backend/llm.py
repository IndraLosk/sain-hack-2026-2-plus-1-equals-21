"""LLM-слой: OpenAI-совместимый клиент (по умолчанию DeepSeek) и guardrail безопасности.

Провайдер выбирается переменной LLM_PROVIDER:
* `deepseek` (по умолчанию) — DEEPSEEK_BASE_URL + DEEPSEEK_API_KEY, модель DEEPSEEK_MODEL;
* `openai` — LLM_BASE_URL + LLM_API_KEY (стенд хакатона).

Общий пул HTTP-соединений на процесс, ретраи с backoff, стриминг,
JSON-режим для guardrail и извлечения памяти.
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

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


# --------------------------------------------------------------------------- #
# Общий HTTP-клиент
# --------------------------------------------------------------------------- #
_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(get_settings().llm_request_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            headers={"User-Agent": "sain-health-agent/2.0"},
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def _backoff(attempt: int) -> None:
    await asyncio.sleep(min(0.6 * (2**attempt), 8.0))


# --------------------------------------------------------------------------- #
# OpenAI-совместимый провайдер
# --------------------------------------------------------------------------- #
class OpenAICompatLLM:
    """Клиент /chat/completions: модель активного провайдера (DeepSeek по умолчанию)."""

    def __init__(self, provider: Optional[str] = None) -> None:
        settings = get_settings()
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_request_timeout
        self.max_retries = max(1, settings.llm_max_retries)
        self.use_provider(provider or settings.provider)

    def use_provider(self, provider: str) -> None:
        """Переключает провайдера на лету (deepseek ↔ openai), не пересоздавая клиент."""
        settings = get_settings()
        name = (provider or settings.provider).strip().lower()
        if name not in AVAILABLE_PROVIDERS:
            name = settings.provider
        self.provider = name
        if name == "deepseek":
            self.model = settings.deepseek_model
            self.base_url = (settings.deepseek_base_url or "").rstrip("/")
            self.api_key = settings.deepseek_api_key
        else:
            self.model = settings.llm_model
            self.base_url = (settings.llm_base_url or "").rstrip("/")
            self.api_key = settings.llm_api_key
        logger.info("LLM provider switched to %s (%s)", name, self.model)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _payload(
        self,
        messages: List[Dict[str, str]],
        *,
        stream: bool,
        temperature: Optional[float],
        json_mode: bool,
        num_predict: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": num_predict or self.max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _require_key(self) -> None:
        if not self.configured:
            key_name = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "LLM_API_KEY"
            raise RuntimeError(f"{key_name} не задан — LLM недоступна")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        num_predict: Optional[int] = None,
    ) -> str:
        self._require_key()
        payload = self._payload(
            messages, stream=False, temperature=temperature, json_mode=json_mode, num_predict=num_predict
        )
        client = get_http_client()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
                if resp.status_code in _RETRY_STATUS:
                    resp.raise_for_status()
                resp.raise_for_status()
                choices = resp.json().get("choices") or []
                return str((choices[0].get("message") or {}).get("content") or "") if choices else ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM chat attempt %s/%s failed: %s", attempt + 1, self.max_retries, exc)
                if attempt + 1 < self.max_retries:
                    await _backoff(attempt)
        raise RuntimeError(f"LLM unavailable: {last_error}")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> AsyncIterator[str]:
        self._require_key()
        payload = self._payload(
            messages, stream=True, temperature=temperature, json_mode=False, num_predict=num_predict
        )
        client = get_http_client()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            produced = False
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        piece = (choices[0].get("delta") or {}).get("content") if choices else ""
                        if piece:
                            produced = True
                            yield piece
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM stream attempt %s/%s failed: %s", attempt + 1, self.max_retries, exc)
                if produced:  # часть ответа уже отдана клиенту — не дублируем текст
                    raise
                if attempt + 1 < self.max_retries:
                    await _backoff(attempt)
        raise RuntimeError(f"LLM stream unavailable: {last_error}")

    async def health(self) -> Dict[str, Any]:
        if not self.configured:
            key_name = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "LLM_API_KEY"
            return {"ok": False, "provider": self.provider, "model": self.model, "error": f"{key_name} is not set"}
        try:
            resp = await get_http_client().get(f"{self.base_url}/models", headers=self._headers(), timeout=10.0)
            resp.raise_for_status()
            names = [m.get("id") for m in (resp.json().get("data") or [])]
            return {"ok": True, "provider": self.provider, "model": self.model, "model_available": self.model in names}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": self.provider, "model": self.model, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Фабрики
# --------------------------------------------------------------------------- #
_llm: Optional[OpenAICompatLLM] = None

# Провайдеры, между которыми можно переключаться из UI.
AVAILABLE_PROVIDERS: Dict[str, Dict[str, str]] = {
    "openai": {"label": "gpt-6-luna", "hint": "стенд хакатона"},
    "deepseek": {"label": "deepseek-chat", "hint": "DeepSeek API"},
}


def get_llm() -> OpenAICompatLLM:
    """Единственный клиент активной модели (создаётся один раз на процесс)."""
    global _llm
    if _llm is None:
        _llm = OpenAICompatLLM()
    return _llm


def set_provider(provider: str) -> Dict[str, Any]:
    """Переключает LLM на лету. Возвращает новое состояние провайдера."""
    name = (provider or "").strip().lower()
    if name not in AVAILABLE_PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    llm = get_llm()
    llm.use_provider(name)
    return provider_info()


def provider_info() -> Dict[str, Any]:
    """Текущий провайдер + список доступных для переключателя в UI."""
    llm = get_llm()
    options = [
        {
            "id": key,
            "label": meta["label"],
            "hint": meta["hint"],
            "configured": bool(
                get_settings().deepseek_api_key if key == "deepseek" else get_settings().llm_api_key
            ),
        }
        for key, meta in AVAILABLE_PROVIDERS.items()
    ]
    return {
        "provider": llm.provider,
        "model": llm.model,
        "configured": bool(llm.api_key),
        "options": options,
    }


# --------------------------------------------------------------------------- #
# Guardrail безопасности
# --------------------------------------------------------------------------- #
_DOSAGE_RE = re.compile(r"\b(дозировк\w*|сколько\s+(мг|миллиграмм|таблеток|капель|г|грамм)\w*)", re.IGNORECASE)
_REGIMEN_RE = re.compile(
    r"\b(схем\w*\s+при[её]м\w*|как\s+принимать|сколько\s+раз\s+в\s+день|курс\w*\s+при[её]м\w*)", re.IGNORECASE
)
_PRESCRIBE_RE = re.compile(
    r"\b(пропиши|назначь|выпиши)\b.{0,40}\b(препарат|лекарств|таблетк|бад|мелатонин|добавк)", re.IGNORECASE
)
_DIAGNOSIS_RE = re.compile(
    r"\b(поставь\s+диагноз|какой\s+у\s+меня\s+диагноз|что\s+у\s+меня\s+за\s+болезнь|диагноз)\b", re.IGNORECASE
)
_OVERRIDE_RE = re.compile(
    r"\b(отмен\w*|бросить|перестать\s+принимать)\b.{0,40}\b(назначен\w*|врач\w*|препарат\w*|таблетк\w*)",
    re.IGNORECASE,
)
_INSTEAD_RE = re.compile(r"\bвместо\s+(того,?\s*)?что\s+назначил\s+врач\b", re.IGNORECASE)

_SAFETY_SYSTEM = (
    "Ты — медицинский фильтр безопасности. Определи, просит ли пользователь то, что нельзя давать в чате самопомощи.\n"
    "Небезопасно (is_unsafe_request=true): 1) дозировка лекарств, БАДов, гормонов, витаминов; 2) схема/курс приёма препаратов; "
    "3) отмена или замена назначений врача; 4) постановка диагноза.\n"
    "Безопасно (false): вопросы о поведении, режиме сна, свете, температуре, кофеине, питании, движении, стрессе.\n"
    "Отвечай строго JSON: is_unsafe_request (boolean), category (dosage|regimen|diagnosis|override_prescription|none), "
    "reason (до 20 слов, по-русски)."
)


def heuristic_guardrail(text: str) -> GuardrailResult:
    """Быстрая проверка без LLM: ловит явные просьбы о дозировке/схеме/диагнозе."""
    lowered = (text or "").strip()
    checks = (
        ("dosage", _DOSAGE_RE),
        ("regimen", _REGIMEN_RE),
        ("override_prescription", _PRESCRIBE_RE),
        ("diagnosis", _DIAGNOSIS_RE),
        ("override_prescription", _OVERRIDE_RE),
        ("override_prescription", _INSTEAD_RE),
    )
    for category, pattern in checks:
        match = pattern.search(lowered)
        if match:
            return GuardrailResult(
                is_unsafe_request=True,
                category=category,  # type: ignore[arg-type]
                reason=f"совпадение с шаблоном: {match.group(0)[:40]}",
            )
    return GuardrailResult(is_unsafe_request=False, category="none", reason="")


def _parse_safety_json(raw: str) -> Optional[SafetyCheckSchema]:
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return SafetyCheckSchema.model_validate(data)
    except Exception:  # noqa: BLE001
        return None


async def llm_guardrail(text: str) -> GuardrailResult:
    """Эвристика + LLM-проверка спорных случаев (structured JSON)."""
    heuristic = heuristic_guardrail(text)
    if heuristic.is_unsafe_request:
        return heuristic
    try:
        raw = await asyncio.wait_for(
            get_llm().chat(
                [
                    {"role": "system", "content": _SAFETY_SYSTEM},
                    {"role": "user", "content": (text or "")[:2000]},
                ],
                temperature=0.0,
                json_mode=True,
                num_predict=120,
            ),
            timeout=20.0,
        )
    except Exception as exc:  # noqa: BLE001 — LLM недоступна: остаёмся на эвристике
        logger.warning("llm guardrail failed, using heuristic: %s", exc)
        return heuristic
    parsed = _parse_safety_json(raw)
    if parsed is None:
        return heuristic
    return GuardrailResult(
        is_unsafe_request=bool(parsed.is_unsafe_request),
        category=parsed.category,
        reason=parsed.reason or "",
    )


__all__ = [
    "AVAILABLE_PROVIDERS",
    "provider_info",
    "set_provider",
    "get_http_client",
    "close_http_client",
    "OpenAICompatLLM",
    "get_llm",
    "heuristic_guardrail",
    "llm_guardrail",
]
