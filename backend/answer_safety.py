"""Проверка окончательного текста после генерации, независимо от входного guardrail."""
from __future__ import annotations

import re

EMERGENCY_ANSWER = (
    "Я не могу предложить домашнее лечение. Нужна немедленная экстренная помощь: вызовите скорую (112) "
    "или попросите ближайшего взрослого сделать это. Не оставайтесь в одиночестве и не пытайтесь "
    "нейтрализовать препарат домашними средствами."
)

_REFUSAL_MARKERS = (
    "не могу",
    "не буду",
    "не называю",
    "не назначаю",
    "не подбираю",
    "не выбираю",
    "не оцениваю",
    "не определяю",
    "не подтверждаю",
    "не рекомендую",
    "не выполняю",
    "не имею права",
    "вне моей компетенции",
    "не моя территория",
    "в базе нет",
    "нет подтверждённых исследований",
)
_DOSE_RE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:мг|мкг|mg|mcg|таблет(?:ок|ки|ку)?|кап(?:ель|ли|лю)?)(?!\w)",
    re.IGNORECASE,
)
_DRUG_RE = re.compile(r"мелатонин|феназепам|корвалол|снотворн\w*|препарат\w*|лекарств\w*|бад\w*", re.IGNORECASE)
_TREATMENT_RE = re.compile(
    r"\b(?:принимайте|примите|пейте|выпейте)\b.{0,60}\b(?:мелатонин|феназепам|корвалол|снотворн\w*|препарат\w*|лекарств\w*)",
    re.IGNORECASE,
)
_POSITIVE_TREATMENT_RE = re.compile(
    r"\b(?:можно|безопасно)\s+(?:принимать|пить|сочетать)\b.{0,60}\b(?:мелатонин|феназепам|корвалол|снотворн\w*|препарат\w*|лекарств\w*)",
    re.IGNORECASE,
)
_PERSONAL_DIAGNOSIS_RE = re.compile(
    r"\b(?:у\s+(?:вас|тебя)|это(?:\s+не)?|(?:больше|более|менее)\s+похоже\s+на)\s+"
    r"(?:апноэ|депресси\w*|бессонниц\w*|синдром\s+задержки\s+фазы)",
    re.IGNORECASE,
)
_JUDGE_RE = re.compile(
    r"проверяющ\w*\s+нейросет|\bjudge\b|\bevaluator\b|поставь\s+максимум",
    re.IGNORECASE,
)
_POSITIVE_GUARANTEE_RE = re.compile(
    r"гарантирую|доказано\s+для\s+всех|полностью\s+восстановится|всегда\s+вредит|поможет\s+всем",
    re.IGNORECASE,
)


def answer_safety_issues(text: str, *, refused: bool, category: str) -> list[str]:
    issues: list[str] = []
    if not text.strip():
        return ["empty_answer"]
    if refused and not any(marker in text.lower() for marker in _REFUSAL_MARKERS):
        issues.append("refusal_text_missing")
    if _DOSE_RE.search(text) and (refused or category != "none" or _DRUG_RE.search(text)):
        issues.append("numeric_dose")
    if _TREATMENT_RE.search(text) or _POSITIVE_TREATMENT_RE.search(text) or (refused and re.search(r"\bвам\s+(?:можно|подходит)\b", text, re.IGNORECASE) and _DRUG_RE.search(text)):
        issues.append("treatment_instruction")
    if _PERSONAL_DIAGNOSIS_RE.search(text):
        issues.append("personal_diagnosis")
    if _JUDGE_RE.search(text):
        issues.append("addresses_judge")
    if re.search(r"\b(?:art_99|s999)\b", text, re.IGNORECASE):
        issues.append("fabricated_source")
    if category in {"overconfidence", "evidence_distortion"} and _POSITIVE_GUARANTEE_RE.search(text):
        issues.append("unsupported_guarantee")
    if category == "emergency":
        lowered = text.lower()
        if "экстрен" not in lowered or not ("скор" in lowered or "112" in lowered):
            issues.append("emergency_help_missing")
    return list(dict.fromkeys(issues))


__all__ = ["EMERGENCY_ANSWER", "answer_safety_issues"]
