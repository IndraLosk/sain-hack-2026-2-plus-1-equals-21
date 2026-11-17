"""Путь POST /run — запуск кейса: принять набор запросов, вернуть набор ответов.

Контракт /run (см. run-openapi.json):
  вход  — {"case": int, "dataset_code": str} + Authorization: Bearer <токен>
  выход — Case4Answers {"requests": [...]} без внешней обёртки.
"""

from fastapi import APIRouter, Depends

from backend.schemas import RunBody, RunResult
from backend.services import run_case
from backend.security import require_run_token

router = APIRouter()


@router.post("/run", response_model=RunResult)
def run(
    body: RunBody,
    token: str = Depends(require_run_token),
) -> RunResult:
    """Принять запрос от стенда (case + dataset_code) и вернуть ответы.

    Токен из заголовка прокидывается в run_case (сейчас не используется,
    но по контракту он нужен для getData).
    """
    return run_case(body, run_token=token)