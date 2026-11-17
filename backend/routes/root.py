"""Базовые пути: корень и health. Только тонкая обёртка над services."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..services import check_health

router = APIRouter()


@router.get("/")
def root():
    return {"service": "sain-case4", "status": "ok"}


@router.get("/health")
def health():
    try:
        return check_health()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "error", "db": "down", "detail": str(exc)},
        )