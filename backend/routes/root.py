"""Базовые пути: GET / — пинг стенда."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {"service": "sain-case4", "status": "ok"}