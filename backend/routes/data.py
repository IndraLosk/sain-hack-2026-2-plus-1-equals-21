"""Пути чтения данных через сервисный слой: статистика, эксперты, статьи, запросы."""

from fastapi import APIRouter

from ..services import stats, list_experts, list_articles, list_requests

router = APIRouter()


@router.get("/stats")
def get_stats():
    return stats()


@router.get("/experts")
def get_experts():
    return list_experts()


@router.get("/articles")
def get_articles():
    return list_articles()


@router.get("/requests")
def get_requests():
    return list_requests()