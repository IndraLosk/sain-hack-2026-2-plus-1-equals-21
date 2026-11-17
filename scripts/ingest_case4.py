"""Загрузка доказательной базы из датасета кейса 4 (data/case4-dev.json).

Единственный источник научных статей приложения — датасет `data/case4-dev.json`
(20 статей, CC BY). Демо-статьи удалены.

Запуск:
    .venv/bin/python scripts/ingest_case4.py            # пересобрать базу с нуля
    .venv/bin/python scripts/ingest_case4.py --keep     # досинхронизировать без очистки
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.vectorstore import add_article_async, get_store  # noqa: E402


def load_dataset(path: Path) -> dict:
    """Датасет может быть как чистым Case4Dataset, так и обёрткой {dataset, reference}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("dataset", raw)


def program_for_article(article: dict, program_ids: list[str]) -> str:
    """Все статьи датасета — про сон/циркадные ритмы, поэтому кладём в программу сна."""
    preferred = "sleep_recovery"
    if preferred in program_ids:
        return preferred
    return program_ids[0]


def build_kwargs(article: dict, program_id: str) -> dict:
    year = int(article.get("year") or 2020)
    # дата публикации: год статьи (датасет даёт только год)
    published = f"{year:04d}-01-01"
    sections = [
        {
            "section_id": str(s.get("section_id")),
            "title": str(s.get("title") or ""),
            "text": str(s.get("text") or ""),
        }
        for s in article.get("sections") or []
        if str(s.get("text") or "").strip()
    ]
    license_info = article.get("license") or {}
    return {
        "program_id": program_id,
        "title": str(article.get("title") or article.get("article_id")),
        "published_date": published,
        "content": "\n\n".join(s["text"] for s in sections),
        "article_id": str(article.get("article_id")),
        "sections": sections,
        "added_at": None,  # None → время индексации (статьи считаются свежими для бейджа NEW)
        "extra_metadata": {
            "authors": ", ".join(article.get("authors") or []),
            "doi": article.get("doi"),
            "year": year,
            "license": license_name(license_info),
            "source": "case4-dev",
        },
    }


def license_name(info: dict) -> str:
    if not info:
        return ""
    return f"{info.get('name', '')} {info.get('version', '')}".strip()


async def main(reset: bool) -> int:
    settings = get_settings()
    dataset_path = Path(settings.dataset_path)
    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path}")
        return 1

    dataset = load_dataset(dataset_path)
    articles = dataset.get("articles") or []
    if not articles:
        print("dataset has no articles")
        return 1

    store = get_store()
    if reset:
        print(f"reset collection {store.collection_name} ({store.count()} chunks)")
        await asyncio.to_thread(store.reset)

    from app.config import get_catalog

    program_ids = [p.id for p in get_catalog().programs]

    total_chunks = 0
    for article in articles:
        program_id = program_for_article(article, program_ids)
        kwargs = build_kwargs(article, program_id)
        article_id, chunks = await add_article_async(**kwargs)
        total_chunks += chunks
        print(f"  {article_id:<8} {chunks:>3} чанков  {kwargs['title'][:60]}")

    print(f"\nИтого: {len(articles)} статей, {total_chunks} чанков, всего в базе: {store.count()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка статей из data/case4-dev.json в ChromaDB")
    parser.add_argument("--keep", action="store_true", help="не очищать коллекцию перед загрузкой")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(reset=not args.keep)))
