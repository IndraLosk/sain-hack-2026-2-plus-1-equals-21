"""Реализация методов работы с базой данных.

Схема полностью определяет таблицы кейса 4. Проектировалась по case4-dev.json:
users, experts, articles, article_sections, requests, chat_history, judge_prompts,
dataset_meta. Всё сериализуемое (JSONB) — license и authors в articles.
"""

import json
import os
from typing import Optional

import psycopg

# --- подключение -------------------------------------------------------------

def get_connection():
    """Взять psycopg-соединение из переменных окружения (compose их задаёт)."""
    return psycopg.connect(
        host=os.environ.get("PG_HOST", "db"),
        port=os.environ.get("PG_PORT", "5432"),
        user=os.environ.get("PG_USER", "sain"),
        password=os.environ.get("PG_PASSWORD", "sain"),
        dbname=os.environ.get("PG_DB", "sain_case4"),
    )


# --- создание схемы ----------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id              TEXT PRIMARY KEY,
    birth_date           DATE,
    timezone             TEXT,
    goal                 TEXT,
    chronotype           TEXT,
    weekday_bedtime      TEXT,
    weekday_wake_time    TEXT,
    average_sleep_hours  DOUBLE PRECISION,
    social_jetlag        DOUBLE PRECISION,
    activity_level       TEXT,
    sedentary_hours      DOUBLE PRECISION,
    training_days        INTEGER,
    works_from_home      BOOLEAN,
    phone_before_sleep   BOOLEAN,
    movement_index       DOUBLE PRECISION,
    sleep_preferences    TEXT,
    disabled_widget_ids  TEXT
);

CREATE TABLE IF NOT EXISTS experts (
    expert_id    TEXT PRIMARY KEY,
    role         TEXT NOT NULL,
    speech_style TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS articles (
    article_id      TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         JSONB NOT NULL DEFAULT '[]',
    year            INTEGER,
    doi             TEXT,
    license_name    TEXT,
    license_version TEXT,
    license_url     TEXT
);

CREATE TABLE IF NOT EXISTS article_sections (
    section_id TEXT NOT NULL,
    article_id TEXT NOT NULL REFERENCES articles(article_id),
    title      TEXT,
    text       TEXT NOT NULL,
    section_no INTEGER,
    PRIMARY KEY (article_id, section_id)
);

CREATE TABLE IF NOT EXISTS requests (
    record_id TEXT PRIMARY KEY,
    user_id   TEXT REFERENCES users(user_id),
    expert_id TEXT REFERENCES experts(expert_id),
    group_id  TEXT,
    text      TEXT NOT NULL,
    role      TEXT NOT NULL,
    at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chat_history (
    message_id TEXT PRIMARY KEY,
    user_id    TEXT REFERENCES users(user_id),
    role       TEXT NOT NULL,
    at         TIMESTAMPTZ,
    text       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judge_prompts (
    prompt_id     TEXT PRIMARY KEY,
    version       TEXT,
    purpose       TEXT,
    model_version TEXT,
    temperature   DOUBLE PRECISION,
    text          TEXT
);

CREATE TABLE IF NOT EXISTS dataset_meta (
    meta_key   TEXT PRIMARY KEY,
    meta_value TEXT
);
"""


def create_schema(conn=None):
    """Создать все таблицы, если их ещё нет. Возвращает список созданных/существующих."""
    own = conn is None
    c = conn or get_connection()
    try:
        with c.cursor() as cur:
            cur.execute(SCHEMA)
        if own:
            c.commit()
        return True
    finally:
        if own:
            c.close()


# --- обработка данных (seed) -------------------------------------------------

def _to_date(v):
    return v if v else None


def seed_from_json(conn, path: str):
    """Прочитать case4 JSON и загрузить данные в БД.

    Идемпотентен для sections/chat_history (upsert), остальные таблицы
    предполагаются пустыми (вызывай truncate/reset перед повторным сидом).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ds = data["dataset"]

    with conn.cursor() as cur:
        cur.execute(SCHEMA)

        for u in ds.get("users", []):
            cur.execute(
                    """
                    INSERT INTO users (
                        user_id, birth_date, timezone, goal, chronotype,
                        weekday_bedtime, weekday_wake_time, average_sleep_hours,
                        social_jetlag, activity_level, sedentary_hours,
                        training_days, works_from_home, phone_before_sleep,
                        movement_index, sleep_preferences, disabled_widget_ids
                    ) VALUES (
                        %(user_id)s, %(birth_date)s, %(timezone)s, %(goal)s,
                        %(chronotype)s, %(weekday_bedtime)s, %(weekday_wake_time)s,
                        %(average_sleep_hours)s, %(social_jetlag)s, %(activity_level)s,
                        %(sedentary_hours)s, %(training_days)s, %(works_from_home)s,
                        %(phone_before_sleep)s, %(movement_index)s,
                        %(sleep_preferences)s, %(disabled_widget_ids)s
                    )
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                {
                    "user_id": u.get("user_id"),
                    "birth_date": _to_date(u.get("birth_date")),
                    **{k: u.get(k) for k in (
                        "timezone", "goal", "chronotype", "weekday_bedtime",
                        "weekday_wake_time", "average_sleep_hours", "social_jetlag",
                        "activity_level", "sedentary_hours", "training_days",
                        "works_from_home", "phone_before_sleep", "movement_index",
                        "sleep_preferences", "disabled_widget_ids",
                    )},
                },
            )

        for e in ds.get("experts", []):
            cur.execute(
                "INSERT INTO experts VALUES (%(expert_id)s, %(role)s, %(speech_style)s) "
                "ON CONFLICT (expert_id) DO NOTHING",
                e,
            )

        for a in ds.get("articles", []):
            cur.execute(
                """
                INSERT INTO articles (article_id, title, authors, year, doi,
                                      license_name, license_version, license_url)
                VALUES (%(article_id)s, %(title)s, %(authors)s::jsonb, %(year)s, %(doi)s,
                        %(license_name)s, %(license_version)s, %(license_url)s)
                ON CONFLICT (article_id) DO NOTHING
                """,
                {
                    "article_id": a.get("article_id"),
                    "title": a.get("title"),
                    "authors": json.dumps(a.get("authors", []), ensure_ascii=False),
                    "year": a.get("year"),
                    "doi": a.get("doi"),
                    "license_name": a.get("license", {}).get("name"),
                    "license_version": a.get("license", {}).get("version"),
                    "license_url": a.get("license", {}).get("url"),
                },
            )
            for idx, sec in enumerate(a.get("sections", []), start=1):
                cur.execute(
                    """
                    INSERT INTO article_sections (section_id, article_id, title, text, section_no)
                    VALUES (%(section_id)s, %(article_id)s, %(title)s, %(text)s, %(section_no)s)
                    ON CONFLICT (article_id, section_id) DO UPDATE
                    SET title=EXCLUDED.title, text=EXCLUDED.text, section_no=EXCLUDED.section_no
                    """,
                    {
                        "section_id": sec.get("section_id") or f"{a['article_id']}_s{idx:03d}",
                        "article_id": a["article_id"],
                        "title": sec.get("title"),
                        "text": sec.get("text"),
                        "section_no": idx,
                    },
                )

        for r in ds.get("requests", []):
            utt = r.get("utterance", {})
            cur.execute(
                """
                INSERT INTO requests (record_id, user_id, expert_id, group_id, text, role, at)
                VALUES (%(record_id)s, %(user_id)s, %(expert_id)s, %(group_id)s,
                        %(text)s, %(role)s, %(at)s)
                ON CONFLICT (record_id) DO NOTHING
                """,
                {
                    "record_id": r.get("record_id"),
                    "user_id": r.get("user_id"),
                    "expert_id": r.get("expert_id"),
                    "group_id": r.get("group_id"),
                    "text": utt.get("text") or "",
                    "role": utt.get("role") or "user",
                    "at": utt.get("at"),
                },
            )

        for c in ds.get("chat_history", []):
            cur.execute(
                """
                INSERT INTO chat_history VALUES (%(message_id)s, %(user_id)s,
                    %(role)s, %(at)s, %(text)s)
                ON CONFLICT (message_id) DO NOTHING
                """,
                c,
            )

        for p in ds.get("judge_prompts", []):
            cur.execute(
                """
                INSERT INTO judge_prompts VALUES (
                    %(prompt_id)s, %(version)s, %(purpose)s,
                    %(model_version)s, %(temperature)s, %(text)s
                )
                ON CONFLICT (prompt_id) DO NOTHING
                """,
                p,
            )

        meta = ds.get("meta", {})
        cur.execute("DELETE FROM dataset_meta")
        cur.execute(
            "INSERT INTO dataset_meta (meta_key, meta_value) VALUES (%s, %s)",
            ("meta", json.dumps(meta, ensure_ascii=False)),
        )


def truncate_all(conn):
    """Очистить все таблицы (с пересозданием схемы).

    Использует DROP + CREATE, чтобы reset применял изменения SCHEMA
    (TRUNCATE структуру не меняет). Вызывать перед create_schema/seed.
    """
    tables = [
        "article_sections", "requests", "chat_history", "judge_prompts",
        "articles", "users", "experts", "dataset_meta",
    ]
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS " + ", ".join(tables) + " CASCADE")


def row_counts(conn):
    """Вернуть словарь {таблица: число строк}."""
    tables = [
        "users", "experts", "articles", "article_sections",
        "requests", "chat_history", "judge_prompts", "dataset_meta",
    ]
    out = {}
    with conn.cursor() as cur:
        for t in tables:
            cur.execute(f'SELECT count(*) FROM "{t}"')
            out[t] = cur.fetchone()[0]
    return out


def sample(conn, table: str, limit: int = 3):
    """Выборка из таблицы как список dict (для отладки через main status)."""
    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM "{table}" LIMIT %s', (limit,))
        cols = [d.name for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]