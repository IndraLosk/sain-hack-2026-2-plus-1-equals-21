"""Сервисный слой (бизнес-логика).

Ничего не знает про HTTP/эндпоинты — здесь только работа с данными через пакет db/.
Сюда выносятся обработчики: health-проверка, статистика, выборки.
Роуты (routes/) вызывают эти функции и отдают результат наружу.
"""

from db import get_connection, row_counts, sample
from backend.schemas import RunBody, RunResult
from backend.client import fetch_dataset


def check_health() -> dict:
    """Проверить подключение к БД. Возвращает статус; исключение пробрасывается наверх."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    return {"status": "ok", "db": "up"}


def stats() -> dict:
    """Число строк по таблицам."""
    with get_connection() as conn:
        return row_counts(conn)


def list_experts():
    """Список экспертов."""
    with get_connection() as conn:
        return sample(conn, "experts", limit=20)


def list_articles():
    """Сводка по статьям: id, год, лицензия."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT article_id, title, year, license_name AS license "
                "FROM articles ORDER BY article_id"
            )
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def list_requests():
    """Запросы набора: запись, пользователь, эксперт, группа, текст."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT record_id, user_id, expert_id, group_id, text "
                "FROM requests ORDER BY record_id"
            )
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def run_case(body: RunBody, run_token: str = "") -> RunResult:
    """Обработать назначенный набор кейса и построить Case4Answers.

    Вход по контракту — только {"case", "dataset_code"}; сам набор данных
    приложение получает через GET /case/4/getData (по dataset_code + run_token).
    Так получается Case4Dataset; затем по каждому запросу набора агент строит
    ответ. Здесь агент ещё не реализован — по запросам ставится заглушка.
    Ключи/токены ИИ и организаторов в этом слое не используются (SAIN_API_BASE
    и токен вписывает пользователь).
    """
    if body.case != 4:
        raise ValueError(f"case {body.case} не поддерживается; ожидается 4")

    dataset = fetch_dataset(body.dataset_code, run_token)

    requests_ = dataset.get("requests") or []
    if not requests_:
        # Отдавать молча пустой {"requests":[]} нельзя — организатор ожидает ответ
        # на каждый запрос набора. Делаем понятную ошибку вместо тихой пустоты.
        raise ValueError(
            f"набор '{body.dataset_code}' не содержит запросов "
            "(dataset['requests'] пуст) — проверьте getData"
        )

    # agent ещё не реализован: для каждого запроса возвращаем минимальный ответ.
    # TODO: для каждого r из dataset['requests'] сгенерить текст/refused/citations
    #       через агента (LLM), опираясь на dataset['articles'].
    answers = []
    answers = []
    for req in requests_:
        answers.append(
            RunAnswer(
                record_id=req["record_id"],
                text="[placeholder]",
                refused=False,
                citations=[],  # TODO: ссылки agent'а
            )
        )
    return RunResult(requests=answers)