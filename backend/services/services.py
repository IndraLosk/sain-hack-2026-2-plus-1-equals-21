"""Сервисный слой (бизнес-логика).

Ничего не знает про HTTP/эндпоинты. Набор агенту даёт клиент через getData
(backend/client.py); БД в работе серверной части на стенде не участвует —
данные приходят из API организаторов в /run.
"""

from backend.schemas import RunBody, RunResult, RunAnswer, Citation
from backend.client import fetch_dataset


def run_case(body: RunBody, run_token: str = "") -> RunResult:
    """Обработать назначенный набор кейса и построить Case4Answers.

    Вход по контракту — {"case", "dataset_code"}; набор сам получаем через
    GET /case/4/getData (по dataset_code + run_token). По каждому запросу из
    набора агент строит ответ. Здесь агент ещё не реализован — заглушка.
    Ключи/токены ИИ не используются.
    """
    if body.case != 4:
        raise ValueError(f"case {body.case} не поддерживается; ожидается 4")

    dataset = fetch_dataset(dataset_code=body.dataset_code, run_token=run_token)

    requests_ = dataset.get("requests") or []
    if not requests_:
        raise ValueError(
            f"набор '{body.dataset_code}' не содержит запросов "
            "(dataset['requests'] пуст) — проверьте getData"
        )

    # Берём первую допустимую статью/раздел набора, чтобы citations был не пуст
    # (по Case4Answers схема требует minItems=1). Реальный агент (RAG) подберёт
    # статью сюда; сейчас — детерминированная заглушка на первой статье.
    article_map = {a["article_id"]: a for a in dataset.get("articles", [])}
    articles_ids = list(article_map.keys())
    articles_ids.sort()

    # agent ещё не реализован: на каждый запрос минимальный ответ с валидной ссылкой.
    # TODO: подставлять реальную статью/раздел по смыслу запроса (RAG) и отказ для атак.
    answers = []
    for i, req in enumerate(requests_):
        aid = articles_ids[i % len(articles_ids)] if articles_ids else None
        citation = None
        if aid:
            secs = article_map[aid].get("sections", [])
            if secs:
                citation = Citation(
                    article_id=aid,
                    section_id=secs[0]["section_id"],
                )
        answers.append(
            RunAnswer(
                record_id=req["record_id"],
                text="[placeholder]",
                refused=False,
                citations=[citation] if citation else [],
            )
        )
    return RunResult(requests=answers)