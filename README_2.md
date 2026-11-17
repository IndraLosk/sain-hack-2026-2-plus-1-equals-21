# «Эксперт как плейлист» — AI-агент на LangGraph

MVP AI-платформы научно обоснованных программ здоровья: пользователь выбирает **программу**
(«плейлист») и **персону эксперта**, а агент отвечает строго по научным статьям из векторной базы,
не назначая дозировок и диагнозов.

Всё работает **локально**, без docker:
* LLM — `qwen2.5:7b` через Ollama;
* эмбеддинги — `BAAI/bge-m3` (sentence-transformers, offline из кэша HF);
* векторная БД — ChromaDB (persistent, `data/chroma`);
* память диалогов — `AsyncSqliteSaver` (`data/checkpoints.sqlite`).

## Быстрый старт

```bash
./run.sh                      # создаст .venv, поднимет ollama при необходимости, запустит uvicorn
# или вручную:
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Открыть: <http://127.0.0.1:8000> (встроенный UI) · Swagger: <http://127.0.0.1:8000/docs>

Самопроверка агента (граф, guardrail, RAG, свежесть базы, память):

```bash
.venv/bin/python check_agent.py
```

## Структура

```
app/
  main.py        FastAPI: /api/v1/catalog, /api/v1/chat/stream (SSE), /api/v1/articles, /run, /health
  agent.py       LangGraph-граф: retriever → guardrail → (safe_redirect | expert_response)
  vectorstore.py ChromaDB + bge-m3 + чанкование 500/100 + гибридный ранжинг со свежестью
  llm.py         Ollama-клиент (stream/non-stream/json) + guardrail (LLM + эвристики)
  config.py      Pydantic-настройки и валидация config/catalog.json при старте
  schemas.py     Pydantic v2 схемы запросов/ответов и событий SSE
config/catalog.json      программы и персоны экспертов (стиль задаётся system_style_prompt)
data/seed_articles.json  демо-доказательная база (сидируется при первом старте)
web/index.html           лёгкий UI: витрина программ, переключатель экспертов, SSE, панель куратора
check_agent.py           end-to-end проверка без HTTP
```

## Граф агента (LangGraph)

```
START → retriever_node → guardrail_node ─┬─ unsafe → safe_redirect_node ─→ END
                                         └─ safe   → expert_response_node ─→ END
```

* **retriever_node** — Chroma-запрос с фильтром `program_id`, топ-3 чанка, бонус свежести
  (`added_at` < 7 дней), максимум 2 чанка на статью, фолбэк на всю базу.
* **guardrail_node** — LLM-проверка с JSON-структурой (`SafetyCheckSchema`: `is_unsafe_request`,
  `category`, `reason`) + детерминированные эвристики (мг/мл/капли/таблетки, «сколько пить»,
  схема приёма, отмена назначений, диагноз). Эвристика имеет приоритет — guardrail не «промолчит».
* **safe_redirect_node** — мягкий отказ + выжимка из статей + **один** поведенческий шаг.
* **expert_response_node** — промпт из изолированных блоков `[FACTS_ONLY]` (чанки + дата) и
  `[PERSONA_STYLE]` (стиль персоны). Anti-hallucination: пустой контекст → фиксированная фраза
  «В моей доказательной базе пока нет подтверждённых исследований по этому вопросу».

Состояние (`AgentState`): `session_id`, `program_id`, `expert_id`,
`messages: Annotated[list, add_messages]`, `is_unsafe_request`, `retrieved_docs`,
`guardrail`, `prompt_messages`, `answer`. Тред = `session_id`, поэтому переключение персоны
не сбрасывает историю.

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/v1/catalog` | программы + эксперты |
| POST | `/api/v1/chat/stream` | SSE-поток ответа |
| GET | `/api/v1/chat/history?session_id=` | история диалога из чекпойнтера |
| DELETE | `/api/v1/chat/history?session_id=` | сброс диалога |
| POST | `/api/v1/articles` | куратор: 202 Accepted, индексация в BackgroundTasks |
| GET | `/api/v1/articles` | список статей в базе (с бейджем `is_new`) |
| DELETE | `/api/v1/articles/{article_id}` | удалить статью из базы |
| GET | `/api/v1/health` | статус Ollama/Chroma/модели |
| POST | `/api/v1/admin/reseed` | пересобрать демо-базу |
| POST | `/run` | контракт стенда кейса 4 → `Case4Answers` |

### SSE-события

```
{"type":"safety_flag","triggered":true}
{"type":"sources","items":[{"article_id":"...","section_id":"...","title":"...","date":"...","is_new":true,"snippet":"..."}]}
{"type":"meta","session_id":"...","expert_id":"...","program_id":"...","is_new_knowledge":true}
{"type":"token","content":"..."}
{"type":"done"}
```

## Проверка критериев приёмки

1. **Переключение стиля** — один вопрос «Как быстрее уснуть?»: сомнолог отвечает академично,
   биохакер — протоколом и метрикой, оба ссылаются на одни и те же `article_id`.
2. **Свежесть базы** — `POST /api/v1/articles` → через пару секунд оба эксперта цитируют новую
   статью с пометкой «Согласно свежему исследованию от …» и бейджем `NEW`.
3. **Безопасность** — «сколько миллиграмм мелатонина» → `safety_flag: true`, в тексте нет ни одной
   дозировки, предлагается поведенческий шаг (температура/свет/время).

## Переменные окружения (`.env`)

```
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=mps
RETRIEVAL_TOP_K=3
FRESHNESS_DAYS=7
```

## Примечания

* Демо-статьи в `data/seed_articles.json` — сокращённые фрагменты открытых публикаций (CC BY),
  используются только как доказательная база для RAG.
* Первый запуск дольше: загружается bge-m3 (~2 ГБ) и сидируется Chroma.
* `HF_HUB_OFFLINE=1` выставляется автоматически — модель берётся из локального кэша HF.
