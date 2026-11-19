#!/usr/bin/env bash
# Проверка стенда/локального сервера для кейса 4 «Эксперт как плейлист».
#
# Что делает:
#   1. GET /health — сервер жив, база проиндексирована (chunks/articles), LLM отвечает;
#   2. GET /api/v1/articles — статьи загружены;
#   3. POST /run — прогоняет dev-набор (по умолчанию берёт локальный data/case4-dev.json),
#      затем полученный ответ отправляет в POST /case/4/validate для проверки формата.
#
# Переменные:
#   BASE_URL  — адрес сервера (localhost по умолчанию)
#   TOKEN     — токен API хакатона (HACK_TEAM_TOKEN на стенде / api.token на ноутбуке)
#   HACK_API  — адрес API организаторов (по умолчанию https://hack-api.be-healthy.tech)
#   DATASET   — путь к локальному dev-набору (по умолчанию data/case4-dev.json)
#
# Пример локального запуска (из ноутбука):
#   TOKEN="$(cat api.token)" STAGING=0 bash scripts/check_stand.sh
#
# Пример проверки уже задеплоенного на стенде бэкенда (адрес открыт до 20.11 12:00):
#   BASE_URL="https://team02.hack.be-healthy.tech" TOKEN="$(cat api.token)" STAGING=0 bash scripts/check_stand.sh
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HACK_API="${HACK_API:-https://hack-api.be-healthy.tech}"
DATASET="${DATASET:-data/case4-dev.json}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

fail() { printf '✗ %s\n' "$1" >&2; exit 1; }
ok()   { printf '✓ %s\n' "$1"; }

# ---------- токен ----------
if [ -z "${TOKEN:-}" ]; then
  for f in api.token "$ROOT/api.token" "$ROOT/../api.token"; do
    if [ -f "$f" ]; then TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
  done
fi
[ -n "${TOKEN:-}" ] || fail "не задан TOKEN (export TOKEN=... или положите рядом api.token)"

echo "== 1. Health =="
H=$(curl -s -m 20 "$BASE_URL/api/v1/health") || fail "health не отвечает на $BASE_URL"
echo "$H" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  status:', d.get('status')); vs=d.get('vector_store',{}); print('  chunks:', vs.get('chunks'), '| articles:', vs.get('articles'), '| retriever:', vs.get('retriever')); print('  llm:', d.get('llm',{}).get('provider'), d.get('llm',{}).get('model'), d.get('llm',{}).get('ok'))" \
  || echo "$H" | head -c 400

echo "== 2. Статьи в базе =="
N=$(curl -s -m 20 "$BASE_URL/api/v1/articles" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
ok "articles: $N"

echo "== 3. Заполняем /run из $DATASET =="
[ -f "$ROOT/$DATASET" ] || fail "нет локального набора $ROOT/$DATASET (укажите DATASET=... или сначала прогоните на стенде)"

# соберём запросы записей в формате case4 (как их отдаёт стенд)
# вырезаем записи; каждый record содержит record_id, expert_id, user_id, utterance.text
python3 - "$ROOT/$DATASET" <<'PY' > /tmp/case4_requests.json
import sys, json
raw = json.load(open(sys.argv[1], encoding="utf-8"))
ds = raw.get("dataset", raw)
reqs = ds.get("requests") or []
print(json.dumps(reqs, ensure_ascii=False))
PY

# локально: шлём прямо список записей (бэкенд сам их переберёт и сам найдёт текст вопроса)
ANS=$(curl -s -m 300 -X POST "$BASE_URL/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @/tmp/case4_requests.json)
echo "$ANS" | python3 -c "import sys,json; d=json.load(sys.stdin); a=d.get('requests',d.get('answers',[])); print('  ответов:', len(a))" \
  || { echo "$ANS" | head -c 400; fail "run вернул не-JSON"; }
printf '%s' "$ANS" > /tmp/case4_answers.json

echo "== 4. Проверка формата через /case/4/validate =="
VALID=$(curl -s -m 60 -X POST "$HACK_API/case/4/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @/tmp/case4_answers.json)
echo "$VALID" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  valid:', d.get('valid')); e=d.get('errors') or []; print('  errors:', len(e)); [print('   -', x) for x in e[:8]]" \
  || echo "$VALID" | head -c 400

echo
echo "Готово. Ответ /run сохранён в /tmp/case4_answers.json"