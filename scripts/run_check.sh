#!/usr/bin/env bash
# Отправляет POST /run ровно так, как это делает стенд на онлайн-показе.
#
# Тело запроса: {"case": 4, "dataset_code": "<код>"} + Authorization: Bearer <токен>.
# Бэкенд сам получает данные через GET /case/4/getData этим токеном и возвращает ответ.
#
# Переменные:
#   BASE_URL — адрес сервера (по умолчанию локальный http://127.0.0.1:8000)
#   TOKEN    — dev-токен хакатона (по умолчанию берётся из ../api.token)
#   DATASET_CODE — dev-код набора (по умолчанию case4-dev-v2)
#
# Пример:
#   bash scripts/run_check.sh                      # локально, сохранит ответ в answer.json
#   BASE_URL="https://team02.hack.be-healthy.tech" bash scripts/run_check.sh
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
DATASET_CODE="${DATASET_CODE:-case4-dev-v2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# токен: из env или из api.token рядом с репозиторием
if [ -z "${TOKEN:-}" ]; then
  for f in "$ROOT/api.token" "$ROOT/../api.token"; do
    if [ -f "$f" ]; then TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
  done
fi
[ -n "${TOKEN:-}" ] || { echo "ERROR: задайте TOKEN (или положите рядом api.token)"; exit 1; }

BODY="{\"case\": 4, \"dataset_code\": \"$DATASET_CODE\"}"

echo "→ POST $BASE_URL/run"
echo "  body: $BODY"
echo "  dataset_code: $DATASET_CODE"
echo "  (токен: из env/файла, ${#TOKEN} символов)"

curl -s -o answer.json -w "HTTP %{http_code} · %{time_total} с\n" \
  -X POST "$BASE_URL/run" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$BODY"

echo "→ сохранено в answer.json"
echo "→ число ответов:"
python3 -c "import json;d=json.load(open('answer.json'));r=d.get('requests',d.get('answers',[]));print('  ',len(r),'ответов')" 2>/dev/null \
  || head -c 300 answer.json