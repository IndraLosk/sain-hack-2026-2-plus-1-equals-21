#!/usr/bin/env bash
# Полная dev-проверка по контракту стенда: getData через /run, затем validate.
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HACK_API="${HACK_API:-https://hack-api.be-healthy.tech}"
DATASET_CODE="${DATASET_CODE:-case4-dev-v2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="${TMP_DIR:-$ROOT/tmp}"
ANSWERS="$TMP_DIR/case4_answers.json"
mkdir -p "$TMP_DIR"

fail() { printf '✗ %s\n' "$1" >&2; exit 1; }
ok()   { printf '✓ %s\n' "$1"; }

if [ -z "${TOKEN:-}" ] && [ -f "$ROOT/api.token" ]; then
  TOKEN="$(tr -d '[:space:]' < "$ROOT/api.token")"
fi
[ -n "${TOKEN:-}" ] || fail "не задан TOKEN (export TOKEN=... или поместите api.token в корень репозитория)"

echo "== 1. Health =="
curl -s -m 20 "$BASE_URL/api/v1/health" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('  status:',d.get('status')); print('  llm:',d.get('llm',{}).get('model'),d.get('llm',{}).get('ok')); print('  guardrail:',d.get('guardrail')); print('  usage:',d.get('llm_usage'))" \
  || fail "health не отвечает"

echo "== 2. POST /run по контракту стенда =="
BODY="{\"case\":4,\"dataset_code\":\"$DATASET_CODE\"}"
HTTP=$(curl -s -m 2700 -o "$ANSWERS" -w '%{http_code}' -X POST "$BASE_URL/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "$BODY") || fail "ошибка HTTP-клиента при /run"
[ "$HTTP" = "200" ] || { head -c 500 "$ANSWERS"; fail " /run вернул HTTP $HTTP"; }
python3 - "$ANSWERS" <<'PY' || fail "некорректный общий JSON /run"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
answers = payload.get("requests") or []
assert answers, "пустой requests"
print("  ответов:", len(answers))
PY

echo "== 3. POST /case/4/validate =="
curl -s -m 60 -o "$TMP_DIR/case4_validation.json" -X POST "$HACK_API/case/4/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  --data-binary @"$ANSWERS" || fail "ошибка HTTP-клиента при validate"
python3 - "$TMP_DIR/case4_validation.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print("  valid:", payload.get("valid"))
for error in (payload.get("errors") or [])[:10]:
    print("   -", error)
raise SystemExit(0 if payload.get("valid") is True else 1)
PY
[ $? -eq 0 ] || fail "validate не подтвердил формат"

ok "Готово. Ответ: $ANSWERS"
