# Sain Case 4 — оркестрация: БД + приложение через единый main.py.
# Использование: make <target>. См. docker-compose.yml.

COMPOSE := docker compose
SERVICE := app

.PHONY: help up build seed reset status serve logs psql db-shell down reset-hard clean

help: ## Показать доступные команды
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-11s %s\n", $$1, $$2}'

########################################
# Сборка и запуск
########################################

up: ## Поднять БД + приложение (main.py: seed → serve)
	$(COMPOSE) up --build -d

build: ## Пересобрать образ приложения
	$(COMPOSE) build

serve: ## Запустить только сервер внутри app-контейнера (данные уже в БД)
	$(COMPOSE) run --rm --no-deps -p 8000:8000 $(SERVICE) python main.py serve --host 0.0.0.0 --port 8000

########################################
# Работа с БД через main.py
########################################

seed: ## Пересоздать схему и залить данные из case4-dev.json
	$(COMPOSE) run --rm $(SERVICE) python main.py seed --json /data/case4-dev.json

reset: ## Очистить таблицы и залить данные заново
	$(COMPOSE) run --rm $(SERVICE) python main.py reset --json /data/case4-dev.json

status: ## Показать число строк по таблицам
	$(COMPOSE) run --rm $(SERVICE) python main.py status

health: ## Проверить подключение к БД
	$(COMPOSE) run --rm $(SERVICE) python main.py health

########################################
# Диагностика
########################################

logs: ## Логи контейнеров
	$(COMPOSE) logs --tail=60 $(SERVICE)

psql: ## Открыть psql внутри postgres-контейнера
	$(COMPOSE) exec db psql -U sain -d sain_case4

db-shell: ## Шелл внутри postgres-контейнера
	$(COMPOSE) exec db sh

status-containers: ## Состояние контейнеров
	$(COMPOSE) ps

down: ## Остановить контейнеры (данные сохраняются)
	$(COMPOSE) down

reset-hard: ## Полностью сбросить данные, volume и пересобрать
	$(COMPOSE) down -v
	$(COMPOSE) up --build -d

clean: ## Убрать контейнеры, volume, образы
	$(COMPOSE) down -v --remove-orphans --rmi local