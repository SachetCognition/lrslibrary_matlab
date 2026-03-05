.PHONY: setup backend-install frontend-install backend-test frontend-test test dev docker-up lint

# Setup everything
setup: backend-install frontend-install

# Backend
backend-install:
	cd backend && pip install -e ".[dev]"

backend-test:
	cd backend && python -m pytest tests/ -v --cov=lrslibrary

backend-lint:
	cd backend && python -m ruff check lrslibrary/ tests/

backend-lint-fix:
	cd backend && python -m ruff check --fix lrslibrary/ tests/

# Frontend
frontend-install:
	cd frontend && npm install

frontend-test:
	cd frontend && npm run test

frontend-lint:
	cd frontend && npm run lint

frontend-build:
	cd frontend && npm run build

# Combined
test: backend-test frontend-test

lint: backend-lint frontend-lint

# Development
dev-backend:
	cd backend && uvicorn lrslibrary.api.app:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Run 'make dev-backend' and 'make dev-frontend' in separate terminals"

# Docker
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
