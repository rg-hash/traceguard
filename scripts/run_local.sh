#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
  echo "Virtual environment not found. Create it before starting TraceGuard."
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "Missing .env file."
  echo "Copy .env.example to .env and enter your local DATABASE_URL."
  exit 1
fi

set -a
source ".env"
set +a

if [ -z "${DATABASE_URL:-}" ]; then
  echo "DATABASE_URL is not set in .env."
  exit 1
fi

source ".venv/bin/activate"

python -c "
from app.database import initialize_database
initialize_database()
print('TraceGuard database schema is ready.')
"

echo "Starting TraceGuard at http://127.0.0.1:${TRACEGUARD_PORT:-8000}"
echo "Dashboard: http://127.0.0.1:${TRACEGUARD_PORT:-8000}/dashboard/"
echo "API docs:  http://127.0.0.1:${TRACEGUARD_PORT:-8000}/docs"

exec uvicorn app.api:app \
  --host 127.0.0.1 \
  --port "${TRACEGUARD_PORT:-8000}" \
  --reload