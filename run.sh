#!/usr/bin/env bash
# Start the Engineering Copilot locally.
#
#   ./run.sh
#
# Creates .venv on first run, installs deps, then serves on http://127.0.0.1:8000
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  echo
  echo "!! No .env found."
  echo "!! Run: cp .env.example .env    then paste a free key from https://openrouter.ai/settings/keys"
  echo "!! Starting anyway -- the UI will tell you the same thing."
  echo
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "==> http://${HOST}:${PORT}"
exec uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload
