#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
MODE=${1:-gpu}
VENV=${R2AI_VENV:-"$ROOT/.venv"}
PYTHON=${R2AI_SETUP_PYTHON:-python3}
if [[ "$MODE" == full ]]; then
  bash "$ROOT/scripts/setup.sh" historical-sft
  bash "$ROOT/scripts/setup_verl.sh"
  "$VENV/bin/python" "$ROOT/scripts/setup_xteaming.py"
  exit 0
fi
if [[ "$MODE" != gpu && "$MODE" != cpu && "$MODE" != historical-sft ]]; then
  echo 'Usage: bash scripts/setup.sh [full|gpu|cpu|historical-sft]' >&2; exit 2
fi
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required; GPU lock tested on Python 3.12"'
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
if [[ "$MODE" == gpu || "$MODE" == historical-sft ]]; then
  "$VENV/bin/python" -m pip install torch==2.9.1 --index-url "${R2AI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements-gpu.txt"
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements-eval.txt"
  export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$ROOT/data/cache/tiktoken}"
  "$VENV/bin/python" -c 'import tiktoken; [tiktoken.get_encoding(n) for n in ("cl100k_base", "o200k_base")]'
  export NLTK_DATA="${NLTK_DATA:-$ROOT/data/cache/nltk_data}"
  if ! "$VENV/bin/python" -c 'import nltk; nltk.data.find("tokenizers/punkt"); nltk.data.find("tokenizers/punkt_tab/english/")' >/dev/null 2>&1; then
    "$VENV/bin/python" -m nltk.downloader -e -q -d "$NLTK_DATA" punkt punkt_tab
  fi
  if [[ "$MODE" == historical-sft ]]; then "$VENV/bin/python" -m pip install -r "$ROOT/requirements-historical-sft.txt"; fi
fi
"$VENV/bin/python" -m pip install -e "$ROOT"
"$VENV/bin/python" -m pip check
echo "Installed: $VENV/bin/python ($MODE)"
