#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f "$ROOT/.env" ]]; then
  R2AI_ORIGINAL_EXPORTS=$(export -p)
  set -a; source "$ROOT/.env"; set +a
  eval "$R2AI_ORIGINAL_EXPORTS"
  unset R2AI_ORIGINAL_EXPORTS
fi
PYTHON=${R2AI_PYTHON:-"${R2AI_VENV:-$ROOT/.venv}/bin/python"}
if [[ ! -x "$PYTHON" ]]; then echo 'Run bash scripts/setup.sh gpu first, or set R2AI_PYTHON to an installed interpreter.' >&2; exit 2; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NLTK_DATA="${NLTK_DATA:-$ROOT/data/cache/nltk_data}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$ROOT/data/cache/tiktoken}"
if [[ $# == 0 ]]; then set -- --config "$ROOT/config.json" --run; fi
exec "$PYTHON" -m r2ai_autoresearch.research.full_evaluate "$@"
