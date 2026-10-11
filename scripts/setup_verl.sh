#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
VERL=${R2AI_VERL_ROOT:-"$ROOT/.runtime/verl"}
REV=c4b389adadc58ce51cb2b63e70df497ca166d77f
PYTHON=${R2AI_SETUP_PYTHON:-python3.12}
if [[ ! -d "$VERL/.git" ]]; then
  git clone https://github.com/verl-project/verl.git "$VERL"
  git -C "$VERL" checkout --detach "$REV"
fi
[[ "$(git -C "$VERL" rev-parse HEAD)" == "$REV" ]] || { echo 'VERL revision differs from the recorded runtime.' >&2; exit 2; }
PATCH="$ROOT/patches/verl_historical_runtime.patch"
if git -C "$VERL" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo 'Historical runtime patch already applied.'
else
  git -C "$VERL" apply --check "$PATCH"
  git -C "$VERL" apply "$PATCH"
fi
"$PYTHON" -m venv "$VERL/.venv"
P="$VERL/.venv/bin/python"
"$P" -m pip install --upgrade pip
"$P" -m pip install torch==2.11.0 --index-url "${R2AI_VERL_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
"$P" -m pip install -c "$ROOT/requirements-verl.txt" -e "$VERL" -r "$ROOT/requirements-verl.txt"
"$P" -m pip install --no-deps -e "$ROOT"
"$P" -m pip check
