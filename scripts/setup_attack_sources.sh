#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
DEST=${RESI_JBS_ROOT:-"$ROOT/.runtime/JailbreakSkill"}
REV=3d164bfc85ea9a39d02e41891734528a8520ad9d
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/BattleWen/JailbreakSkill.git "$DEST"
  git -C "$DEST" checkout --detach "$REV"
fi
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$REV" ]] || { echo 'JailbreakSkill revision differs from the tested source.' >&2; exit 2; }
"${RESI_PYTHON:-$ROOT/.venv/bin/python}" -m pip install -r "$DEST/requirements.txt"
echo "JailbreakSkill installed in $DEST; upstream terms apply."
