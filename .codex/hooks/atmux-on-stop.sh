#!/usr/bin/env bash
set -euo pipefail

HOOK="${ATMUX_ON_STOP_HOOK:-$HOME/.atmux/src/atmux/adapter/scripts/on-stop}"
[[ -x "$HOOK" ]] || exit 0
"$HOOK" "$@"
