#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
usage: scripts/check_sml_behavior_selection.sh [PATH...]

Scans Boost.SML actor source for hidden runtime behavior selection that should be
modeled as explicit sm.hpp guards/states/transitions.

When PATH is omitted, scans src/emel.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "error: required tool missing: rg" >&2
  exit 2
fi

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=("src/emel")
fi

for target in "${targets[@]}"; do
  if [[ ! -e "$ROOT_DIR/$target" && ! -e "$target" ]]; then
    echo "error: scan target missing: $target" >&2
    exit 2
  fi
done

status=0

run_scan() {
  local title="$1"
  local pattern="$2"
  shift 2

  local output
  if output=$(rg -n --color never "$@" -- "$pattern" "${targets[@]}" 2>/dev/null); then
    echo "BLOCK: $title"
    echo "$output"
    echo
    status=1
  fi
}

# Runtime callable bundles are the failure mode where a public actor appears in
# the maintained path, but route/backend behavior still hides behind a table.
run_scan \
  "runtime function-pointer backend or route contract" \
  '\bruntime_[A-Za-z0-9_]*backend\b|using[[:space:]]+[A-Za-z0-9_]+_fn[[:space:]]*=[^;]*\(\s*\*\s*\)' \
  --glob 'src/emel/**/{events,context,guards,actions,detail,sm}.{hpp,cpp}'

# Action/guard calls through a stored backend/route pointer select behavior
# outside the transition graph. Guards may decide, but not by dispatching through
# runtime-indexed callable tables.
run_scan \
  "backend or route pointer invocation from actor logic" \
  '->[[:space:]]*(supports_|recognition_ready|encode|decode|detokenize|route|run|dispatch|select)[A-Za-z0-9_]*[[:space:]]*\(' \
  --glob 'src/emel/**/{guards,actions,detail,sm}.{hpp,cpp}'

# Static tables of route callbacks are still runtime behavior indirection when a
# state machine action/guard later invokes them.
run_scan \
  "backend or route callback table binding" \
  '\.[[:space:]]*(supports_|recognition_ready|encode|decode|detokenize|route|run|dispatch|select)[A-Za-z0-9_]*[[:space:]]*=[[:space:]]*&' \
  --glob 'src/emel/**/{events,context,guards,actions,detail,sm}.{hpp,cpp}'

# Runtime branching in actions is never acceptable. Compile-time branches are
# allowed and filtered by the negative sub-pattern.
action_branch_output="$(
  rg -n --color never --glob 'src/emel/**/actions.hpp' \
    '(^|[^A-Za-z0-9_])if[[:space:]]*\(|else[[:space:]]+if|switch[[:space:]]*\(' \
    "${targets[@]}" 2>/dev/null || true
)"
action_branch_output="$(
  printf '%s\n' "$action_branch_output" | rg -v 'if[[:space:]]+constexpr' || true
)"
if [[ -n "$action_branch_output" ]]; then
  echo "BLOCK: runtime branching in actions.hpp"
  echo "$action_branch_output"
  echo
  status=1
fi

if [[ $status -eq 0 ]]; then
  echo "SML behavior-selection scan passed for: ${targets[*]}"
fi

exit "$status"
