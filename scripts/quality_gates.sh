#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMING_FILE="$ROOT_DIR/snapshots/quality_gates/timing.txt"

if [[ -z "${EMEL_QUALITY_GATES_INNER:-}" ]]; then
  timeout_cmd=()
  if command -v timeout >/dev/null 2>&1; then
    timeout_cmd=(timeout 300s)
  elif command -v gtimeout >/dev/null 2>&1; then
    timeout_cmd=(gtimeout 300s)
  else
    echo "error: timeout tool missing (install coreutils for gtimeout on macOS)" >&2
    exit 1
  fi

  export EMEL_QUALITY_GATES_INNER=1
  exec "${timeout_cmd[@]}" "$0" "$@"
fi

timing_lines=()
total_start="$(date +%s)"

write_timing_snapshot() {
  local total_now
  total_now="$(( $(date +%s) - total_start ))"
  mkdir -p "$(dirname "$TIMING_FILE")"
  {
    echo "# quality_gates timing (seconds)"
    for line in "${timing_lines[@]}"; do
      echo "$line"
    done
    echo "total $total_now"
  } > "$TIMING_FILE"
}

run_step() {
  local name="$1"
  shift
  local start
  local end
  local duration
  start="$(date +%s)"
  "$@"
  end="$(date +%s)"
  duration="$(( end - start ))"
  timing_lines+=("$name $duration")
  write_timing_snapshot
}

run_step build_with_zig "$ROOT_DIR/scripts/build_with_zig.sh"
run_step test_with_coverage "$ROOT_DIR/scripts/test_with_coverage.sh"
run_step paritychecker "$ROOT_DIR/scripts/paritychecker.sh"
# Temporarily disabled (SML UBSAN issue under asan_ubsan).
# TODO: re-enable once stateforward/sml.cpp fix lands.
run_step fuzz_smoke "$ROOT_DIR/scripts/fuzz_smoke.sh"
run_step lint_snapshot "$ROOT_DIR/scripts/lint_snapshot.sh"
run_step bench_snapshot env \
  EMEL_BENCH_ITERS=10000 \
  EMEL_BENCH_RUNS=3 \
  EMEL_BENCH_WARMUP_ITERS=1000 \
  EMEL_BENCH_WARMUP_RUNS=1 \
  "$ROOT_DIR/scripts/bench.sh" --snapshot --compare-update
run_step generate_docs "$ROOT_DIR/scripts/generate_docs.sh"
