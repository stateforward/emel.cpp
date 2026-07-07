#!/usr/bin/env bash
set -euo pipefail

# Determinism gate (docs/determinism.md): builds tools/determinism_check and
# runs it TWICE (two separate processes, fresh address spaces) against a small
# maintained generation fixture. The binary itself hard-fails unless repeated
# runs, and a freshly constructed session, produce bitwise-identical token
# streams and logits checksums in both maintained selection modes
# (preselected_argmax and sample_logits). This script additionally asserts the
# emitted `determinism_evidence` lines are identical across the two processes,
# proving cross-process determinism on the same host/build.

# shellcheck source=scripts/build_jobs.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_jobs.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_DETERMINISM_BUILD_DIR:-$ROOT_DIR/build/determinism_check}"
MODEL_PATH="${EMEL_DETERMINISM_MODEL:-$ROOT_DIR/tests/models/LFM2.5-230M-Q8_0.gguf}"
REPEATS="${EMEL_DETERMINISM_REPEATS:-3}"
TOKENS="${EMEL_DETERMINISM_TOKENS:-16}"

for tool in zig cmake ninja; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: determinism fixture missing: $MODEL_PATH" >&2
  echo "error: fetch the maintained fixture per tests/models/README.md" >&2
  exit 1
fi

zig_bin="$(command -v zig)"

cmake -S "$ROOT_DIR/tools/determinism_check" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$zig_bin" \
  -DCMAKE_C_COMPILER_ARG1=cc \
  -DCMAKE_CXX_COMPILER="$zig_bin" \
  -DCMAKE_CXX_COMPILER_ARG1=c++ \
  -DEMEL_ROOT_DIR="$ROOT_DIR"
cmake --build "$BUILD_DIR" --parallel "$EMEL_BUILD_JOBS"

binary="$BUILD_DIR/emel_determinism_check"
if [[ ! -x "$binary" ]]; then
  echo "error: determinism check binary missing: $binary" >&2
  exit 1
fi

run_log_a="$BUILD_DIR/determinism_run_a.log"
run_log_b="$BUILD_DIR/determinism_run_b.log"

echo "check_determinism: process A"
"$binary" "$MODEL_PATH" "$REPEATS" "$TOKENS" | tee "$run_log_a"
echo "check_determinism: process B"
"$binary" "$MODEL_PATH" "$REPEATS" "$TOKENS" | tee "$run_log_b"

evidence_a="$(grep '^determinism_evidence ' "$run_log_a" || true)"
evidence_b="$(grep '^determinism_evidence ' "$run_log_b" || true)"

if [[ -z "$evidence_a" || -z "$evidence_b" ]]; then
  echo "error: determinism evidence lines missing" >&2
  exit 1
fi

if [[ "$evidence_a" != "$evidence_b" ]]; then
  echo "error: determinism evidence diverged across processes" >&2
  diff <(printf '%s\n' "$evidence_a") <(printf '%s\n' "$evidence_b") >&2 || true
  exit 1
fi

echo "check_determinism: PASS (cross-process evidence identical, repeats=$REPEATS tokens=$TOKENS)"
