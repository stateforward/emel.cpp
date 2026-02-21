#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mode="${1:-}"
if [[ -n "$mode" && "$mode" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 1
fi

CHECK_MODE=false
if [[ "$mode" == "--check" ]]; then
  CHECK_MODE=true
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

require_tools() {
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
}

run_docsgen() {
  local build_dir="${DOCSGEN_BUILD_DIR:-$ROOT_DIR/build/docsgen}"
  local docsgen_dir="$ROOT_DIR/tools/docsgen"
  local docsgen_bin="$build_dir/docsgen"
  local bench_cc
  local bench_cxx
  local bench_cc_arg="cc"
  local bench_cxx_arg="c++"

  require_tools cmake ninja zig

  bench_cc="$(command -v zig)"
  bench_cxx="$bench_cc"

  cmake -S "$docsgen_dir" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$bench_cc" \
    -DCMAKE_CXX_COMPILER="$bench_cxx" \
    -DCMAKE_ASM_COMPILER="$bench_cc" \
    -DCMAKE_C_COMPILER_ARG1="$bench_cc_arg" \
    -DCMAKE_CXX_COMPILER_ARG1="$bench_cxx_arg" \
    -DCMAKE_ASM_COMPILER_ARG1="$bench_cc_arg"

  cmake --build "$build_dir" --parallel --target docsgen

  if $CHECK_MODE; then
    "$docsgen_bin" --root "$ROOT_DIR" --check
  else
    "$docsgen_bin" --root "$ROOT_DIR"
  fi
}

run_docsgen
