#!/usr/bin/env bash
set -euo pipefail

for tool in cmake ctest ninja zig; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/paritychecker_zig"
zig_bin="$(command -v zig)"
selected_runners=()

usage() {
  cat >&2 <<'USAGE'
usage: scripts/paritychecker.sh [--runner=<name>|--mode=<name>]...

Runs the maintained paritychecker gate. With no runner filters, the full
paritychecker test target runs. Runner filters execute the maintained doctest
cases for selected parity runners only.

Supported runners:
  tokenizer, gbnf_parser, kernel, jinja, generation, all
USAGE
}

add_runner() {
  local runner="$1"
  local existing
  if [[ -z "$runner" ]]; then
    echo "error: empty parity runner" >&2
    exit 1
  fi
  if [[ "$runner" == "all" ]]; then
    selected_runners=()
    return
  fi
  case "$runner" in
    gbnf)
      runner="gbnf_parser"
      ;;
    tokenizer|gbnf_parser|kernel|jinja|generation)
      ;;
    *)
      echo "error: unsupported parity runner: $runner" >&2
      usage
      exit 1
      ;;
  esac
  for existing in "${selected_runners[@]+${selected_runners[@]}}"; do
    if [[ "$existing" == "$runner" ]]; then
      return
    fi
  done
  selected_runners+=("$runner")
}

for arg in "$@"; do
  case "$arg" in
    --runner=*|--mode=*)
      add_runner "${arg#*=}"
      ;;
    --all)
      add_runner all
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown paritychecker.sh argument: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

cmake -S "$ROOT_DIR/tools/paritychecker" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$zig_bin" \
  -DCMAKE_C_COMPILER_ARG1=cc \
  -DCMAKE_CXX_COMPILER="$zig_bin" \
  -DCMAKE_CXX_COMPILER_ARG1=c++ \
  -DGGML_METAL=OFF \
  -DLLAMA_METAL=OFF

cmake --build "$BUILD_DIR" --parallel
if [[ ${#selected_runners[@]} -eq 0 ]]; then
  ctest --test-dir "$BUILD_DIR" --output-on-failure -R paritychecker_tests
  exit $?
fi

test_binary="$BUILD_DIR/paritychecker_tests"
if [[ ! -x "$test_binary" ]]; then
  echo "error: missing paritychecker test binary: $test_binary" >&2
  exit 1
fi

for runner in "${selected_runners[@]}"; do
  echo "paritychecker: runner=$runner" >&2
  case "$runner" in
    tokenizer)
      "$test_binary" --test-case="*tokens across tiny models*"
      ;;
    gbnf_parser)
      "$test_binary" --test-case="*gbnf parser outputs*"
      ;;
    kernel)
      "$test_binary" --test-case="*kernel outputs*"
      ;;
    jinja)
      "$test_binary" --test-case="*jinja parser and formatter outputs*"
      ;;
    generation)
      "$test_binary" --test-case="*generation*"
      ;;
  esac
done
