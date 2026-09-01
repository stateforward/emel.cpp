#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/build_jobs.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_jobs.sh"

for tool in cmake ninja clang clang++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${EMEL_FUZZ_BUILD_DIR:-$ROOT_DIR/build/fuzz}"
FUZZ_CLEAN="${EMEL_FUZZ_CLEAN:-0}"
FUZZ_OUTER_TIMEOUT="${EMEL_FUZZ_OUTER_TIMEOUT:-30s}"

detect_fuzzer_toolchain() {
  if [[ -n "${CC:-}" && -n "${CXX:-}" ]]; then
    echo "$CC" "$CXX"
    return
  fi

  local brew_llvm_root=""
  if [[ "$(uname -s)" == "Darwin" ]]; then
    for candidate in /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
      if [[ -x "$candidate/bin/clang" && -x "$candidate/bin/clang++" ]]; then
        brew_llvm_root="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$brew_llvm_root" ]]; then
    echo "$brew_llvm_root/bin/clang" "$brew_llvm_root/bin/clang++"
    return
  fi

  echo "clang" "clang++"
}

read -r fuzz_cc fuzz_cxx < <(detect_fuzzer_toolchain)
fuzz_timeout_cmd=()
if command -v timeout >/dev/null 2>&1; then
  fuzz_timeout_cmd=(timeout "$FUZZ_OUTER_TIMEOUT")
elif command -v gtimeout >/dev/null 2>&1; then
  fuzz_timeout_cmd=(gtimeout "$FUZZ_OUTER_TIMEOUT")
else
  echo "error: timeout tool missing (install coreutils for gtimeout on macOS)" >&2
  exit 1
fi
fuzz_cxx_flags=""
fuzz_link_flags=""
fuzz_platform_flags=()
fuzz_probe_flags=()
fuzz_root="$(cd "$(dirname "$fuzz_cc")/.." && pwd)"
if [[ -d "$fuzz_root/lib/c++" ]]; then
  fuzz_cxx_flags="-stdlib=libc++ -I${fuzz_root}/include/c++/v1"
  fuzz_link_flags="-stdlib=libc++ -L${fuzz_root}/lib/c++ -Wl,-rpath,${fuzz_root}/lib/c++ -lc++ -lc++abi"
  fuzz_probe_flags+=(
    -stdlib=libc++
    "-I${fuzz_root}/include/c++/v1"
    "-L${fuzz_root}/lib/c++"
    "-Wl,-rpath,${fuzz_root}/lib/c++"
    -lc++
    -lc++abi
  )
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v xcrun >/dev/null 2>&1; then
    echo "error: xcrun is required to locate the active macOS SDK" >&2
    exit 1
  fi
  fuzz_macos_sysroot="$(xcrun --sdk macosx --show-sdk-path)"
  if [[ ! -d "$fuzz_macos_sysroot" ]]; then
    echo "error: active macOS SDK does not exist: $fuzz_macos_sysroot" >&2
    exit 1
  fi
  fuzz_platform_flags+=("-DCMAKE_OSX_SYSROOT=$fuzz_macos_sysroot")
  fuzz_probe_flags+=(-isysroot "$fuzz_macos_sysroot")
fi
check_libfuzzer_runtime() {
  local probe_dir probe_log
  probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/emel-fuzzer-probe.XXXXXX")"
  probe_log="$probe_dir/link.log"

  if ! printf '%s\n' \
    'extern "C" int LLVMFuzzerTestOneInput(const unsigned char *, unsigned long) { return 0; }' | \
    "$fuzz_cxx" "${fuzz_probe_flags[@]}" -x c++ -fsanitize=fuzzer \
      -o "$probe_dir/fuzzer_probe" - >"$probe_log" 2>&1; then
    echo "error: selected C++ compiler cannot link a libFuzzer executable with -fsanitize=fuzzer: $fuzz_cxx" >&2
    cat "$probe_log" >&2
    rm -rf "$probe_dir"
    return 1
  fi

  rm -rf "$probe_dir"
}

check_libfuzzer_runtime

if [[ "$FUZZ_CLEAN" == "1" ]]; then
  rm -rf "$BUILD_DIR"
fi

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$fuzz_cc" \
  -DCMAKE_CXX_COMPILER="$fuzz_cxx" \
  -DCMAKE_CXX_FLAGS="$fuzz_cxx_flags" \
  -DCMAKE_EXE_LINKER_FLAGS="$fuzz_link_flags" \
  "${fuzz_platform_flags[@]}" \
  -DEMEL_ENABLE_FUZZ=ON \
  -DEMEL_ENABLE_TESTS=OFF

cmake --build "$BUILD_DIR" --parallel "$EMEL_BUILD_JOBS"

run_fuzzer() {
  local name="$1"
  local corpus="$2"
  local status=0
  mkdir -p "$corpus"
  "${fuzz_timeout_cmd[@]}" \
    "$BUILD_DIR/$name" \
    -seed=1 \
    -max_total_time=10 \
    -max_len=4096 \
    "$corpus" || status=$?
  if [[ "$status" -eq 124 ]]; then
    echo "error: fuzzer timed out after $FUZZ_OUTER_TIMEOUT: $name" >&2
  fi
  return "$status"
}

if [[ -x "$BUILD_DIR/emel_fuzz_gguf_parser" ]]; then
  run_fuzzer emel_fuzz_gguf_parser "$ROOT_DIR/tests/fuzz/corpus/gguf_parser"
fi
run_fuzzer emel_fuzz_gbnf_parser "$ROOT_DIR/tests/fuzz/corpus/gbnf_parser"
run_fuzzer emel_fuzz_jinja_parser "$ROOT_DIR/tests/fuzz/corpus/jinja_parser"
run_fuzzer emel_fuzz_jinja_formatter "$ROOT_DIR/tests/fuzz/corpus/jinja_formatter"
