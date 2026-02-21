#!/usr/bin/env bash
set -euo pipefail

for tool in cmake ninja clang clang++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/fuzz"

detect_fuzzer_toolchain() {
  if [[ -n "${CC:-}" && -n "${CXX:-}" ]]; then
    echo "$CC" "$CXX"
    return
  fi

  local brew_llvm_root=""
  for candidate in /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
    if [[ -x "$candidate/bin/clang" && -x "$candidate/bin/clang++" ]]; then
      if ls "$candidate"/lib/clang/*/lib/darwin/libclang_rt.fuzzer_osx.a >/dev/null 2>&1; then
        brew_llvm_root="$candidate"
        break
      fi
    fi
  done
  if [[ -n "$brew_llvm_root" ]]; then
    echo "$brew_llvm_root/bin/clang" "$brew_llvm_root/bin/clang++"
    return
  fi

  echo "clang" "clang++"
}

read -r fuzz_cc fuzz_cxx < <(detect_fuzzer_toolchain)
fuzz_cxx_flags=""
fuzz_link_flags=""
fuzz_root="$(cd "$(dirname "$fuzz_cc")/.." && pwd)"
if [[ -d "$fuzz_root/lib/c++" ]]; then
  fuzz_cxx_flags="-stdlib=libc++ -I${fuzz_root}/include/c++/v1"
  fuzz_link_flags="-stdlib=libc++ -L${fuzz_root}/lib/c++ -Wl,-rpath,${fuzz_root}/lib/c++ -lc++ -lc++abi"
fi
if [[ "$fuzz_cc" == "clang" ]]; then
  if ! ls /opt/homebrew/opt/llvm/lib/clang/*/lib/darwin/libclang_rt.fuzzer_osx.a >/dev/null 2>&1 && \
     ! ls /usr/local/opt/llvm/lib/clang/*/lib/darwin/libclang_rt.fuzzer_osx.a >/dev/null 2>&1; then
    echo "error: libFuzzer runtime not found for clang (install llvm via Homebrew)." >&2
    exit 1
  fi
fi

rm -rf "$BUILD_DIR"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$fuzz_cc" \
  -DCMAKE_CXX_COMPILER="$fuzz_cxx" \
  -DCMAKE_CXX_FLAGS="$fuzz_cxx_flags" \
  -DCMAKE_EXE_LINKER_FLAGS="$fuzz_link_flags" \
  -DEMEL_ENABLE_FUZZ=ON \
  -DEMEL_ENABLE_TESTS=OFF

cmake --build "$BUILD_DIR" --parallel

run_fuzzer() {
  local name="$1"
  local corpus="$2"
  "$BUILD_DIR/$name" -seed=1 -max_total_time=10 -max_len=4096 "$corpus"
}

run_fuzzer emel_fuzz_gguf_parser "$ROOT_DIR/tests/fuzz/corpus/gguf_parser"
run_fuzzer emel_fuzz_gbnf_parser "$ROOT_DIR/tests/fuzz/corpus/gbnf_parser"
