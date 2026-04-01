#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${EMEL_EMBEDDED_SIZE_BUILD_ROOT:-$ROOT_DIR/build/embedded_size}"
REFERENCE_REPOSITORY="${EMEL_EMBEDDED_SIZE_REFERENCE_REPOSITORY:-https://github.com/ggml-org/llama.cpp.git}"

USE_ZIG=true
JSON_OUTPUT=false
SNAPSHOT_UPDATE=false
REFERENCE_REF="${EMEL_EMBEDDED_SIZE_REF:-}"
SNAPSHOT_PATH="${EMEL_EMBEDDED_SIZE_SNAPSHOT_PATH:-$ROOT_DIR/snapshots/embedded_size/summary.txt}"

usage() {
  cat <<'USAGE'
usage: scripts/embedded_size.sh [--zig|--system] [--ref=<git-ref>] [--json] [--snapshot-update]

Build emel and llama.cpp separately with embedded-oriented static/min-size flags
and compare their static payload sizes.

Output fields:
  raw_bytes       on-disk archive bytes before stripping
  stripped_bytes  on-disk archive bytes after strip in a temporary copy
  section_bytes   summed object-code/data bytes reported by `size`

Notes:
  - This reports static payload, not a final application image.
  - emel is currently header-heavy, so archive payload can undercount code that
    is instantiated only in downstream translation units.
  - the reference total includes ggml because llama.cpp depends on it.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --zig) USE_ZIG=true ;;
  --system) USE_ZIG=false ;;
  --json) JSON_OUTPUT=true ;;
  --snapshot-update) SNAPSHOT_UPDATE=true ;;
    --ref=*) REFERENCE_REF="${arg#--ref=}" ;;
    --snapshot-path=*) SNAPSHOT_PATH="${arg#--snapshot-path=}" ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      usage
      exit 1
      ;;
  esac
done

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
}

for tool in cmake ninja git size strip; do
  require_tool "$tool"
done

prepare_toolchain() {
  bench_cc="${EMBEDDED_SIZE_CC:-cc}"
  bench_cxx="${EMBEDDED_SIZE_CXX:-c++}"
  bench_cc_arg=""
  bench_cxx_arg=""
  bench_asm_arg=""
  if $USE_ZIG; then
    require_tool zig
    bench_cc="$(command -v zig)"
    bench_cxx="$bench_cc"
    bench_cc_arg="cc"
    bench_cxx_arg="c++"
    bench_asm_arg="cc"
  fi
}

if [[ -z "$REFERENCE_REF" ]]; then
  ref_file="$ROOT_DIR/tools/bench/reference_ref.txt"
  if [[ -f "$ref_file" ]]; then
    REFERENCE_REF="$(head -n 1 "$ref_file" | tr -d '[:space:]')"
  fi
fi

if [[ -z "$REFERENCE_REF" ]]; then
  REFERENCE_REF="master"
fi

platform="$(uname -s)"
strip_args=()
case "$platform" in
  Darwin)
    strip_args=(-S -x)
    ;;
  *)
    strip_args=(--strip-debug --strip-unneeded)
    ;;
esac

common_compile_flags="-ffunction-sections -fdata-sections"

prepare_reference_checkout() {
  local ref_dir="$1"
  mkdir -p "$(dirname "$ref_dir")"
  if [[ -d "$ref_dir/.git" ]]; then
    git -C "$ref_dir" fetch --tags origin
  else
    rm -rf "$ref_dir"
    git clone "$REFERENCE_REPOSITORY" "$ref_dir"
  fi
  git -C "$ref_dir" checkout --detach "$REFERENCE_REF"
}

configure_emel_build() {
  local build_dir="$1"
  local cmake_args=(
    -S "$ROOT_DIR"
    -B "$build_dir"
    -G Ninja
    -DCMAKE_BUILD_TYPE=MinSizeRel
    -DEMEL_ENABLE_TESTS=OFF
    "-DCMAKE_C_FLAGS=$common_compile_flags"
    "-DCMAKE_CXX_FLAGS=$common_compile_flags"
    "-DCMAKE_C_COMPILER=$bench_cc"
    "-DCMAKE_CXX_COMPILER=$bench_cxx"
  )
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$build_dir" --parallel --target emel
}

configure_reference_build() {
  local ref_source_dir="$1"
  local build_dir="$2"
  local cmake_args=(
    -S "$ref_source_dir"
    -B "$build_dir"
    -G Ninja
    -DCMAKE_BUILD_TYPE=MinSizeRel
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_BUILD_COMMON=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_TOOLS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_TOOLS_INSTALL=OFF
    -DLLAMA_TESTS_INSTALL=OFF
    -DLLAMA_OPENSSL=OFF
    -DGGML_STATIC=ON
    -DGGML_BUILD_TESTS=OFF
    -DGGML_BUILD_EXAMPLES=OFF
    -DGGML_METAL=OFF
    -DGGML_BLAS=OFF
    -DGGML_ACCELERATE=OFF
    -DGGML_OPENMP=OFF
    -DGGML_NATIVE=OFF
    "-DCMAKE_C_FLAGS=$common_compile_flags"
    "-DCMAKE_CXX_FLAGS=$common_compile_flags"
    "-DCMAKE_C_COMPILER=$bench_cc"
    "-DCMAKE_CXX_COMPILER=$bench_cxx"
    "-DCMAKE_ASM_COMPILER=$bench_cc"
  )
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_asm_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  cmake "${cmake_args[@]}"
  cmake --build "$build_dir" --parallel --target llama ggml ggml-base ggml-cpu
}

raw_bytes() {
  wc -c < "$1" | awk '{print $1}'
}

stripped_bytes() {
  local input_path="$1"
  local temp_path="$2"
  local raw_size
  local stripped_size
  raw_size="$(raw_bytes "$input_path")"
  cp "$input_path" "$temp_path"
  strip "${strip_args[@]}" "$temp_path"
  stripped_size="$(raw_bytes "$temp_path")"
  if [[ "$stripped_size" -gt "$raw_size" ]]; then
    printf '%s\n' "$raw_size"
    return
  fi
  printf '%s\n' "$stripped_size"
}

section_bytes() {
  size "$1" | awk 'NR > 1 && NF >= 6 { sum += $(NF - 2) } END { print sum + 0 }'
}

sum_metric() {
  local fn_name="$1"
  shift
  local total=0
  local path
  for path in "$@"; do
    total="$(( total + $("$fn_name" "$path") ))"
  done
  printf '%s\n' "$total"
}

print_artifact_block() {
  local label="$1"
  local raw_total="$2"
  local stripped_total="$3"
  local section_total="$4"
  shift 4
  printf '%s\n' "$label"
  printf '  raw_bytes: %s\n' "$raw_total"
  printf '  stripped_bytes: %s\n' "$stripped_total"
  printf '  section_bytes: %s\n' "$section_total"
  printf '  artifacts:\n'
  local path
  for path in "$@"; do
    printf '    - %s\n' "${path#$ROOT_DIR/}"
  done
}

prepare_toolchain

emel_build_dir="$BUILD_ROOT/emel"
reference_checkout_dir="$BUILD_ROOT/reference-src"
reference_build_dir="$BUILD_ROOT/reference"
temp_strip_dir="$BUILD_ROOT/stripped"

mkdir -p "$BUILD_ROOT" "$temp_strip_dir"
prepare_reference_checkout "$reference_checkout_dir"
configure_emel_build "$emel_build_dir"
configure_reference_build "$reference_checkout_dir" "$reference_build_dir"

emel_lib="$emel_build_dir/libemel.a"
if [[ ! -f "$emel_lib" ]]; then
  echo "error: missing built emel archive: $emel_lib" >&2
  exit 1
fi

reference_libs=()
while IFS= read -r lib_path; do
  reference_libs+=("$lib_path")
done < <(find "$reference_build_dir" -type f \( -name 'libllama.a' -o -name 'libggml*.a' \) | sort)

if [[ "${#reference_libs[@]}" -eq 0 ]]; then
  echo "error: missing built reference archives under $reference_build_dir" >&2
  exit 1
fi

emel_raw_total="$(raw_bytes "$emel_lib")"
emel_stripped_total="$(stripped_bytes "$emel_lib" "$temp_strip_dir/libemel.a")"
emel_section_total="$(section_bytes "$emel_lib")"

reference_raw_total="$(sum_metric raw_bytes "${reference_libs[@]}")"
reference_section_total="$(sum_metric section_bytes "${reference_libs[@]}")"
reference_stripped_total=0
for lib_path in "${reference_libs[@]}"; do
  lib_name="$(basename "$lib_path")"
  reference_stripped_total="$(( reference_stripped_total + $(stripped_bytes "$lib_path" "$temp_strip_dir/$lib_name") ))"
done

raw_ratio="$(awk -v a="$emel_raw_total" -v b="$reference_raw_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"
stripped_ratio="$(awk -v a="$emel_stripped_total" -v b="$reference_stripped_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"
section_ratio="$(awk -v a="$emel_section_total" -v b="$reference_section_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"

if $SNAPSHOT_UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT_PATH")"
  {
    printf '# embedded_size_config: reference_ref=%s toolchain=%s build_type=MinSizeRel compile_flags=%s\n' \
      "$REFERENCE_REF" "$bench_cxx" "-ffunction-sections,-fdata-sections"
    printf '# embedded_size_emel: raw_bytes=%s stripped_bytes=%s section_bytes=%s\n' \
      "$emel_raw_total" "$emel_stripped_total" "$emel_section_total"
    printf '# embedded_size_reference: raw_bytes=%s stripped_bytes=%s section_bytes=%s\n' \
      "$reference_raw_total" "$reference_stripped_total" "$reference_section_total"
    printf '# embedded_size_ratio: raw=%s stripped=%s section=%s\n' \
      "$raw_ratio" "$stripped_ratio" "$section_ratio"
  } > "$SNAPSHOT_PATH"
fi

if $JSON_OUTPUT; then
  printf '{\n'
  printf '  "mode": "embedded_static_payload",\n'
  printf '  "reference_ref": "%s",\n' "$REFERENCE_REF"
  printf '  "toolchain": "%s",\n' "$bench_cxx"
  printf '  "emel": {\n'
  printf '    "raw_bytes": %s,\n' "$emel_raw_total"
  printf '    "stripped_bytes": %s,\n' "$emel_stripped_total"
  printf '    "section_bytes": %s,\n' "$emel_section_total"
  printf '    "artifacts": ["%s"]\n' "${emel_lib#$ROOT_DIR/}"
  printf '  },\n'
  printf '  "reference": {\n'
  printf '    "raw_bytes": %s,\n' "$reference_raw_total"
  printf '    "stripped_bytes": %s,\n' "$reference_stripped_total"
  printf '    "section_bytes": %s,\n' "$reference_section_total"
  printf '    "artifacts": [\n'
  for idx in "${!reference_libs[@]}"; do
    suffix=','
    if [[ "$idx" -eq "$(( ${#reference_libs[@]} - 1 ))" ]]; then
      suffix=''
    fi
    printf '      "%s"%s\n' "${reference_libs[$idx]#$ROOT_DIR/}" "$suffix"
  done
  printf '    ]\n'
  printf '  },\n'
  printf '  "ratio": {\n'
  printf '    "raw": %s,\n' "$raw_ratio"
  printf '    "stripped": %s,\n' "$stripped_ratio"
  printf '    "section": %s\n' "$section_ratio"
  printf '  }\n'
  printf '}\n'
  exit 0
fi

printf 'Embedded Static Payload Comparison\n'
printf 'reference_ref: %s\n' "$REFERENCE_REF"
printf 'toolchain: %s\n' "$bench_cxx"
printf 'build_type: MinSizeRel\n'
printf 'compile_flags: %s\n' "$common_compile_flags"
printf '\n'

print_artifact_block "emel" "$emel_raw_total" "$emel_stripped_total" "$emel_section_total" \
  "$emel_lib"
printf '\n'
print_artifact_block "reference (llama + ggml)" \
  "$reference_raw_total" "$reference_stripped_total" "$reference_section_total" \
  "${reference_libs[@]}"
printf '\n'
printf 'ratios\n'
printf '  raw: %sx\n' "$raw_ratio"
printf '  stripped: %sx\n' "$stripped_ratio"
printf '  section: %sx\n' "$section_ratio"
printf '\n'
printf 'notes\n'
printf '  - emel archive payload is a lower-bound for deployed code because major runtime paths are header-instantiated in downstream translation units.\n'
printf '  - reference payload includes ggml static archives because llama.cpp depends on them for the runtime.\n'
printf '  - this is a static payload estimate for embedded builds, not a final flashed image or firmware map.\n'
if $SNAPSHOT_UPDATE; then
  printf '\n'
  printf 'snapshot\n'
  printf '  - updated: %s\n' "${SNAPSHOT_PATH#$ROOT_DIR/}"
fi
