#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${EMEL_EMBEDDED_SIZE_BUILD_ROOT:-$ROOT_DIR/build/embedded_size}"
REFERENCE_REPOSITORY="${EMEL_EMBEDDED_SIZE_REFERENCE_REPOSITORY:-https://github.com/ggml-org/llama.cpp.git}"
REFERENCE_REF="${EMEL_EMBEDDED_SIZE_REF:-}"
SNAPSHOT_PATH="${EMEL_EMBEDDED_SIZE_SNAPSHOT_PATH:-$ROOT_DIR/snapshots/embedded_size/summary.txt}"

USE_ZIG=true
JSON_OUTPUT=false
SNAPSHOT_UPDATE=false

BUILD_TYPE="MinSizeRel"
MEASUREMENT_MODE="linked_executable"
MEASUREMENT_SCOPE="e2e_inference"
WORKLOAD_NAME="qwen3_0_6b_prompt_hello_max_tokens_1"
MODEL_FIXTURE_REL="tests/models/Qwen3-0.6B-Q8_0.gguf"
PROMPT_TEXT="hello"
MAX_TOKENS=1

usage() {
  cat <<'USAGE'
usage: scripts/embedded_size.sh [--zig|--system] [--ref=<git-ref>] [--json] [--snapshot-update]

Build final emel and llama.cpp Qwen3 E2E runner executables with size-oriented
flags, optionally smoke-run them against the maintained local fixture, and
report final linked binary sizes.

Output fields:
  raw_bytes       on-disk executable bytes before stripping
  stripped_bytes  on-disk executable bytes after strip in a temporary copy
  section_bytes   live code/data section bytes reported by `size`

Notes:
  - This measures executable size for the maintained Qwen3-0.6B `hello` ->
    first-token path, not static libraries.
  - The runtime smoke uses `tests/models/Qwen3-0.6B-Q8_0.gguf` when present.
  - Both executables still include the platform runtime selected by the toolchain.
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
host_arch="$(uname -m)"
strip_args=()
linker_gc_flags=()
case "$platform" in
  Darwin)
    strip_args=(-S -x)
    linker_gc_flags=(-Wl,-dead_strip)
    ;;
  *)
    strip_args=(--strip-debug --strip-unneeded)
    linker_gc_flags=(-Wl,--gc-sections)
    ;;
esac

case "$host_arch" in
  arm64|aarch64) probe_backend="aarch64" ;;
  x86_64|amd64) probe_backend="x86_64" ;;
  *)
    echo "error: unsupported host architecture for embedded probe: $host_arch" >&2
    exit 1
    ;;
esac

common_compile_flags="-ffunction-sections -fdata-sections"
common_compile_flags_csv="-ffunction-sections,-fdata-sections"
linker_gc_flags_csv="$(IFS=,; printf '%s' "${linker_gc_flags[*]}")"

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
  REFERENCE_REF="$(git -C "$ref_dir" rev-parse HEAD)"
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
  if ! strip "${strip_args[@]}" "$temp_path" >/dev/null 2>&1; then
    printf '%s\n' "$raw_size"
    return
  fi
  stripped_size="$(raw_bytes "$temp_path")"
  if [[ "$stripped_size" -gt "$raw_size" ]]; then
    printf '%s\n' "$raw_size"
    return
  fi
  printf '%s\n' "$stripped_size"
}

section_bytes() {
  local input_path="$1"
  case "$platform" in
    Darwin)
      size -m "$input_path" | awk '
        /^Segment __PAGEZERO:/ { skip = 1; next }
        /^Segment __LINKEDIT:/ { skip = 1; next }
        /^Segment / { skip = 0; next }
        /^[[:space:]]*total / { if (!skip) sum += $2 }
        END { print sum + 0 }'
      ;;
    *)
      size "$input_path" | awk 'NR > 1 && NF >= 6 { sum += $1 + $2 + $3 } END { print sum + 0 }'
      ;;
  esac
}

run_probe_binary() {
  local binary="$1"
  local model_path="$2"
  if ! (ulimit -s 8192; "$binary" "$model_path" >/dev/null 2>&1); then
    echo "error: probe executable failed: $binary" >&2
    exit 1
  fi
}

configure_cmake_project() {
  local source_dir="$1"
  local build_dir="$2"
  local with_asm="$3"
  shift 3

  local cmake_args=(
    -S "$source_dir"
    -B "$build_dir"
    -G Ninja
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    -DCMAKE_CXX_SCAN_FOR_MODULES=OFF
    "-DCMAKE_C_FLAGS=$common_compile_flags"
    "-DCMAKE_CXX_FLAGS=$common_compile_flags"
    "-DCMAKE_EXE_LINKER_FLAGS=${linker_gc_flags[*]}"
    "-DCMAKE_C_COMPILER=$bench_cc"
    "-DCMAKE_CXX_COMPILER=$bench_cxx"
  )
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  if [[ "$with_asm" == "with_asm" && -n "$bench_asm_arg" ]]; then
    cmake_args+=("-DCMAKE_ASM_COMPILER=$bench_cc")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_asm_arg")
  fi
  cmake_args+=("$@")

  cmake "${cmake_args[@]}"
}

prepare_toolchain

reference_checkout_dir="$BUILD_ROOT/reference-src"
emel_probe_source_dir="$ROOT_DIR/tools/embedded_size/emel_probe"
emel_probe_build_dir="$BUILD_ROOT/emel_probe_build"
reference_probe_source_dir="$ROOT_DIR/tools/embedded_size/reference_probe"
reference_probe_build_dir="$BUILD_ROOT/reference_probe_build"
temp_strip_dir="$BUILD_ROOT/stripped"

mkdir -p "$BUILD_ROOT" "$temp_strip_dir"
rm -rf "$emel_probe_build_dir" "$reference_probe_build_dir"
prepare_reference_checkout "$reference_checkout_dir"

configure_cmake_project "$emel_probe_source_dir" "$emel_probe_build_dir" without_asm \
  "-DEMEL_ROOT_DIR=$ROOT_DIR"
cmake --build "$emel_probe_build_dir" --parallel --target emel_qwen3_e2e_probe

configure_cmake_project "$reference_probe_source_dir" "$reference_probe_build_dir" with_asm \
  "-DREFERENCE_CHECKOUT_DIR=$reference_checkout_dir"
cmake --build "$reference_probe_build_dir" --parallel --target reference_qwen3_e2e_probe

emel_binary="$emel_probe_build_dir/emel_qwen3_e2e_probe"
reference_binary="$reference_probe_build_dir/reference_qwen3_e2e_probe"
model_fixture_path="$ROOT_DIR/$MODEL_FIXTURE_REL"
runtime_smoke="skipped_missing_fixture"

if [[ ! -f "$emel_binary" ]]; then
  echo "error: missing built emel probe executable: $emel_binary" >&2
  exit 1
fi

if [[ ! -f "$reference_binary" ]]; then
  echo "error: missing built reference probe executable: $reference_binary" >&2
  exit 1
fi

if [[ -f "$model_fixture_path" ]]; then
  run_probe_binary "$emel_binary" "$model_fixture_path"
  run_probe_binary "$reference_binary" "$model_fixture_path"
  runtime_smoke="passed"
fi

emel_raw_total="$(raw_bytes "$emel_binary")"
emel_stripped_total="$(stripped_bytes "$emel_binary" "$temp_strip_dir/emel_qwen3_e2e_probe")"
emel_section_total="$(section_bytes "$emel_binary")"

reference_raw_total="$(raw_bytes "$reference_binary")"
reference_stripped_total="$(stripped_bytes "$reference_binary" "$temp_strip_dir/reference_qwen3_e2e_probe")"
reference_section_total="$(section_bytes "$reference_binary")"

reference_raw_ratio="$(awk -v a="$emel_raw_total" -v b="$reference_raw_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"
reference_stripped_ratio="$(awk -v a="$emel_stripped_total" -v b="$reference_stripped_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"
reference_section_ratio="$(awk -v a="$emel_section_total" -v b="$reference_section_total" 'BEGIN { if (b == 0) { print "0.000"; } else { printf "%.3f", a / b; } }')"

if $SNAPSHOT_UPDATE; then
  mkdir -p "$(dirname "$SNAPSHOT_PATH")"
  {
    printf '# embedded_size_config: reference_ref=%s toolchain=%s build_type=%s compile_flags=%s\n' \
      "$REFERENCE_REF" "$bench_cxx" "$BUILD_TYPE" "$common_compile_flags_csv"
    printf '# embedded_size_measurement: mode=%s scope=%s workload=%s backend=%s link_flags=%s model_fixture=%s prompt=%s max_tokens=%s runtime_smoke=%s\n' \
      "$MEASUREMENT_MODE" "$MEASUREMENT_SCOPE" "$WORKLOAD_NAME" "$probe_backend" \
      "$linker_gc_flags_csv" "$MODEL_FIXTURE_REL" "$PROMPT_TEXT" "$MAX_TOKENS" "$runtime_smoke"
    printf '# embedded_size_emel: raw_bytes=%s stripped_bytes=%s section_bytes=%s binary=%s\n' \
      "$emel_raw_total" "$emel_stripped_total" "$emel_section_total" "${emel_binary#$ROOT_DIR/}"
    printf '# embedded_size_reference: raw_bytes=%s stripped_bytes=%s section_bytes=%s binary=%s\n' \
      "$reference_raw_total" "$reference_stripped_total" "$reference_section_total" \
      "${reference_binary#$ROOT_DIR/}"
    printf '# embedded_size_ratio: raw=%s stripped=%s section=%s\n' \
      "$reference_raw_ratio" "$reference_stripped_ratio" "$reference_section_ratio"
  } > "$SNAPSHOT_PATH"
fi

if $JSON_OUTPUT; then
  printf '{\n'
  printf '  "mode": "%s",\n' "$MEASUREMENT_MODE"
  printf '  "scope": "%s",\n' "$MEASUREMENT_SCOPE"
  printf '  "workload": "%s",\n' "$WORKLOAD_NAME"
  printf '  "backend": "%s",\n' "$probe_backend"
  printf '  "reference_ref": "%s",\n' "$REFERENCE_REF"
  printf '  "toolchain": "%s",\n' "$bench_cxx"
  printf '  "build_type": "%s",\n' "$BUILD_TYPE"
  printf '  "compile_flags": "%s",\n' "$common_compile_flags_csv"
  printf '  "link_flags": "%s",\n' "$linker_gc_flags_csv"
  printf '  "model_fixture": "%s",\n' "$MODEL_FIXTURE_REL"
  printf '  "prompt": "%s",\n' "$PROMPT_TEXT"
  printf '  "max_tokens": %s,\n' "$MAX_TOKENS"
  printf '  "runtime_smoke": "%s",\n' "$runtime_smoke"
  printf '  "emel": {\n'
  printf '    "raw_bytes": %s,\n' "$emel_raw_total"
  printf '    "stripped_bytes": %s,\n' "$emel_stripped_total"
  printf '    "section_bytes": %s,\n' "$emel_section_total"
  printf '    "binary": "%s"\n' "${emel_binary#$ROOT_DIR/}"
  printf '  },\n'
  printf '  "reference": {\n'
  printf '    "raw_bytes": %s,\n' "$reference_raw_total"
  printf '    "stripped_bytes": %s,\n' "$reference_stripped_total"
  printf '    "section_bytes": %s,\n' "$reference_section_total"
  printf '    "binary": "%s"\n' "${reference_binary#$ROOT_DIR/}"
  printf '  },\n'
  printf '  "ratio": {\n'
  printf '    "raw": %s,\n' "$reference_raw_ratio"
  printf '    "stripped": %s,\n' "$reference_stripped_ratio"
  printf '    "section": %s\n' "$reference_section_ratio"
  printf '  }\n'
  printf '}\n'
fi
