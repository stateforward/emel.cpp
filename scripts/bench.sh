#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/build_jobs.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_jobs.sh"
# shellcheck source=scripts/zig_toolchain.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/zig_toolchain.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"

SNAPSHOT=false
COMPARE=false
COMPARE_UPDATE=false
UPDATE=false
TEST_TOOLS=false
USE_ZIG=true
MODE_FLAG=""
SUITE_FILTER=""
COMBINED=false
MEMORY_MAX_RAW=""
MEMORY_MAX_BYTES=""
RUN_ONLY=false
DEFAULT_GENERATION_WORKLOAD_ID="${EMEL_BENCH_DEFAULT_GENERATION_WORKLOAD_ID:-lfm2_single_user_hello_max_tokens_1_v1}"
DEFAULT_DIARIZATION_ITERS="${EMEL_BENCH_DEFAULT_DIARIZATION_ITERS:-1}"
DEFAULT_DIARIZATION_RUNS="${EMEL_BENCH_DEFAULT_DIARIZATION_RUNS:-3}"

usage() {
  cat <<'USAGE'
usage: scripts/bench.sh [--snapshot] [--compare] [--compare-update] [--update] [--test-tools] [--zig|--system] [--llama-only|--emel-only] [--generation-only|--suite=<name>]

  --snapshot   run EMEL benchmark snapshot gate
  --compare    build and run reference comparison
  --compare-update update reference comparison snapshot
  --update     update snapshot baseline (requires --snapshot)
  --test-tools configure an unfiltered bench-tools build and run focused bench tool tests
  --zig        use zig cc/zig c++ as the toolchain (default)
  --system     use system cc/c++
  --llama-only run only the reference benchmarks
  --emel-only  run only the EMEL benchmarks
  --generation-only run only the generation benchmark suite
  --suite=...  run only the named benchmark suite
  --memory-max=<bytes|NNpct|none> weight_streaming only: wrap the runner in a
               systemd-run user scope with MemoryMax (and MemorySwapMax=0);
               NNpct derives bytes from the fixture size; none runs unwrapped.
               Runs the suite directly (no snapshot/compare gating).
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --snapshot) SNAPSHOT=true ;;
    --compare) COMPARE=true ;;
    --compare-update) COMPARE=true; COMPARE_UPDATE=true ;;
    --update) UPDATE=true ;;
    --test-tools) TEST_TOOLS=true ;;
    --zig) USE_ZIG=true ;;
    --system) USE_ZIG=false ;;
    --llama-only) MODE_FLAG="--mode=reference" ;;
    --emel-only) MODE_FLAG="--mode=emel" ;;
    --generation-only) SUITE_FILTER="generation" ;;
    --suite=*) SUITE_FILTER="${arg#--suite=}" ;;
    --memory-max=*) MEMORY_MAX_RAW="${arg#--memory-max=}" ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      usage
      exit 1
      ;;
  esac
done

ref_file="$TOOLS_DIR/reference_ref.txt"
ref_value=""
if [[ -f "$ref_file" ]]; then
  ref_value="$(head -n 1 "$ref_file" | tr -d '[:space:]')"
fi
if [[ -n "${BENCH_REF_OVERRIDE:-}" ]]; then
  ref_value="${BENCH_REF_OVERRIDE}"
fi
if [[ -z "$ref_value" ]]; then
  ref_value="master"
fi

bench_suite_build_dir() {
  local suite="$1"
  local safe_suite

  if [[ -z "$suite" ]]; then
    printf "%s\n" "$ROOT_DIR/build/bench_tools_ninja"
    return
  fi

  # The unfiltered runner contains every suite and selects at run time via
  # EMEL_BENCH_SUITE (run_bench_runner). When it is already built, reuse it
  # instead of configuring a per-suite tree - each per-suite tree clones and
  # compiles its own llama.cpp reference (~1GB and many minutes apiece).
  # Per-suite trees remain the cold-start path so single-suite iteration on a
  # fresh checkout stays cheap.
  if [[ -x "$ROOT_DIR/build/bench_tools_ninja/bench_runner" ]]; then
    printf "%s\n" "$ROOT_DIR/build/bench_tools_ninja"
    return
  fi

  safe_suite="${suite//[^A-Za-z0-9_]/_}"
  printf "%s\n" "$ROOT_DIR/build/bench_tools_ninja_${safe_suite}"
}

# Normalized host architecture the bench runner is built for. The compare gate
# uses it to exempt foreign-arch baseline rows the runner cannot emit on this
# host (case_supported_on_host in tools/bench/bench_runner.cpp). Prefer the
# runner-emitted "# bench_host_arch:" marker in the snapshot file; fall back to
# a normalized uname -m so an older runner still gates correctly.
resolve_bench_host_arch() {
  local snapshot_file="$1"
  local arch=""
  if [[ -n "$snapshot_file" && -f "$snapshot_file" ]]; then
    arch="$(awk -F': ' '/^# bench_host_arch: / { print $2; exit }' "$snapshot_file" \
      | tr -d '[:space:]')"
  fi
  if [[ -z "$arch" ]]; then
    case "$(uname -m)" in
      x86_64|amd64) arch="x86_64" ;;
      arm64|aarch64) arch="aarch64" ;;
      *) arch="host" ;;
    esac
  fi
  printf "%s\n" "$arch"
}

if [[ -n "$MEMORY_MAX_RAW" && "$SUITE_FILTER" != "weight_streaming" ]]; then
  echo "error: --memory-max requires --suite=weight_streaming" >&2
  exit 1
fi

if [[ "$SUITE_FILTER" == "weight_streaming" ]]; then
  # Opt-in suite: rows are emitted only when this is set, so default
  # snapshot/compare runs carry no baseline requirement.
  export EMEL_BENCH_WEIGHT_STREAMING=1
fi

if [[ -n "$MEMORY_MAX_RAW" && "$MEMORY_MAX_RAW" != "none" ]]; then
  weight_streaming_fixture="$ROOT_DIR/tests/models/LFM2.5-230M-Q8_0.gguf"
  if [[ "$MEMORY_MAX_RAW" == *pct ]]; then
    memory_max_pct="${MEMORY_MAX_RAW%pct}"
    if ! [[ "$memory_max_pct" =~ ^[0-9]+$ ]]; then
      echo "error: invalid --memory-max value '$MEMORY_MAX_RAW'" >&2
      exit 1
    fi
    if [[ ! -f "$weight_streaming_fixture" ]]; then
      echo "error: --memory-max percentage needs the fixture: $weight_streaming_fixture" >&2
      exit 1
    fi
    fixture_bytes="$(stat -c%s "$weight_streaming_fixture" 2>/dev/null ||                      stat -f%z "$weight_streaming_fixture")"
    MEMORY_MAX_BYTES=$((fixture_bytes * memory_max_pct / 100))
  elif [[ "$MEMORY_MAX_RAW" =~ ^[0-9]+$ ]]; then
    MEMORY_MAX_BYTES="$MEMORY_MAX_RAW"
  else
    echo "error: invalid --memory-max value '$MEMORY_MAX_RAW'" >&2
    exit 1
  fi
  # A requested limit without a working user cgroup wrapper is a hard failure
  # (missing-tools rule), never a silent unwrapped run.
  if ! command -v systemd-run >/dev/null 2>&1; then
    echo "error: required tool missing: systemd-run (needed for --memory-max)" >&2
    exit 1
  fi
  if ! systemd-run --user --scope -p MemoryMax=64M --collect true       >/dev/null 2>&1; then
    echo "error: systemd-run user scope probe failed; --memory-max needs user cgroup delegation" >&2
    exit 1
  fi
fi

if [[ -n "$MEMORY_MAX_RAW" ]]; then
  if $SNAPSHOT || $COMPARE || $TEST_TOOLS || $UPDATE; then
    echo "error: --memory-max runs the suite directly and cannot combine with --snapshot/--compare/--update/--test-tools" >&2
    exit 1
  fi
  RUN_ONLY=true
fi

if ! $TEST_TOOLS && ! $SNAPSHOT && ! $COMPARE && ! $RUN_ONLY; then
  COMPARE=true
fi

if $UPDATE && ! $SNAPSHOT; then
  echo "error: --update requires --snapshot" >&2
  exit 1
fi

if $SNAPSHOT && $COMPARE; then
  COMBINED=true
fi

if $COMBINED && [[ -n "$MODE_FLAG" ]]; then
  echo "error: --llama-only/--emel-only cannot be used with --snapshot and --compare together" >&2
  exit 1
fi

if $COMPARE_UPDATE && [[ -n "$SUITE_FILTER" ]]; then
  echo "error: --compare-update cannot be combined with --suite or --generation-only" >&2
  exit 1
fi

if [[ "$SUITE_FILTER" == "speech_lm_moshi" ]] && ! $TEST_TOOLS; then
  if $UPDATE || $COMPARE_UPDATE; then
    echo "error: speech_lm_moshi is an EMEL load-contract guard with no snapshot update path" >&2
    exit 1
  fi
  if [[ "$MODE_FLAG" == "--mode=reference" ]]; then
    echo "error: speech_lm_moshi has no reference lane" >&2
    exit 1
  fi
  moshi_lm_args=()
  if $RUN_ONLY; then
    moshi_lm_args+=(--run-only)
  fi
  if $USE_ZIG; then
    moshi_lm_args+=(--zig)
  else
    moshi_lm_args+=(--system)
  fi
  bash "$ROOT_DIR/scripts/bench_moshi_lm_compare.sh" "${moshi_lm_args[@]}"
  exit $?
fi

prepare_toolchain() {
  bench_cc="${BENCH_CC:-cc}"
  bench_cxx="${BENCH_CXX:-c++}"
  bench_c_flags=""
  bench_cxx_flags=""
  bench_cc_arg=""
  bench_cxx_arg=""
  bench_asm_arg=""
  if $USE_ZIG; then
    bench_cc="$(command -v zig)"
    bench_cxx="$bench_cc"
    bench_cc_arg="cc"
    bench_cxx_arg="c++"
    bench_asm_arg="cc"
    bench_c_flags="-fno-sanitize=undefined"
    bench_cxx_flags="-fno-sanitize=undefined"
  fi
}

run_bench_runner() {
  local build_dir="$1"
  local generation_workload_id
  local diarization_iters
  local diarization_runs
  local -a runner_prefix=()
  shift
  generation_workload_id="${EMEL_GENERATION_WORKLOAD_ID:-$DEFAULT_GENERATION_WORKLOAD_ID}"
  diarization_iters="${EMEL_BENCH_DIARIZATION_ITERS:-$DEFAULT_DIARIZATION_ITERS}"
  diarization_runs="${EMEL_BENCH_DIARIZATION_RUNS:-$DEFAULT_DIARIZATION_RUNS}"
  if [[ -n "$MEMORY_MAX_BYTES" ]]; then
    # The scope inherits this shell's environment; MemorySwapMax=0 keeps swap
    # from masking the memory pressure the lane is meant to feel.
    export EMEL_BENCH_MEMORY_MAX="$MEMORY_MAX_BYTES"
    runner_prefix=(systemd-run --user --scope -p "MemoryMax=$MEMORY_MAX_BYTES"
                   -p MemorySwapMax=0 --same-dir --collect)
  fi
  if [[ -n "$SUITE_FILTER" ]]; then
    EMEL_GENERATION_WORKLOAD_ID="$generation_workload_id" \
      EMEL_BENCH_DIARIZATION_ITERS="$diarization_iters" \
      EMEL_BENCH_DIARIZATION_RUNS="$diarization_runs" \
      EMEL_BENCH_SUITE="$SUITE_FILTER" \
      ${runner_prefix[@]+"${runner_prefix[@]}"} "$build_dir/bench_runner" "$@"
    return
  fi
  EMEL_GENERATION_WORKLOAD_ID="$generation_workload_id" \
    EMEL_BENCH_DIARIZATION_ITERS="$diarization_iters" \
    EMEL_BENCH_DIARIZATION_RUNS="$diarization_runs" \
    ${runner_prefix[@]+"${runner_prefix[@]}"} "$build_dir/bench_runner" "$@"
}

configure_bench_build() {
  local build_dir="$1"
  local build_suite_filter="$SUITE_FILTER"

  # The default tree is always configured unfiltered: it serves every suite
  # via the runtime EMEL_BENCH_SUITE selector, and a filtered reconfigure
  # would thrash its cached objects.
  if [[ "$build_dir" == "$ROOT_DIR/build/bench_tools_ninja" ]]; then
    build_suite_filter=""
  fi

  cmake_args=(-S "$TOOLS_DIR" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release
              -DEMEL_ENABLE_TESTS=OFF
              -DREF_IMPL_REF="$ref_value"
              -DEMEL_BENCH_SUITE_FILTER="$build_suite_filter")
  cmake_args+=("-DCMAKE_C_COMPILER=$bench_cc")
  cmake_args+=("-DCMAKE_CXX_COMPILER=$bench_cxx")
  cmake_args+=("-DCMAKE_ASM_COMPILER=$bench_cc")
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_cc_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  if [[ -n "$bench_c_flags" ]]; then
    cmake_args+=("-DCMAKE_C_FLAGS=$bench_c_flags")
  fi
  if [[ -n "$bench_cxx_flags" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS=$bench_cxx_flags")
  fi
  if $USE_ZIG; then
    cmake_args+=("${EMEL_ZIG_CMAKE_PLATFORM_ARGS[@]}")
  fi

  cmake "${cmake_args[@]}" >&2
  cmake --build "$build_dir" --parallel "$EMEL_BUILD_JOBS" --target bench_runner >&2
}

update_snapshot_baseline() {
  local baseline="$1"
  local current="$2"
  local merged

  mkdir -p "$(dirname "$baseline")"
  if [[ -z "$SUITE_FILTER" ]]; then
    {
      printf "# ref=%s\n" "$ref_value"
      printf "# toolchain=%s\n" "$bench_cxx"
      cat "$current"
    } > "$baseline"
    echo "updated $baseline"
    return
  fi

  if [[ ! -f "$baseline" ]]; then
    echo "error: missing baseline $baseline (run scripts/bench.sh --snapshot --update)" >&2
    exit 1
  fi

  merged="$(mktemp)"
  awk -v ref="$ref_value" -v toolchain="$bench_cxx" '
    function entry_name(line,    n, fields) {
      if (line == "" || line ~ /^#/) {
        return "";
      }
      n = split(line, fields, " ");
      return fields[1];
    }
    FNR == NR {
      name = entry_name($0);
      if (name != "") {
        curr[name] = $0;
        order[++order_count] = name;
      }
      next;
    }
    {
      if ($0 ~ /^# ref=/) {
        print "# ref=" ref;
        next;
      }
      if ($0 ~ /^# toolchain=/) {
        print "# toolchain=" toolchain;
        next;
      }
      name = entry_name($0);
      if (name != "" && (name in curr)) {
        print curr[name];
        seen[name] = 1;
        next;
      }
      print $0;
    }
    END {
      for (i = 1; i <= order_count; ++i) {
        name = order[i];
        if (!(name in seen)) {
          print curr[name];
        }
      }
    }
  ' "$current" "$baseline" > "$merged"
  mv "$merged" "$baseline"
  echo "updated $baseline (merged suite $SUITE_FILTER)"
}

if $TEST_TOOLS; then
  if $SNAPSHOT || $COMPARE || $COMPARE_UPDATE || $UPDATE || [[ -n "$MODE_FLAG" ||
      -n "$SUITE_FILTER" ]]; then
    echo "error: --test-tools cannot be combined with benchmark run mode or suite options" >&2
    exit 1
  fi

  for tool in cmake ninja git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
  if $USE_ZIG && ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi

  prepare_toolchain

  build_dir="${BENCH_TOOLS_TEST_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
  configure_bench_build "$build_dir"
  cmake --build "$build_dir" --parallel "$EMEL_BUILD_JOBS" --target bench_runner_tests quality_gates_tests >&2
  ctest --test-dir "$build_dir" -R 'quality_gates_tests|bench_runner_tests' --output-on-failure
  exit 0
fi

if $COMBINED; then
  for tool in cmake ninja git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
  if $USE_ZIG && ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi

  prepare_toolchain

  build_dir="${BENCH_COMPARE_BUILD_DIR:-$(bench_suite_build_dir "$SUITE_FILTER")}"
  configure_bench_build "$build_dir"

  compare_output="$(mktemp)"
  run_bench_runner "$build_dir" --mode=compare > "$compare_output"

  current_snapshot="$(mktemp)"
  trap 'rm -f "$compare_output" "$current_snapshot"' EXIT
  awk '
    /^#/ {
      skip_next = ($0 ~ /proof_status=measurement_only/);
      next;
    }
    /^[^#]/ {
      if (skip_next) {
        skip_next = 0;
        next;
      }
      name = $1;
      emel = $3;
      if (name != "" && emel != "") {
        tokens_per_second = "";
        if ($4 == "ns/op" && $5 ~ /^\([0-9.]+$/ && $6 == "tokens/s),") {
          tokens_per_second = $5;
          sub(/^\(/, "", tokens_per_second);
        }
        if (tokens_per_second != "") {
          printf("%s ns_per_op=%s tokens_per_second=%s\n",
                 name, emel, tokens_per_second);
        } else {
          printf("%s ns_per_op=%s\n", name, emel);
        }
      }
      skip_next = 0;
    }
  ' "$compare_output" > "$current_snapshot"

  TOLERANCE="${BENCH_TOLERANCE:-0.30}"
  ABS_TOLERANCE_NS="${BENCH_ABS_TOLERANCE_NS:-5000}"
  BASELINE="$ROOT_DIR/snapshots/bench/benchmarks.txt"
  host_arch="$(resolve_bench_host_arch "$compare_output")"

  new_sms=()
  base_ref="${BENCH_BASE_REF:-origin/main}"
  if ! git -C "$ROOT_DIR" rev-parse --verify "$base_ref" >/dev/null 2>&1; then
    if git -C "$ROOT_DIR" rev-parse --verify main >/dev/null 2>&1; then
      base_ref="main"
    else
      base_ref="HEAD"
      echo "warning: unable to resolve base ref, using HEAD (set BENCH_BASE_REF to override)" >&2
    fi
  fi

  if [[ "$base_ref" != "HEAD" ]]; then
    while IFS= read -r line; do
      new_sms+=("$line")
    done < <(git -C "$ROOT_DIR" diff --name-status "$base_ref...HEAD" -- 'src/emel/**/sm.hpp' \
      | awk '$1 == "A" { print $2 }')
  fi

  ready_names=()
  for sm in "${new_sms[@]+${new_sms[@]}}"; do
    marker="$(grep -E "benchmark: (scaffold|designed|ready)" "$ROOT_DIR/$sm" | head -n 1 || true)"
    if [[ -z "$marker" ]]; then
      echo "error: missing benchmark marker in $sm" >&2
      exit 1
    fi
    if [[ "$marker" == *"benchmark: ready"* ]]; then
      rel="${sm#src/emel/}"
      name="${rel%/sm.hpp}"
      ready_names+=("$name")
    fi
  done

  for name in "${ready_names[@]+${ready_names[@]}}"; do
    if ! grep -q "^${name} " "$current_snapshot"; then
      echo "error: missing benchmark entry for $name" >&2
      exit 1
    fi
  done

  if $UPDATE; then
    update_snapshot_baseline "$BASELINE" "$current_snapshot"
  else
    if [[ ! -f "$BASELINE" ]]; then
      echo "error: missing baseline $BASELINE (run scripts/bench.sh --snapshot --update)" >&2
      exit 1
    fi

    awk -v tol="$TOLERANCE" -v abs_tol="$ABS_TOLERANCE_NS" \
      -v strict_regression="${EMEL_BENCH_STRICT_REGRESSION:-0}" \
      -v scoped="$([[ -n "$SUITE_FILTER" ]] && echo 1 || echo 0)" \
      -v host_arch="$host_arch" \
      -f "$ROOT_DIR/scripts/bench_compare_gate.awk" \
      "$BASELINE" "$current_snapshot"
  fi

  if $COMPARE_UPDATE; then
    compare_baseline="$ROOT_DIR/snapshots/bench/benchmarks_compare.txt"
    {
      printf "# ref=%s\n" "$ref_value"
      printf "# toolchain=%s\n" "$bench_cxx"
      cat "$compare_output"
    } > "$compare_baseline"
    echo "updated $compare_baseline"
  else
    cat "$compare_output"
  fi

  exit 0
fi

if $SNAPSHOT; then
  TOLERANCE="${BENCH_TOLERANCE:-0.30}"
  ABS_TOLERANCE_NS="${BENCH_ABS_TOLERANCE_NS:-5000}"
  BASELINE="$ROOT_DIR/snapshots/bench/benchmarks.txt"
  CURRENT="$(mktemp)"
  trap 'rm -f "$CURRENT"' EXIT

  for tool in cmake ninja git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
  if $USE_ZIG && ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi

  base_ref="${BENCH_BASE_REF:-origin/main}"
  if ! git -C "$ROOT_DIR" rev-parse --verify "$base_ref" >/dev/null 2>&1; then
    if git -C "$ROOT_DIR" rev-parse --verify main >/dev/null 2>&1; then
      base_ref="main"
    else
      base_ref="HEAD"
      echo "warning: unable to resolve base ref, using HEAD (set BENCH_BASE_REF to override)" >&2
    fi
  fi

  new_sms=()
  if [[ "$base_ref" != "HEAD" ]]; then
    while IFS= read -r line; do
      new_sms+=("$line")
    done < <(git -C "$ROOT_DIR" diff --name-status "$base_ref...HEAD" -- 'src/emel/**/sm.hpp' \
      | awk '$1 == "A" { print $2 }')
  fi

  ready_names=()
  for sm in "${new_sms[@]+${new_sms[@]}}"; do
    marker="$(grep -E "benchmark: (scaffold|designed|ready)" "$ROOT_DIR/$sm" | head -n 1 || true)"
    if [[ -z "$marker" ]]; then
      echo "error: missing benchmark marker in $sm" >&2
      exit 1
    fi
    if [[ "$marker" == *"benchmark: ready"* ]]; then
      rel="${sm#src/emel/}"
      name="${rel%/sm.hpp}"
      ready_names+=("$name")
    fi
  done

  build_dir="${BENCH_BUILD_DIR:-$(bench_suite_build_dir "$SUITE_FILTER")}"
  bench_cc="${BENCH_CC:-cc}"
  bench_cxx="${BENCH_CXX:-c++}"
  bench_c_flags=""
  bench_cxx_flags=""
  bench_cc_arg=""
  bench_cxx_arg=""
  bench_asm_arg=""
  if $USE_ZIG; then
    bench_cc="$(command -v zig)"
    bench_cxx="$bench_cc"
    bench_cc_arg="cc"
    bench_cxx_arg="c++"
    bench_asm_arg="cc"
    bench_c_flags="-fno-sanitize=undefined"
    bench_cxx_flags="-fno-sanitize=undefined"
  fi

  cmake_args=(-S "$TOOLS_DIR" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release
              -DEMEL_ENABLE_TESTS=OFF
              -DREF_IMPL_REF="$ref_value"
              -DEMEL_BENCH_SUITE_FILTER="$SUITE_FILTER")
  cmake_args+=("-DCMAKE_C_COMPILER=$bench_cc")
  cmake_args+=("-DCMAKE_CXX_COMPILER=$bench_cxx")
  cmake_args+=("-DCMAKE_ASM_COMPILER=$bench_cc")
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_cc_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  if [[ -n "$bench_c_flags" ]]; then
    cmake_args+=("-DCMAKE_C_FLAGS=$bench_c_flags")
  fi
  if [[ -n "$bench_cxx_flags" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS=$bench_cxx_flags")
  fi
  if $USE_ZIG; then
    cmake_args+=("${EMEL_ZIG_CMAKE_PLATFORM_ARGS[@]}")
  fi

  cmake "${cmake_args[@]}"

  cmake --build "$build_dir" --parallel "$EMEL_BUILD_JOBS" --target bench_runner

  run_bench_runner "$build_dir" --mode=emel > "$CURRENT"

  for name in "${ready_names[@]+${ready_names[@]}}"; do
    if ! grep -q "^${name} " "$CURRENT"; then
      echo "error: missing benchmark entry for $name" >&2
      exit 1
    fi
  done

  if $UPDATE; then
    update_snapshot_baseline "$BASELINE" "$CURRENT"
  else
    if [[ ! -f "$BASELINE" ]]; then
      echo "error: missing baseline $BASELINE (run scripts/bench.sh --snapshot --update)" >&2
      exit 1
    fi

    host_arch="$(resolve_bench_host_arch "$CURRENT")"
    awk -v tol="$TOLERANCE" -v abs_tol="$ABS_TOLERANCE_NS" \
      -v strict_regression="${EMEL_BENCH_STRICT_REGRESSION:-0}" \
      -v scoped="$([[ -n "$SUITE_FILTER" ]] && echo 1 || echo 0)" \
      -v host_arch="$host_arch" \
      -f "$ROOT_DIR/scripts/bench_compare_gate.awk" \
      "$BASELINE" "$CURRENT"
  fi
fi

if $COMPARE; then
  for tool in cmake ninja git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
  if $USE_ZIG && ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi

  compare_build_dir="${BENCH_COMPARE_BUILD_DIR:-$(bench_suite_build_dir "$SUITE_FILTER")}"
  bench_cc="${BENCH_CC:-cc}"
  bench_cxx="${BENCH_CXX:-c++}"
  bench_c_flags=""
  bench_cxx_flags=""
  bench_cc_arg=""
  bench_cxx_arg=""
  bench_asm_arg=""
  if $USE_ZIG; then
    bench_cc="$(command -v zig)"
    bench_cxx="$bench_cc"
    bench_cc_arg="cc"
    bench_cxx_arg="c++"
    bench_asm_arg="cc"
    bench_c_flags="-fno-sanitize=undefined"
    bench_cxx_flags="-fno-sanitize=undefined"
  fi

  cmake_args=(-S "$TOOLS_DIR" -B "$compare_build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release
              -DEMEL_ENABLE_TESTS=OFF
              -DREF_IMPL_REF="$ref_value"
              -DEMEL_BENCH_SUITE_FILTER="$SUITE_FILTER")
  cmake_args+=("-DCMAKE_C_COMPILER=$bench_cc")
  cmake_args+=("-DCMAKE_CXX_COMPILER=$bench_cxx")
  cmake_args+=("-DCMAKE_ASM_COMPILER=$bench_cc")
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_cc_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  if [[ -n "$bench_c_flags" ]]; then
    cmake_args+=("-DCMAKE_C_FLAGS=$bench_c_flags")
  fi
  if [[ -n "$bench_cxx_flags" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS=$bench_cxx_flags")
  fi
  if $USE_ZIG; then
    cmake_args+=("${EMEL_ZIG_CMAKE_PLATFORM_ARGS[@]}")
  fi

  cmake "${cmake_args[@]}" >&2
  cmake --build "$compare_build_dir" --parallel "$EMEL_BUILD_JOBS" --target bench_runner >&2
  if $COMPARE_UPDATE; then
    compare_baseline="$ROOT_DIR/snapshots/bench/benchmarks_compare.txt"
    {
      printf "# ref=%s\n" "$ref_value"
      printf "# toolchain=%s\n" "$bench_cxx"
      run_bench_runner "$compare_build_dir" --mode=compare
    } > "$compare_baseline"
    echo "updated $compare_baseline"
  else
    if [[ -n "$MODE_FLAG" ]]; then
      run_bench_runner "$compare_build_dir" "$MODE_FLAG"
    else
      run_bench_runner "$compare_build_dir" --mode=compare
    fi
  fi
fi

if $RUN_ONLY; then
  for tool in cmake ninja git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
  if $USE_ZIG && ! command -v zig >/dev/null 2>&1; then
    echo "error: zig not found (use --system to use system compilers)" >&2
    exit 1
  fi

  # Reuse the default unfiltered build (runtime EMEL_BENCH_SUITE selects the
  # suite) instead of configuring a per-suite tree: the reference build is
  # multi-gigabyte and the run-only mode should not duplicate it.
  run_only_build_dir="${BENCH_BUILD_DIR:-$(bench_suite_build_dir "")}"
  prepare_toolchain

  cmake_args=(-S "$TOOLS_DIR" -B "$run_only_build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release
              -DEMEL_ENABLE_TESTS=OFF
              -DREF_IMPL_REF="$ref_value"
              -DEMEL_BENCH_SUITE_FILTER=)
  cmake_args+=("-DCMAKE_C_COMPILER=$bench_cc")
  cmake_args+=("-DCMAKE_CXX_COMPILER=$bench_cxx")
  cmake_args+=("-DCMAKE_ASM_COMPILER=$bench_cc")
  if [[ -n "$bench_cc_arg" ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER_ARG1=$bench_cc_arg")
    cmake_args+=("-DCMAKE_ASM_COMPILER_ARG1=$bench_cc_arg")
  fi
  if [[ -n "$bench_cxx_arg" ]]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_ARG1=$bench_cxx_arg")
  fi
  if [[ -n "$bench_c_flags" ]]; then
    cmake_args+=("-DCMAKE_C_FLAGS=$bench_c_flags")
  fi
  if [[ -n "$bench_cxx_flags" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS=$bench_cxx_flags")
  fi
  if $USE_ZIG; then
    cmake_args+=("${EMEL_ZIG_CMAKE_PLATFORM_ARGS[@]}")
  fi

  cmake "${cmake_args[@]}" >&2
  cmake --build "$run_only_build_dir" --parallel "$EMEL_BUILD_JOBS" --target bench_runner >&2
  if [[ -n "$MEMORY_MAX_BYTES" ]]; then
    # Capped runs measure the EMEL lanes only: the llama.cpp baseline comes
    # from the unwrapped --memory-max=none run (its 385MiB compute buffer would
    # otherwise dominate any meaningful MemoryMax).
    run_bench_runner "$run_only_build_dir" --mode=emel
  else
    run_bench_runner "$run_only_build_dir" --mode=compare
  fi
fi
