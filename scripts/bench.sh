#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"

SNAPSHOT=false
COMPARE=false
COMPARE_UPDATE=false
UPDATE=false
USE_ZIG=true
MODE_FLAG=""
SUITE_FILTER=""
COMBINED=false

usage() {
  cat <<'USAGE'
usage: scripts/bench.sh [--snapshot] [--compare] [--compare-update] [--update] [--zig|--system] [--llama-only|--emel-only] [--generation-only|--suite=<name>]

  --snapshot   run EMEL benchmark snapshot gate
  --compare    build and run reference comparison
  --compare-update update reference comparison snapshot
  --update     update snapshot baseline (requires --snapshot)
  --zig        use zig cc/zig c++ as the toolchain (default)
  --system     use system cc/c++
  --llama-only run only the reference benchmarks
  --emel-only  run only the EMEL benchmarks
  --generation-only run only the generation benchmark suite
  --suite=...  run only the named benchmark suite
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --snapshot) SNAPSHOT=true ;;
    --compare) COMPARE=true ;;
    --compare-update) COMPARE=true; COMPARE_UPDATE=true ;;
    --update) UPDATE=true ;;
    --zig) USE_ZIG=true ;;
    --system) USE_ZIG=false ;;
    --llama-only) MODE_FLAG="--mode=reference" ;;
    --emel-only) MODE_FLAG="--mode=emel" ;;
    --generation-only) SUITE_FILTER="generation" ;;
    --suite=*) SUITE_FILTER="${arg#--suite=}" ;;
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

if ! $SNAPSHOT && ! $COMPARE; then
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
  shift
  if [[ -n "$SUITE_FILTER" ]]; then
    EMEL_BENCH_SUITE="$SUITE_FILTER" "$build_dir/bench_runner" "$@"
    return
  fi
  "$build_dir/bench_runner" "$@"
}

configure_bench_build() {
  local build_dir="$1"

  cmake_args=(-S "$TOOLS_DIR" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release
              -DEMEL_ENABLE_TESTS=OFF
              -DREF_IMPL_REF="$ref_value")
  if [[ -n "$SUITE_FILTER" ]]; then
    cmake_args+=("-DEMEL_BENCH_SUITE_FILTER=$SUITE_FILTER")
  fi
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

  cmake "${cmake_args[@]}" >&2
  cmake --build "$build_dir" --parallel --target bench_runner >&2
}

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

  build_dir="${BENCH_COMPARE_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
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
        printf("%s ns_per_op=%s\n", name, emel);
      }
      skip_next = 0;
    }
  ' "$compare_output" > "$current_snapshot"

  TOLERANCE="${BENCH_TOLERANCE:-0.10}"
  BASELINE="$ROOT_DIR/snapshots/bench/benchmarks.txt"

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
    mkdir -p "$(dirname "$BASELINE")"
    {
      printf "# ref=%s\n" "$ref_value"
      printf "# toolchain=%s\n" "$bench_cxx"
      cat "$current_snapshot"
    } > "$BASELINE"
    echo "updated $BASELINE"
  else
    if [[ ! -f "$BASELINE" ]]; then
      echo "error: missing baseline $BASELINE (run scripts/bench.sh --snapshot --update)" >&2
      exit 1
    fi

    awk -v tol="$TOLERANCE" -v scoped="$([[ -n "$SUITE_FILTER" ]] && echo 1 || echo 0)" '
    function parse_base(line,    n, fields, name, ns, i, pair) {
      n = split(line, fields, " ");
      name = fields[1];
      for (i = 2; i <= n; ++i) {
        if (fields[i] ~ /^ns_per_op=/) {
          split(fields[i], pair, "=");
          ns = pair[2];
          break;
        }
      }
      if (name == "" || ns == "") {
        return;
      }
      base[name] = ns;
    }
    function parse_curr(line,    n, fields, name, ns, i, pair) {
      n = split(line, fields, " ");
      name = fields[1];
      for (i = 2; i <= n; ++i) {
        if (fields[i] ~ /^ns_per_op=/) {
          split(fields[i], pair, "=");
          ns = pair[2];
          break;
        }
      }
      if (name == "" || ns == "") {
        return;
      }
      curr[name] = ns;
    }
    FNR == NR {
      if ($0 ~ /^#/) {
        skip_base = ($0 ~ /proof_status=measurement_only/);
        next;
      }
      if (skip_base) {
        skip_base = 0;
        next;
      }
      parse_base($0);
      next;
    }
    {
      if ($0 ~ /^#/) {
        skip_curr = ($0 ~ /proof_status=measurement_only/);
        next;
      }
      if (skip_curr) {
        skip_curr = 0;
        next;
      }
      parse_curr($0);
      next;
    }
    END {
      fail = 0;
      compared = 0;
      for (name in curr) {
        if (!(name in base)) {
          print "error: new benchmark entry without baseline: " name > "/dev/stderr";
          fail = 1;
          continue;
        }
        compared += 1;
        limit = base[name] * (1 + tol);
        if (curr[name] > limit) {
          printf("error: benchmark regression %s (%.3f > %.3f)\n", name, curr[name], limit) > "/dev/stderr";
          fail = 1;
        }
      }
      if (scoped && compared == 0) {
        print "error: no benchmark entries matched selected suite" > "/dev/stderr";
        fail = 1;
      }
      if (!scoped) {
        for (name in base) {
          if (!(name in curr)) {
            print "error: missing benchmark entry for " name > "/dev/stderr";
            fail = 1;
          }
        }
      }
      exit fail;
    }
    ' "$BASELINE" "$current_snapshot"
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
  TOLERANCE="${BENCH_TOLERANCE:-0.10}"
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

  build_dir="${BENCH_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
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
              -DREF_IMPL_REF="$ref_value")
  if [[ -n "$SUITE_FILTER" ]]; then
    cmake_args+=("-DEMEL_BENCH_SUITE_FILTER=$SUITE_FILTER")
  fi
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

  cmake "${cmake_args[@]}"

  cmake --build "$build_dir" --parallel --target bench_runner

  run_bench_runner "$build_dir" --mode=emel > "$CURRENT"

  for name in "${ready_names[@]+${ready_names[@]}}"; do
    if ! grep -q "^${name} " "$CURRENT"; then
      echo "error: missing benchmark entry for $name" >&2
      exit 1
    fi
  done

  if $UPDATE; then
    mkdir -p "$(dirname "$BASELINE")"
    {
      printf "# ref=%s\n" "$ref_value"
      printf "# toolchain=%s\n" "$bench_cxx"
      cat "$CURRENT"
    } > "$BASELINE"
    echo "updated $BASELINE"
  else
    if [[ ! -f "$BASELINE" ]]; then
      echo "error: missing baseline $BASELINE (run scripts/bench.sh --snapshot --update)" >&2
      exit 1
    fi

    awk -v tol="$TOLERANCE" -v scoped="$([[ -n "$SUITE_FILTER" ]] && echo 1 || echo 0)" '
    function parse_base(line,    n, fields, name, ns, i, pair) {
      n = split(line, fields, " ");
      name = fields[1];
      for (i = 2; i <= n; ++i) {
        if (fields[i] ~ /^ns_per_op=/) {
          split(fields[i], pair, "=");
          ns = pair[2];
          break;
        }
      }
      if (name == "" || ns == "") {
        return;
      }
      base[name] = ns;
    }
    function parse_curr(line,    n, fields, name, ns, i, pair) {
      n = split(line, fields, " ");
      name = fields[1];
      for (i = 2; i <= n; ++i) {
        if (fields[i] ~ /^ns_per_op=/) {
          split(fields[i], pair, "=");
          ns = pair[2];
          break;
        }
      }
      if (name == "" || ns == "") {
        return;
      }
      curr[name] = ns;
    }
    FNR == NR {
      if ($0 ~ /^#/) {
        skip_base = ($0 ~ /proof_status=measurement_only/);
        next;
      }
      if (skip_base) {
        skip_base = 0;
        next;
      }
      parse_base($0);
      next;
    }
    {
      if ($0 ~ /^#/) {
        skip_curr = ($0 ~ /proof_status=measurement_only/);
        next;
      }
      if (skip_curr) {
        skip_curr = 0;
        next;
      }
      parse_curr($0);
      next;
    }
    END {
      fail = 0;
      compared = 0;
      for (name in curr) {
        if (!(name in base)) {
          print "error: new benchmark entry without baseline: " name > "/dev/stderr";
          fail = 1;
          continue;
        }
        compared += 1;
        limit = base[name] * (1 + tol);
        if (curr[name] > limit) {
          printf("error: benchmark regression %s (%.3f > %.3f)\n", name, curr[name], limit) > "/dev/stderr";
          fail = 1;
        }
      }
      if (scoped && compared == 0) {
        print "error: no benchmark entries matched selected suite" > "/dev/stderr";
        fail = 1;
      }
      if (!scoped) {
        for (name in base) {
          if (!(name in curr)) {
            print "error: missing benchmark entry for " name > "/dev/stderr";
            fail = 1;
          }
        }
      }
      exit fail;
    }
    ' "$BASELINE" "$CURRENT"
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

  compare_build_dir="${BENCH_COMPARE_BUILD_DIR:-$ROOT_DIR/build/bench_tools_ninja}"
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
              -DREF_IMPL_REF="$ref_value")
  if [[ -n "$SUITE_FILTER" ]]; then
    cmake_args+=("-DEMEL_BENCH_SUITE_FILTER=$SUITE_FILTER")
  fi
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

  cmake "${cmake_args[@]}" >&2
  cmake --build "$compare_build_dir" --parallel --target bench_runner >&2
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
