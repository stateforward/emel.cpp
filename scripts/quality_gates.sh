#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMING_FILE="$ROOT_DIR/snapshots/quality_gates/timing.txt"
QUALITY_GATES_TIMEOUT="${EMEL_QUALITY_GATES_TIMEOUT:-1800s}"
QUALITY_GATES_SCOPE="${EMEL_QUALITY_GATES_SCOPE:-changed}"
QUALITY_GATES_CHANGED_FILES="${EMEL_QUALITY_GATES_CHANGED_FILES:-}"
QUALITY_GATES_BENCH_SUITE="${EMEL_QUALITY_GATES_BENCH_SUITE:-}"
QUALITY_GATES_DOCS="${EMEL_QUALITY_GATES_DOCS:-auto}"
QUALITY_GATES_BENCH_ITERS="${EMEL_QUALITY_GATES_BENCH_ITERS:-1000}"
QUALITY_GATES_BENCH_RUNS="${EMEL_QUALITY_GATES_BENCH_RUNS:-3}"
QUALITY_GATES_BENCH_WARMUP_ITERS="${EMEL_QUALITY_GATES_BENCH_WARMUP_ITERS:-100}"
QUALITY_GATES_BENCH_WARMUP_RUNS="${EMEL_QUALITY_GATES_BENCH_WARMUP_RUNS:-1}"
QUALITY_GATES_BENCH_TOLERANCE="${EMEL_QUALITY_GATES_BENCH_TOLERANCE:-0.30}"
QUALITY_GATES_ALLOW_BENCH_REGRESSION="${EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION:-0}"
QUALITY_GATES_PARITY="${EMEL_QUALITY_GATES_PARITY:-auto}"
QUALITY_GATES_FUZZ="${EMEL_QUALITY_GATES_FUZZ:-auto}"
PARITY_DEPENDENCY_MANIFEST_BASELINE="${EMEL_PARITY_DEPENDENCY_MANIFEST_BASELINE:-}"
PARITY_DEPENDENCY_MANIFEST_CURRENT="${EMEL_PARITY_DEPENDENCY_MANIFEST_CURRENT:-}"
PARITYCHECKER_BINARY="${EMEL_PARITYCHECKER_BINARY:-$ROOT_DIR/build/paritychecker_zig/paritychecker}"
PARITY_DEPENDENCY_MANIFEST_UNCERTAIN="${EMEL_PARITY_DEPENDENCY_MANIFEST_UNCERTAIN:-0}"
BENCH_DEPENDENCY_MANIFEST_BASELINE="${EMEL_BENCH_DEPENDENCY_MANIFEST_BASELINE:-}"
BENCH_DEPENDENCY_MANIFEST_CURRENT="${EMEL_BENCH_DEPENDENCY_MANIFEST_CURRENT:-}"
BENCH_RUNNER_BINARY="${EMEL_BENCH_RUNNER_BINARY:-$ROOT_DIR/build/bench_tools_ninja/bench_runner}"
BENCH_DEPENDENCY_MANIFEST_UNCERTAIN="${EMEL_BENCH_DEPENDENCY_MANIFEST_UNCERTAIN:-0}"
if [[ -z "$PARITY_DEPENDENCY_MANIFEST_BASELINE" ]]; then
  PARITY_DEPENDENCY_MANIFEST_BASELINE="$ROOT_DIR/tools/paritychecker/dependency_manifest.txt"
fi
if [[ -z "$PARITY_DEPENDENCY_MANIFEST_CURRENT" ]]; then
  PARITY_DEPENDENCY_MANIFEST_CURRENT="$ROOT_DIR/build/paritychecker_zig/parity_dependency_manifest.current.txt"
fi
if [[ -z "$BENCH_DEPENDENCY_MANIFEST_BASELINE" ]]; then
  BENCH_DEPENDENCY_MANIFEST_BASELINE="$ROOT_DIR/tools/bench/dependency_manifest.txt"
fi
if [[ -z "$BENCH_DEPENDENCY_MANIFEST_CURRENT" ]]; then
  BENCH_DEPENDENCY_MANIFEST_CURRENT="$ROOT_DIR/build/bench_tools_ninja/bench_dependency_manifest.current.txt"
fi

if [[ -z "${EMEL_QUALITY_GATES_INNER:-}" ]]; then
  timeout_cmd=()
  if command -v timeout >/dev/null 2>&1; then
    timeout_cmd=(timeout "$QUALITY_GATES_TIMEOUT")
  elif command -v gtimeout >/dev/null 2>&1; then
    timeout_cmd=(gtimeout "$QUALITY_GATES_TIMEOUT")
  else
    echo "error: timeout tool missing (install coreutils for gtimeout on macOS)" >&2
    exit 1
  fi

  export EMEL_QUALITY_GATES_INNER=1
  exec "${timeout_cmd[@]}" "$0" "$@"
fi

timing_lines=()
total_start="$(date +%s)"
bench_status=0

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

run_step_allow_fail() {
  local name="$1"
  shift
  local start
  local end
  local duration
  local status=0
  start="$(date +%s)"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  end="$(date +%s)"
  duration="$(( end - start ))"
  timing_lines+=("$name $duration")
  write_timing_snapshot
  return "$status"
}

record_skipped_step() {
  local name="$1"
  local reason="$2"
  echo "skipping ${name}: ${reason}" >&2
  timing_lines+=("$name 0")
  write_timing_snapshot
}

run_domain_boundary_gate() {
  "$ROOT_DIR/scripts/check_domain_boundaries.sh"
}

run_sml_surface_gate() {
  "$ROOT_DIR/scripts/check_legacy_sml_surface.sh"
}

is_coverage_excluded_src_file() {
  local file="$1"
  case "$file" in
    src/emel/generator/*.hpp|src/emel/generator/**/*.hpp)
      return 0
      ;;
    src/emel/*/sm.hpp|src/emel/**/*/sm.hpp)
      return 0
      ;;
  esac
  return 1
}

changed_files=()
coverage_src_files=()
test_shards=()
bench_suites=()
bench_full=false
bench_all_suites=false
docs_needed=false
parity_needed=false
fuzz_needed=false
unknown_test_shard_src=false

add_changed_file() {
  local file="$1"
  local existing
  if [[ -z "$file" ]]; then
    return
  fi
  for existing in ${changed_files[@]+"${changed_files[@]}"}; do
    if [[ "$existing" == "$file" ]]; then
      return
    fi
  done
  changed_files+=("$file")
  case "$file" in
    src/emel/*.c|src/emel/*.cc|src/emel/*.cpp|src/emel/*.cxx|\
    src/emel/*.h|src/emel/*.hh|src/emel/*.hpp|\
    src/emel/**/*.c|src/emel/**/*.cc|src/emel/**/*.cpp|src/emel/**/*.cxx|\
    src/emel/**/*.h|src/emel/**/*.hh|src/emel/**/*.hpp)
      infer_test_shard_for_src "$file"
      if ! is_coverage_excluded_src_file "$file"; then
        coverage_src_files+=("$file")
      fi
      ;;
  esac
  case "$file" in
    tests/text/generator/*)
      add_test_shard generator_and_runtime
      ;;
    tests/text/encoders/plamo2_tests.cpp)
      add_test_shard text_encoder_plamo2
      ;;
    tests/text/encoders/*)
      add_test_shard text_encoders
      ;;
    tests/text/tokenizer/*)
      add_test_shard text_tokenizer
      ;;
    tests/text/conditioner/*|tests/text/detokenizer/*|tests/text/formatter/*|\
    tests/text/jinja/*|tests/text/renderer/*|tests/text/unicode/*)
      add_test_shard text_runtime
      ;;
    tests/text/*)
      add_test_shard text
      ;;
  esac
}

add_test_shard() {
  local shard="$1"
  local existing
  if [[ -z "$shard" ]]; then
    return
  fi
  for existing in ${test_shards[@]+"${test_shards[@]}"}; do
    if [[ "$existing" == "$shard" ]]; then
      return
    fi
  done
  test_shards+=("$shard")
}

infer_test_shard_for_src() {
  local file="$1"
  case "$file" in
    src/emel/model/*|src/emel/model*.hpp|src/emel/gguf/*|src/emel/gbnf/*|src/emel/batch/*)
      add_test_shard model_and_batch
      ;;
    src/emel/generator/*|src/emel/generator/**/*)
      add_test_shard generator_and_runtime
      ;;
    src/emel/text/generator/*|src/emel/embeddings/*|src/emel/logits/*|src/emel/token/*)
      add_test_shard generator_and_runtime
      ;;
    src/emel/diarization/*)
      add_test_shard diarization
      ;;
    src/emel/speech/*|src/emel/speech/**/*)
      add_test_shard speech
      ;;
    src/emel/sm/*)
      add_test_shard sm
      ;;
    src/emel/text/encoders/plamo2/*|src/emel/text/encoders/plamo2/**/*)
      add_test_shard text_encoder_plamo2
      ;;
    src/emel/text/encoders/*|src/emel/text/encoders/**/*)
      add_test_shard text_encoders
      ;;
    src/emel/text/tokenizer/*|src/emel/text/tokenizer/**/*)
      add_test_shard text_tokenizer
      ;;
    src/emel/text/conditioner/*|src/emel/text/detokenizer/*|src/emel/text/formatter/*|\
    src/emel/text/jinja/*|src/emel/text/renderer/*|\
    src/emel/text/conditioner/**/*|src/emel/text/detokenizer/**/*|\
    src/emel/text/formatter/**/*|src/emel/text/jinja/**/*|src/emel/text/renderer/**/*)
      add_test_shard text_runtime
      ;;
    src/emel/text/*)
      add_test_shard text
      ;;
    src/emel/kernel/*|src/emel/graph/*|src/emel/memory/*|src/emel/tensor/*)
      add_test_shard kernel_and_graph
      ;;
    src/emel/machines.hpp)
      ;;
    src/emel/*)
      unknown_test_shard_src=true
      ;;
  esac
}

add_bench_suite() {
  local suite="$1"
  local existing
  if [[ -z "$suite" ]]; then
    return
  fi
  if [[ "$suite" == "full" ]]; then
    bench_all_suites=true
    return
  fi
  for existing in ${bench_suites[@]+"${bench_suites[@]}"}; do
    if [[ "$existing" == "$suite" ]]; then
      return
    fi
  done
  bench_suites+=("$suite")
}

bench_suite_supported_for_host() {
  local suite="$1"
  local host_arch
  host_arch="$(uname -m)"
  case "$suite" in
    kernel_x86_64)
      [[ "$host_arch" == "x86_64" || "$host_arch" == "amd64" ]]
      ;;
    kernel_aarch64)
      [[ "$host_arch" == "aarch64" || "$host_arch" == "arm64" ]]
      ;;
    sm_any)
      [[ -n "${EMEL_BENCH_INTERNAL:-}" && "${EMEL_BENCH_INTERNAL:-}" != "0" ]]
      ;;
    *)
      return 0
      ;;
  esac
}

add_all_benchmark_suites_from_manifest() {
  local line
  local token
  local runner
  local priority_runner

  if [[ ! -f "$BENCH_DEPENDENCY_MANIFEST_BASELINE" ]]; then
    bench_full=true
    return
  fi

  for priority_runner in \
      gbnf_rule_parser \
      jinja_formatter \
      jinja_parser \
      logits_sampler \
      logits_validator \
      kernel_aarch64 \
      kernel_x86_64 \
      batch_planner \
      sm_any; do
    add_benchmark_suite_from_manifest "$priority_runner"
  done

  while IFS= read -r line; do
    [[ "$line" == record\ * ]] || continue
    runner=""
    for token in $line; do
      case "$token" in
        runner=*)
          runner="${token#runner=}"
          ;;
      esac
    done
    if [[ -z "$runner" || "$runner" == "all" ]]; then
      continue
    fi
    if ! bench_suite_supported_for_host "$runner"; then
      continue
    fi
    add_bench_suite "$runner"
  done < "$BENCH_DEPENDENCY_MANIFEST_BASELINE"

  if [[ ${#bench_suites[@]} -eq 0 ]]; then
    bench_full=true
  fi
}

add_benchmark_suite_from_manifest() {
  local wanted_runner="$1"
  local line
  local token
  local runner

  while IFS= read -r line; do
    [[ "$line" == record\ * ]] || continue
    runner=""
    for token in $line; do
      case "$token" in
        runner=*)
          runner="${token#runner=}"
          ;;
      esac
    done
    if [[ "$runner" != "$wanted_runner" ]]; then
      continue
    fi
    if ! bench_suite_supported_for_host "$runner"; then
      return
    fi
    add_bench_suite "$runner"
    return
  done < "$BENCH_DEPENDENCY_MANIFEST_BASELINE"
}

bench_dependency_manifest_record_matches_file() {
  local manifest_path="$1"
  local changed_file="$2"

  [[ -n "$manifest_path" ]] || return 1
  [[ "$changed_file" == "$manifest_path" || "$changed_file" == "$manifest_path"/* ]]
}

bench_dependency_manifest_apply_changed_files() {
  local file
  local line
  local token
  local runner
  local path
  local matched

  if [[ ! -f "$BENCH_DEPENDENCY_MANIFEST_BASELINE" ]]; then
    bench_full=true
    return
  fi

  for file in "${changed_files[@]+${changed_files[@]}}"; do
    matched=false
    while IFS= read -r line; do
      [[ "$line" == record\ * ]] || continue
      runner=""
      path=""
      for token in $line; do
        case "$token" in
          runner=*)
            runner="${token#runner=}"
            ;;
          path=*)
            path="${token#path=}"
            ;;
        esac
      done
      if [[ -z "$runner" || -z "$path" ]]; then
        continue
      fi
      if ! bench_dependency_manifest_record_matches_file "$path" "$file"; then
        continue
      fi
      matched=true
      if [[ "$runner" == "all" ]]; then
        bench_all_suites=true
        add_all_benchmark_suites_from_manifest
        return
      fi
      add_bench_suite "$runner"
    done < "$BENCH_DEPENDENCY_MANIFEST_BASELINE"

    case "$file" in
      tools/bench/*|tools/bench/**/*)
        if [[ "$matched" == "false" ]]; then
          bench_full=true
          return
        fi
        ;;
    esac
  done
}

collect_changed_files() {
  local base_ref
  local file

  if [[ "$QUALITY_GATES_SCOPE" == "full" ]]; then
    return
  fi

  if [[ -n "$QUALITY_GATES_CHANGED_FILES" ]]; then
    while IFS= read -r file; do
      add_changed_file "$file"
    done < <(printf '%s\n' "$QUALITY_GATES_CHANGED_FILES" | tr ':,' '\n')
    return
  fi

  if ! command -v git >/dev/null 2>&1; then
    bench_full=true
    docs_needed=true
    return
  fi

  base_ref="${EMEL_QUALITY_GATES_BASE_REF:-origin/main}"
  if ! git -C "$ROOT_DIR" rev-parse --verify "$base_ref" >/dev/null 2>&1; then
    if git -C "$ROOT_DIR" rev-parse --verify main >/dev/null 2>&1; then
      base_ref="main"
    else
      base_ref="HEAD"
      echo "warning: unable to resolve quality gate base ref, using HEAD" >&2
    fi
  fi

  while IFS= read -r file; do
    add_changed_file "$file"
  done < <(
    {
      if [[ "$base_ref" != "HEAD" ]]; then
        git -C "$ROOT_DIR" diff --name-only --diff-filter=ACMR "$base_ref...HEAD"
      fi
      git -C "$ROOT_DIR" diff --name-only --diff-filter=ACMR
      git -C "$ROOT_DIR" diff --name-only --cached --diff-filter=ACMR
      git -C "$ROOT_DIR" ls-files --others --exclude-standard
    } | awk 'NF && !seen[$0] { seen[$0] = 1; print $0 }'
  )
}

infer_quality_gate_scope() {
  local file
  local raw_suite

  if [[ "$QUALITY_GATES_SCOPE" == "full" ]]; then
    if [[ -n "$QUALITY_GATES_BENCH_SUITE" ]]; then
      while IFS= read -r raw_suite; do
        add_bench_suite "$raw_suite"
      done < <(printf '%s\n' "$QUALITY_GATES_BENCH_SUITE" | tr ':,' '\n')
    else
      bench_all_suites=true
      add_all_benchmark_suites_from_manifest
    fi
    docs_needed=true
    parity_needed=true
    fuzz_needed=true
    return
  fi

  collect_changed_files

  if [[ -n "$QUALITY_GATES_BENCH_SUITE" ]]; then
    while IFS= read -r raw_suite; do
      add_bench_suite "$raw_suite"
    done < <(printf '%s\n' "$QUALITY_GATES_BENCH_SUITE" | tr ':,' '\n')
  fi

  for file in "${changed_files[@]+${changed_files[@]}}"; do
    case "$file" in
      src/emel/generator/*|src/emel/generator/**/*)
        ;;
      docs/templates/*|tools/docsgen/*|snapshots/bench/*|snapshots/embedded_size/*|src/emel/**/sm.hpp)
        docs_needed=true
        ;;
    esac

    case "$file" in
      scripts/paritychecker.sh|tools/paritychecker/*|tools/paritychecker/**/*|\
      src/emel/gguf/*|src/emel/gguf/**/*|\
      src/emel/model/*|src/emel/model/**/*|\
      src/emel/text/*|src/emel/text/**/*|\
      src/emel/token/*|src/emel/token/**/*)
        case "$file" in
          src/emel/model/sortformer/*|src/emel/model/sortformer/**/*)
            ;;
          src/emel/text/encoders/plamo2/*|src/emel/text/encoders/plamo2/**/*)
            # The maintained GGUF parity fixture set has no PLAMO2 tokenizer model, so the
            # reference paritychecker sweep is not source-backed for this encoder family.
            ;;
          *)
            parity_needed=true
            ;;
        esac
        ;;
    esac

    case "$file" in
      scripts/fuzz_smoke.sh|tests/fuzz/*|tests/fuzz/**/*|\
      src/emel/gbnf/*|src/emel/gbnf/**/*|\
      src/emel/gguf/*|src/emel/gguf/**/*|\
      src/emel/text/jinja/*|src/emel/text/jinja/**/*|\
      src/emel/text/formatter/*|src/emel/text/formatter/**/*)
        fuzz_needed=true
        ;;
    esac

    if [[ -n "$QUALITY_GATES_BENCH_SUITE" ]]; then
      continue
    fi

    case "$file" in
      src/emel/generator/*|src/emel/generator/**/*)
        ;;
      src/emel/diarization/*|src/emel/model/sortformer/*|tests/diarization/*|\
      tools/bench/diarization*|scripts/bench_diarization_compare.sh|\
      scripts/setup_diarization_pytorch_ref_env.sh)
        add_bench_suite diarization_sortformer
        ;;
      src/emel/text/generator/*|tools/bench/generation*|scripts/bench_generation*)
        add_bench_suite generation
        ;;
      src/emel/batch/*|tests/batch/*)
        add_bench_suite batch_planner
        ;;
      src/emel/kernel/aarch64/*|tests/kernel/aarch64*)
        add_bench_suite kernel_aarch64
        ;;
      src/emel/kernel/x86_64/*|tests/kernel/x86_64*)
        add_bench_suite kernel_x86_64
        ;;
      src/emel/kernel/*|src/emel/graph/*|src/emel/memory/*|src/emel/tensor/*|\
      tests/kernel/*|tests/graph/*|tests/memory/*|tests/tensor/*)
        bench_full=true
        ;;
      src/emel/*)
        bench_full=true
        ;;
    esac
  done

  if [[ -z "$QUALITY_GATES_BENCH_SUITE" && "$bench_full" == "false" ]]; then
    bench_dependency_manifest_apply_changed_files
  fi
}

run_parity_gate() {
  local manifest_full=false

  if [[ "$QUALITY_GATES_PARITY" != "always" ]]; then
    if parity_dependency_manifest_requires_full_gate; then
      manifest_full=true
      parity_needed=true
    fi
  fi

  case "$QUALITY_GATES_PARITY" in
    always)
      run_step paritychecker "$ROOT_DIR/scripts/paritychecker.sh"
      ;;
    never)
      if $manifest_full; then
        run_step paritychecker "$ROOT_DIR/scripts/paritychecker.sh"
      else
        record_skipped_step paritychecker "disabled by EMEL_QUALITY_GATES_PARITY=never"
      fi
      ;;
    auto)
      if $parity_needed; then
        run_step paritychecker "$ROOT_DIR/scripts/paritychecker.sh"
      else
        record_skipped_step paritychecker "no paritychecker-affecting changed files"
      fi
      ;;
    *)
      echo "error: unknown EMEL_QUALITY_GATES_PARITY value '$QUALITY_GATES_PARITY'" >&2
      exit 1
      ;;
  esac
}

parity_dependency_manifest_requires_full_gate() {
  local output
  local status
  local check_args

  check_args=("--check-dependency-manifest" "$PARITY_DEPENDENCY_MANIFEST_BASELINE")
  case "$PARITY_DEPENDENCY_MANIFEST_UNCERTAIN" in
    1|true|yes)
      check_args+=("--dependency-manifest-uncertain")
      ;;
  esac

  if [[ ! -x "$PARITYCHECKER_BINARY" ]]; then
    echo "dependency manifest requires full parity gate: reason=uncertain paritychecker binary missing" >&2
    return 0
  fi

  mkdir -p "$(dirname "$PARITY_DEPENDENCY_MANIFEST_CURRENT")"
  if output="$("$PARITYCHECKER_BINARY" \
      --write-dependency-manifest "$PARITY_DEPENDENCY_MANIFEST_CURRENT" 2>&1)"; then
    echo "$output" >&2
  else
    status=$?
    echo "$output" >&2
    echo "dependency manifest requires full parity gate: reason=uncertain emit_status=$status" >&2
    return 0
  fi

  if output="$("$PARITYCHECKER_BINARY" "${check_args[@]}" 2>&1)"; then
    echo "$output" >&2
    return 1
  fi

  status=$?
  echo "$output" >&2
  if [[ "$status" -eq 3 ]]; then
    echo "dependency manifest requires full parity gate: $output" >&2
    return 0
  fi

  echo "dependency manifest requires full parity gate: reason=uncertain check_status=$status" >&2
  return 0
}

bench_dependency_manifest_check_needed() {
  if $bench_full || $bench_all_suites || [[ ${#bench_suites[@]} -gt 0 ]]; then
    return 0
  fi
  return 1
}

bench_dependency_manifest_requires_full_gate() {
  local output
  local status
  local check_args

  check_args=("--check-dependency-manifest" "$BENCH_DEPENDENCY_MANIFEST_BASELINE")
  case "$BENCH_DEPENDENCY_MANIFEST_UNCERTAIN" in
    1|true|yes)
      check_args+=("--dependency-manifest-uncertain")
      ;;
  esac

  if [[ ! -x "$BENCH_RUNNER_BINARY" ]]; then
    echo "dependency manifest requires full benchmark gate: reason=uncertain bench_runner binary missing" >&2
    return 0
  fi

  mkdir -p "$(dirname "$BENCH_DEPENDENCY_MANIFEST_CURRENT")"
  if output="$("$BENCH_RUNNER_BINARY" \
      --write-dependency-manifest "$BENCH_DEPENDENCY_MANIFEST_CURRENT" 2>&1)"; then
    echo "$output" >&2
  else
    status=$?
    echo "$output" >&2
    echo "dependency manifest requires full benchmark gate: reason=uncertain emit_status=$status" >&2
    return 0
  fi

  if output="$("$BENCH_RUNNER_BINARY" "${check_args[@]}" 2>&1)"; then
    echo "$output" >&2
    return 1
  fi

  status=$?
  echo "$output" >&2
  if [[ "$status" -eq 3 ]]; then
    echo "dependency manifest requires full benchmark gate: $output" >&2
    return 0
  fi

  echo "dependency manifest requires full benchmark gate: reason=uncertain check_status=$status" >&2
  return 0
}

run_fuzz_gate() {
  case "$QUALITY_GATES_FUZZ" in
    always)
      run_step fuzz_smoke "$ROOT_DIR/scripts/fuzz_smoke.sh"
      ;;
    never)
      record_skipped_step fuzz_smoke "disabled by EMEL_QUALITY_GATES_FUZZ=never"
      ;;
    auto)
      if $fuzz_needed; then
        run_step fuzz_smoke "$ROOT_DIR/scripts/fuzz_smoke.sh"
      else
        record_skipped_step fuzz_smoke "no fuzz-affecting changed files"
      fi
      ;;
    *)
      echo "error: unknown EMEL_QUALITY_GATES_FUZZ value '$QUALITY_GATES_FUZZ'" >&2
      exit 1
      ;;
  esac
}

run_benchmark_gates() {
  local suite
  local status=0
  local bench_iters
  local bench_runs
  local bench_warmup_iters
  local bench_warmup_runs
  local bench_tolerance

  if bench_dependency_manifest_check_needed; then
    if bench_dependency_manifest_requires_full_gate; then
      bench_full=true
    fi
  fi

  if ! $bench_full && $bench_all_suites && [[ ${#bench_suites[@]} -eq 0 ]]; then
    add_all_benchmark_suites_from_manifest
  fi

  if $bench_full; then
    if run_step_allow_fail bench_snapshot env \
      EMEL_BENCH_ITERS="$QUALITY_GATES_BENCH_ITERS" \
      EMEL_BENCH_RUNS="$QUALITY_GATES_BENCH_RUNS" \
      EMEL_BENCH_WARMUP_ITERS="$QUALITY_GATES_BENCH_WARMUP_ITERS" \
      EMEL_BENCH_WARMUP_RUNS="$QUALITY_GATES_BENCH_WARMUP_RUNS" \
      BENCH_TOLERANCE="$QUALITY_GATES_BENCH_TOLERANCE" \
      "$ROOT_DIR/scripts/bench.sh" --snapshot --compare; then
      return 0
    else
      status=$?
    fi
    return "$status"
  fi

  if [[ ${#bench_suites[@]} -eq 0 ]]; then
    record_skipped_step bench_snapshot "no benchmark-affecting changed files"
    return 0
  fi

  for suite in "${bench_suites[@]}"; do
    bench_iters="$QUALITY_GATES_BENCH_ITERS"
    bench_runs="$QUALITY_GATES_BENCH_RUNS"
    bench_warmup_iters="$QUALITY_GATES_BENCH_WARMUP_ITERS"
    bench_warmup_runs="$QUALITY_GATES_BENCH_WARMUP_RUNS"
    bench_tolerance="$QUALITY_GATES_BENCH_TOLERANCE"
    case "$suite" in
      diarization_sortformer)
        bench_iters="${EMEL_QUALITY_GATES_DIARIZATION_BENCH_ITERS:-1}"
        bench_runs="${EMEL_QUALITY_GATES_DIARIZATION_BENCH_RUNS:-5}"
        bench_warmup_iters="${EMEL_QUALITY_GATES_DIARIZATION_BENCH_WARMUP_ITERS:-1}"
        bench_warmup_runs="${EMEL_QUALITY_GATES_DIARIZATION_BENCH_WARMUP_RUNS:-1}"
        bench_tolerance="${EMEL_QUALITY_GATES_DIARIZATION_BENCH_TOLERANCE:-0.30}"
        ;;
      whisper_compare)
        if run_step_allow_fail "bench_snapshot_${suite}" \
          "$ROOT_DIR/scripts/bench_whisper_compare.sh"; then
          continue
        else
          status=$?
          continue
        fi
        ;;
      whisper_single_thread)
        if run_step_allow_fail "bench_snapshot_${suite}" \
          "$ROOT_DIR/scripts/bench_whisper_single_thread.sh"; then
          continue
        else
          status=$?
          continue
        fi
        ;;
    esac
    if run_step_allow_fail "bench_snapshot_${suite}" env \
      EMEL_BENCH_ITERS="$bench_iters" \
      EMEL_BENCH_RUNS="$bench_runs" \
      EMEL_BENCH_WARMUP_ITERS="$bench_warmup_iters" \
      EMEL_BENCH_WARMUP_RUNS="$bench_warmup_runs" \
      BENCH_TOLERANCE="$bench_tolerance" \
      "$ROOT_DIR/scripts/bench.sh" --snapshot --compare --suite="$suite"; then
      continue
    else
      status=$?
    fi
  done
  return "$status"
}

infer_quality_gate_scope

coverage_changed_files="${EMEL_QUALITY_GATES_COVERAGE_CHANGED_FILES:-}"
if [[ -z "$coverage_changed_files" &&
      "$QUALITY_GATES_SCOPE" != "full" &&
      ${#coverage_src_files[@]} -gt 0 ]]; then
  coverage_changed_files="$(printf '%s\n' "${coverage_src_files[@]}")"
fi
coverage_changed_only_default=1
if [[ "$QUALITY_GATES_SCOPE" == "full" ]]; then
  coverage_changed_only_default=0
fi
coverage_changed_only="${EMEL_QUALITY_GATES_COVERAGE_CHANGED_ONLY:-$coverage_changed_only_default}"
test_shard_list="${EMEL_QUALITY_GATES_TEST_SHARDS:-}"
if [[ -z "$test_shard_list" &&
      "$QUALITY_GATES_SCOPE" != "full" &&
      "$unknown_test_shard_src" == "false" &&
      ${#test_shards[@]} -gt 0 ]]; then
  test_shard_list="$(IFS=,; echo "${test_shards[*]}")"
fi

run_step domain_boundaries run_domain_boundary_gate
run_step legacy_sml_surface run_sml_surface_gate
run_step build_with_zig env \
  EMEL_ZIG_TEST_SHARDS="$test_shard_list" \
  "$ROOT_DIR/scripts/build_with_zig.sh"
if run_benchmark_gates; then
  bench_status=0
else
  bench_status=$?
fi
if [[ "$coverage_changed_only" == "1" &&
      "$QUALITY_GATES_SCOPE" != "full" &&
      -z "$coverage_changed_files" ]]; then
  record_skipped_step test_with_coverage "no changed src/emel files"
else
  run_step test_with_coverage env \
    EMEL_COVERAGE_CHANGED_ONLY="$coverage_changed_only" \
    EMEL_COVERAGE_CLEAN="${EMEL_QUALITY_GATES_COVERAGE_CLEAN:-0}" \
    EMEL_COVERAGE_CHANGED_FILES="$coverage_changed_files" \
    EMEL_COVERAGE_GCOV_JOBS="${EMEL_QUALITY_GATES_COVERAGE_GCOV_JOBS:-}" \
    EMEL_COVERAGE_TEST_EXTRA_ARG="${EMEL_QUALITY_GATES_COVERAGE_TEST_EXTRA_ARG:-}" \
    EMEL_COVERAGE_TEST_SHARDS="$test_shard_list" \
    "$ROOT_DIR/scripts/test_with_coverage.sh"
fi
run_parity_gate
# Temporarily disabled (SML UBSAN issue under asan_ubsan).
# TODO: re-enable once stateforward/sml.cpp fix lands.
run_fuzz_gate
run_step lint_snapshot "$ROOT_DIR/scripts/lint_snapshot.sh"
case "$QUALITY_GATES_DOCS" in
  always)
    run_step generate_docs "$ROOT_DIR/scripts/generate_docs.sh"
    ;;
  never)
    record_skipped_step generate_docs "disabled by EMEL_QUALITY_GATES_DOCS=never"
    ;;
  auto)
    if $docs_needed; then
      run_step generate_docs "$ROOT_DIR/scripts/generate_docs.sh"
    else
      record_skipped_step generate_docs "no docsgen-affecting changed files"
    fi
    ;;
  *)
    echo "error: unknown EMEL_QUALITY_GATES_DOCS value '$QUALITY_GATES_DOCS'" >&2
    exit 1
    ;;
esac

if [[ $bench_status -ne 0 ]]; then
  if [[ "$QUALITY_GATES_ALLOW_BENCH_REGRESSION" == "1" ]]; then
    echo "warning: benchmark snapshot regression ignored by explicit override" >&2
  else
    echo "error: benchmark snapshot gate failed" >&2
    exit "$bench_status"
  fi
fi
