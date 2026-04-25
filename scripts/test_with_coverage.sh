#!/usr/bin/env bash
set -euo pipefail

LINE_COVERAGE_MIN="${LINE_COVERAGE_MIN:-90}"
BRANCH_COVERAGE_MIN="${BRANCH_COVERAGE_MIN:-50}"
COVERAGE_BUILD_DIR="${EMEL_COVERAGE_BUILD_DIR:-build/coverage}"
COVERAGE_CLEAN="${EMEL_COVERAGE_CLEAN:-0}"
COVERAGE_CHANGED_ONLY="${EMEL_COVERAGE_CHANGED_ONLY:-0}"
COVERAGE_BASE_REF="${EMEL_COVERAGE_BASE_REF:-origin/main}"
COVERAGE_CHANGED_FILES="${EMEL_COVERAGE_CHANGED_FILES:-}"
COVERAGE_TEST_REGEX="${EMEL_COVERAGE_TEST_REGEX:-}"
COVERAGE_GCOV_JOBS="${EMEL_COVERAGE_GCOV_JOBS:-}"
COVERAGE_TEST_EXTRA_ARG="${EMEL_COVERAGE_TEST_EXTRA_ARG:-}"
COVERAGE_TEST_SHARDS="${EMEL_COVERAGE_TEST_SHARDS:-}"

if [[ -z "$COVERAGE_TEST_EXTRA_ARG" && "$COVERAGE_CHANGED_ONLY" == "1" ]]; then
  COVERAGE_TEST_EXTRA_ARG="--test-case-exclude=*sortformer pipeline runs maintained pcm to probabilities and segments*"
fi

# Resolve Homebrew LLVM when binaries exist but are not in PATH.
if ! command -v llvm-cov >/dev/null 2>&1 || ! command -v llvm-profdata >/dev/null 2>&1; then
  for llvm_bin in /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin; do
    if [ -x "$llvm_bin/llvm-cov" ] && [ -x "$llvm_bin/llvm-profdata" ]; then
      export PATH="$llvm_bin:$PATH"
      break
    fi
  done
fi

for tool in cmake ctest gcovr clang-format llvm-cov llvm-profdata gcc g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done
if [[ "$COVERAGE_CHANGED_ONLY" == "1" ]] && ! command -v git >/dev/null 2>&1; then
  echo "error: required tool missing: git" >&2
  exit 1
fi

if [[ "$COVERAGE_CLEAN" == "1" ]]; then
  rm -rf "$COVERAGE_BUILD_DIR"
fi

coverage_filters=(--filter src)
coverage_search_paths=("$COVERAGE_BUILD_DIR")
changed_files=()
changed_shards=()
selected_test_dirs=()
selected_test_sources=()
unknown_changed_src=0

add_changed_shard() {
  local shard="$1"
  local existing
  for existing in ${changed_shards[@]+"${changed_shards[@]}"}; do
    if [[ "$existing" == "$shard" ]]; then
      return
    fi
  done
  changed_shards+=("$shard")
}

add_selected_test_dir() {
  local dir="$1"
  local existing
  for existing in ${selected_test_dirs[@]+"${selected_test_dirs[@]}"}; do
    if [[ "$existing" == "$dir" ]]; then
      return
    fi
  done
  selected_test_dirs+=("$dir")
}

add_selected_test_source() {
  local source="$1"
  local existing
  for existing in ${selected_test_sources[@]+"${selected_test_sources[@]}"}; do
    if [[ "$existing" == "$source" ]]; then
      return
    fi
  done
  selected_test_sources+=("$source")
}

add_test_dirs_for_shard() {
  local shard="$1"
  case "$shard" in
    model_and_batch)
      add_selected_test_dir tests/model
      add_selected_test_dir tests/gguf
      add_selected_test_dir tests/gbnf
      add_selected_test_dir tests/batch
      ;;
    generator_and_runtime)
      add_selected_test_dir tests/generator
      add_selected_test_dir tests/embeddings
      add_selected_test_dir tests/logits
      add_selected_test_dir tests/token
      ;;
    diarization)
      add_selected_test_dir tests/diarization
      ;;
    sm)
      add_selected_test_dir tests/sm
      ;;
    text)
      add_selected_test_dir tests/text
      ;;
    text_encoder_plamo2)
      add_selected_test_dir tests/text/encoders
      add_selected_test_source tests/text/encoders/plamo2_tests.cpp
      ;;
    text_encoders)
      add_selected_test_dir tests/text/encoders
      ;;
    text_tokenizer)
      add_selected_test_dir tests/text/tokenizer
      ;;
    text_runtime)
      add_selected_test_dir tests/text/conditioner
      add_selected_test_dir tests/text/detokenizer
      add_selected_test_dir tests/text/formatter
      add_selected_test_dir tests/text/jinja
      add_selected_test_dir tests/text/renderer
      add_selected_test_dir tests/text/unicode
      ;;
    kernel_and_graph)
      add_selected_test_dir tests/kernel
      add_selected_test_dir tests/graph
      add_selected_test_dir tests/memory
      add_selected_test_dir tests/tensor
      ;;
  esac
}

if [[ "$COVERAGE_CHANGED_ONLY" == "1" ]]; then
  base_ref="$COVERAGE_BASE_REF"
  if ! git rev-parse --verify "$base_ref" >/dev/null 2>&1; then
    if git rev-parse --verify main >/dev/null 2>&1; then
      base_ref="main"
    else
      base_ref="HEAD"
      echo "warning: unable to resolve coverage base ref, using HEAD" >&2
    fi
  fi

  if [[ -n "$COVERAGE_CHANGED_FILES" ]]; then
    while IFS= read -r file; do
      if [[ -n "$file" ]]; then
        changed_files+=("$file")
      fi
    done < <(printf '%s\n' "$COVERAGE_CHANGED_FILES" | tr ':,' '\n')
  else
    while IFS= read -r file; do
      changed_files+=("$file")
    done < <(
      {
        git diff --name-only --diff-filter=ACMR "$base_ref...HEAD" -- src
        git diff --name-only --diff-filter=ACMR -- src
        git diff --name-only --cached --diff-filter=ACMR -- src
        git ls-files --others --exclude-standard -- src
      } | awk '
        /\.(c|cc|cpp|cxx|h|hh|hpp)$/ && !seen[$0] {
          seen[$0] = 1;
          print $0;
        }
      '
    )
  fi

  if [[ ${#changed_files[@]} -eq 0 ]]; then
    echo "no changed src files found; skipping coverage threshold check"
    exit 0
  fi

  coverage_filters=()
  echo "coverage scoped to changed src files:"
  for file in "${changed_files[@]}"; do
    echo "  $file"
    escaped_file="$(printf '%s\n' "$file" | sed 's/[][(){}.^$+*?|\\]/\\&/g')"
    coverage_filters+=(--filter "^${escaped_file}$")

    case "$file" in
      src/emel/model/*|src/emel/model*.hpp|src/emel/gguf/*|src/emel/gbnf/*|src/emel/batch/*)
        add_changed_shard model_and_batch
        ;;
      src/emel/generator/*|src/emel/embeddings/*|src/emel/logits/*|src/emel/token/*)
        add_changed_shard generator_and_runtime
        ;;
      src/emel/diarization/*)
        add_changed_shard diarization
        ;;
      src/emel/sm/*)
        add_changed_shard sm
        ;;
      src/emel/text/encoders/plamo2/*|src/emel/text/encoders/plamo2/**/*)
        add_changed_shard text_encoder_plamo2
        ;;
      src/emel/text/encoders/*|src/emel/text/encoders/**/*)
        add_changed_shard text_encoders
        ;;
      src/emel/text/tokenizer/*|src/emel/text/tokenizer/**/*)
        add_changed_shard text_tokenizer
        ;;
      src/emel/text/conditioner/*|src/emel/text/detokenizer/*|src/emel/text/formatter/*|\
      src/emel/text/jinja/*|src/emel/text/renderer/*|\
      src/emel/text/conditioner/**/*|src/emel/text/detokenizer/**/*|\
      src/emel/text/formatter/**/*|src/emel/text/jinja/**/*|src/emel/text/renderer/**/*)
        add_changed_shard text_runtime
        ;;
      src/emel/text/*)
        add_changed_shard text
        ;;
      src/emel/kernel/*|src/emel/graph/*|src/emel/memory/*|src/emel/tensor/*)
        add_changed_shard kernel_and_graph
        ;;
      src/emel/machines.hpp)
        ;;
      *)
        unknown_changed_src=1
        ;;
    esac
  done
fi

if [[ "$COVERAGE_CHANGED_ONLY" == "1" &&
      "$unknown_changed_src" == "0" &&
      ${#changed_shards[@]} -gt 0 &&
      -z "$COVERAGE_TEST_SHARDS" ]]; then
  COVERAGE_TEST_SHARDS="$(IFS=,; echo "${changed_shards[*]}")"
fi

if [[ "$COVERAGE_CHANGED_ONLY" == "1" &&
      "$unknown_changed_src" == "0" &&
      -n "$COVERAGE_TEST_SHARDS" ]]; then
  while IFS= read -r shard; do
    add_test_dirs_for_shard "$shard"
  done < <(printf '%s\n' "$COVERAGE_TEST_SHARDS" | tr ':,' '\n')
fi

cmake -S . -B "$COVERAGE_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_FLAGS="--coverage -O0" \
  -DCMAKE_CXX_FLAGS="--coverage -O0" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
  -DEMEL_TEST_EXTRA_ARG="$COVERAGE_TEST_EXTRA_ARG" \
  -DEMEL_TEST_SHARDS="$COVERAGE_TEST_SHARDS"

cmake --build "$COVERAGE_BUILD_DIR" --parallel

if [[ "$COVERAGE_CHANGED_ONLY" == "1" &&
      "$unknown_changed_src" == "0" &&
      ${#changed_files[@]} -gt 0 ]]; then
  candidate_search_paths=()
  for file in "${changed_files[@]}"; do
    case "$file" in
      *.c|*.cc|*.cpp|*.cxx)
        gcno_path="$COVERAGE_BUILD_DIR/CMakeFiles/emel.dir/$file.gcno"
        if [[ -f "$gcno_path" ]]; then
          candidate_search_paths+=("$gcno_path")
        fi
        ;;
    esac
  done
  if [[ ${#selected_test_sources[@]} -gt 0 ]]; then
    for source in "${selected_test_sources[@]}"; do
      test_gcno_path="$COVERAGE_BUILD_DIR/CMakeFiles/emel_tests_bin.dir/$source.gcno"
      if [[ -f "$test_gcno_path" ]]; then
        candidate_search_paths+=("$test_gcno_path")
      fi
    done
  else
    for dir in "${selected_test_dirs[@]+${selected_test_dirs[@]}}"; do
      test_gcno_dir="$COVERAGE_BUILD_DIR/CMakeFiles/emel_tests_bin.dir/$dir"
      if [[ -d "$test_gcno_dir" ]]; then
        candidate_search_paths+=("$test_gcno_dir")
      fi
    done
  fi
  if [[ ${#candidate_search_paths[@]} -gt 0 ]]; then
    coverage_search_paths=("${candidate_search_paths[@]}")
  fi
fi

cpu_count=2
if command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  cpu_count="$(getconf _NPROCESSORS_ONLN || echo 2)"
elif command -v sysctl >/dev/null 2>&1; then
  cpu_count="$(sysctl -n hw.ncpu || echo 2)"
fi
if [[ -z "$cpu_count" || "$cpu_count" -lt 1 ]]; then
  cpu_count=2
fi
ctest_jobs=1
if [[ -z "$COVERAGE_GCOV_JOBS" ]]; then
  COVERAGE_GCOV_JOBS="$cpu_count"
fi

find "$COVERAGE_BUILD_DIR" -name '*.gcda' -delete
find "$COVERAGE_BUILD_DIR" -maxdepth 1 -type d -name 'profiles*' -exec rm -rf {} +

if [[ -z "$COVERAGE_TEST_REGEX" ]]; then
  if [[ "$COVERAGE_CHANGED_ONLY" == "1" &&
        "$unknown_changed_src" == "0" &&
        ${#changed_shards[@]} -gt 0 ]]; then
    shard_regex=""
    for shard in "${changed_shards[@]}"; do
      if [[ -n "$shard_regex" ]]; then
        shard_regex+="|"
      fi
      shard_regex+="$shard"
    done
    COVERAGE_TEST_REGEX="^emel_tests_(${shard_regex})$"
  else
    COVERAGE_TEST_REGEX="^emel_tests"
  fi
fi

echo "running coverage test regex: ${COVERAGE_TEST_REGEX}"
ctest --test-dir "$COVERAGE_BUILD_DIR" --output-on-failure -R "$COVERAGE_TEST_REGEX" -j "$ctest_jobs"

if [[ "$COVERAGE_CHANGED_ONLY" == "1" &&
      "$unknown_changed_src" == "0" &&
      ${#changed_shards[@]} -gt 0 ]]; then
  all_test_dirs=(
    tests/model
    tests/gguf
    tests/gbnf
    tests/batch
    tests/generator
    tests/embeddings
    tests/logits
    tests/token
    tests/diarization
    tests/sm
    tests/text
    tests/text/encoders
    tests/text/tokenizer
    tests/text/conditioner
    tests/text/detokenizer
    tests/text/formatter
    tests/text/jinja
    tests/text/renderer
    tests/text/unicode
    tests/kernel
    tests/graph
    tests/memory
    tests/tensor
  )
  for dir in "${all_test_dirs[@]}"; do
    keep=0
    for selected in "${selected_test_dirs[@]}"; do
      if [[ "$selected" == "$dir" || "$selected" == "$dir"/* ]]; then
        keep=1
        break
      fi
    done
    if [[ "$keep" == "0" ]]; then
      find "$COVERAGE_BUILD_DIR/CMakeFiles/emel_tests_bin.dir/$dir" \
        -name '*.gcda' -delete 2>/dev/null || true
    fi
  done
  rm -f "$COVERAGE_BUILD_DIR/CMakeFiles/emel_tests_bin.dir/tests/doctest_main.cpp.gcda"
fi

echo "enforcing coverage thresholds: line >= ${LINE_COVERAGE_MIN}%, branch >= ${BRANCH_COVERAGE_MIN}%"

gcovr \
  --root . \
  -j "$COVERAGE_GCOV_JOBS" \
  "${coverage_filters[@]}" \
  --exclude tests \
  --exclude 'src/emel/.*/sm.hpp' \
  --gcov-ignore-parse-errors suspicious_hits.warn_once_per_file \
  --exclude-throw-branches \
  --exclude-unreachable-branches \
  --txt-summary \
  --print-summary \
  --fail-under-line "$LINE_COVERAGE_MIN" \
  --fail-under-branch "$BRANCH_COVERAGE_MIN" \
  "${coverage_search_paths[@]}"
