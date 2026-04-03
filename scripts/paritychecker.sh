#!/usr/bin/env bash
set -euo pipefail

for tool in cmake ctest ninja zig; do
  if [[ "$tool" == "zig" && "$(uname -s)" == "Darwin" ]]; then
    continue
  fi
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool missing: $tool" >&2
    exit 1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN="${EMEL_PARITYCHECKER_TOOLCHAIN:-auto}"
MODE="${EMEL_PARITYCHECKER_MODE:-full}"
IMPLEMENTATION_MODEL="${EMEL_PARITYCHECKER_MODEL:-}"
IMPLEMENTATION_TEXT="${EMEL_PARITYCHECKER_TEXT:-hello}"
IMPLEMENTATION_MAX_TOKENS="${EMEL_PARITYCHECKER_MAX_TOKENS:-1}"
IMPLEMENTATION_FAILURE_SURFACE="${EMEL_PARITYCHECKER_IMPLEMENTATION_FAILURE_SURFACE:-summary}"
IMPLEMENTATION_LITE_CASES="${EMEL_PARITYCHECKER_IMPLEMENTATION_LITE_CASES:-smoke}"
if [[ "$TOOLCHAIN" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    TOOLCHAIN="system"
  else
    TOOLCHAIN="zig"
  fi
fi

usage() {
  cat <<'USAGE'
usage: scripts/paritychecker.sh [--implementation|--full] [--model <path>] [--text <text>]
                                [--max-tokens <count>] [--help]

  default mode runs the full parity gate across every configured reference lane
  --implementation  run a fast implementation probe on one lane with focused generation only
  --full            force the full parity gate
  --model           maintained fixture path or target model for implementation mode
  --text            prompt text for implementation mode (default: hello)
  --max-tokens      max generation tokens for implementation mode (default: 1)

environment:
  EMEL_PARITYCHECKER_MODE=implementation
  EMEL_PARITYCHECKER_MODEL=tests/models/Bonsai-1.7B.gguf
  EMEL_PARITYCHECKER_TEXT=hello
  EMEL_PARITYCHECKER_MAX_TOKENS=1
  EMEL_PARITYCHECKER_IMPLEMENTATION_FAILURE_SURFACE=summary|lite|full
  EMEL_PARITYCHECKER_IMPLEMENTATION_LITE_CASES=smoke|full
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --implementation)
      MODE="implementation"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "error: --model requires a path" >&2
        exit 1
      fi
      IMPLEMENTATION_MODEL="$2"
      shift 2
      ;;
    --text)
      if [[ $# -lt 2 ]]; then
        echo "error: --text requires a value" >&2
        exit 1
      fi
      IMPLEMENTATION_TEXT="$2"
      shift 2
      ;;
    --max-tokens)
      if [[ $# -lt 2 ]]; then
        echo "error: --max-tokens requires a value" >&2
        exit 1
      fi
      IMPLEMENTATION_MAX_TOKENS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$IMPLEMENTATION_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --max-tokens must be a positive integer" >&2
  exit 1
fi

default_ref_repository="https://github.com/ggml-org/llama.cpp.git"
default_ref_value="master"
explicit_ref_repository="${REF_IMPL_REPOSITORY:-}"
explicit_ref_value="${REF_IMPL_REF:-}"

trim_file_value() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 1
  fi
  head -n 1 "$path" | tr -d '[:space:]'
}

if value="$(trim_file_value "$ROOT_DIR/tools/paritychecker/reference_repo.txt")"; then
  default_ref_repository="$value"
fi
if value="$(trim_file_value "$ROOT_DIR/tools/paritychecker/reference_ref.txt")"; then
  default_ref_value="$value"
fi

configure_lane() {
  local build_dir="$1"
  local ref_repository="$2"
  local ref_value="$3"
  shift 3
  local -a build_targets=("$@")
  local -a cmake_args=(
    -S "$ROOT_DIR/tools/paritychecker"
    -B "$build_dir"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_METAL=OFF
    -DLLAMA_METAL=OFF
  )
  if [[ -n "$ref_repository" ]]; then
    cmake_args+=("-DREF_IMPL_REPOSITORY=$ref_repository")
  fi
  if [[ -n "$ref_value" ]]; then
    cmake_args+=("-DREF_IMPL_REF=$ref_value")
  fi

  if [[ "$TOOLCHAIN" == "zig" ]]; then
    zig_bin="$(command -v zig)"
    cmake_args+=(
      "-DCMAKE_C_COMPILER=$zig_bin"
      -DCMAKE_C_COMPILER_ARG1=cc
      "-DCMAKE_CXX_COMPILER=$zig_bin"
      -DCMAKE_CXX_COMPILER_ARG1=c++
      -DCMAKE_C_FLAGS=-fno-sanitize=undefined
      -DCMAKE_CXX_FLAGS=-fno-sanitize=undefined
    )
  elif [[ "$TOOLCHAIN" == "system" ]]; then
    cmake_args+=(
      "-DCMAKE_C_COMPILER=${PARITYCHECKER_CC:-cc}"
      "-DCMAKE_CXX_COMPILER=${PARITYCHECKER_CXX:-c++}"
    )
  else
    echo "error: unsupported EMEL_PARITYCHECKER_TOOLCHAIN '$TOOLCHAIN'" >&2
    exit 1
  fi

  cmake "${cmake_args[@]}"
  if [[ "${#build_targets[@]}" -eq 0 ]]; then
    cmake --build "$build_dir" --parallel
  else
    cmake --build "$build_dir" --parallel --target "${build_targets[@]}"
  fi
}

fixture_records() {
  awk '
    /inline constexpr maintained_fixture k_.*_generation_fixture = {/ {
      in_fixture = 1
      fixture_name = ""
      fixture_slug = ""
      fixture_rel = ""
      reference_engine = ""
      reference_repository = ""
      reference_ref = ""
      generation_parity_contract = ""
      next
    }
    in_fixture && /\.name = "/ {
      if (match($0, /"([^"]+)"/)) {
        fixture_name = substr($0, RSTART + 1, RLENGTH - 2)
      }
      next
    }
    in_fixture && /\.slug = "/ {
      if (match($0, /"([^"]+)"/)) {
        fixture_slug = substr($0, RSTART + 1, RLENGTH - 2)
      }
      next
    }
    in_fixture && /\.fixture_rel = "/ {
      if (match($0, /"([^"]+)"/)) {
        fixture_rel = substr($0, RSTART + 1, RLENGTH - 2)
      }
      next
    }
    in_fixture && /\.reference_engine = "/ {
      if (match($0, /"([^"]*)"/)) {
        reference_engine = substr($0, RSTART + 1, RLENGTH - 2)
      }
      next
    }
    in_fixture && /\.reference_repository = "/ {
      if (match($0, /"([^"]*)"/)) {
        reference_repository = substr($0, RSTART + 1, RLENGTH - 2)
      }
      next
    }
    in_fixture && /\.reference_ref = / {
      if (match($0, /"([^"]*)"/)) {
        reference_ref = substr($0, RSTART + 1, RLENGTH - 2)
      }
      next
    }
    in_fixture && /\.generation_parity_contract = "/ {
      if (match($0, /"([^"]*)"/)) {
        generation_parity_contract = substr($0, RSTART + 1, RLENGTH - 2)
      }
      next
    }
    in_fixture && /^};/ {
      if (fixture_rel != "" && reference_engine != "" && reference_repository != "") {
        print fixture_name "|" fixture_slug "|" fixture_rel "|" reference_engine "|" \
              reference_repository "|" reference_ref "|" generation_parity_contract
      }
      in_fixture = 0
    }
  ' "$ROOT_DIR/tools/generation_fixture_registry.hpp"
}

lane_matches_fixture() {
  local lane_repository="$1"
  local lane_ref="$2"
  local fixture_repository="$3"
  local fixture_ref="$4"

  if [[ "$fixture_repository" != "$lane_repository" ]]; then
    return 1
  fi
  if [[ -n "$fixture_ref" && "$fixture_ref" != "$lane_ref" ]]; then
    return 1
  fi
  return 0
}

canonicalize_path() {
  local raw_path="$1"
  local abs_path="$raw_path"
  if [[ "$abs_path" != /* ]]; then
    abs_path="$ROOT_DIR/$abs_path"
  fi
  local abs_dir
  abs_dir="$(cd "$(dirname "$abs_path")" && pwd)"
  printf '%s/%s\n' "$abs_dir" "$(basename "$abs_path")"
}

resolve_implementation_lane() {
  local model_path="$1"

  if [[ -n "$explicit_ref_repository" || -n "$explicit_ref_value" ]]; then
    effective_ref_repository="$explicit_ref_repository"
    effective_ref_value="$explicit_ref_value"
    if [[ -z "$effective_ref_repository" ]]; then
      effective_ref_repository="$default_ref_repository"
    fi
    if [[ -z "$effective_ref_value" ]]; then
      effective_ref_value="$default_ref_value"
    fi
    return 0
  fi

  effective_ref_repository="$default_ref_repository"
  effective_ref_value="$default_ref_value"

  if [[ -z "$model_path" ]]; then
    return 0
  fi

  local model_path_abs
  model_path_abs="$(canonicalize_path "$model_path")"
  local model_basename
  model_basename="$(basename "$model_path")"
  local basename_matches=0
  local basename_match_repository=""
  local basename_match_ref=""

  while IFS='|' read -r fixture_name fixture_slug fixture_rel _ fixture_repository fixture_ref _; do
    local fixture_path_abs
    fixture_path_abs="$(canonicalize_path "$fixture_rel")"
    if [[ "$fixture_path_abs" == "$model_path_abs" ]]; then
      effective_ref_repository="$fixture_repository"
      effective_ref_value="$fixture_ref"
      if [[ -z "$effective_ref_value" ]]; then
        effective_ref_value="$default_ref_value"
      fi
      return 0
    fi
    if [[ "$model_basename" == "$fixture_name" || "$model_basename" == "$fixture_slug" ]]; then
      basename_matches="$((basename_matches + 1))"
      basename_match_repository="$fixture_repository"
      basename_match_ref="$fixture_ref"
    fi
  done < <(fixture_records)

  if [[ "$basename_matches" -eq 1 ]]; then
    effective_ref_repository="$basename_match_repository"
    effective_ref_value="$basename_match_ref"
    if [[ -z "$effective_ref_value" ]]; then
      effective_ref_value="$default_ref_value"
    fi
  fi
}

implementation_models_for_lane() {
  local lane_repository="$1"
  local lane_ref="$2"

  if [[ -n "$IMPLEMENTATION_MODEL" ]]; then
    printf '%s\n' "$IMPLEMENTATION_MODEL"
    return 0
  fi

  while IFS='|' read -r _ _ fixture_rel _ fixture_repository fixture_ref _; do
    if lane_matches_fixture "$lane_repository" "$lane_ref" "$fixture_repository" "$fixture_ref"; then
      printf '%s\n' "$ROOT_DIR/$fixture_rel"
    fi
  done < <(fixture_records)
}

run_lane_generation_contract() {
  local build_dir="$1"
  local lane_repository="$2"
  local lane_ref="$3"

  while IFS='|' read -r _ _ fixture_rel _ fixture_repository fixture_ref generation_parity_contract; do
    if [[ "$generation_parity_contract" != "live_reference_generation" ]]; then
      continue
    fi
    if ! lane_matches_fixture "$lane_repository" "$lane_ref" "$fixture_repository" "$fixture_ref"; then
      continue
    fi
    "$build_dir/paritychecker" \
      --generation \
      --model "$ROOT_DIR/$fixture_rel" \
      --text hello \
      --max-tokens 10
  done < <(fixture_records)
}

run_lane_suite() {
  local build_dir="$1"
  local lane_repository="$2"
  local lane_ref="$3"

  configure_lane "$build_dir" "$lane_repository" "$lane_ref" paritychecker paritychecker_tests
  ctest --test-dir "$build_dir" --output-on-failure -R paritychecker_tests
  run_lane_generation_contract "$build_dir" "$lane_repository" "$lane_ref"
}

run_implementation_suite() {
  local build_dir="$1"
  local lane_repository="$2"
  local lane_ref="$3"

  configure_lane "$build_dir" "$lane_repository" "$lane_ref" paritychecker

  echo "implementation parity: repo=$lane_repository ref=$lane_ref text=$IMPLEMENTATION_TEXT max_tokens=$IMPLEMENTATION_MAX_TOKENS"

  local ran_any=false
  local -a paritychecker_env=()
  if [[ "$IMPLEMENTATION_FAILURE_SURFACE" == "summary" ]]; then
    paritychecker_env=(env EMEL_PARITYCHECKER_FAILURE_SURFACE=summary)
  elif [[ "$IMPLEMENTATION_FAILURE_SURFACE" == "lite" ]]; then
    if [[ "$IMPLEMENTATION_LITE_CASES" != "smoke" && "$IMPLEMENTATION_LITE_CASES" != "full" ]]; then
      echo "error: unsupported EMEL_PARITYCHECKER_IMPLEMENTATION_LITE_CASES '$IMPLEMENTATION_LITE_CASES'" >&2
      exit 1
    fi
    paritychecker_env=(
      env
      EMEL_PARITYCHECKER_FAILURE_SURFACE=lite
      EMEL_PARITYCHECKER_LITE_DEBUG_CASES="$IMPLEMENTATION_LITE_CASES"
    )
  elif [[ "$IMPLEMENTATION_FAILURE_SURFACE" != "full" ]]; then
    echo "error: unsupported EMEL_PARITYCHECKER_IMPLEMENTATION_FAILURE_SURFACE '$IMPLEMENTATION_FAILURE_SURFACE'" >&2
    exit 1
  fi
  while IFS= read -r model_path; do
    if [[ -z "$model_path" ]]; then
      continue
    fi
    ran_any=true
    "${paritychecker_env[@]}" "$build_dir/paritychecker" \
      --generation \
      --model "$model_path" \
      --text "$IMPLEMENTATION_TEXT" \
      --max-tokens "$IMPLEMENTATION_MAX_TOKENS"
  done < <(implementation_models_for_lane "$lane_repository" "$lane_ref")

  if [[ "$ran_any" == false ]]; then
    echo "error: no implementation-mode fixtures resolved for repo=$lane_repository ref=$lane_ref" >&2
    exit 1
  fi
}

if [[ "$MODE" == "implementation" ]]; then
  resolve_implementation_lane "$IMPLEMENTATION_MODEL"
  run_implementation_suite \
    "$ROOT_DIR/build/paritychecker_${TOOLCHAIN}_implementation" \
    "$effective_ref_repository" \
    "$effective_ref_value"
  exit 0
fi

if [[ -n "$explicit_ref_repository" || -n "$explicit_ref_value" ]]; then
  effective_ref_repository="$explicit_ref_repository"
  effective_ref_value="$explicit_ref_value"
  if [[ -z "$effective_ref_repository" ]]; then
    effective_ref_repository="$default_ref_repository"
  fi
  if [[ -z "$effective_ref_value" ]]; then
    effective_ref_value="$default_ref_value"
  fi
  run_lane_suite "$ROOT_DIR/build/paritychecker_${TOOLCHAIN}" \
    "$effective_ref_repository" \
    "$effective_ref_value"
  exit 0
fi

lane_repositories=("$default_ref_repository")
lane_refs=("$default_ref_value")
seen_keys=("${default_ref_repository}|${default_ref_value}")

while IFS='|' read -r _ _ _ _ reference_repository reference_ref _; do
  lane_key="${reference_repository}|${reference_ref}"
  already_seen=false
  for seen_key in "${seen_keys[@]}"; do
    if [[ "$seen_key" == "$lane_key" ]]; then
      already_seen=true
      break
    fi
  done
  if [[ "$already_seen" == false ]]; then
    seen_keys+=("$lane_key")
    lane_repositories+=("$reference_repository")
    lane_refs+=("$reference_ref")
  fi
done < <(fixture_records)

for idx in "${!lane_repositories[@]}"; do
  run_lane_suite \
    "$ROOT_DIR/build/paritychecker_${TOOLCHAIN}_lane_$((idx + 1))" \
    "${lane_repositories[$idx]}" \
    "${lane_refs[$idx]}"
done
