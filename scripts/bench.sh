#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools/bench"

SNAPSHOT=false
COMPARE=false
UPDATE=false

usage() {
  cat <<'USAGE'
usage: scripts/bench.sh [--snapshot] [--compare] [--update]

  --snapshot   run EMEL benchmark snapshot gate
  --compare    build and run reference comparison
  --update     update snapshot baseline (requires --snapshot)
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --snapshot) SNAPSHOT=true ;;
    --compare) COMPARE=true ;;
    --update) UPDATE=true ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      usage
      exit 1
      ;;
  esac
done

if ! $SNAPSHOT && ! $COMPARE; then
  usage
  exit 1
fi

if $UPDATE && ! $SNAPSHOT; then
  echo "error: --update requires --snapshot" >&2
  exit 1
fi

if $SNAPSHOT; then
  TOLERANCE="${BENCH_TOLERANCE:-0.05}"
  BASELINE="$ROOT_DIR/snapshots/bench/benchmarks.txt"
  CURRENT="$(mktemp)"
  trap 'rm -f "$CURRENT"' EXIT

  for tool in zig cmake ninja git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done

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
    marker="$(grep -E "benchmark: (scaffold|ready)" "$ROOT_DIR/$sm" | head -n 1 || true)"
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

  zig_bin="$(command -v zig)"
  build_dir="${BENCH_BUILD_DIR:-$ROOT_DIR/build/bench}"

  cmake -S "$ROOT_DIR" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$zig_bin" \
    -DCMAKE_C_COMPILER_ARG1=cc \
    -DCMAKE_CXX_COMPILER="$zig_bin" \
    -DCMAKE_CXX_COMPILER_ARG1=c++ \
    -DEMEL_ENABLE_TESTS=ON

  cmake --build "$build_dir" --parallel --target emel_bench_bin

  "$build_dir/emel_bench_bin" > "$CURRENT"

  for name in "${ready_names[@]+${ready_names[@]}}"; do
    if ! grep -q "^${name} " "$CURRENT"; then
      echo "error: missing benchmark entry for $name" >&2
      exit 1
    fi
  done

  if $UPDATE; then
    mkdir -p "$(dirname "$BASELINE")"
    cp "$CURRENT" "$BASELINE"
    echo "updated $BASELINE"
  else
    if [[ ! -f "$BASELINE" ]]; then
      echo "error: missing baseline $BASELINE (run scripts/bench.sh --snapshot --update)" >&2
      exit 1
    fi

    awk -v tol="$TOLERANCE" '
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
      parse_base($0);
      next;
    }
    {
      parse_curr($0);
      next;
    }
    END {
      fail = 0;
      for (name in base) {
        if (!(name in curr)) {
          print "error: missing benchmark entry for " name > "/dev/stderr";
          fail = 1;
          continue;
        }
        limit = base[name] * (1 + tol);
        if (curr[name] > limit) {
          printf("error: benchmark regression %s (%.3f > %.3f)\n", name, curr[name], limit) > "/dev/stderr";
          fail = 1;
        }
      }
      for (name in curr) {
        if (!(name in base)) {
          print "error: new benchmark entry without baseline: " name > "/dev/stderr";
          fail = 1;
        }
      }
      exit fail;
    }
    ' "$BASELINE" "$CURRENT"
  fi
fi

if $COMPARE; then
  for tool in cmake; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done

  compare_build_dir="${BENCH_COMPARE_BUILD_DIR:-$ROOT_DIR/build/tools_bench}"

  cmake -S "$TOOLS_DIR" -B "$compare_build_dir" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$compare_build_dir" --target bench_compare
fi
