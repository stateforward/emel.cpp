#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <emel_bench_bin> <reference_bench_bin>" >&2
  exit 1
fi

EMEL_BIN="$1"
REF_BIN="$2"

EMEL_OUT="$(mktemp)"
REF_OUT="$(mktemp)"
trap 'rm -f "$EMEL_OUT" "$REF_OUT"' EXIT

"$EMEL_BIN" > "$EMEL_OUT"
"$REF_BIN" > "$REF_OUT"

extract_ns() {
  awk -v name="$1" '
  $1 == name {
    for (i = 2; i <= NF; ++i) {
      if ($i ~ /^ns_per_op=/) {
        split($i, pair, "=");
        print pair[2];
        exit 0;
      }
    }
  }
  END { exit 1; }
  ' "$2"
}

cases=("buffer/allocator_reserve_n" "buffer/allocator_alloc_graph" "buffer/allocator_full")

for case_name in "${cases[@]}"; do
  emel_ns="$(extract_ns "$case_name" "$EMEL_OUT" || true)"
  ref_ns="$(extract_ns "$case_name" "$REF_OUT" || true)"

  if [[ -z "$emel_ns" ]]; then
    echo "error: missing $case_name in emel output" >&2
    exit 1
  fi
  if [[ -z "$ref_ns" ]]; then
    echo "error: missing $case_name in reference output" >&2
    exit 1
  fi

  ratio=$(awk -v emel="$emel_ns" -v ref="$ref_ns" 'BEGIN { printf("%.3f", emel / ref); }')

  printf "%s emel_ns_per_op=%s ref_ns_per_op=%s ratio=%sx\n" \
    "$case_name" "$emel_ns" "$ref_ns" "$ratio"

done
