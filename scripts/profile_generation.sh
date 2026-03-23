#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="${EMEL_PROFILE_DIR:-$ROOT_DIR/tmp/profiles}"
TRACE_TEMPLATE="${EMEL_PROFILE_TEMPLATE:-Time Profiler}"
CASE_INDEX="${EMEL_PROFILE_CASE_INDEX:-7}"
OUT_BASENAME="${EMEL_PROFILE_BASENAME:-generation_$(date +%Y%m%d_%H%M%S)}"
TIME_LIMIT="${EMEL_PROFILE_TIME_LIMIT:-}"
GENERATE_FLAMEGRAPH=true

usage() {
  cat <<'USAGE'
usage: scripts/profile_generation.sh [--case-index N] [--out-basename NAME] [--time-limit DURATION]
                                    [--no-flamegraph]

Profiles the maintained generation benchmark group with macOS xctrace Time Profiler.

Outputs:
  tmp/profiles/<name>.trace
  tmp/profiles/<name>.stdout
  tmp/profiles/<name>_time_profile.xml
  tmp/profiles/<name>_summary.txt

Optional outputs when FlameGraph helpers are available:
  tmp/profiles/<name>.folded
  tmp/profiles/<name>.svg
  tmp/profiles/<name>_bench_only.folded
  tmp/profiles/<name>_bench_only.svg

Environment overrides:
  EMEL_PROFILE_CASE_INDEX
  EMEL_PROFILE_BASENAME
  EMEL_PROFILE_DIR
  EMEL_PROFILE_TEMPLATE
  EMEL_PROFILE_TIME_LIMIT
  FLAMEGRAPH_DIR
  STACKCOLLAPSE_INSTRUMENTS
  FLAMEGRAPH_PL

The default case index is 7, which is the current generation case group in tools/bench.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --case-index=*)
      CASE_INDEX="${arg#*=}"
      ;;
    --out-basename=*)
      OUT_BASENAME="${arg#*=}"
      ;;
    --time-limit=*)
      TIME_LIMIT="${arg#*=}"
      ;;
    --no-flamegraph)
      GENERATE_FLAMEGRAPH=false
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$arg'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_tools() {
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required tool missing: $tool" >&2
      exit 1
    fi
  done
}

resolve_optional_tool() {
  local explicit_path="$1"
  local fallback_name="$2"
  local flamegraph_dir="${FLAMEGRAPH_DIR:-}"

  if [[ -n "$explicit_path" ]]; then
    echo "$explicit_path"
    return 0
  fi
  if command -v "$fallback_name" >/dev/null 2>&1; then
    command -v "$fallback_name"
    return 0
  fi
  if [[ -n "$flamegraph_dir" && -x "$flamegraph_dir/$fallback_name" ]]; then
    echo "$flamegraph_dir/$fallback_name"
    return 0
  fi
  return 1
}

require_tools xctrace python3 rg

mkdir -p "$PROFILE_DIR"

trace_path="$PROFILE_DIR/${OUT_BASENAME}.trace"
stdout_path="$PROFILE_DIR/${OUT_BASENAME}.stdout"
xml_path="$PROFILE_DIR/${OUT_BASENAME}_time_profile.xml"
summary_path="$PROFILE_DIR/${OUT_BASENAME}_summary.txt"
folded_path="$PROFILE_DIR/${OUT_BASENAME}.folded"
svg_path="$PROFILE_DIR/${OUT_BASENAME}.svg"
bench_only_folded_path="$PROFILE_DIR/${OUT_BASENAME}_bench_only.folded"
bench_only_svg_path="$PROFILE_DIR/${OUT_BASENAME}_bench_only.svg"

rm -rf "$trace_path"
rm -f "$stdout_path" "$xml_path" "$summary_path" \
  "$folded_path" "$svg_path" "$bench_only_folded_path" "$bench_only_svg_path"

# Reuse the maintained bench workflow so the profile always targets an up-to-date bench_runner.
EMEL_BENCH_CASE_INDEX="$CASE_INDEX" \
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
  "$ROOT_DIR/scripts/bench.sh" --compare --emel-only >/dev/null

xctrace_args=(
  record
  --template "$TRACE_TEMPLATE"
  --output "$trace_path"
  --target-stdout "$stdout_path"
  --env "EMEL_BENCH_CASE_INDEX=$CASE_INDEX"
  --env "EMEL_BENCH_ITERS=1"
  --env "EMEL_BENCH_RUNS=1"
  --env "EMEL_BENCH_WARMUP_ITERS=0"
  --env "EMEL_BENCH_WARMUP_RUNS=0"
  --launch --
  "$ROOT_DIR/build/bench_tools_ninja/bench_runner"
  --mode=emel
)

if [[ -n "$TIME_LIMIT" ]]; then
  xctrace_args=(record --template "$TRACE_TEMPLATE" --time-limit "$TIME_LIMIT" \
    --output "$trace_path" --target-stdout "$stdout_path" \
    --env "EMEL_BENCH_CASE_INDEX=$CASE_INDEX" \
    --env "EMEL_BENCH_ITERS=1" \
    --env "EMEL_BENCH_RUNS=1" \
    --env "EMEL_BENCH_WARMUP_ITERS=0" \
    --env "EMEL_BENCH_WARMUP_RUNS=0" \
    --launch -- "$ROOT_DIR/build/bench_tools_ninja/bench_runner" --mode=emel)
fi

xctrace "${xctrace_args[@]}"

xctrace export \
  --input "$trace_path" \
  --xpath '//trace-toc/run/data/table[@schema="time-profile"]' \
  --output "$xml_path" >/dev/null

python3 - "$xml_path" "$summary_path" <<'PY'
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

xml_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
root = ET.parse(xml_path).getroot()

inclusive = Counter()
leaf = Counter()
bench_inclusive = Counter()
bench_leaf = Counter()

def is_bench_frame(name: str) -> bool:
    return (
        "emel::" in name
        or "boost::ext::sml::" in name
        or "llama" in name
        or "ggml" in name
        or "bench_runner" in name
    )

rows = root.findall(".//row")
for row in rows:
    frames = [f.attrib.get("name", "") for f in row.findall(".//backtrace/frame")]
    frames = [name for name in frames if name]
    if not frames:
        continue

    leaf[frames[0]] += 1
    for name in dict.fromkeys(frames):
      inclusive[name] += 1

    bench_frames = [name for name in frames if is_bench_frame(name)]
    if not bench_frames:
        continue
    bench_leaf[bench_frames[0]] += 1
    for name in dict.fromkeys(bench_frames):
      bench_inclusive[name] += 1

lines = [f"total_samples {len(rows)}", "top_inclusive_samples"]
for name, count in inclusive.most_common(20):
    lines.append(f"{count} {name}")
lines.append("top_leaf_samples")
for name, count in leaf.most_common(20):
    lines.append(f"{count} {name}")
lines.append("top_bench_inclusive_samples")
for name, count in bench_inclusive.most_common(20):
    lines.append(f"{count} {name}")
lines.append("top_bench_leaf_samples")
for name, count in bench_leaf.most_common(20):
    lines.append(f"{count} {name}")

summary_path.write_text("\n".join(lines) + "\n")
PY

if ! rg -q '^generation/' "$stdout_path"; then
  echo "error: expected generation benchmark rows in $stdout_path" >&2
  exit 1
fi

flamegraph_note="flamegraph skipped"
if $GENERATE_FLAMEGRAPH; then
  stackcollapse_tool="$(resolve_optional_tool "${STACKCOLLAPSE_INSTRUMENTS:-}" \
    stackcollapse-instruments.pl || true)"
  flamegraph_tool="$(resolve_optional_tool "${FLAMEGRAPH_PL:-}" flamegraph.pl || true)"
  if [[ -n "$stackcollapse_tool" && -n "$flamegraph_tool" ]]; then
    "$stackcollapse_tool" "$xml_path" > "$folded_path"
    "$flamegraph_tool" "$folded_path" > "$svg_path"
    rg 'emel::|boost::ext::sml::|llama|ggml|bench_runner' "$folded_path" \
      > "$bench_only_folded_path" || true
    if [[ -s "$bench_only_folded_path" ]]; then
      "$flamegraph_tool" "$bench_only_folded_path" > "$bench_only_svg_path"
    fi
    flamegraph_note="flamegraph generated"
  fi
fi

printf 'trace: %s\n' "$trace_path"
printf 'stdout: %s\n' "$stdout_path"
printf 'time_profile_xml: %s\n' "$xml_path"
printf 'summary: %s\n' "$summary_path"
printf 'flamegraph: %s\n' "$flamegraph_note"
if [[ -f "$svg_path" ]]; then
  printf 'svg: %s\n' "$svg_path"
fi
if [[ -f "$bench_only_svg_path" ]]; then
  printf 'bench_only_svg: %s\n' "$bench_only_svg_path"
fi

printf '\ngeneration_rows\n'
sed -n '1,20p' "$stdout_path"
