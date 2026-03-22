#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REQUIRED_BASELINE_FIELDS = (
    "source_commit",
    "baseline_ref",
    "case",
    "baseline_emel_ns",
    "baseline_reference_ns",
    "baseline_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a preserved pre-flash baseline artifact against the current compare snapshot."
    )
    parser.add_argument("--baseline", required=True, help="Path to the preserved baseline artifact.")
    parser.add_argument("--current", required=True, help="Path to the current compare snapshot.")
    parser.add_argument("--case", required=True, help="Canonical compare case name.")
    return parser.parse_args()


def read_text(path_str: str) -> str:
    path = Path(path_str)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"missing file: {path}")


def parse_baseline(path_str: str, expected_case: str) -> float:
    fields: dict[str, str] = {}
    for raw_line in read_text(path_str).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator != "=" or not key:
            raise SystemExit(f"invalid baseline line: {raw_line}")
        fields[key] = value

    missing = [field for field in REQUIRED_BASELINE_FIELDS if field not in fields]
    if missing:
        raise SystemExit(f"missing baseline fields: {', '.join(missing)}")
    if fields["case"] != expected_case:
        raise SystemExit(
            f"baseline case mismatch: expected {expected_case}, found {fields['case']}"
        )

    try:
        return float(fields["baseline_emel_ns"])
    except ValueError as exc:
        raise SystemExit(f"invalid baseline_emel_ns: {fields['baseline_emel_ns']}") from exc


def parse_current(path_str: str, expected_case: str) -> float:
    row_pattern = re.compile(
        rf"^{re.escape(expected_case)} emel\.cpp ([0-9]+(?:\.[0-9]+)?) ns/op, "
        r"llama\.cpp ([0-9]+(?:\.[0-9]+)?) ns/op, ratio=([0-9]+(?:\.[0-9]+)?)x$"
    )
    for raw_line in read_text(path_str).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = row_pattern.match(line)
        if match is None:
            continue
        return float(match.group(1))
    raise SystemExit(f"missing current compare row for case: {expected_case}")


def main() -> int:
    args = parse_args()
    baseline_emel_ns = parse_baseline(args.baseline, args.case)
    current_emel_ns = parse_current(args.current, args.case)

    if current_emel_ns >= baseline_emel_ns:
        raise SystemExit(
            "current_emel_ns must be lower than baseline_emel_ns for flash improvement proof"
        )

    speedup = baseline_emel_ns / current_emel_ns
    latency_drop_pct = ((baseline_emel_ns - current_emel_ns) / baseline_emel_ns) * 100.0

    print(
        f"case={args.case} "
        f"baseline_emel_ns={baseline_emel_ns:.3f} "
        f"current_emel_ns={current_emel_ns:.3f} "
        f"speedup={speedup:.3f}x "
        f"latency_drop_pct={latency_drop_pct:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
