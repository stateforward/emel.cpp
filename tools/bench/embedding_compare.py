#!/usr/bin/env python3

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "embedding_compare/v1"
SUMMARY_SCHEMA = "embedding_compare_summary/v1"


@dataclass(frozen=True)
class command_result:
  returncode: int
  error_kind: str = ""
  error_message: str = ""


def repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--reference-backend")
  parser.add_argument("--backend-manifest", type=Path)
  parser.add_argument("--emel-runner", type=Path, default=repo_root() / "build" / "bench_tools_ninja" / "embedding_generator_bench_runner")
  parser.add_argument("--emel-input", type=Path)
  parser.add_argument("--reference-input", type=Path)
  parser.add_argument("--case-filter", default="")
  parser.add_argument("--variant-id", default="")
  args = parser.parse_args()
  if args.case_filter and args.variant_id:
    parser.error("--variant-id and --case-filter are mutually exclusive")
  return args


def load_manifest(manifest_path: Path) -> dict[str, object]:
  return json.loads(manifest_path.read_text(encoding="utf-8"))


def find_manifest(backend_id: str) -> tuple[Path, dict[str, object]]:
  manifests = sorted((repo_root() / "tools" / "bench" / "reference_backends").glob("*.json"))
  for manifest_path in manifests:
    manifest = load_manifest(manifest_path)
    if manifest.get("id") == backend_id:
      return manifest_path, manifest
  raise FileNotFoundError(f"unknown reference backend: {backend_id}")


def command_to_strings(command: object) -> list[str]:
  if not isinstance(command, list) or not command:
    raise ValueError("manifest command must be a non-empty array")
  return [str(part) for part in command]


def error_record(*,
                 lane: str,
                 backend_id: str,
                 backend_language: str,
                 error_kind: str,
                 error_message: str,
                 note: str = "") -> dict[str, object]:
  return {
    "schema": SCHEMA,
    "record_type": "error",
    "status": "error",
    "case_name": f"{lane}/{backend_id}",
    "compare_group": "backend",
    "lane": lane,
    "backend_id": backend_id,
    "backend_language": backend_language,
    "comparison_mode": "parity",
    "model_id": "",
    "fixture_id": "",
    "modality": "",
    "ns_per_op": 0.0,
    "prepare_ns_per_op": 0.0,
    "encode_ns_per_op": 0.0,
    "publish_ns_per_op": 0.0,
    "output_tokens": 0,
    "output_dim": 0,
    "output_checksum": 0,
    "iterations": 0,
    "runs": 0,
    "output_path": "",
    "note": note,
    "error_kind": error_kind,
    "error_message": error_message,
  }


def append_jsonl_record(path: Path, record: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True))
    handle.write("\n")


def ensure_error_record(path: Path, record: dict[str, object]) -> None:
  records = parse_jsonl_records(path,
                                lane=str(record.get("lane", "")),
                                backend_id=str(record.get("backend_id", "")),
                                backend_language=str(record.get("backend_language", "")))
  if any(existing.get("record_type") == "error" for existing in records):
    return
  append_jsonl_record(path, record)


def run_command(command: list[str],
                env_updates: dict[str, str],
                stdout_path: Path,
                stderr_path: Path) -> command_result:
  env = os.environ.copy()
  env.update(env_updates)
  stdout_path.parent.mkdir(parents=True, exist_ok=True)
  stderr_path.parent.mkdir(parents=True, exist_ok=True)
  try:
    process = subprocess.run(
      command,
      cwd=repo_root(),
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=False,
    )
  except FileNotFoundError as exc:
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text(f"{exc}\n", encoding="utf-8")
    return command_result(returncode=127,
                          error_kind="missing_executable",
                          error_message=str(exc))
  except OSError as exc:
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text(f"{exc}\n", encoding="utf-8")
    return command_result(returncode=126, error_kind="os_error", error_message=str(exc))
  stdout_path.write_text(process.stdout, encoding="utf-8")
  stderr_path.write_text(process.stderr, encoding="utf-8")
  return command_result(returncode=process.returncode)


def parse_jsonl_records(path: Path,
                       *,
                       lane: str = "",
                       backend_id: str = "",
                       backend_language: str = "") -> list[dict[str, object]]:
  records: list[dict[str, object]] = []
  if not path.exists():
    return records
  record_lane = lane or "unknown"
  record_backend_id = backend_id or f"{record_lane}.input"
  for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    line = raw_line.strip()
    if not line or line.startswith("#") or not line.startswith("{"):
      continue
    try:
      payload = json.loads(line)
    except json.JSONDecodeError as exc:
      records.append(
        error_record(
          lane=record_lane,
          backend_id=record_backend_id,
          backend_language=backend_language,
          error_kind="malformed_jsonl",
          error_message=f"{path}:{line_number}: {exc.msg}",
        )
      )
      break
    if payload.get("schema") != SCHEMA:
      continue
    records.append(payload)
  return records


def load_float_vector(path: Path) -> list[float]:
  buffer = array.array("f")
  with path.open("rb") as handle:
    buffer.fromfile(handle, path.stat().st_size // 4)
  return list(buffer)


def cosine(lhs: list[float], rhs: list[float]) -> float:
  numerator = 0.0
  lhs_norm = 0.0
  rhs_norm = 0.0
  for left, right in zip(lhs, rhs):
    numerator += left * right
    lhs_norm += left * left
    rhs_norm += right * right
  if lhs_norm == 0.0 or rhs_norm == 0.0:
    return 0.0
  return numerator / math.sqrt(lhs_norm * rhs_norm)


def l2(lhs: list[float], rhs: list[float]) -> float:
  total = 0.0
  for left, right in zip(lhs, rhs):
    diff = left - right
    total += diff * diff
  return math.sqrt(total)


def max_abs(lhs: list[float], rhs: list[float]) -> float:
  maximum = 0.0
  for left, right in zip(lhs, rhs):
    maximum = max(maximum, abs(left - right))
  return maximum


def select_records(records: list[dict[str, object]],
                   *,
                   lane: str,
                   compare_group: str) -> list[dict[str, object]]:
  selected: list[dict[str, object]] = []
  for record in records:
    if record.get("lane") == lane and record.get("compare_group") == compare_group:
      selected.append(record)
  return selected


def summarize_group(compare_group: str,
                    emel_record: dict[str, object] | None,
                    reference_record: dict[str, object] | None) -> dict[str, object]:
  summary: dict[str, object] = {
    "compare_group": compare_group,
    "comparison_status": "missing",
    "reason": "",
    "emel": emel_record,
    "reference": reference_record,
    "similarity": None,
  }
  if emel_record is not None and emel_record.get("record_type") == "error":
    summary["comparison_status"] = "error"
    summary["reason"] = "emel_lane_error"
    return summary
  if reference_record is not None and reference_record.get("record_type") == "error":
    summary["comparison_status"] = "error"
    summary["reason"] = "reference_lane_error"
    return summary
  if emel_record is None:
    summary["reason"] = "missing_emel_record"
    return summary
  if reference_record is None:
    summary["reason"] = "missing_reference_record"
    return summary
  if emel_record.get("comparison_mode") != "parity" or reference_record.get("comparison_mode") != "parity":
    summary["comparison_status"] = "unavailable"
    summary["reason"] = "non_parity_backend"
    return summary
  emel_output_path = str(emel_record.get("output_path", ""))
  reference_output_path = str(reference_record.get("output_path", ""))
  if not emel_output_path or not reference_output_path:
    summary["comparison_status"] = "unavailable"
    summary["reason"] = "missing_output_vectors"
    return summary
  try:
    lhs = load_float_vector(Path(emel_output_path))
    rhs = load_float_vector(Path(reference_output_path))
  except FileNotFoundError:
    summary["comparison_status"] = "unavailable"
    summary["reason"] = "missing_output_vectors"
    return summary
  if len(lhs) != len(rhs):
    summary["comparison_status"] = "unavailable"
    summary["reason"] = "dimension_mismatch"
    return summary
  summary["comparison_status"] = "computed"
  summary["reason"] = "ok"
  summary["similarity"] = {
    "cosine": cosine(lhs, rhs),
    "l2": l2(lhs, rhs),
    "max_abs": max_abs(lhs, rhs),
  }
  return summary


def summarize_group_matches(compare_group: str,
                            emel_records: list[dict[str, object]],
                            reference_records: list[dict[str, object]]) -> list[dict[str, object]]:
  if not emel_records and not reference_records:
    return []
  if not emel_records:
    return [summarize_group(compare_group, None, reference_record)
            for reference_record in reference_records]
  if not reference_records:
    return [summarize_group(compare_group, emel_record, None) for emel_record in emel_records]
  if len(emel_records) == 1:
    emel_record = emel_records[0]
    return [summarize_group(compare_group, emel_record, reference_record)
            for reference_record in reference_records]
  if len(reference_records) == 1:
    reference_record = reference_records[0]
    return [summarize_group(compare_group, emel_record, reference_record)
            for emel_record in emel_records]

  summaries: list[dict[str, object]] = []
  pair_count = max(len(emel_records), len(reference_records))
  for index in range(pair_count):
    emel_record = emel_records[index] if index < len(emel_records) else None
    reference_record = reference_records[index] if index < len(reference_records) else None
    summaries.append(summarize_group(compare_group, emel_record, reference_record))
  return summaries


def build_summary(emel_records: list[dict[str, object]],
                  reference_records: list[dict[str, object]],
                  manifest: dict[str, object] | None,
                  manifest_path: Path | None) -> dict[str, object]:
  compare_groups = {
    str(record.get("compare_group", ""))
    for record in emel_records + reference_records
    if record.get("compare_group")
  }
  groups = []
  failure = not compare_groups
  failure = failure or any(
    record.get("record_type") == "error" for record in emel_records + reference_records
  )
  for compare_group in sorted(compare_groups):
    group_summaries = summarize_group_matches(
      compare_group,
      select_records(emel_records, lane="emel", compare_group=compare_group),
      select_records(reference_records, lane="reference", compare_group=compare_group),
    )
    for group_summary in group_summaries:
      if group_summary["comparison_status"] in ("error", "missing"):
        failure = True
      groups.append(group_summary)
  return {
    "schema": SUMMARY_SCHEMA,
    "reference_backend": manifest,
    "reference_manifest_path": str(manifest_path) if manifest_path else "",
    "groups": groups,
    "failed": failure,
  }


def print_text_summary(summary: dict[str, object]) -> None:
  backend = summary.get("reference_backend") or {}
  backend_id = backend.get("id", "<input-only>")
  print(f"# reference_backend: {backend_id}")
  for group in summary["groups"]:
    line = f"{group['compare_group']} status={group['comparison_status']} reason={group['reason']}"
    similarity = group.get("similarity")
    if similarity:
      line += (
        f" cosine={similarity['cosine']:.6f}"
        f" l2={similarity['l2']:.6f}"
        f" max_abs={similarity['max_abs']:.6f}"
      )
    print(line)


def main() -> int:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.output_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  manifest_path: Path | None = None
  manifest: dict[str, object] | None = None
  if args.reference_backend:
    manifest_path, manifest = find_manifest(args.reference_backend)
  elif args.backend_manifest:
    manifest_path = args.backend_manifest
    manifest = load_manifest(manifest_path)

  emel_jsonl = args.emel_input or (raw_dir / "emel.jsonl")
  reference_jsonl = args.reference_input or (raw_dir / "reference.jsonl")

  env_updates = {
    "EMEL_EMBEDDING_BENCH_FORMAT": "jsonl",
  }
  if args.variant_id:
    env_updates["EMEL_BENCH_VARIANT_ID"] = args.variant_id
  if args.case_filter:
    env_updates["EMEL_BENCH_CASE_FILTER"] = args.case_filter

  if args.emel_input is None:
    emel_vector_dir = args.output_dir / "vectors" / "emel"
    emel_env = dict(env_updates)
    emel_env["EMEL_EMBEDDING_RESULT_DIR"] = str(emel_vector_dir)
    emel_stderr = raw_dir / "emel.stderr.txt"
    emel_result = run_command([str(args.emel_runner)], emel_env, emel_jsonl, emel_stderr)
    if emel_result.returncode != 0:
      error_kind = emel_result.error_kind or "runner_failed"
      error_message = emel_result.error_message or (
        f"EMEL runner failed with exit code {emel_result.returncode}"
      )
      ensure_error_record(
        emel_jsonl,
        error_record(lane="emel",
                     backend_id="emel.generator",
                     backend_language="cpp",
                     error_kind=error_kind,
                     error_message=error_message),
      )

  if args.reference_input is None:
    if manifest is None:
      print("error: reference backend manifest is required when --reference-input is omitted",
            file=sys.stderr)
      return 1
    reference_jsonl.parent.mkdir(parents=True, exist_ok=True)
    reference_jsonl.write_text("", encoding="utf-8")
    reference_build_failed = False
    if "build_command" in manifest:
      build_stdout = raw_dir / "reference.build.stdout.txt"
      build_stderr = raw_dir / "reference.build.stderr.txt"
      build_result = run_command(command_to_strings(manifest["build_command"]),
                                 dict(env_updates),
                                 build_stdout,
                                 build_stderr)
      if build_result.returncode != 0:
        error_kind = build_result.error_kind or "backend_build_failed"
        error_message = build_result.error_message or (
          f"reference backend build failed with exit code {build_result.returncode}"
        )
        ensure_error_record(
          reference_jsonl,
          error_record(lane="reference",
                       backend_id=str(manifest.get("id", "reference.backend")),
                       backend_language=str(manifest.get("language", "")),
                       error_kind=error_kind,
                       error_message=error_message),
        )
        reference_build_failed = True
    reference_vector_dir = args.output_dir / "vectors" / "reference"
    reference_env = dict(env_updates)
    reference_env["EMEL_EMBEDDING_RESULT_DIR"] = str(reference_vector_dir)
    reference_stderr = raw_dir / "reference.stderr.txt"
    if not reference_build_failed:
      reference_result = run_command(command_to_strings(manifest["run_command"]),
                                     reference_env,
                                     reference_jsonl,
                                     reference_stderr)
      if reference_result.returncode not in (0, 1):
        error_kind = reference_result.error_kind or "backend_run_failed"
        error_message = reference_result.error_message or (
          f"reference backend run failed with exit code {reference_result.returncode}"
        )
        ensure_error_record(
          reference_jsonl,
          error_record(lane="reference",
                       backend_id=str(manifest.get("id", "reference.backend")),
                       backend_language=str(manifest.get("language", "")),
                       error_kind=error_kind,
                       error_message=error_message),
        )

  emel_records = parse_jsonl_records(emel_jsonl,
                                     lane="emel",
                                     backend_id="emel.generator",
                                     backend_language="cpp")
  reference_records = parse_jsonl_records(
    reference_jsonl,
    lane="reference",
    backend_id=str(manifest.get("id", "reference.input")) if manifest else "reference.input",
    backend_language=str(manifest.get("language", "")) if manifest else "",
  )
  summary = build_summary(emel_records, reference_records, manifest, manifest_path)
  summary_path = args.output_dir / "compare_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  print_text_summary(summary)
  return 1 if summary["failed"] else 0


if __name__ == "__main__":
  raise SystemExit(main())
