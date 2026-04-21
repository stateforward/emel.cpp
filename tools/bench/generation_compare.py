#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "generation_compare/v1"
SUMMARY_SCHEMA = "generation_compare_summary/v1"

SUMMARY_METADATA_FIELDS = (
  "workload_id",
  "workload_manifest_path",
  "model_id",
  "fixture_id",
  "prompt_fixture_id",
  "prompt_fixture_path",
  "prompt_id",
  "formatter_mode",
  "formatter_contract",
  "sampling_id",
  "stop_id",
  "seed",
  "max_output_tokens",
)

SUMMARY_MISMATCH_REASONS = {
  "workload_manifest_path": "workload_manifest_mismatch",
}


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
  parser.add_argument("--emel-runner",
                      type=Path,
                      default=repo_root() / "build" / "bench_tools_ninja" / "bench_runner")
  parser.add_argument("--emel-input", type=Path)
  parser.add_argument("--reference-input", type=Path)
  parser.add_argument("--workload-id", default="")
  return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict[str, object]:
  return json.loads(manifest_path.read_text(encoding="utf-8"))


def find_workload_manifest(workload_id: str) -> dict[str, object] | None:
  if not workload_id:
    return None
  manifests = sorted((repo_root() / "tools" / "bench" / "generation_workloads").glob("*.json"))
  for manifest_path in manifests:
    manifest = load_manifest(manifest_path)
    if manifest.get("id") == workload_id:
      return manifest
  return None


def is_single_lane_workload(workload_manifest: dict[str, object] | None) -> bool:
  if workload_manifest is None:
    return False
  return (
    workload_manifest.get("comparison_mode") != "parity" or
    not bool(workload_manifest.get("comparable", False))
  )


def manifest_surface(manifest: dict[str, object]) -> str:
  return str(manifest.get("surface", ""))


def find_manifest(backend_id: str) -> tuple[Path, dict[str, object]]:
  manifests = sorted((repo_root() / "tools" / "bench" / "reference_backends").glob("*.json"))
  for manifest_path in manifests:
    manifest = load_manifest(manifest_path)
    if manifest.get("id") != backend_id:
      continue
    if manifest_surface(manifest) != SCHEMA:
      continue
    return manifest_path, manifest
  raise FileNotFoundError(f"unknown generation reference backend: {backend_id}")


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
    "workload_id": "",
    "workload_manifest_path": "",
    "comparison_mode": "parity",
    "model_id": "",
    "fixture_id": "",
    "prompt_fixture_id": "",
    "prompt_fixture_path": "",
    "prompt_id": "",
    "formatter_mode": "",
    "formatter_contract": "",
    "sampling_id": "",
    "stop_id": "",
    "seed": 0,
    "comparable": False,
    "max_output_tokens": 0,
    "ns_per_op": 0.0,
    "prepare_ns_per_op": 0.0,
    "encode_ns_per_op": 0.0,
    "publish_ns_per_op": 0.0,
    "output_tokens": 0,
    "output_bytes": 0,
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


def read_generation_output_text(record: dict[str, object]) -> str:
  output_path = str(record.get("output_path", ""))
  if not output_path:
    return ""
  try:
    return Path(output_path).read_text(encoding="utf-8")
  except FileNotFoundError:
    return ""


def copy_summary_metadata(summary: dict[str, object], record: dict[str, object] | None) -> None:
  if record is None:
    return
  for field in SUMMARY_METADATA_FIELDS:
    summary[field] = record.get(field, "")


def mismatch_reason(emel_record: dict[str, object], reference_record: dict[str, object]) -> str:
  for field in SUMMARY_METADATA_FIELDS:
    if str(emel_record.get(field, "")) != str(reference_record.get(field, "")):
      return SUMMARY_MISMATCH_REASONS.get(field, f"{field}_mismatch")
  return ""


def compare_prefix_bytes(lhs: str, rhs: str) -> int:
  lhs_bytes = lhs.encode("utf-8")
  rhs_bytes = rhs.encode("utf-8")
  prefix = 0
  limit = min(len(lhs_bytes), len(rhs_bytes))
  while prefix < limit and lhs_bytes[prefix] == rhs_bytes[prefix]:
    prefix += 1
  return prefix


def select_records(records: list[dict[str, object]], *, lane: str) -> list[dict[str, object]]:
  selected: list[dict[str, object]] = []
  for record in records:
    if record.get("lane") != lane:
      continue
    selected.append(record)
  return selected


def summarize_group(compare_group: str,
                    emel_record: dict[str, object] | None,
                    reference_record: dict[str, object] | None) -> dict[str, object]:
  summary: dict[str, object] = {
    "compare_group": compare_group,
    "comparison_status": "missing",
    "reason": "",
    "workload_id": "",
    "workload_manifest_path": "",
    "model_id": "",
    "fixture_id": "",
    "prompt_fixture_id": "",
    "prompt_fixture_path": "",
    "prompt_id": "",
    "formatter_mode": "",
    "formatter_contract": "",
    "sampling_id": "",
    "stop_id": "",
    "seed": "",
    "max_output_tokens": "",
    "exact_output_match": False,
    "exact_checksum_match": False,
    "shared_prefix_bytes": 0,
    "shared_prefix_fraction": 0.0,
    "output_tokens_delta": None,
    "output_bytes_delta": None,
    "emel": emel_record,
    "reference": reference_record,
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
    copy_summary_metadata(summary, reference_record)
    if reference_record is not None and (
        reference_record.get("comparison_mode") != "parity" or
        not bool(reference_record.get("comparable", False))):
      summary["comparison_status"] = "non_comparable"
      summary["reason"] = "single_lane_reference_workload"
      return summary
    summary["reason"] = "missing_emel_record"
    return summary
  if reference_record is None:
    copy_summary_metadata(summary, emel_record)
    if (emel_record.get("comparison_mode") != "parity" or
        not bool(emel_record.get("comparable", False))):
      summary["comparison_status"] = "non_comparable"
      summary["reason"] = "single_lane_emel_workload"
      return summary
    summary["reason"] = "missing_reference_record"
    return summary

  copy_summary_metadata(summary, emel_record)

  if (emel_record.get("comparison_mode") != "parity" or
      reference_record.get("comparison_mode") != "parity" or
      not bool(emel_record.get("comparable", False)) or
      not bool(reference_record.get("comparable", False))):
    summary["comparison_status"] = "non_comparable"
    summary["reason"] = "non_parity_workload"
    return summary

  reason = mismatch_reason(emel_record, reference_record)
  if reason:
    summary["comparison_status"] = "non_comparable"
    summary["reason"] = reason
    return summary

  emel_text = read_generation_output_text(emel_record)
  reference_text = read_generation_output_text(reference_record)
  emel_output_bytes = int(emel_record.get("output_bytes", 0))
  reference_output_bytes = int(reference_record.get("output_bytes", 0))
  both_empty_outputs = emel_output_bytes == 0 and reference_output_bytes == 0
  exact_checksum_match = (
    int(emel_record.get("output_checksum", 0)) == int(reference_record.get("output_checksum", 0)) and
    int(emel_record.get("output_checksum", 0)) != 0
  )
  exact_output_match = (both_empty_outputs or bool(emel_text) or bool(reference_text)) and (
    emel_text == reference_text
  )
  shared_prefix_bytes = compare_prefix_bytes(emel_text, reference_text) if emel_text or reference_text else 0
  longest_output = max(len(emel_text.encode("utf-8")), len(reference_text.encode("utf-8")))
  shared_prefix_fraction = 0.0
  if longest_output > 0:
    shared_prefix_fraction = shared_prefix_bytes / longest_output

  summary["exact_output_match"] = exact_output_match
  summary["exact_checksum_match"] = exact_checksum_match
  summary["shared_prefix_bytes"] = shared_prefix_bytes
  summary["shared_prefix_fraction"] = shared_prefix_fraction
  summary["output_tokens_delta"] = abs(
    int(emel_record.get("output_tokens", 0)) - int(reference_record.get("output_tokens", 0)))
  summary["output_bytes_delta"] = abs(
    int(emel_record.get("output_bytes", 0)) - int(reference_record.get("output_bytes", 0)))

  if exact_output_match or exact_checksum_match:
    summary["comparison_status"] = "exact_match"
    summary["reason"] = "ok"
    return summary

  summary["comparison_status"] = "bounded_drift"
  summary["reason"] = "output_mismatch"
  return summary


def build_summary(emel_records: list[dict[str, object]],
                  reference_records: list[dict[str, object]],
                  manifest: dict[str, object] | None,
                  manifest_path: Path | None) -> dict[str, object]:
  emel_records = select_records(emel_records, lane="emel")
  reference_records = select_records(reference_records, lane="reference")
  compare_groups = {
    str(record.get("compare_group", ""))
    for record in emel_records + reference_records
    if record.get("compare_group")
  }
  groups = []
  failed = not compare_groups
  emel_by_group = {str(record.get("compare_group", "")): record for record in emel_records}
  reference_by_group = {
    str(record.get("compare_group", "")): record for record in reference_records
  }
  for compare_group in sorted(compare_groups):
    group_summary = summarize_group(compare_group,
                                    emel_by_group.get(compare_group),
                                    reference_by_group.get(compare_group))
    if group_summary["comparison_status"] in ("error", "missing"):
      failed = True
    groups.append(group_summary)
  return {
    "schema": SUMMARY_SCHEMA,
    "reference_backend": manifest,
    "reference_manifest_path": str(manifest_path) if manifest_path else "",
    "groups": groups,
    "failed": failed,
  }


def print_text_summary(summary: dict[str, object]) -> None:
  backend = summary.get("reference_backend") or {}
  backend_id = backend.get("id", "<input-only>")
  print(f"# reference_backend: {backend_id}")
  for group in summary["groups"]:
    line = f"{group['compare_group']} status={group['comparison_status']} reason={group['reason']}"
    if group["comparison_status"] in ("exact_match", "bounded_drift"):
      line += (
        f" exact_output_match={str(group['exact_output_match']).lower()}"
        f" exact_checksum_match={str(group['exact_checksum_match']).lower()}"
        f" prefix_bytes={group['shared_prefix_bytes']}"
        f" prefix_fraction={group['shared_prefix_fraction']:.6f}"
        f" output_tokens_delta={group['output_tokens_delta']}"
        f" output_bytes_delta={group['output_bytes_delta']}"
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
    if manifest_surface(manifest) not in ("", SCHEMA):
      print("error: backend manifest surface is not generation_compare/v1", file=sys.stderr)
      return 1

  emel_jsonl = args.emel_input or (raw_dir / "emel.jsonl")
  reference_jsonl = args.reference_input or (raw_dir / "reference.jsonl")
  selected_workload_manifest = find_workload_manifest(args.workload_id)
  skip_reference_lane = is_single_lane_workload(selected_workload_manifest)

  env_updates = {
    "EMEL_GENERATION_BENCH_FORMAT": "jsonl",
    "EMEL_BENCH_SUITE": "generation",
  }
  if args.workload_id:
    env_updates["EMEL_GENERATION_WORKLOAD_ID"] = args.workload_id

  if args.emel_input is None:
    emel_output_dir = args.output_dir / "outputs" / "emel"
    emel_env = dict(env_updates)
    emel_env["EMEL_GENERATION_RESULT_DIR"] = str(emel_output_dir)
    emel_stderr = raw_dir / "emel.stderr.txt"
    emel_result = run_command([str(args.emel_runner), "--mode=emel"],
                              emel_env,
                              emel_jsonl,
                              emel_stderr)
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
    if skip_reference_lane:
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
    reference_output_dir = args.output_dir / "outputs" / "reference"
    reference_env = dict(env_updates)
    reference_env["EMEL_GENERATION_RESULT_DIR"] = str(reference_output_dir)
    reference_stderr = raw_dir / "reference.stderr.txt"
    if not reference_build_failed:
      reference_result = run_command(command_to_strings(manifest["run_command"]),
                                     reference_env,
                                     reference_jsonl,
                                     reference_stderr)
      if reference_result.returncode != 0:
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
