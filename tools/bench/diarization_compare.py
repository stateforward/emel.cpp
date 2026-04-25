#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "diarization_compare/v1"
SUMMARY_SCHEMA = "diarization_compare_summary/v1"

SUMMARY_METADATA_FIELDS = (
  "comparison_mode",
  "model_id",
  "fixture_id",
  "workload_id",
)
REFERENCE_ROLE_FIELD = "reference_role"

SORTFORMER_COMPARE_GROUP = "diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono"
SORTFORMER_MODEL_ID = "diar_streaming_sortformer_4spk_v2_1_gguf"
SORTFORMER_FIXTURE_ID = "ami_en2002b_mix_headset_137.00_152.04_16khz_mono"
SORTFORMER_WORKLOAD_ID = "diarization_sortformer_pipeline_v1"
DEFAULT_BENCHMARK_ITERATIONS = 1
DEFAULT_BENCHMARK_RUNS = 3
DEFAULT_BENCHMARK_WARMUP_ITERATIONS = 1
DEFAULT_BENCHMARK_WARMUP_RUNS = 1
SORTFORMER_CHUNK_LEN = 188
SORTFORMER_CHUNK_RIGHT_CONTEXT = 1
SORTFORMER_FIFO_LEN = 0
SORTFORMER_SPKCACHE_UPDATE_PERIOD = 188
SORTFORMER_SPKCACHE_LEN = 188
SORTFORMER_ONNX_OUTPUT_CONTRACT = "probabilities"


@dataclass(frozen=True)
class command_result:
  returncode: int
  error_kind: str = ""
  error_message: str = ""


def repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def positive_int(value: str) -> int:
  parsed = int(value, 10)
  if parsed < 0:
    raise argparse.ArgumentTypeError("value must be non-negative")
  return parsed


def env_int(name: str, default: int) -> int:
  raw = os.environ.get(name, "")
  if not raw:
    return default
  return positive_int(raw)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--emel-runner",
                      type=Path,
                      default=repo_root() / "build" / "bench_tools_ninja" / "bench_runner")
  parser.add_argument("--onnx-reference-runner",
                      type=Path,
                      default=repo_root() / "tools" / "bench" /
                      "diarization_sortformer_onnx_reference.py")
  parser.add_argument("--pytorch-reference-runner",
                      type=Path,
                      default=repo_root() / "tools" / "bench" /
                      "diarization_sortformer_pytorch_reference.py")
  parser.add_argument("--pytorch-reference-python", type=Path, default=Path(sys.executable))
  parser.add_argument("--onnx-reference-model", type=Path)
  parser.add_argument("--onnx-reference-features", type=Path)
  parser.add_argument("--onnx-reference-expected-sha256", default="")
  parser.add_argument("--pytorch-reference-model", default="")
  parser.add_argument("--pytorch-reference-audio",
                      type=Path,
                      default=repo_root() / "tests" / "fixtures" / "diarization" /
                      "ami_en2002b_mix_headset_137.00_152.04_16khz_mono.wav")
  parser.add_argument("--pytorch-reference-device", default="cpu")
  parser.add_argument("--emel-input", type=Path)
  parser.add_argument("--reference-input", type=Path)
  parser.add_argument("--onnx-reference-input", type=Path)
  parser.add_argument("--pytorch-reference-input", type=Path)
  parser.add_argument("--benchmark-iterations",
                      type=positive_int,
                      default=env_int("EMEL_DIARIZATION_COMPARE_ITERS",
                                      DEFAULT_BENCHMARK_ITERATIONS))
  parser.add_argument("--benchmark-runs",
                      type=positive_int,
                      default=env_int("EMEL_DIARIZATION_COMPARE_RUNS",
                                      DEFAULT_BENCHMARK_RUNS))
  parser.add_argument("--benchmark-warmup-iterations",
                      type=positive_int,
                      default=env_int("EMEL_DIARIZATION_COMPARE_WARMUP_ITERS",
                                      DEFAULT_BENCHMARK_WARMUP_ITERATIONS))
  parser.add_argument("--benchmark-warmup-runs",
                      type=positive_int,
                      default=env_int("EMEL_DIARIZATION_COMPARE_WARMUP_RUNS",
                                      DEFAULT_BENCHMARK_WARMUP_RUNS))
  return parser.parse_args()


def error_record(*,
                 lane: str,
                 backend_id: str,
                 backend_language: str,
                 error_kind: str,
                 error_message: str,
                 compare_group: str = "",
                 model_id: str = "",
                 fixture_id: str = "",
                 workload_id: str = "",
                 reference_role: str = "",
                 note: str = "") -> dict[str, object]:
  return {
    "schema": SCHEMA,
    "record_type": "error",
    "status": "error",
    "case_name": f"{lane}/{backend_id}",
    "compare_group": compare_group or "backend",
    "lane": lane,
    "backend_id": backend_id,
    "backend_language": backend_language,
    "comparison_mode": "parity",
    "reference_role": reference_role,
    "model_id": model_id,
    "fixture_id": fixture_id,
    "workload_id": workload_id,
    "comparable": False,
    "ns_per_op": 0.0,
    "ns_min_per_op": 0.0,
    "ns_mean_per_op": 0.0,
    "ns_max_per_op": 0.0,
    "prepare_ns_per_op": 0.0,
    "encode_ns_per_op": 0.0,
    "publish_ns_per_op": 0.0,
    "output_bytes": 0,
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


def read_output_text(record: dict[str, object]) -> str:
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
      return f"{field}_mismatch"
  return ""


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
  reference_backend_id = ""
  reference_role = ""
  if reference_record is not None:
    reference_backend_id = str(reference_record.get("backend_id", ""))
    reference_role = str(reference_record.get(REFERENCE_ROLE_FIELD, ""))
  summary: dict[str, object] = {
    "compare_group": compare_group,
    "reference_backend_id": reference_backend_id,
    "reference_role": reference_role,
    "comparison_status": "missing",
    "reason": "",
    "comparison_mode": "",
    "model_id": "",
    "fixture_id": "",
    "workload_id": "",
    "exact_output_match": False,
    "exact_checksum_match": False,
    "exact_output_dim_match": False,
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
    summary["reason"] = "missing_emel_record"
    return summary
  if reference_record is None:
    copy_summary_metadata(summary, emel_record)
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

  emel_output_text = read_output_text(emel_record)
  reference_output_text = read_output_text(reference_record)
  exact_output_match = bool(emel_output_text or reference_output_text) and (
    emel_output_text == reference_output_text
  )
  exact_checksum_match = (
    int(emel_record.get("output_checksum", 0)) == int(reference_record.get("output_checksum", 0))
    and int(emel_record.get("output_checksum", 0)) != 0
  )
  exact_output_dim_match = (
    int(emel_record.get("output_dim", 0)) == int(reference_record.get("output_dim", 0))
    and int(emel_record.get("output_dim", 0)) != 0
  )

  summary["exact_output_match"] = exact_output_match
  summary["exact_checksum_match"] = exact_checksum_match
  summary["exact_output_dim_match"] = exact_output_dim_match
  summary["output_bytes_delta"] = abs(
    int(emel_record.get("output_bytes", 0)) - int(reference_record.get("output_bytes", 0)))

  if exact_checksum_match and exact_output_dim_match:
    summary["comparison_status"] = "exact_match"
    summary["reason"] = "ok"
    return summary

  summary["comparison_status"] = "bounded_drift"
  summary["reason"] = "output_mismatch"
  return summary


def build_summary(emel_records: list[dict[str, object]],
                  reference_records: list[dict[str, object]]) -> dict[str, object]:
  emel_records = select_records(emel_records, lane="emel")
  reference_records = select_records(reference_records, lane="reference")
  emel_records = [
    record for record in emel_records
    if record.get("record_type") == "error" or record.get("comparison_mode") == "parity"
  ]
  reference_records = [
    record for record in reference_records
    if record.get("record_type") == "error" or record.get("comparison_mode") == "parity"
  ]
  compare_groups = {
    str(record.get("compare_group", ""))
    for record in emel_records + reference_records
    if record.get("compare_group")
  }
  groups = []
  failed = not compare_groups
  emel_by_group = {str(record.get("compare_group", "")): record for record in emel_records}
  reference_by_group: dict[str, list[dict[str, object]]] = {}
  for record in reference_records:
    compare_group = str(record.get("compare_group", ""))
    if not compare_group:
      continue
    reference_by_group.setdefault(compare_group, []).append(record)
  for compare_group in sorted(compare_groups):
    references = reference_by_group.get(compare_group, [])
    if not references:
      group_summary = summarize_group(compare_group, emel_by_group.get(compare_group), None)
      if group_summary["comparison_status"] in ("error", "missing"):
        failed = True
      groups.append(group_summary)
      continue
    for reference_record in sorted(references, key=lambda record: str(record.get("backend_id", ""))):
      group_summary = summarize_group(compare_group,
                                      emel_by_group.get(compare_group),
                                      reference_record)
      if group_summary["comparison_status"] in ("error", "missing"):
        failed = True
      if (group_summary.get("reference_role") == "parity_reference" and
          group_summary["comparison_status"] != "exact_match"):
        failed = True
      groups.append(group_summary)
  return {
    "schema": SUMMARY_SCHEMA,
    "groups": groups,
    "failed": failed,
  }


def print_text_summary(summary: dict[str, object]) -> None:
  for group in summary["groups"]:
    line = f"{group['compare_group']} status={group['comparison_status']} reason={group['reason']}"
    if group["comparison_status"] in ("exact_match", "bounded_drift"):
      line += (
        f" exact_output_match={str(group['exact_output_match']).lower()}"
        f" exact_checksum_match={str(group['exact_checksum_match']).lower()}"
        f" exact_output_dim_match={str(group['exact_output_dim_match']).lower()}"
        f" output_bytes_delta={group['output_bytes_delta']}"
      )
    reference_backend_id = str(group.get("reference_backend_id", ""))
    if reference_backend_id:
      line += f" reference_backend={reference_backend_id}"
    reference_role = str(group.get("reference_role", ""))
    if reference_role:
      line += f" reference_role={reference_role}"
    print(line)


def main() -> int:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.output_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)

  emel_jsonl = args.emel_input or (raw_dir / "emel.jsonl")
  reference_jsonl = args.reference_input or (raw_dir / "reference.jsonl")
  onnx_reference_jsonl = args.onnx_reference_input or (raw_dir / "onnx_reference.jsonl")
  pytorch_reference_jsonl = args.pytorch_reference_input or (raw_dir / "pytorch_reference.jsonl")
  onnx_reference_features = args.onnx_reference_features or (raw_dir / "onnx_features.f32")

  env_updates = {
    "EMEL_DIARIZATION_BENCH_FORMAT": "jsonl",
    "EMEL_BENCH_SUITE": "diarization_sortformer",
    "EMEL_BENCH_ITERS": str(max(1, args.benchmark_iterations)),
    "EMEL_BENCH_RUNS": str(max(1, args.benchmark_runs)),
    "EMEL_BENCH_WARMUP_ITERS": str(args.benchmark_warmup_iterations),
    "EMEL_BENCH_WARMUP_RUNS": str(args.benchmark_warmup_runs),
  }

  if args.emel_input is None:
    emel_output_dir = args.output_dir / "outputs" / "emel"
    emel_env = dict(env_updates)
    emel_env["EMEL_DIARIZATION_RESULT_DIR"] = str(emel_output_dir)
    if args.onnx_reference_model is not None and args.onnx_reference_features is None:
      emel_env["EMEL_DIARIZATION_ONNX_FEATURE_INPUT"] = str(onnx_reference_features)
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
                     backend_id="emel.diarization.sortformer",
                     backend_language="cpp",
                     error_kind=error_kind,
                     error_message=error_message,
                     compare_group=SORTFORMER_COMPARE_GROUP,
                     model_id=SORTFORMER_MODEL_ID,
                     fixture_id=SORTFORMER_FIXTURE_ID,
                     workload_id=SORTFORMER_WORKLOAD_ID),
      )

  if args.reference_input is None:
    reference_output_dir = args.output_dir / "outputs" / "reference"
    reference_env = dict(env_updates)
    reference_env["EMEL_DIARIZATION_RESULT_DIR"] = str(reference_output_dir)
    reference_stderr = raw_dir / "reference.stderr.txt"
    reference_result = run_command([str(args.emel_runner), "--mode=reference"],
                                   reference_env,
                                   reference_jsonl,
                                   reference_stderr)
    if reference_result.returncode != 0:
      error_kind = reference_result.error_kind or "runner_failed"
      error_message = reference_result.error_message or (
        f"reference runner failed with exit code {reference_result.returncode}"
      )
      ensure_error_record(
        reference_jsonl,
        error_record(lane="reference",
                     backend_id="recorded.diarization.baseline",
                     backend_language="recorded",
                     error_kind=error_kind,
                     error_message=error_message,
                     compare_group=SORTFORMER_COMPARE_GROUP,
                     model_id=SORTFORMER_MODEL_ID,
                     fixture_id=SORTFORMER_FIXTURE_ID,
                     workload_id=SORTFORMER_WORKLOAD_ID,
                     reference_role="regression_snapshot"),
      )

  if args.onnx_reference_model is not None and args.onnx_reference_input is None:
    onnx_output_dir = args.output_dir / "outputs" / "onnx_reference"
    onnx_stderr = raw_dir / "onnx_reference.stderr.txt"
    onnx_command = [
      sys.executable,
      str(args.onnx_reference_runner),
      "--model",
      str(args.onnx_reference_model),
      "--features",
      str(onnx_reference_features),
      "--segments-output-dir",
      str(onnx_output_dir),
      "--output-contract",
      SORTFORMER_ONNX_OUTPUT_CONTRACT,
      "--iterations",
      str(max(1, args.benchmark_iterations)),
      "--runs",
      str(max(1, args.benchmark_runs)),
      "--warmup-iterations",
      str(args.benchmark_warmup_iterations),
      "--warmup-runs",
      str(args.benchmark_warmup_runs),
    ]
    if args.onnx_reference_expected_sha256:
      onnx_command.extend(["--expected-sha256", args.onnx_reference_expected_sha256])
    onnx_result = run_command(
      onnx_command,
      {},
      onnx_reference_jsonl,
      onnx_stderr,
    )
    if onnx_result.returncode != 0:
      error_kind = onnx_result.error_kind or "runner_failed"
      error_message = onnx_result.error_message or (
        f"ONNX reference runner failed with exit code {onnx_result.returncode}"
      )
      ensure_error_record(
        onnx_reference_jsonl,
        error_record(lane="reference",
                     backend_id="onnx.sortformer.v2_1",
                     backend_language="python/onnxruntime",
                     error_kind=error_kind,
                     error_message=error_message,
                     compare_group=SORTFORMER_COMPARE_GROUP,
                     model_id=SORTFORMER_MODEL_ID,
                     fixture_id=SORTFORMER_FIXTURE_ID,
                     workload_id=SORTFORMER_WORKLOAD_ID,
                     reference_role="benchmark_reference"),
      )

  if args.pytorch_reference_model and args.pytorch_reference_input is None:
    pytorch_output_dir = args.output_dir / "outputs" / "pytorch_reference"
    pytorch_stderr = raw_dir / "pytorch_reference.stderr.txt"
    pytorch_command = [
      str(args.pytorch_reference_python),
      str(args.pytorch_reference_runner),
      "--model",
      args.pytorch_reference_model,
      "--audio",
      str(args.pytorch_reference_audio),
      "--segments-output-dir",
      str(pytorch_output_dir),
      "--device",
      args.pytorch_reference_device,
      "--chunk-len",
      str(SORTFORMER_CHUNK_LEN),
      "--chunk-right-context",
      str(SORTFORMER_CHUNK_RIGHT_CONTEXT),
      "--fifo-len",
      str(SORTFORMER_FIFO_LEN),
      "--spkcache-update-period",
      str(SORTFORMER_SPKCACHE_UPDATE_PERIOD),
      "--spkcache-len",
      str(SORTFORMER_SPKCACHE_LEN),
    ]
    pytorch_result = run_command(
      pytorch_command,
      {},
      pytorch_reference_jsonl,
      pytorch_stderr,
    )
    if pytorch_result.returncode != 0:
      error_kind = pytorch_result.error_kind or "runner_failed"
      error_message = pytorch_result.error_message or (
        f"PyTorch reference runner failed with exit code {pytorch_result.returncode}"
      )
      ensure_error_record(
        pytorch_reference_jsonl,
        error_record(lane="reference",
                     backend_id="pytorch.nemo.sortformer.v2_1",
                     backend_language="python/pytorch+nemo",
                     error_kind=error_kind,
                     error_message=error_message,
                     compare_group=SORTFORMER_COMPARE_GROUP,
                     model_id=SORTFORMER_MODEL_ID,
                     fixture_id=SORTFORMER_FIXTURE_ID,
                     workload_id=SORTFORMER_WORKLOAD_ID,
                     reference_role="parity_reference"),
      )

  emel_records = parse_jsonl_records(emel_jsonl,
                                     lane="emel",
                                     backend_id="emel.diarization.sortformer",
                                     backend_language="cpp")
  reference_records = parse_jsonl_records(reference_jsonl,
                                          lane="reference",
                                          backend_id="recorded.diarization.baseline",
                                          backend_language="recorded")
  if args.onnx_reference_model is not None or args.onnx_reference_input is not None:
    onnx_reference_records = parse_jsonl_records(onnx_reference_jsonl,
                                                 lane="reference",
                                                 backend_id="onnx.sortformer.v2_1",
                                                 backend_language="python/onnxruntime")
    reference_records.extend(onnx_reference_records)
  if args.pytorch_reference_model or args.pytorch_reference_input is not None:
    pytorch_reference_records = parse_jsonl_records(pytorch_reference_jsonl,
                                                    lane="reference",
                                                    backend_id="pytorch.nemo.sortformer.v2_1",
                                                    backend_language="python/pytorch+nemo")
    reference_records.extend(pytorch_reference_records)
  summary = build_summary(emel_records, reference_records)
  summary_path = args.output_dir / "compare_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  print_text_summary(summary)
  return 1 if summary["failed"] else 0


if __name__ == "__main__":
  sys.exit(main())
