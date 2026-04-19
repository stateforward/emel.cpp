#!/usr/bin/env python3

from __future__ import annotations

import argparse
import array
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


SCHEMA = "embedding_compare/v1"
FORMAT_ENV = "EMEL_EMBEDDING_BENCH_FORMAT"
RESULT_DIR_ENV = "EMEL_EMBEDDING_RESULT_DIR"
CASE_FILTER_ENV = "EMEL_BENCH_CASE_FILTER"


def repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def fixture_root() -> Path:
  return repo_root() / "tests" / "embeddings" / "fixtures" / "te75m"


def emit_jsonl() -> bool:
  return os.environ.get(FORMAT_ENV, "") == "jsonl"


def case_enabled(case_name: str) -> bool:
  case_filter = os.environ.get(CASE_FILTER_ENV, "")
  return not case_filter or case_filter in case_name


def sanitize_name(name: str) -> str:
  return "".join(ch if ch.isalnum() else "_" for ch in name)


def checksum_bytes(data: bytes) -> int:
  hash_value = 1469598103934665603
  for byte in data:
    hash_value ^= byte
    hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
  return hash_value


def load_float_vector(path: Path) -> list[float]:
  values: list[float] = []
  for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line:
      continue
    values.append(float(line))
  return values


def dump_output(record: dict[str, object]) -> None:
  output_dir = os.environ.get(RESULT_DIR_ENV, "")
  output_values = record.get("output_values", [])
  if not output_dir or not output_values:
    return
  root = Path(output_dir)
  root.mkdir(parents=True, exist_ok=True)
  output_path = root / (
    f"{sanitize_name(str(record['backend_id']))}__{sanitize_name(str(record['case_name']))}.f32"
  )
  buffer = array.array("f", output_values)
  with output_path.open("wb") as handle:
    buffer.tofile(handle)
  record["output_path"] = str(output_path)


def print_record(record: dict[str, object]) -> None:
  payload = {
    "schema": SCHEMA,
    "record_type": record.get("record_type", "result"),
    "status": record.get("status", "ok"),
    "case_name": record.get("case_name", ""),
    "compare_group": record.get("compare_group", ""),
    "lane": record.get("lane", "reference"),
    "backend_id": record.get("backend_id", ""),
    "backend_language": record.get("backend_language", "python"),
    "comparison_mode": record.get("comparison_mode", ""),
    "model_id": record.get("model_id", ""),
    "fixture_id": record.get("fixture_id", ""),
    "modality": record.get("modality", ""),
    "ns_per_op": record.get("ns_per_op", 0.0),
    "prepare_ns_per_op": record.get("prepare_ns_per_op", 0.0),
    "encode_ns_per_op": record.get("encode_ns_per_op", 0.0),
    "publish_ns_per_op": record.get("publish_ns_per_op", 0.0),
    "output_tokens": record.get("output_tokens", 0),
    "output_dim": record.get("output_dim", 0),
    "output_checksum": record.get("output_checksum", 0),
    "iterations": record.get("iterations", 0),
    "runs": record.get("runs", 0),
    "output_path": record.get("output_path", ""),
    "note": record.get("note", ""),
    "error_kind": record.get("error_kind", ""),
    "error_message": record.get("error_message", ""),
  }
  print(json.dumps(payload, sort_keys=True))


def result_record(*,
                  backend_id: str,
                  case_name: str,
                  compare_group: str,
                  modality: str,
                  model_id: str,
                  fixture_id: str,
                  values: list[float],
                  note: str = "") -> dict[str, object]:
  binary = array.array("f", values).tobytes()
  return {
    "record_type": "result",
    "status": "ok",
    "case_name": case_name,
    "compare_group": compare_group,
    "lane": "reference",
    "backend_id": backend_id,
    "backend_language": "python",
    "comparison_mode": "parity",
    "model_id": model_id,
    "fixture_id": fixture_id,
    "modality": modality,
    "ns_per_op": 0.0,
    "prepare_ns_per_op": 0.0,
    "encode_ns_per_op": 0.0,
    "publish_ns_per_op": 0.0,
    "output_tokens": 1,
    "output_dim": len(values),
    "output_checksum": checksum_bytes(binary),
    "iterations": 1,
    "runs": 1,
    "output_values": values,
    "output_path": "",
    "note": note,
    "error_kind": "",
    "error_message": "",
  }


def error_record(*,
                 backend_id: str,
                 error_kind: str,
                 error_message: str,
                 note: str = "") -> dict[str, object]:
  return {
    "record_type": "error",
    "status": "error",
    "case_name": f"reference/python/{backend_id}",
    "compare_group": "backend",
    "lane": "reference",
    "backend_id": backend_id,
    "backend_language": "python",
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
    "output_values": [],
    "output_path": "",
    "note": note,
    "error_kind": error_kind,
    "error_message": error_message,
  }


def emit_records(records: list[dict[str, object]]) -> int:
  if not emit_jsonl():
    print("error: EMEL_EMBEDDING_BENCH_FORMAT=jsonl is required", file=sys.stderr)
    return 1
  exit_code = 0
  for record in records:
    dump_output(record)
    print_record(record)
    if record.get("record_type") == "error":
      exit_code = 1
  return exit_code


def te75m_goldens_records() -> list[dict[str, object]]:
  root = fixture_root()
  mapping = [
    (
      "reference/python/steady_request/te75m_goldens_text_red_square_full_dim",
      "text/red_square/full_dim",
      "text",
      root / "red-square.text.1280.txt",
    ),
    (
      "reference/python/steady_request/te75m_goldens_image_red_square_full_dim",
      "image/red_square/full_dim",
      "image",
      root / "red-square.image.1280.txt",
    ),
    (
      "reference/python/steady_request/te75m_goldens_audio_pure_tone_440hz_full_dim",
      "audio/pure_tone_440hz/full_dim",
      "audio",
      root / "pure-tone-440hz.audio.1280.txt",
    ),
  ]
  records: list[dict[str, object]] = []
  for case_name, compare_group, modality, path in mapping:
    if not case_enabled(case_name):
      continue
    if not path.exists():
      records.append(error_record(
        backend_id="python.reference.te75m_goldens",
        error_kind="missing_fixture",
        error_message=f"missing upstream golden fixture: {path}",
        note="golden_fixture_required",
      ))
      return records
    records.append(result_record(
      backend_id="python.reference.te75m_goldens",
      case_name=case_name,
      compare_group=compare_group,
      modality=modality,
      model_id="augmem/TE-75M.safetensors",
      fixture_id=str(path.relative_to(repo_root())),
      values=load_float_vector(path),
      note="stored_upstream_python_goldens",
    ))
  return records


def te75m_live_records() -> list[dict[str, object]]:
  generator_path = fixture_root() / "generate_upstream_goldens.py"
  if not generator_path.exists():
    return [error_record(
      backend_id="python.reference.te75m_live",
      error_kind="missing_script",
      error_message=f"missing live TE backend script: {generator_path}",
    )]

  try:
    spec = importlib.util.spec_from_file_location("emel_te75m_generate_upstream_goldens",
                                                  generator_path)
    if spec is None or spec.loader is None:
      raise RuntimeError(f"failed to load module spec from {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
  except Exception as exc:  # noqa: BLE001
    return [error_record(
      backend_id="python.reference.te75m_live",
      error_kind="python_environment_error",
      error_message=str(exc),
      note="live_python_backend_import_failed",
    )]

  temp_root = Path(tempfile.mkdtemp(prefix="emel-te75m-live-"))
  efficientat_root = temp_root / "EfficientAT"
  try:
    with contextlib.redirect_stdout(sys.stderr):
      module.ensure_efficientat_clone(efficientat_root)
      module.generate_goldens(temp_root, efficientat_root)
  except Exception as exc:  # noqa: BLE001
    shutil.rmtree(temp_root, ignore_errors=True)
    return [error_record(
      backend_id="python.reference.te75m_live",
      error_kind="python_backend_execution_error",
      error_message=str(exc),
      note="live_python_backend_execution_failed",
    )]

  mapping = [
    (
      "reference/python/steady_request/te75m_live_text_red_square_full_dim",
      "text/red_square/full_dim",
      "text",
      temp_root / "red-square.text.1280.txt",
    ),
    (
      "reference/python/steady_request/te75m_live_image_red_square_full_dim",
      "image/red_square/full_dim",
      "image",
      temp_root / "red-square.image.1280.txt",
    ),
    (
      "reference/python/steady_request/te75m_live_audio_pure_tone_440hz_full_dim",
      "audio/pure_tone_440hz/full_dim",
      "audio",
      temp_root / "pure-tone-440hz.audio.1280.txt",
    ),
  ]
  records: list[dict[str, object]] = []
  for case_name, compare_group, modality, path in mapping:
    if not case_enabled(case_name):
      continue
    if not path.exists():
      records.append(error_record(
        backend_id="python.reference.te75m_live",
        error_kind="missing_live_output",
        error_message=f"expected generated live output missing: {path}",
        note="live_python_backend_generation_incomplete",
      ))
      break
    records.append(result_record(
      backend_id="python.reference.te75m_live",
      case_name=case_name,
      compare_group=compare_group,
      modality=modality,
      model_id="augmem/TE-75M.safetensors",
      fixture_id="hf://augmem/TE-75M/TE-75M.safetensors",
      values=load_float_vector(path),
      note="live_upstream_python_inference",
    ))
  shutil.rmtree(temp_root, ignore_errors=True)
  return records


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--backend",
    required=True,
    choices=("te75m_goldens", "te75m_live"),
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  if args.backend == "te75m_goldens":
    return emit_records(te75m_goldens_records())
  if args.backend == "te75m_live":
    return emit_records(te75m_live_records())
  print(f"error: unsupported backend {args.backend}", file=sys.stderr)
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
