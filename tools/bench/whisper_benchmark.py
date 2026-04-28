#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


SCHEMA = "whisper_benchmark/v1"
SUMMARY_SCHEMA = "whisper_benchmark_summary/v1"
COMPARE_GROUP = "whisper/tiny/q8_0/phase99_440hz_16khz_mono"
WHISPER_CPP_REF = "v1.7.6"
WHISPER_CPP_COMMIT = "a8d002cfd879315632a579e73f0148d06959de36"
EXPECTED_TOKENIZER_SHA256 = "dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759"
EMEL_BACKEND_ID = "emel.speech.recognizer.whisper"
EMEL_RUNTIME_SURFACE = "speech/recognizer+speech/recognizer_routes/whisper"


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def checksum_text(text: str) -> int:
  return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--emel-runner", type=Path, required=True)
  parser.add_argument("--emel-model", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path, required=True)
  parser.add_argument("--reference-cli", type=Path, required=True)
  parser.add_argument("--reference-model", type=Path, required=True)
  parser.add_argument("--audio", type=Path, required=True)
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--iterations", type=int, default=20)
  parser.add_argument("--performance-tolerance-ppm", type=int, default=20_000)
  return parser.parse_args()


def require_file(path: Path, description: str, executable: bool = False) -> None:
  if not path.exists():
    raise FileNotFoundError(f"{description} missing: {path}")
  if executable and not path.is_file():
    raise FileNotFoundError(f"{description} is not a file: {path}")
  if executable and (path.stat().st_mode & 0o111) == 0:
    raise PermissionError(f"{description} is not executable: {path}")


def require_tokenizer_checksum(path: Path) -> None:
  actual = sha256_file(path)
  if actual != EXPECTED_TOKENIZER_SHA256:
    raise ValueError(
      f"EMEL Whisper tokenizer sha256 mismatch: {actual} != {EXPECTED_TOKENIZER_SHA256}")


def run_capture(command: list[str], stdout_path: Path, stderr_path: Path) -> tuple[int, int]:
  start = time.perf_counter_ns()
  result = subprocess.run(command,
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          check=False)
  elapsed = time.perf_counter_ns() - start
  stdout_path.write_text(result.stdout, encoding="utf-8")
  stderr_path.write_text(result.stderr, encoding="utf-8")
  return result.returncode, elapsed


def read_jsonl_record(path: Path, schema: str) -> dict[str, object]:
  for raw in path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
      continue
    record = json.loads(raw)
    if record.get("schema") == schema:
      return record
  raise ValueError(f"no {schema} record in {path}")


def parse_reference_timings(stderr_text: str) -> dict[str, int]:
  timings: dict[str, int] = {}
  patterns = {
    "load_ns": r"load time =\s*([0-9.]+) ms",
    "mel_ns": r"mel time =\s*([0-9.]+) ms",
    "sample_ns": r"sample time =\s*([0-9.]+) ms",
    "encode_ns": r"encode time =\s*([0-9.]+) ms",
    "decode_ns": r"decode time =\s*([0-9.]+) ms",
    "batchd_ns": r"batchd time =\s*([0-9.]+) ms",
    "prompt_ns": r"prompt time =\s*([0-9.]+) ms",
    "reported_total_ns": r"total time =\s*([0-9.]+) ms",
  }
  for key, pattern in patterns.items():
    match = re.search(pattern, stderr_text)
    if match:
      timings[key] = int(float(match.group(1)) * 1_000_000.0)
  return timings


def host_identity() -> dict[str, str]:
  uname = platform.uname()
  return {
    "system": uname.system,
    "release": uname.release,
    "machine": uname.machine,
    "processor": uname.processor,
    "platform": platform.platform(),
  }


def git_identity(repo_root: Path) -> dict[str, object]:
  try:
    rev = subprocess.check_output(
      ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
      text=True,
      stderr=subprocess.DEVNULL).strip()
    dirty = subprocess.run(["git", "-C", str(repo_root), "diff", "--quiet"],
                           check=False).returncode != 0
    return {"commit": rev, "dirty": dirty}
  except Exception:
    return {"commit": "unknown", "dirty": True}


def append_record(path: Path, record: dict[str, object]) -> None:
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True))
    handle.write("\n")


def base_record(*,
                lane: str,
                backend_id: str,
                backend_language: str,
                model_path: Path,
                audio_path: Path,
                warmups: int,
                iterations: int,
                iteration_index: int,
                process_wall_time_ns: int,
                output_path: Path,
                transcript: str,
                status: str = "ok",
                error_kind: str = "",
                error_message: str = "") -> dict[str, object]:
  record = {
    "schema": SCHEMA,
    "record_type": "benchmark" if status == "ok" else "error",
    "status": status,
    "case_name": f"{lane}/{backend_id}",
    "compare_group": COMPARE_GROUP,
    "lane": lane,
    "backend_id": backend_id,
    "backend_language": backend_language,
    "benchmark_mode": "single_thread_cpu",
    "thread_count": 1,
    "processor_count": 1,
    "cpu_only": True,
    "model_id": "oxide_lab_whisper_tiny_q8_0",
    "model_path": str(model_path),
    "model_sha256": sha256_file(model_path) if model_path.exists() else "",
    "audio_fixture_id": "phase99_440hz_16khz_mono",
    "audio_path": str(audio_path),
    "audio_sha256": sha256_file(audio_path) if audio_path.exists() else "",
    "warmup_count": warmups,
    "iteration_count": iterations,
    "iteration_index": iteration_index,
    "process_wall_time_ns": process_wall_time_ns,
    "transcript": transcript,
    "output_bytes": len(transcript.encode("utf-8")),
    "output_checksum": checksum_text(transcript),
    "output_path": str(output_path),
    "timestamp_metadata": "unsupported",
    "host": host_identity(),
    "error_kind": error_kind,
    "error_message": error_message,
  }
  if lane == "emel":
    record["runtime_surface"] = EMEL_RUNTIME_SURFACE
  return record


def run_emel(args: argparse.Namespace,
             raw_dir: Path,
             outputs_dir: Path,
             iteration_index: int,
             record_path: Path) -> None:
  iter_output = outputs_dir / "emel" / f"iter_{iteration_index}"
  iter_output.mkdir(parents=True, exist_ok=True)
  stdout_path = raw_dir / f"emel_iter_{iteration_index}.stdout.txt"
  stderr_path = raw_dir / f"emel_iter_{iteration_index}.stderr.txt"
  returncode, elapsed = run_capture([
    str(args.emel_runner),
    "--model",
    str(args.emel_model),
    "--audio",
    str(args.audio),
    "--tokenizer",
    str(args.tokenizer),
    "--output-dir",
    str(iter_output),
  ], stdout_path, stderr_path)
  if returncode != 0:
    record = base_record(lane="emel",
                         backend_id=EMEL_BACKEND_ID,
                         backend_language="cpp",
                         model_path=args.emel_model,
                         audio_path=args.audio,
                         warmups=args.warmups,
                         iterations=args.iterations,
                         iteration_index=iteration_index,
                         process_wall_time_ns=elapsed,
                         output_path=iter_output / "transcript.txt",
                         transcript="",
                         status="error",
                         error_kind="runner_failed",
                         error_message=f"exit={returncode}")
    append_record(record_path, record)
    return
  compare_record = read_jsonl_record(stdout_path, "whisper_compare/v1")
  transcript = str(compare_record.get("transcript", ""))
  record = base_record(lane="emel",
                       backend_id=EMEL_BACKEND_ID,
                       backend_language="cpp",
                       model_path=args.emel_model,
                       audio_path=args.audio,
                       warmups=args.warmups,
                       iterations=args.iterations,
                       iteration_index=iteration_index,
                       process_wall_time_ns=elapsed,
                       output_path=iter_output / "transcript.txt",
                       transcript=transcript)
  for key in ("wall_time_ns", "model_load_ns", "audio_load_ns", "binding_ns", "contract_ns",
              "recognize_ns", "encode_ns", "decode_ns", "publish_ns", "selected_token",
              "encoder_frames", "encoder_width", "encoder_digest", "decoder_digest"):
    if key in compare_record:
      record[key] = compare_record[key]
  record["runtime_surface"] = EMEL_RUNTIME_SURFACE
  repo_root = Path(__file__).resolve().parents[2]
  record["backend_version"] = "emel-local"
  record["backend_build"] = git_identity(repo_root)
  append_record(record_path, record)


def run_reference(args: argparse.Namespace,
                  raw_dir: Path,
                  outputs_dir: Path,
                  iteration_index: int,
                  record_path: Path) -> None:
  iter_output = outputs_dir / "reference" / f"iter_{iteration_index}" / "transcript"
  iter_output.parent.mkdir(parents=True, exist_ok=True)
  stdout_path = raw_dir / f"reference_iter_{iteration_index}.stdout.txt"
  stderr_path = raw_dir / f"reference_iter_{iteration_index}.stderr.txt"
  returncode, elapsed = run_capture([
    str(args.reference_cli),
    "--model",
    str(args.reference_model),
    "--file",
    str(args.audio),
    "--threads",
    "1",
    "--processors",
    "1",
    "--no-gpu",
    "--audio-ctx",
    "50",
    "--beam-size",
    "1",
    "--best-of",
    "1",
    "--no-fallback",
    "--output-txt",
    "--output-file",
    str(iter_output),
  ], stdout_path, stderr_path)
  transcript_path = iter_output.with_suffix(".txt")
  transcript = ""
  if transcript_path.exists():
    transcript = transcript_path.read_text(encoding="utf-8").strip()
  reference_status = "ok" if returncode == 0 else "error"
  reference_error_kind = "" if returncode == 0 else "runner_failed"
  reference_error_message = "" if returncode == 0 else f"exit={returncode}"
  if returncode == 0 and not transcript_path.exists():
    reference_status = "error"
    reference_error_kind = "missing_transcript"
    reference_error_message = f"missing transcript output: {transcript_path}"
  record = base_record(lane="reference",
                       backend_id="whisper_cpp_asr",
                       backend_language="cpp",
                       model_path=args.reference_model,
                       audio_path=args.audio,
                       warmups=args.warmups,
                       iterations=args.iterations,
                       iteration_index=iteration_index,
                       process_wall_time_ns=elapsed,
                       output_path=transcript_path,
                       transcript=transcript,
                       status=reference_status,
                       error_kind=reference_error_kind,
                       error_message=reference_error_message)
  record.update(parse_reference_timings(stderr_path.read_text(encoding="utf-8")))
  record["backend_version"] = WHISPER_CPP_REF
  record["backend_commit"] = WHISPER_CPP_COMMIT
  append_record(record_path, record)


def summarize(records: list[dict[str, object]], lane: str) -> dict[str, object]:
  lane_records = [
    record for record in records
    if record.get("lane") == lane and record.get("record_type") == "benchmark"
  ]
  if not lane_records:
    return {"lane": lane, "status": "missing"}
  wall_times = [int(record.get("process_wall_time_ns", 0)) for record in lane_records]
  return {
    "lane": lane,
    "status": "ok",
    "backend_id": str(lane_records[-1].get("backend_id", "")),
    "iterations": len(lane_records),
    "min_process_wall_time_ns": min(wall_times),
    "mean_process_wall_time_ns": sum(wall_times) // len(wall_times),
    "max_process_wall_time_ns": max(wall_times),
    "model_sha256": str(lane_records[-1].get("model_sha256", "")),
    "transcript": str(lane_records[-1].get("transcript", "")),
    "output_checksum": lane_records[-1].get("output_checksum", 0),
    "runtime_surface": str(lane_records[-1].get("runtime_surface", "")),
  }


def first_iteration_mismatch(
    records: list[dict[str, object]]) -> dict[str, object] | None:
  emel_records = {
    int(record.get("iteration_index", -1)): record
    for record in records
    if record.get("lane") == "emel" and record.get("record_type") == "benchmark"
  }
  reference_records = {
    int(record.get("iteration_index", -1)): record
    for record in records
    if record.get("lane") == "reference" and record.get("record_type") == "benchmark"
  }
  for iteration_index in sorted(emel_records.keys() & reference_records.keys()):
    emel_record = emel_records[iteration_index]
    reference_record = reference_records[iteration_index]
    if emel_record.get("model_sha256") != reference_record.get("model_sha256"):
      return {
        "iteration_index": iteration_index,
        "reason": "model_mismatch",
        "emel_model_sha256": str(emel_record.get("model_sha256", "")),
        "reference_model_sha256": str(reference_record.get("model_sha256", "")),
      }
    if emel_record.get("transcript") != reference_record.get("transcript"):
      return {
        "iteration_index": iteration_index,
        "reason": "transcript_mismatch",
        "emel_transcript": str(emel_record.get("transcript", "")),
        "reference_transcript": str(reference_record.get("transcript", "")),
      }
  return None


def performance_regression(emel_summary: dict[str, object],
                           reference_summary: dict[str, object],
                           tolerance_ppm: int) -> dict[str, object] | None:
  if emel_summary.get("status") != "ok" or reference_summary.get("status") != "ok":
    return None
  emel_mean = int(emel_summary.get("mean_process_wall_time_ns", 0))
  reference_mean = int(reference_summary.get("mean_process_wall_time_ns", 0))
  ratio_ppm = 0
  if reference_mean > 0:
    ratio_ppm = (emel_mean * 1_000_000) // reference_mean
  comparison = {
    "emel_mean_process_wall_time_ns": emel_mean,
    "reference_mean_process_wall_time_ns": reference_mean,
    "emel_minus_reference_ns": emel_mean - reference_mean,
    "emel_to_reference_ppm": ratio_ppm,
    "tolerance_ppm": tolerance_ppm,
  }
  if reference_mean > 0 and ratio_ppm > 1_000_000 + tolerance_ppm:
    return comparison
  return None


def read_records(paths: tuple[Path, ...]) -> list[dict[str, object]]:
  records: list[dict[str, object]] = []
  for path in paths:
    if not path.exists():
      continue
    for line in path.read_text(encoding="utf-8").splitlines():
      if line.strip():
        records.append(json.loads(line))
  return records


def main() -> int:
  args = parse_args()
  if args.iterations <= 0 or args.warmups < 0 or args.performance_tolerance_ppm < 0:
    raise ValueError(
      "iterations must be positive; warmups and tolerance must be non-negative")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.output_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  outputs_dir = args.output_dir / "outputs"
  outputs_dir.mkdir(parents=True, exist_ok=True)

  require_file(args.emel_runner, "EMEL Whisper parity runner", executable=True)
  require_file(args.emel_model, "EMEL Whisper model")
  require_file(args.tokenizer, "EMEL Whisper tokenizer")
  require_tokenizer_checksum(args.tokenizer)
  require_file(args.reference_cli, "whisper.cpp CLI", executable=True)
  require_file(args.reference_model, "whisper.cpp reference model")
  require_file(args.audio, "Whisper audio fixture")

  emel_record_path = raw_dir / "emel_benchmark.jsonl"
  reference_record_path = raw_dir / "reference_benchmark.jsonl"
  emel_warmup_record_path = raw_dir / "emel_warmup.jsonl"
  reference_warmup_record_path = raw_dir / "reference_warmup.jsonl"
  emel_record_path.write_text("", encoding="utf-8")
  reference_record_path.write_text("", encoding="utf-8")
  emel_warmup_record_path.write_text("", encoding="utf-8")
  reference_warmup_record_path.write_text("", encoding="utf-8")

  for warmup in range(args.warmups):
    run_emel(args, raw_dir, outputs_dir, -(warmup + 1), emel_warmup_record_path)
    run_reference(args, raw_dir, outputs_dir, -(warmup + 1), reference_warmup_record_path)

  for iteration in range(args.iterations):
    run_emel(args, raw_dir, outputs_dir, iteration, emel_record_path)
    run_reference(args, raw_dir, outputs_dir, iteration, reference_record_path)

  measured_records = read_records((emel_record_path, reference_record_path))
  all_records = read_records((emel_warmup_record_path, reference_warmup_record_path,
                              emel_record_path, reference_record_path))
  failed = [record for record in all_records if record.get("record_type") == "error"]
  mismatch = first_iteration_mismatch(measured_records)
  emel_summary = summarize(measured_records, "emel")
  reference_summary = summarize(measured_records, "reference")
  perf_regression = performance_regression(emel_summary, reference_summary,
                                           args.performance_tolerance_ppm)
  summary_status = "ok"
  reason = "ok"
  if failed:
    summary_status = "error"
    reason = "lane_error"
  elif emel_summary.get("status") != "ok" or reference_summary.get("status") != "ok":
    summary_status = "error"
    reason = "missing_lane"
  elif mismatch is not None:
    summary_status = "error"
    reason = str(mismatch.get("reason", "iteration_mismatch"))
  elif perf_regression is not None:
    summary_status = "error"
    reason = "performance_regression"

  summary = {
    "schema": SUMMARY_SCHEMA,
    "compare_group": COMPARE_GROUP,
    "benchmark_mode": "single_thread_cpu",
    "thread_count": 1,
    "processor_count": 1,
    "cpu_only": True,
    "warmup_count": args.warmups,
    "iteration_count": args.iterations,
    "performance_tolerance_ppm": args.performance_tolerance_ppm,
    "status": summary_status,
    "reason": reason,
    "host": host_identity(),
    "emel": emel_summary,
    "reference": reference_summary,
  }
  if failed:
    summary["first_error"] = {
      "lane": str(failed[0].get("lane", "")),
      "iteration_index": failed[0].get("iteration_index", 0),
      "error_kind": str(failed[0].get("error_kind", "")),
      "error_message": str(failed[0].get("error_message", "")),
    }
  if mismatch is not None:
    summary["first_mismatch"] = mismatch
  if perf_regression is not None:
    summary["performance_comparison"] = perf_regression
  (args.output_dir / "benchmark_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  print(f"{COMPARE_GROUP} benchmark_status={summary_status} reason={reason}")
  return 1 if summary_status == "error" else 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except Exception as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(1)
