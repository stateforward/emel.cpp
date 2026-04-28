#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCHEMA = "whisper_compare/v1"
SUMMARY_SCHEMA = "whisper_compare_summary/v1"
COMPARE_GROUP = "whisper/tiny/q8_0/phase99_440hz_16khz_mono"
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


def run_capture(command: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
  result = subprocess.run(command,
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          check=False)
  stdout_path.write_text(result.stdout, encoding="utf-8")
  stderr_path.write_text(result.stderr, encoding="utf-8")
  return result


def append_record(path: Path, record: dict[str, object]) -> None:
  with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True))
    handle.write("\n")


def output_record(*,
                  lane: str,
                  backend_id: str,
                  backend_language: str,
                  transcript: str,
                  output_path: Path,
                  model_path: Path,
                  audio_path: Path,
                  status: str = "ok",
                  error_kind: str = "",
                  error_message: str = "") -> dict[str, object]:
  record = {
    "schema": SCHEMA,
    "record_type": "result" if status == "ok" else "error",
    "status": status,
    "case_name": f"{lane}/{backend_id}",
    "compare_group": COMPARE_GROUP,
    "lane": lane,
    "backend_id": backend_id,
    "backend_language": backend_language,
    "comparison_mode": "parity",
    "model_id": "oxide_lab_whisper_tiny_q8_0",
    "model_path": str(model_path),
    "model_sha256": sha256_file(model_path) if model_path.exists() else "",
    "audio_fixture_id": "phase99_440hz_16khz_mono",
    "audio_path": str(audio_path),
    "audio_sha256": sha256_file(audio_path) if audio_path.exists() else "",
    "transcript": transcript,
    "token_count": 1 if transcript else 0,
    "timestamp_metadata": "unsupported",
    "output_bytes": len(transcript.encode("utf-8")),
    "output_checksum": checksum_text(transcript),
    "output_path": str(output_path),
    "error_kind": error_kind,
    "error_message": error_message,
  }
  if lane == "emel":
    record["runtime_surface"] = EMEL_RUNTIME_SURFACE
  return record


def read_jsonl_record(path: Path) -> dict[str, object]:
  for raw in path.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
      continue
    record = json.loads(raw)
    if record.get("schema") == SCHEMA:
      return record
  raise ValueError(f"no {SCHEMA} record in {path}")


def normalize_emel_record(record: dict[str, object],
                          model_path: Path,
                          audio_path: Path) -> dict[str, object]:
  transcript = str(record.get("transcript", ""))
  record["model_path"] = str(model_path)
  record["model_sha256"] = sha256_file(model_path) if model_path.exists() else ""
  record["audio_path"] = str(audio_path)
  record["audio_sha256"] = sha256_file(audio_path) if audio_path.exists() else ""
  record["backend_id"] = EMEL_BACKEND_ID
  record["case_name"] = f"emel/{EMEL_BACKEND_ID}"
  record["output_bytes"] = len(transcript.encode("utf-8"))
  record["output_checksum"] = checksum_text(transcript)
  record.setdefault("error_kind", "")
  record.setdefault("error_message", "")
  record["runtime_surface"] = EMEL_RUNTIME_SURFACE
  return record


def main() -> int:
  args = parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  raw_dir = args.output_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  output_dir = args.output_dir / "outputs"
  output_dir.mkdir(parents=True, exist_ok=True)

  require_file(args.emel_runner, "EMEL Whisper parity runner", executable=True)
  require_file(args.emel_model, "EMEL Whisper model")
  require_file(args.tokenizer, "EMEL Whisper tokenizer")
  require_tokenizer_checksum(args.tokenizer)
  require_file(args.reference_cli, "whisper.cpp CLI", executable=True)
  require_file(args.reference_model, "whisper.cpp reference model")
  require_file(args.audio, "Whisper audio fixture")

  emel_jsonl = raw_dir / "emel.jsonl"
  reference_jsonl = raw_dir / "reference.jsonl"
  summary_path = args.output_dir / "summary.json"
  emel_jsonl.write_text("", encoding="utf-8")
  reference_jsonl.write_text("", encoding="utf-8")

  emel_result = run_capture([
    str(args.emel_runner),
    "--model",
    str(args.emel_model),
    "--audio",
    str(args.audio),
    "--tokenizer",
    str(args.tokenizer),
    "--output-dir",
    str(output_dir / "emel"),
  ], raw_dir / "emel.stdout.txt", raw_dir / "emel.stderr.txt")
  if emel_result.returncode != 0:
    append_record(emel_jsonl, output_record(lane="emel",
                                            backend_id=EMEL_BACKEND_ID,
                                            backend_language="cpp",
                                            transcript="",
                                            output_path=output_dir / "emel" / "transcript.txt",
                                            model_path=args.emel_model,
                                            audio_path=args.audio,
                                            status="error",
                                            error_kind="runner_failed",
                                            error_message=f"exit={emel_result.returncode}"))
  else:
    emel_stdout_path = raw_dir / "emel.stdout.txt"
    emel_record = read_jsonl_record(emel_stdout_path)
    emel_record = normalize_emel_record(emel_record, args.emel_model, args.audio)
    emel_jsonl.write_text(json.dumps(emel_record, sort_keys=True) + "\n", encoding="utf-8")

  reference_output = output_dir / "reference" / "transcript"
  reference_output.parent.mkdir(parents=True, exist_ok=True)
  reference_result = run_capture([
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
    str(reference_output),
  ], raw_dir / "reference.stdout.txt", raw_dir / "reference.stderr.txt")
  reference_transcript_path = reference_output.with_suffix(".txt")
  reference_transcript = ""
  if reference_transcript_path.exists():
    reference_transcript = reference_transcript_path.read_text(encoding="utf-8").strip()
  if reference_result.returncode != 0:
    append_record(reference_jsonl, output_record(lane="reference",
                                                 backend_id="whisper_cpp_asr",
                                                 backend_language="cpp",
                                                 transcript="",
                                                 output_path=reference_transcript_path,
                                                 model_path=args.reference_model,
                                                 audio_path=args.audio,
                                                 status="error",
                                                 error_kind="runner_failed",
                                                 error_message=f"exit={reference_result.returncode}"))
  else:
    append_record(reference_jsonl, output_record(lane="reference",
                                                 backend_id="whisper_cpp_asr",
                                                 backend_language="cpp",
                                                 transcript=reference_transcript,
                                                 output_path=reference_transcript_path,
                                                 model_path=args.reference_model,
                                                 audio_path=args.audio))

  emel_record = read_jsonl_record(emel_jsonl)
  reference_record = read_jsonl_record(reference_jsonl)
  normalization_manifest_path = args.output_dir / "normalized" / "manifest.json"
  normalization_manifest: dict[str, object] = {}
  if normalization_manifest_path.exists():
    normalization_manifest = json.loads(
        normalization_manifest_path.read_text(encoding="utf-8"))
  status = "exact_match"
  reason = "ok"
  if emel_record.get("record_type") == "error":
    status = "error"
    reason = "emel_lane_error"
  elif reference_record.get("record_type") == "error":
    status = "error"
    reason = "reference_lane_error"
  elif emel_record.get("output_checksum") != reference_record.get("output_checksum"):
    status = "bounded_drift"
    reason = "transcript_mismatch"

  summary = {
    "schema": SUMMARY_SCHEMA,
    "compare_group": COMPARE_GROUP,
    "comparison_status": status,
    "reason": reason,
    "emel": emel_record,
    "reference": reference_record,
    "model_normalization": normalization_manifest,
  }
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
  print(f"{COMPARE_GROUP} status={status} reason={reason}")
  return 0 if status == "exact_match" else 1


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except Exception as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(1)
