#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


SCHEMA = "diarization_compare/v1"
BACKEND_ID = "onnx.sortformer.v2_1"
COMPARE_GROUP = "diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono"
MODEL_ID = "diar_streaming_sortformer_4spk_v2_1_gguf"
FIXTURE_ID = "ami_en2002b_mix_headset_137.00_152.04_16khz_mono"
WORKLOAD_ID = "diarization_sortformer_pipeline_v1"
FEATURE_FRAMES = 1504
FEATURE_BINS = 128
OUTPUT_FRAMES = 188
SPEAKERS = 4
THRESHOLD = 0.5
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


def non_negative_int(value: str) -> int:
  parsed = int(value, 10)
  if parsed < 0:
    raise argparse.ArgumentTypeError("value must be non-negative")
  return parsed


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--features", type=Path, required=True)
  parser.add_argument("--segments-output-dir", type=Path)
  parser.add_argument("--expected-sha256", default="")
  parser.add_argument("--iterations", type=non_negative_int, default=1)
  parser.add_argument("--runs", type=non_negative_int, default=1)
  parser.add_argument("--warmup-iterations", type=non_negative_int, default=0)
  parser.add_argument("--warmup-runs", type=non_negative_int, default=0)
  return parser.parse_args()


def error_record(error_kind: str, error_message: str) -> dict[str, object]:
  return {
    "schema": SCHEMA,
    "record_type": "error",
    "status": "error",
    "case_name": f"reference/{BACKEND_ID}",
    "compare_group": COMPARE_GROUP,
    "lane": "reference",
    "backend_id": BACKEND_ID,
    "backend_language": "python/onnxruntime",
    "comparison_mode": "parity",
    "reference_role": "benchmark_reference",
    "model_id": MODEL_ID,
    "fixture_id": FIXTURE_ID,
    "workload_id": WORKLOAD_ID,
    "comparable": False,
    "ns_per_op": 0.0,
    "prepare_ns_per_op": 0.0,
    "encode_ns_per_op": 0.0,
    "publish_ns_per_op": 0.0,
    "output_bytes": 0,
    "output_dim": 0,
    "output_checksum": 0,
    "iterations": 0,
    "runs": 0,
    "output_path": "",
    "note": "proof_status=onnx_reference_error",
    "error_kind": error_kind,
    "error_message": error_message,
  }


def print_record(record: dict[str, object]) -> None:
  print(json.dumps(record, sort_keys=True))


def sha256_file(path: Path) -> str:
  hasher = hashlib.sha256()
  with path.open("rb") as handle:
    while True:
      chunk = handle.read(1 << 20)
      if not chunk:
        break
      hasher.update(chunk)
  return hasher.hexdigest()


def decode_segments(probabilities) -> list[tuple[int, int, int]]:
  segments: list[tuple[int, int, int]] = []
  for speaker in range(SPEAKERS):
    start_frame = -1
    for frame in range(OUTPUT_FRAMES):
      active = float(probabilities[frame, speaker]) >= THRESHOLD
      if active and start_frame < 0:
        start_frame = frame
      elif not active and start_frame >= 0:
        segments.append((speaker, start_frame, frame))
        start_frame = -1
    if start_frame >= 0:
      segments.append((speaker, start_frame, OUTPUT_FRAMES))
  return segments


def checksum_segments(segments: list[tuple[int, int, int]]) -> int:
  checksum = FNV_OFFSET
  for speaker, start_frame, end_frame in segments:
    checksum ^= speaker + 1
    checksum = (checksum * FNV_PRIME) & UINT64_MASK
    checksum ^= start_frame + 1
    checksum = (checksum * FNV_PRIME) & UINT64_MASK
    checksum ^= end_frame + 1
    checksum = (checksum * FNV_PRIME) & UINT64_MASK
  return checksum


def output_to_probabilities(output):
  try:
    import numpy as np
  except ImportError as exc:
    raise RuntimeError(f"numpy is required for ONNX reference execution: {exc}") from exc

  min_value = float(np.min(output))
  max_value = float(np.max(output))
  if min_value >= 0.0 and max_value <= 1.0:
    return output, "onnx_output_probabilities"
  return 1.0 / (1.0 + np.exp(-output)), "onnx_output_logits_sigmoid"


def write_segments(output_dir: Path | None, segments: list[tuple[int, int, int]]) -> tuple[str, int]:
  text = "".join(
    f"speaker={speaker} start_frame={start_frame} end_frame={end_frame}\n"
    for speaker, start_frame, end_frame in segments
  )
  if output_dir is None:
    return "", len(text.encode("utf-8"))
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / f"{BACKEND_ID.replace('.', '_')}__sortformer.segments.txt"
  path.write_text(text, encoding="utf-8")
  return str(path), len(text.encode("utf-8"))


def load_features(path: Path):
  try:
    import numpy as np
  except ImportError as exc:
    raise RuntimeError(f"numpy is required for ONNX reference execution: {exc}") from exc

  features = np.fromfile(path, dtype=np.float32)
  expected = FEATURE_FRAMES * FEATURE_BINS
  if features.size != expected:
    raise RuntimeError(
      f"feature input has {features.size} float32 values, expected {expected} "
      f"({FEATURE_FRAMES}x{FEATURE_BINS})"
    )
  return features.reshape((1, FEATURE_FRAMES, FEATURE_BINS))


def main() -> int:
  args = parse_args()
  if not args.model.exists():
    print_record(error_record("missing_model", f"ONNX model not found: {args.model}"))
    return 1
  if not args.features.exists():
    print_record(error_record("missing_feature_input", f"feature input not found: {args.features}"))
    return 1

  actual_sha256 = sha256_file(args.model)
  if args.expected_sha256 and actual_sha256 != args.expected_sha256:
    print_record(error_record("model_checksum_mismatch",
                              f"expected {args.expected_sha256} but found {actual_sha256}"))
    return 1

  try:
    import numpy as np
    import onnxruntime as ort
  except ImportError as exc:
    print_record(error_record("missing_dependency", str(exc)))
    return 1

  try:
    chunk = load_features(args.features).astype(np.float32, copy=False)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
      str(args.model),
      sess_options=session_options,
      providers=["CPUExecutionProvider"],
    )
    actual_providers = ",".join(session.get_providers())
    feeds = {
      "chunk": chunk,
      "chunk_lengths": np.asarray([FEATURE_FRAMES], dtype=np.int32),
      "spkcache": np.zeros((1, 0, 512), dtype=np.float32),
      "spkcache_lengths": np.asarray([0], dtype=np.int32),
      "fifo": np.zeros((1, 0, 512), dtype=np.float32),
      "fifo_lengths": np.asarray([0], dtype=np.int32),
    }
    iterations = max(1, args.iterations)
    runs = max(1, args.runs)
    for _ in range(args.warmup_runs):
      for _ in range(args.warmup_iterations):
        session.run(None, feeds)
    samples: list[float] = []
    outputs = None
    for _ in range(runs):
      start_ns = time.perf_counter_ns()
      for _ in range(iterations):
        outputs = session.run(None, feeds)
      samples.append(float(time.perf_counter_ns() - start_ns) / float(iterations))
    samples.sort()
    elapsed_ns = samples[len(samples) // 2]
    if outputs is None:
      raise RuntimeError("ONNX reference produced no outputs")
    logits = np.asarray(outputs[0], dtype=np.float32)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] != SPEAKERS:
      raise RuntimeError(f"unexpected logits shape: {logits.shape}")
    if logits.shape[1] < OUTPUT_FRAMES:
      raise RuntimeError(f"logits contain {logits.shape[1]} frames, expected at least {OUTPUT_FRAMES}")
    comparable_output = logits[0, -OUTPUT_FRAMES:, :]
    probabilities, output_contract = output_to_probabilities(comparable_output)
    segments = decode_segments(probabilities)
    checksum = checksum_segments(segments)
    output_path, output_bytes = write_segments(args.segments_output_dir, segments)
  except Exception as exc:  # noqa: BLE001 - runner errors are emitted as compare records.
    print_record(error_record("onnx_execution_failed", str(exc)))
    return 1

  print_record({
    "schema": SCHEMA,
    "record_type": "result",
    "status": "ok",
    "case_name": COMPARE_GROUP,
    "compare_group": COMPARE_GROUP,
    "lane": "reference",
    "backend_id": BACKEND_ID,
    "backend_language": "python/onnxruntime",
    "comparison_mode": "parity",
    "reference_role": "benchmark_reference",
    "model_id": MODEL_ID,
    "fixture_id": FIXTURE_ID,
    "workload_id": WORKLOAD_ID,
    "comparable": True,
    "ns_per_op": elapsed_ns,
    "prepare_ns_per_op": 0.0,
    "encode_ns_per_op": elapsed_ns,
    "publish_ns_per_op": 0.0,
    "output_bytes": output_bytes,
    "output_dim": len(segments),
    "output_checksum": checksum,
    "iterations": iterations,
    "runs": runs,
    "output_path": output_path,
    "note": (
      "proof_status=onnxruntime_cpu "
      "thread_contract=intra_op=1 inter_op=1 execution_mode=sequential "
      f"benchmark_config=iterations:{iterations},runs:{runs},"
      f"warmup_iterations:{args.warmup_iterations},warmup_runs:{args.warmup_runs} "
      f"actual_providers={actual_providers} "
      "feature_contract=emel_maintained_features "
      f"onnx_model={args.model} "
      f"onnx_sha256={actual_sha256} "
      f"output_contract={output_contract} "
      "onnx_repo=ooobo/diar_streaming_sortformer_4spk-v2.1-onnx"
    ),
    "error_kind": "",
    "error_message": "",
  })
  return 0


if __name__ == "__main__":
  sys.exit(main())
