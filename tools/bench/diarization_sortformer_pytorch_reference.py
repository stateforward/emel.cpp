#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SCHEMA = "diarization_compare/v1"
BACKEND_ID = "pytorch.nemo.sortformer.v2_1"
COMPARE_GROUP = "diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono"
MODEL_ID = "diar_streaming_sortformer_4spk_v2_1_gguf"
FIXTURE_ID = "ami_en2002b_mix_headset_137.00_152.04_16khz_mono"
WORKLOAD_ID = "diarization_sortformer_pipeline_v1"
SPEAKERS = 4
THRESHOLD = 0.5
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
  parser.add_argument("--audio", type=Path, required=True)
  parser.add_argument("--segments-output-dir", type=Path)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--sample-rate", type=int, default=16000)
  parser.add_argument("--chunk-len", type=int, default=340)
  parser.add_argument("--chunk-right-context", type=int, default=40)
  parser.add_argument("--fifo-len", type=int, default=40)
  parser.add_argument("--spkcache-update-period", type=int, default=300)
  parser.add_argument("--spkcache-len", type=int, default=188)
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
    "backend_language": "python/pytorch+nemo",
    "comparison_mode": "parity",
    "reference_role": "parity_reference",
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
    "note": "proof_status=pytorch_reference_error",
    "error_kind": error_kind,
    "error_message": error_message,
  }


def print_record(record: dict[str, object]) -> None:
  print(json.dumps(record, sort_keys=True))


def decode_segments(probabilities) -> list[tuple[int, int, int]]:
  segments: list[tuple[int, int, int]] = []
  frame_count = int(probabilities.shape[0])
  for speaker in range(SPEAKERS):
    start_frame = -1
    for frame in range(frame_count):
      active = float(probabilities[frame, speaker]) >= THRESHOLD
      if active and start_frame < 0:
        start_frame = frame
      elif not active and start_frame >= 0:
        segments.append((speaker, start_frame, frame))
        start_frame = -1
    if start_frame >= 0:
      segments.append((speaker, start_frame, frame_count))
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


def as_numpy_probabilities(predicted_probs):
  try:
    import numpy as np
    import torch
  except ImportError as exc:
    raise RuntimeError(f"numpy and torch are required for PyTorch reference execution: {exc}") from exc

  probs = predicted_probs[0] if isinstance(predicted_probs, (list, tuple)) else predicted_probs
  if isinstance(probs, torch.Tensor):
    probs = probs.detach().cpu().numpy()
  probabilities = np.asarray(probs, dtype=np.float32)
  if probabilities.ndim == 3 and probabilities.shape[0] == 1:
    probabilities = probabilities[0]
  if probabilities.ndim != 2:
    raise RuntimeError(f"unexpected probability rank: {probabilities.shape}")
  if probabilities.shape[1] != SPEAKERS and probabilities.shape[0] == SPEAKERS:
    probabilities = probabilities.transpose()
  if probabilities.shape[1] != SPEAKERS:
    raise RuntimeError(f"unexpected probability shape: {probabilities.shape}")
  min_value = float(np.min(probabilities))
  max_value = float(np.max(probabilities))
  if min_value < 0.0 or max_value > 1.0:
    raise RuntimeError(
      f"probabilities are outside [0, 1]: min={min_value} max={max_value}"
    )
  return probabilities


def restore_model(args: argparse.Namespace):
  try:
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
  except ImportError as exc:
    raise RuntimeError(f"NeMo ASR and PyTorch are required: {exc}") from exc

  map_location = torch.device(args.device)
  if args.model.endswith(".nemo") or Path(args.model).exists():
    model = SortformerEncLabelModel.restore_from(
      restore_path=args.model,
      map_location=map_location,
      strict=False,
    )
  else:
    model = SortformerEncLabelModel.from_pretrained(args.model)
    model = model.to(map_location)
  model.eval()
  modules = model.sortformer_modules
  modules.chunk_len = args.chunk_len
  modules.chunk_right_context = args.chunk_right_context
  modules.fifo_len = args.fifo_len
  modules.spkcache_update_period = args.spkcache_update_period
  modules.spkcache_len = args.spkcache_len
  modules._check_streaming_parameters()
  return model


def main() -> int:
  args = parse_args()
  if not args.audio.exists():
    print_record(error_record("missing_audio", f"audio fixture not found: {args.audio}"))
    return 1

  try:
    model = restore_model(args)
    start_ns = time.perf_counter_ns()
    _, predicted_probs = model.diarize(
      audio=[str(args.audio)],
      batch_size=1,
      sample_rate=args.sample_rate,
      include_tensor_outputs=True,
    )
    elapsed_ns = float(time.perf_counter_ns() - start_ns)
    probabilities = as_numpy_probabilities(predicted_probs)
    segments = decode_segments(probabilities)
    checksum = checksum_segments(segments)
    output_path, output_bytes = write_segments(args.segments_output_dir, segments)
  except Exception as exc:  # noqa: BLE001 - runner errors are emitted as compare records.
    message = str(exc)
    error_kind = "missing_dependency" if "required" in message else "pytorch_execution_failed"
    print_record(error_record(error_kind, message))
    return 1

  print_record({
    "schema": SCHEMA,
    "record_type": "result",
    "status": "ok",
    "case_name": COMPARE_GROUP,
    "compare_group": COMPARE_GROUP,
    "lane": "reference",
    "backend_id": BACKEND_ID,
    "backend_language": "python/pytorch+nemo",
    "comparison_mode": "parity",
    "reference_role": "parity_reference",
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
    "iterations": 1,
    "runs": 1,
    "output_path": output_path,
    "note": (
      "proof_status=pytorch_nemo "
      "timing_scope=diarize_only_excludes_model_load "
      f"source_model={args.model} "
      f"audio={args.audio} "
      f"device={args.device} "
      f"sample_rate={args.sample_rate} "
      f"chunk_len={args.chunk_len} "
      f"chunk_right_context={args.chunk_right_context} "
      f"fifo_len={args.fifo_len} "
      f"spkcache_update_period={args.spkcache_update_period} "
      f"spkcache_len={args.spkcache_len} "
      "source=https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1"
    ),
    "error_kind": "",
    "error_message": "",
  })
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
