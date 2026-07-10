#!/usr/bin/env python3
"""CPU-only fixed-seed PersonaPlex E2E comparison (personaplex_compare/v1)."""

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path


def run_lane(command: list[str], stdout_path: Path, stderr_path: Path) -> float:
  started = time.monotonic()
  with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
    result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
  elapsed = time.monotonic() - started
  if result.returncode != 0:
    raise RuntimeError(
        f"lane failed with status {result.returncode}: {' '.join(command)}\n"
        f"stderr: {stderr_path}")
  return elapsed


def parse_codes(line: str) -> list[int]:
  return [int(value) for value in line.split("codes=", 1)[1].strip().split(",")]


def parse_emel_log(path: Path) -> tuple[list[list[int]], list[list[int]], list[int]]:
  input_frames = []
  output_frames = []
  text_tokens = []
  pattern = re.compile(r"text=(-?\d+).*codes=([0-9,-]+)")
  for line in path.read_text().splitlines():
    if line.startswith("EMEL_INPUT "):
      input_frames.append(parse_codes(line))
    elif line.startswith("EMEL_OUTPUT "):
      match = pattern.search(line)
      if match is None:
        raise ValueError(f"malformed EMEL graph line: {line}")
      text_tokens.append(int(match.group(1)))
      output_frames.append([int(value) for value in match.group(2).split(",")])
  return input_frames, output_frames, text_tokens


def parse_reference_log(path: Path) -> tuple[list[list[int]], list[list[int]], list[int]]:
  input_frames = []
  output_frames = []
  text_tokens = []
  pattern = re.compile(r"text=(-?\d+).*codes=([0-9,-]+)")
  for line in path.read_text().splitlines():
    if line.startswith("input "):
      input_frames.append(parse_codes(line))
    elif line.startswith("output "):
      match = pattern.search(line)
      if match is None:
        raise ValueError(f"malformed reference output line: {line}")
      text_tokens.append(int(match.group(1)))
      output_frames.append([int(value) for value in match.group(2).split(",")])
  return input_frames, output_frames, text_tokens


def match_fraction(left: list[list[int]], right: list[list[int]], streams: int) -> float:
  if len(left) != len(right) or not left:
    return 0.0
  matched = 0
  total = 0
  for left_frame, right_frame in zip(left, right):
    width = min(streams, len(left_frame), len(right_frame))
    matched += sum(a == b for a, b in zip(left_frame[:width], right_frame[:width]))
    total += width
  return matched / total if total else 0.0


def common_token_prefix(left: list[list[int]], right: list[list[int]]) -> int:
  count = 0
  for left_frame, right_frame in zip(left, right):
    for left_token, right_token in zip(left_frame, right_frame):
      if left_token != right_token:
        return count
      count += 1
    if len(left_frame) != len(right_frame):
      return count
  return count


def read_wav(path: Path) -> tuple[int, list[float]]:
  with wave.open(str(path), "rb") as source:
    if (source.getnchannels(), source.getsampwidth()) != (1, 2):
      raise ValueError(f"expected mono s16 WAV: {path}")
    sample_rate = source.getframerate()
    raw = source.readframes(source.getnframes())
  samples = struct.unpack(f"<{len(raw) // 2}h", raw)
  return sample_rate, [sample / 32768.0 for sample in samples]


def audio_metrics(samples: list[float], sample_rate: int) -> dict[str, object]:
  energy = sum(sample * sample for sample in samples)
  rms = math.sqrt(energy / len(samples)) if samples else 0.0
  peak = max((abs(sample) for sample in samples), default=0.0)
  window = max(1, int(sample_rate * 0.08))
  frame_rms = []
  for begin in range(0, len(samples), window):
    frame = samples[begin:begin + window]
    frame_energy = sum(sample * sample for sample in frame)
    frame_rms.append(math.sqrt(frame_energy / len(frame)))
  active = sum(value >= 0.01 for value in frame_rms)
  return {
      "samples": len(samples),
      "seconds": len(samples) / sample_rate,
      "peak": peak,
      "rms": rms,
      "active_ratio": active / len(frame_rms) if frame_rms else 0.0,
      "frame_rms": frame_rms,
  }


def correlation(left: list[float], right: list[float]) -> float:
  if len(left) != len(right) or len(left) < 2:
    return 0.0
  left_mean = sum(left) / len(left)
  right_mean = sum(right) / len(right)
  numerator = sum((a - left_mean) * (b - right_mean)
                  for a, b in zip(left, right))
  left_energy = sum((value - left_mean) ** 2 for value in left)
  right_energy = sum((value - right_mean) ** 2 for value in right)
  denominator = math.sqrt(left_energy * right_energy)
  return numerator / denominator if denominator else 0.0


def best_energy_correlation(left: list[float], right: list[float], max_lag: int) -> tuple[float, int]:
  best = (-2.0, 0)
  for lag in range(-max_lag, max_lag + 1):
    if lag < 0:
      current = correlation(left[-lag:], right[:len(right) + lag])
    elif lag > 0:
      current = correlation(left[:len(left) - lag], right[lag:])
    else:
      current = correlation(left, right)
    if current > best[0]:
      best = (current, lag)
  return best


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while block := source.read(1024 * 1024):
      digest.update(block)
  return digest.hexdigest()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--emel-runner", required=True)
  parser.add_argument("--reference-driver", required=True)
  parser.add_argument("--emel-mimi", required=True)
  parser.add_argument("--reference-mimi", required=True)
  parser.add_argument("--emel-lm", required=True)
  parser.add_argument("--reference-lm", required=True)
  parser.add_argument("--emel-voice", required=True)
  parser.add_argument("--reference-voice", required=True)
  parser.add_argument("--config", required=True)
  parser.add_argument("--inference-config", required=True)
  parser.add_argument("--audio", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--frames", type=int, default=125)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--threads", type=int, default=1)
  parser.add_argument("--audio-temperature", type=float, default=0.8)
  parser.add_argument("--text-temperature", type=float, default=0.7)
  parser.add_argument("--audio-top-k", type=int, default=250)
  parser.add_argument("--text-top-k", type=int, default=25)
  parser.add_argument("--min-output-match", type=float, default=0.15)
  parser.add_argument("--min-first-four-match", type=float, default=0.30)
  parser.add_argument("--min-energy-correlation", type=float, default=0.30)
  parser.add_argument("--max-active-ratio-delta", type=float, default=0.35)
  args = parser.parse_args()
  if args.frames <= 0 or args.seed <= 0 or args.threads <= 0:
    parser.error("frames, seed, and threads must be positive")

  inference_path = Path(args.inference_config)
  inference = json.loads(inference_path.read_text(encoding="utf-8"))
  required_inference = (
      "dep_q", "max_blocks", "block_tokens", "prompt_text_token",
      "frame_samples")
  if any(not isinstance(inference.get(key), int)
         for key in required_inference):
    parser.error("inference config integer contract is incomplete")
  public_n_q = inference["dep_q"]
  max_blocks = inference["max_blocks"]
  block_tokens = inference["block_tokens"]
  prompt_text_token = inference["prompt_text_token"]
  frame_samples = inference["frame_samples"]
  if (public_n_q <= 0 or max_blocks <= 0 or block_tokens <= 0 or
      frame_samples <= 0):
    parser.error(
        "inference dep_q, max_blocks, block_tokens, and frame_samples must "
        "be positive")

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  emel_wav = output_dir / "emel_personaplex.wav"
  reference_wav = output_dir / "moshi_cpp_personaplex.wav"
  emel_stdout = output_dir / "emel.stdout.log"
  emel_stderr = output_dir / "emel.stderr.log"
  reference_stdout = output_dir / "moshi_cpp.stdout.log"
  reference_stderr = output_dir / "moshi_cpp.stderr.log"

  emel_elapsed = run_lane([
      args.emel_runner,
      args.emel_mimi,
      args.emel_lm,
      args.emel_voice,
      args.audio,
      str(emel_wav),
      str(args.frames),
      str(args.seed),
      str(args.audio_temperature),
      str(args.text_temperature),
      str(args.audio_top_k),
      str(args.text_top_k),
      str(max_blocks),
      str(block_tokens),
      str(prompt_text_token),
  ], emel_stdout, emel_stderr)
  reference_elapsed = run_lane([
      args.reference_driver,
      "personaplex",
      "--config", args.config,
      "--lm", args.reference_lm,
      "--mimi", args.reference_mimi,
      "--voice", args.reference_voice,
      "--audio", args.audio,
      "--out", str(reference_wav),
      "--frames", str(args.frames),
      "--threads", str(args.threads),
      "--seed", str(args.seed),
      "--n-q", str(public_n_q),
      "--audio-temperature", str(args.audio_temperature),
      "--text-temperature", str(args.text_temperature),
  ], reference_stdout, reference_stderr)

  emel_input, emel_output, emel_text = parse_emel_log(emel_stderr)
  ref_input, ref_output, ref_text = parse_reference_log(reference_stdout)
  input_rate, input_samples = read_wav(Path(args.audio))
  if input_rate != 24000:
    raise RuntimeError("input WAV sample rate must be 24 kHz")
  audio_input_frames = (
      len(input_samples) + frame_samples - 1) // frame_samples
  emel_wav_input = emel_input[:audio_input_frames]
  ref_public_input = [frame[:public_n_q]
                      for frame in ref_input[:audio_input_frames]]
  input_match = match_fraction(
      emel_wav_input, ref_public_input, public_n_q)
  output_match = match_fraction(emel_output, ref_output, public_n_q)
  first_four_match = match_fraction(
      emel_output, ref_output, min(4, public_n_q))
  text_match = (sum(a == b for a, b in zip(emel_text, ref_text)) /
                len(emel_text) if len(emel_text) == len(ref_text) and emel_text
                else 0.0)

  emel_rate, emel_samples = read_wav(emel_wav)
  reference_rate, reference_samples = read_wav(reference_wav)
  if emel_rate != reference_rate:
    raise RuntimeError("output WAV sample-rate mismatch")
  emel_audio = audio_metrics(emel_samples, emel_rate)
  reference_audio = audio_metrics(reference_samples, reference_rate)
  emel_log_energy = [math.log(max(value, 1.0e-8))
                     for value in emel_audio.pop("frame_rms")]
  reference_log_energy = [math.log(max(value, 1.0e-8))
                          for value in reference_audio.pop("frame_rms")]
  energy_correlation, energy_lag = best_energy_correlation(
      emel_log_energy, reference_log_energy, 25)
  active_ratio_delta = abs(
      float(emel_audio["active_ratio"]) - float(reference_audio["active_ratio"]))

  reasons = []
  if len(emel_input) != args.frames or len(ref_input) != args.frames:
    reasons.append("input frame count mismatch")
  if len(emel_output) != args.frames or len(ref_output) != args.frames:
    reasons.append("output frame count mismatch")
  if input_match != 1.0:
    reasons.append(f"same-WAV input token match is {input_match:.6f}, expected 1.0")
  if output_match < args.min_output_match:
    reasons.append(
        f"output token match {output_match:.6f} below {args.min_output_match:.6f}")
  if first_four_match < args.min_first_four_match:
    reasons.append(
        f"first-four token match {first_four_match:.6f} below "
        f"{args.min_first_four_match:.6f}")
  if energy_correlation < args.min_energy_correlation:
    reasons.append(
        f"energy correlation {energy_correlation:.6f} below "
        f"{args.min_energy_correlation:.6f}")
  if active_ratio_delta > args.max_active_ratio_delta:
    reasons.append(
        f"active ratio delta {active_ratio_delta:.6f} above "
        f"{args.max_active_ratio_delta:.6f}")
  if float(emel_audio["rms"]) <= 0.0 or float(reference_audio["rms"]) <= 0.0:
    reasons.append("one or both output WAVs are silent")

  report = {
      "surface": "personaplex_compare/v1",
      "status": "pass" if not reasons else "fail",
      "reasons": reasons,
      "cpu_threads": args.threads,
      "seed": args.seed,
      "frames": args.frames,
      "inference": {
          "path": str(inference_path.resolve()),
          "sha256": sha256(inference_path),
          "public_n_q": public_n_q,
          "max_blocks": max_blocks,
          "block_tokens": block_tokens,
          "prompt_text_token": prompt_text_token,
          "frame_samples": frame_samples,
      },
      "input": {
          "path": str(Path(args.audio).resolve()),
          "sha256": sha256(Path(args.audio)),
          "emel_frames": len(emel_input),
          "reference_frames": len(ref_input),
          "audio_frames": audio_input_frames,
          "compared_public_tokens": audio_input_frames * public_n_q,
          "public_codebook_match_fraction": input_match,
      },
      "tokens": {
          "emel_output_frames": len(emel_output),
          "reference_output_frames": len(ref_output),
          "common_prefix_tokens": common_token_prefix(emel_output, ref_output),
          "public_codebook_match_fraction": output_match,
          "first_four_codebook_match_fraction": first_four_match,
          "text_token_match_fraction": text_match,
      },
      "audio": {
          "emel": emel_audio,
          "reference": reference_audio,
          "best_log_energy_correlation": energy_correlation,
          "best_energy_lag_frames_80ms": energy_lag,
          "active_ratio_delta": active_ratio_delta,
          "emel_wav": str(emel_wav.resolve()),
          "reference_wav": str(reference_wav.resolve()),
          "emel_sha256": sha256(emel_wav),
          "reference_sha256": sha256(reference_wav),
      },
      "timing": {
          "emel_wall_seconds": emel_elapsed,
          "reference_wall_seconds": reference_elapsed,
          "reference_over_emel": reference_elapsed / emel_elapsed,
      },
  }
  report_path = output_dir / "personaplex_compare.json"
  report_path.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))
  print(f"report: {report_path.resolve()}")
  return 0 if not reasons else 1


if __name__ == "__main__":
  try:
    sys.exit(main())
  except (OSError, RuntimeError, ValueError) as error:
    sys.exit(f"error: {error}")
