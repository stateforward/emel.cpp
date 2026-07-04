#!/usr/bin/env python3
"""Two-lane Mimi codec comparison (surface: mimi_compare/v1).

Runs the EMEL-owned runner (mimi_emel_parity_runner, enriched GGUF) and the
reference driver (moshi_reference_driver, raw GGUF) as separate subprocesses
per the two-lane isolation policy, then compares per-frame RVQ codes and
optionally the decode reconstructions.

For f16 models the EMEL codec consumes the same effective operand pipeline
as the reference (f16 conv weights + f16 im2col, bf16 K/V attention with
bf16-rounded q and softmax weights, and exact ports of ggml's vec_dot,
softmax, rope, and GELU numerics), so encode comparison is expected to be
token-exact and CI gates with --require-token-exact.
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SURFACE = "mimi_compare/v1"


def parse_codes(text: str) -> list[list[int]]:
  frames = []
  for line in text.splitlines():
    if "codes=" not in line:
      continue
    frames.append([int(x) for x in line.split("codes=")[1].strip().split(",")])
  return frames


def read_f32(path: Path) -> list[float]:
  raw = path.read_bytes()
  return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def run_lane(cmd: list[str]) -> str:
  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    sys.exit(f"lane failed ({' '.join(cmd)}):\n{result.stderr}")
  return result.stdout


def run_lane_capture(cmd: list[str]) -> tuple[str, str]:
  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    sys.exit(f"lane failed ({' '.join(cmd)}):\n{result.stderr}")
  return result.stdout, result.stderr


def parse_metric(text: str, key: str) -> float | None:
  needle = f"{key}="
  for line in text.splitlines():
    if needle not in line:
      continue
    token = line.split(needle, 1)[1].split()[0]
    try:
      return float(token)
    except ValueError:
      return None
  return None


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--emel-runner", required=True)
  parser.add_argument("--reference-driver", required=True)
  parser.add_argument("--emel-model", required=True,
                      help="enriched mimi GGUF (EMEL lane)")
  parser.add_argument("--reference-model", required=True,
                      help="raw mimi GGUF (reference lane)")
  parser.add_argument("--audio", required=True,
                      help="24 kHz mono s16 WAV")
  parser.add_argument("--n-q", type=int, default=16)
  parser.add_argument("--compare-decode", action="store_true",
                      help="also decode each lane's own codes and compare "
                      "reconstructions (PSNR between lanes)")
  parser.add_argument("--require-token-exact", action="store_true",
                      help="fail unless codes match exactly (activates once "
                      "the EMEL lane consumes the f16 operand class)")
  parser.add_argument("--min-code-match", type=float, default=0.0,
                      help="minimum code match fraction to pass [0..1]")
  parser.add_argument("--prefix-streams", type=int, default=0,
                      help="also gate the first N RVQ streams separately; "
                      "early residual layers are numerically stable across "
                      "float implementations while deep-layer codes cascade")
  parser.add_argument("--min-decode-psnr-db", type=float, default=0.0,
                      help="fail when --compare-decode measures a decode "
                           "PSNR below this floor (0 disables the gate)")
  parser.add_argument("--min-prefix-match", type=float, default=0.0,
                      help="minimum match fraction over --prefix-streams")
  parser.add_argument("--timing-audio", default=None,
                      help="24 kHz mono s16 WAV used for a measurement-only "
                      "per-frame timing pass in both lanes (each lane "
                      "self-reports its steady-state loop time)")
  parser.add_argument("--reference-label", default="reference",
                      help="reference lane display name for timing rows")
  parser.add_argument("--json-out", default=None)
  args = parser.parse_args()

  emel_out = run_lane([
      args.emel_runner, "--model", args.emel_model, "--audio", args.audio
  ])
  ref_out = run_lane([
      args.reference_driver, "encode", "--mimi", args.reference_model,
      "--audio", args.audio, "--n-q", str(args.n_q)
  ])
  emel_frames = parse_codes(emel_out)
  ref_frames = parse_codes(ref_out)

  report = {
      "surface": SURFACE,
      "emel_frames": len(emel_frames),
      "reference_frames": len(ref_frames),
      "operand_class_note":
          "emel lane consumes the reference f16/bf16 operand pipeline; "
          "encode comparison is token-exact kernel parity",
  }
  if len(emel_frames) != len(ref_frames) or not emel_frames:
    report["status"] = "fail"
    report["reason"] = "frame count mismatch or no frames"
    print(json.dumps(report, indent=2))
    return 1

  total = matched = 0
  prefix_total = prefix_matched = 0
  per_frame = []
  for emel_codes, ref_codes in zip(emel_frames, ref_frames):
    if len(emel_codes) != len(ref_codes):
      report["status"] = "fail"
      report["reason"] = "stream count mismatch"
      print(json.dumps(report, indent=2))
      return 1
    frame_match = sum(1 for a, b in zip(emel_codes, ref_codes) if a == b)
    per_frame.append(frame_match / len(emel_codes))
    matched += frame_match
    total += len(emel_codes)
    if args.prefix_streams > 0:
      prefix = min(args.prefix_streams, len(emel_codes))
      prefix_matched += sum(
          1 for a, b in zip(emel_codes[:prefix], ref_codes[:prefix]) if a == b)
      prefix_total += prefix
  report["code_match_fraction"] = matched / total
  report["per_frame_match"] = per_frame
  report["token_exact"] = matched == total
  if prefix_total > 0:
    report["prefix_streams"] = args.prefix_streams
    report["prefix_match_fraction"] = prefix_matched / prefix_total

  if args.compare_decode:
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      emel_pcm_path = tmp_path / "emel.f32"
      ref_pcm_path = tmp_path / "ref.f32"
      ref_codes_path = tmp_path / "ref_codes.txt"
      ref_codes_path.write_text(ref_out)
      run_lane([
          args.emel_runner, "--model", args.emel_model, "--audio", args.audio,
          "--decode-out", str(emel_pcm_path)
      ])
      run_lane([
          args.reference_driver, "decode", "--mimi", args.reference_model,
          "--codes", str(ref_codes_path), "--out", str(ref_pcm_path), "--n-q",
          str(args.n_q)
      ])
      emel_pcm = read_f32(emel_pcm_path)
      ref_pcm = read_f32(ref_pcm_path)
      # A truncated or extended stream from either lane must fail the run: a
      # prefix-only comparison would hide decode regressions behind the
      # encode-token gates.
      if len(emel_pcm) != len(ref_pcm):
        raise SystemExit(
            f"decode pcm length mismatch: emel={len(emel_pcm)} "
            f"reference={len(ref_pcm)}")
      n = len(emel_pcm)
      if n == 0:
        report["decode_psnr_db"] = None
      else:
        err = sum((a - b) ** 2 for a, b in zip(emel_pcm, ref_pcm)) / n
        peak = max(abs(x) for x in ref_pcm) or 1.0
        report["decode_psnr_db"] = (10.0 * math.log10(peak * peak / err)
                                    if err > 0 else float("inf"))

  if args.timing_audio:
    # Measurement-only pass: each lane self-reports its steady-state
    # per-frame loop time on a longer signal, excluding model load.
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      emel_timing_out = run_lane([
          args.emel_runner, "--model", args.emel_model,
          "--audio", args.timing_audio, "--timing",
          "--decode-out", str(tmp_path / "emel_timing.f32")
      ])
      ref_codes_out, ref_encode_err = run_lane_capture([
          args.reference_driver, "encode", "--mimi", args.reference_model,
          "--audio", args.timing_audio, "--n-q", str(args.n_q)
      ])
      ref_timing_codes = tmp_path / "ref_timing_codes.txt"
      ref_timing_codes.write_text(ref_codes_out)
      _, ref_decode_err = run_lane_capture([
          args.reference_driver, "decode", "--mimi", args.reference_model,
          "--codes", str(ref_timing_codes),
          "--out", str(tmp_path / "ref_timing.f32"), "--n-q", str(args.n_q)
      ])
    timing = {
        "emel_encode_ns_per_frame":
            parse_metric(emel_timing_out, "emel_encode_ns_per_frame"),
        "emel_decode_ns_per_frame":
            parse_metric(emel_timing_out, "emel_decode_ns_per_frame"),
        "reference_encode_ns_per_frame": None,
        "reference_decode_ns_per_frame": None,
    }
    ref_encode_ms = parse_metric(ref_encode_err, "encode_ms_per_frame")
    ref_decode_ms = parse_metric(ref_decode_err, "decode_ms_per_frame")
    if ref_encode_ms is not None:
      timing["reference_encode_ns_per_frame"] = ref_encode_ms * 1.0e6
    if ref_decode_ms is not None:
      timing["reference_decode_ns_per_frame"] = ref_decode_ms * 1.0e6
    report["timing"] = timing
    for stage in ("encode", "decode"):
      emel_ns = timing[f"emel_{stage}_ns_per_frame"]
      ref_ns = timing[f"reference_{stage}_ns_per_frame"]
      if emel_ns is None or ref_ns is None or emel_ns <= 0:
        continue
      print(f"speech_codec_mimi/{stage}_frame_timing "
            f"emel.cpp {emel_ns:.0f} ns/op, "
            f"{args.reference_label} {ref_ns:.0f} ns/op, "
            f"ratio={ref_ns / emel_ns:.3f}x")

  passed = True
  if args.require_token_exact and not report["token_exact"]:
    passed = False
    report["reason"] = "token-exact required but codes differ"
  if report["code_match_fraction"] < args.min_code_match:
    passed = False
    report["reason"] = (f"code match {report['code_match_fraction']:.3f} "
                        f"below threshold {args.min_code_match}")
  if prefix_total > 0 and report["prefix_match_fraction"] < args.min_prefix_match:
    passed = False
    report["reason"] = (
        f"prefix match {report['prefix_match_fraction']:.3f} over first "
        f"{args.prefix_streams} streams below threshold "
        f"{args.min_prefix_match}")
  if args.compare_decode and args.min_decode_psnr_db > 0.0:
    # A decoder returning silence or wrong PCM with the right length would
    # otherwise pass on the encode gates alone.
    psnr = report.get("decode_psnr_db")
    if psnr is None or psnr < args.min_decode_psnr_db:
      passed = False
      report["reason"] = (
          f"decode psnr {psnr} dB below threshold "
          f"{args.min_decode_psnr_db} dB")
  report["status"] = "pass" if passed else "fail"

  output = json.dumps(report, indent=2)
  print(output)
  if args.json_out:
    Path(args.json_out).write_text(output + "\n")
  return 0 if passed else 1


if __name__ == "__main__":
  sys.exit(main())
