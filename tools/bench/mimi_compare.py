#!/usr/bin/env python3
"""Two-lane Mimi codec comparison (surface: mimi_compare/v1).

Runs the EMEL-owned runner (mimi_emel_parity_runner, enriched GGUF) and the
reference driver (moshi_reference_driver, raw GGUF) as separate subprocesses
per the two-lane isolation policy, then compares per-frame RVQ codes and
optionally the decode reconstructions.

Current status (honest label): the reference executes ggml's f16 conv
pipeline (f16 im2col + f16 mul_mat; moshi.cpp requires f16 conv weights and
aborts on f32), while the EMEL lane computes in f32 after lossless f16->f32
canonicalization. These are different effective operand classes, so
code-match percentage is reported as a similarity metric, NOT claimed as
kernel parity. Token-exact comparison gating (--require-token-exact)
activates once the EMEL codec consumes the same f16 operand class.
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
          "reference runs ggml f16 conv pipeline; emel lane runs f32 - "
          "code match is a similarity metric, not kernel parity",
  }
  if len(emel_frames) != len(ref_frames) or not emel_frames:
    report["status"] = "fail"
    report["reason"] = "frame count mismatch or no frames"
    print(json.dumps(report, indent=2))
    return 1

  total = matched = 0
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
  report["code_match_fraction"] = matched / total
  report["per_frame_match"] = per_frame
  report["token_exact"] = matched == total

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
      n = min(len(emel_pcm), len(ref_pcm))
      if n == 0:
        report["decode_psnr_db"] = None
      else:
        err = sum((a - b) ** 2 for a, b in zip(emel_pcm[:n], ref_pcm[:n])) / n
        peak = max(abs(x) for x in ref_pcm[:n]) or 1.0
        report["decode_psnr_db"] = (10.0 * math.log10(peak * peak / err)
                                    if err > 0 else float("inf"))

  passed = True
  if args.require_token_exact and not report["token_exact"]:
    passed = False
    report["reason"] = "token-exact required but codes differ"
  if report["code_match_fraction"] < args.min_code_match:
    passed = False
    report["reason"] = (f"code match {report['code_match_fraction']:.3f} "
                        f"below threshold {args.min_code_match}")
  report["status"] = "pass" if passed else "fail"

  output = json.dumps(report, indent=2)
  print(output)
  if args.json_out:
    Path(args.json_out).write_text(output + "\n")
  return 0 if passed else 1


if __name__ == "__main__":
  sys.exit(main())
