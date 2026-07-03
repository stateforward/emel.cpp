#!/usr/bin/env python3
"""PersonaPlex-MLX Mimi reference driver (surface: mimi_compare/v1).

Reference-lane-only code: drives the pinned personaplex-mlx MLX Mimi codec
(personaplex_mlx.models.mimi, mimi_202407 config) as a separate process and
mirrors moshi_reference_driver's frame-level CLI, so
tools/bench/mimi_compare.py consumes either reference lane unchanged:

  personaplex_mlx_mimi_driver.py encode --mimi <safetensors> --audio <wav> \
      [--n-q N]
  personaplex_mlx_mimi_driver.py decode --mimi <safetensors> --codes <txt> \
      --out <f32> [--n-q N]

Encode prints one `frame=<n> codes=<c0>,<c1>,...` line per 1920-sample frame
(24 kHz / 12.5 Hz), truncating any trailing partial frame exactly like the
moshi.cpp reference driver. Decode reads that format and appends one f32 LE
frame of PCM per code frame to --out.

Runs inside the venv created by scripts/setup_personaplex_mlx_reference.sh
(pinned via tools/bench/personaplex_mlx_ref.txt). Never imported by the EMEL
lane; two-lane isolation per AGENTS.md.
"""

import argparse
import sys
import time
import wave

import numpy as np

import mlx.core as mx
from personaplex_mlx import models

FRAME_SIZE = 1920  # 24 kHz sample rate / 12.5 Hz frame rate

# The e351c8d8-checkpoint125 safetensors carries the full 32-codebook
# quantizer; build the model at checkpoint size so load stays strict, then
# slice/feed the requested --n-q codes. Residual VQ layers only depend on
# earlier layers, and the split quantizer decodes by input code count, so
# the first n_q codes are identical to an n_q-configured model's.
CHECKPOINT_NUM_CODEBOOKS = 32


def load_wav_24khz_mono_s16(path: str) -> np.ndarray:
  with wave.open(path, "rb") as wav:
    if (wav.getframerate() != 24000 or wav.getnchannels() != 1 or
        wav.getsampwidth() != 2):
      sys.exit(f"error: need 24 kHz mono s16 wav: {path}")
    raw = wav.readframes(wav.getnframes())
  return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_mimi(weights: str):
  mimi = models.mimi.Mimi(models.mimi.mimi_202407(CHECKPOINT_NUM_CODEBOOKS))
  mimi.load_pytorch_weights(weights, strict=True)
  # warmup() runs a full (non-streaming) encode/decode to materialize the
  # graph; reset the streaming state it leaves behind before stepping.
  mimi.warmup()
  mimi.reset_all()
  return mimi


def parse_codes_file(path: str, n_q: int) -> list[list[int]]:
  frames = []
  for line in open(path, "r", encoding="utf-8"):
    if "codes=" not in line:
      continue
    codes = [int(x) for x in line.split("codes=")[1].strip().split(",")]
    if len(codes) != n_q:
      sys.exit(f"error: expected {n_q} codes per frame, got {len(codes)}")
    frames.append(codes)
  if not frames:
    sys.exit(f"error: no code frames in {path}")
  return frames


def run_encode(args: argparse.Namespace) -> int:
  pcm = load_wav_24khz_mono_s16(args.audio)
  frames = len(pcm) // FRAME_SIZE
  if frames == 0:
    sys.exit("error: audio shorter than one frame")
  if args.n_q > CHECKPOINT_NUM_CODEBOOKS:
    sys.exit(f"error: --n-q {args.n_q} exceeds checkpoint codebooks "
             f"{CHECKPOINT_NUM_CODEBOOKS}")
  mimi = build_mimi(args.mimi)
  started = time.perf_counter()
  for index in range(frames):
    chunk = mx.array(
        pcm[index * FRAME_SIZE:(index + 1) * FRAME_SIZE]).reshape(
            1, 1, FRAME_SIZE)
    codes = mimi.encode_step(chunk)
    mx.eval(codes)
    if (codes.shape[0] != 1 or codes.shape[1] != CHECKPOINT_NUM_CODEBOOKS or
        codes.shape[2] != 1):
      sys.exit(f"error: unexpected encode_step shape {codes.shape} "
               f"at frame {index}")
    row = ",".join(str(int(codes[0, stream, 0])) for stream in range(args.n_q))
    print(f"frame={index} codes={row}")
  elapsed_ms = (time.perf_counter() - started) * 1000.0
  print(f"encoded {frames} frames (frame_size={FRAME_SIZE}) "
        f"encode_ms_per_frame={elapsed_ms / frames:.3f}",
        file=sys.stderr)
  return 0


def run_decode(args: argparse.Namespace) -> int:
  frames = parse_codes_file(args.codes, args.n_q)
  mimi = build_mimi(args.mimi)
  started = time.perf_counter()
  with open(args.out, "wb") as out:
    for codes in frames:
      step = mx.array(codes, dtype=mx.int32).reshape(1, args.n_q, 1)
      pcm = mimi.decode_step(step)
      mx.eval(pcm)
      if pcm.shape[0] != 1 or pcm.shape[1] != 1 or pcm.shape[2] != FRAME_SIZE:
        sys.exit(f"error: unexpected decode_step shape {pcm.shape}")
      out.write(np.asarray(pcm[0, 0]).astype(np.float32).tobytes())
  elapsed_ms = (time.perf_counter() - started) * 1000.0
  print(f"decoded {len(frames)} frames (frame_size={FRAME_SIZE}) "
        f"decode_ms_per_frame={elapsed_ms / len(frames):.3f}",
        file=sys.stderr)
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)

  encode = sub.add_parser("encode")
  encode.add_argument("--mimi", required=True, help="mimi safetensors weights")
  encode.add_argument("--audio", required=True, help="24 kHz mono s16 wav")
  encode.add_argument("--n-q", type=int, default=16)

  decode = sub.add_parser("decode")
  decode.add_argument("--mimi", required=True, help="mimi safetensors weights")
  decode.add_argument("--codes", required=True, help="frame=/codes= text file")
  decode.add_argument("--out", required=True, help="f32 LE PCM output path")
  decode.add_argument("--n-q", type=int, default=16)

  args = parser.parse_args()
  if args.command == "encode":
    return run_encode(args)
  return run_decode(args)


if __name__ == "__main__":
  sys.exit(main())
