#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_VERSION = 3
GGUF_ALIGNMENT = 32
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_Q8_0 = 8


@dataclass(frozen=True)
class TensorRecord:
  name: str
  dims: tuple[int, ...]
  dtype: int
  data: bytes


class Reader:
  def __init__(self, data: bytes) -> None:
    self.data = data
    self.offset = 0

  def read(self, size: int) -> bytes:
    if size < 0:
      raise ValueError("negative read size")
    if self.offset + size > len(self.data):
      raise ValueError("unexpected EOF")
    out = self.data[self.offset:self.offset + size]
    self.offset += size
    return out

  def u32(self) -> int:
    return struct.unpack_from("<I", self.read(4))[0]

  def i32(self) -> int:
    return struct.unpack_from("<i", self.read(4))[0]


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def dtype_bytes(dtype: int, dims: tuple[int, ...]) -> int:
  elements = 1
  for dim in dims:
    elements *= dim
  if dtype == DTYPE_F32:
    return elements * 4
  if dtype == DTYPE_F16:
    return elements * 2
  if dtype == DTYPE_Q8_0:
    if not dims or dims[0] % 32 != 0:
      raise ValueError(f"q8_0 tensor has invalid first dimension: {dims}")
    return (elements // 32) * 34
  raise ValueError(f"unsupported tensor dtype: {dtype}")


def canonical_name(name: str) -> str:
  replacements = {
    "encoder.positional_embedding": "model.encoder.embed_positions.weight",
    "encoder.conv1.weight": "model.encoder.conv1.weight",
    "encoder.conv1.bias": "model.encoder.conv1.bias",
    "encoder.conv2.weight": "model.encoder.conv2.weight",
    "encoder.conv2.bias": "model.encoder.conv2.bias",
    "encoder.ln_post.weight": "model.encoder.layer_norm.weight",
    "encoder.ln_post.bias": "model.encoder.layer_norm.bias",
    "decoder.positional_embedding": "model.decoder.embed_positions.weight",
    "decoder.token_embedding.weight": "model.decoder.embed_tokens.weight",
    "decoder.ln.weight": "model.decoder.layer_norm.weight",
    "decoder.ln.bias": "model.decoder.layer_norm.bias",
  }
  if name in replacements:
    return replacements[name]

  for block in range(4):
    encoder_prefix = f"encoder.blocks.{block}."
    decoder_prefix = f"decoder.blocks.{block}."
    cross_prefix = f"decoder.blocks.{block}.cross_attn."
    if name.startswith(encoder_prefix):
      suffix = name[len(encoder_prefix):]
      return "model.encoder.layers.%d.%s" % (block, canonical_encoder_suffix(suffix))
    if name.startswith(cross_prefix):
      suffix = name[len(cross_prefix):]
      return "model.decoder.layers.%d.encoder_attn.%s" % (
          block, canonical_attention_suffix(suffix))
    if name.startswith(decoder_prefix):
      suffix = name[len(decoder_prefix):]
      return "model.decoder.layers.%d.%s" % (block, canonical_decoder_suffix(suffix))

  raise ValueError(f"unsupported Whisper tensor name: {name}")


def canonical_encoder_suffix(suffix: str) -> str:
  return {
    "attn_ln.weight": "self_attn_layer_norm.weight",
    "attn_ln.bias": "self_attn_layer_norm.bias",
    "attn.query.weight": "self_attn.q_proj.weight",
    "attn.query.bias": "self_attn.q_proj.bias",
    "attn.key.weight": "self_attn.k_proj.weight",
    "attn.value.weight": "self_attn.v_proj.weight",
    "attn.value.bias": "self_attn.v_proj.bias",
    "attn.out.weight": "self_attn.out_proj.weight",
    "attn.out.bias": "self_attn.out_proj.bias",
    "mlp_ln.weight": "final_layer_norm.weight",
    "mlp_ln.bias": "final_layer_norm.bias",
    "mlp.0.weight": "fc1.weight",
    "mlp.0.bias": "fc1.bias",
    "mlp.2.weight": "fc2.weight",
    "mlp.2.bias": "fc2.bias",
  }[suffix]


def canonical_decoder_suffix(suffix: str) -> str:
  if suffix.startswith("cross_attn_ln."):
    return "encoder_attn_layer_norm." + suffix[len("cross_attn_ln."):]
  return {
    "attn_ln.weight": "self_attn_layer_norm.weight",
    "attn_ln.bias": "self_attn_layer_norm.bias",
    "attn.query.weight": "self_attn.q_proj.weight",
    "attn.query.bias": "self_attn.q_proj.bias",
    "attn.key.weight": "self_attn.k_proj.weight",
    "attn.value.weight": "self_attn.v_proj.weight",
    "attn.value.bias": "self_attn.v_proj.bias",
    "attn.out.weight": "self_attn.out_proj.weight",
    "attn.out.bias": "self_attn.out_proj.bias",
    "mlp_ln.weight": "final_layer_norm.weight",
    "mlp_ln.bias": "final_layer_norm.bias",
    "mlp.0.weight": "fc1.weight",
    "mlp.0.bias": "fc1.bias",
    "mlp.2.weight": "fc2.weight",
    "mlp.2.bias": "fc2.bias",
  }[suffix]


def canonical_attention_suffix(suffix: str) -> str:
  return {
    "query.weight": "q_proj.weight",
    "query.bias": "q_proj.bias",
    "key.weight": "k_proj.weight",
    "value.weight": "v_proj.weight",
    "value.bias": "v_proj.bias",
    "out.weight": "out_proj.weight",
    "out.bias": "out_proj.bias",
  }[suffix]


def canonical_dims(name: str, dims: tuple[int, ...]) -> tuple[int, ...]:
  if name.endswith(".bias") and len(dims) == 2 and dims[0] == 1:
    return (dims[1],)
  if name.endswith(".weight") and len(dims) == 2 and dims[0] == 1:
    return (dims[1],)
  return dims


def read_source(path: Path) -> tuple[dict[str, int], list[TensorRecord]]:
  reader = Reader(path.read_bytes())
  if reader.read(4) != b"lmgg":
    raise ValueError(f"{path} is not a whisper.cpp-compatible Whisper binary")

  hparams = {
    "n_vocab": reader.i32(),
    "n_audio_ctx": reader.i32(),
    "n_audio_state": reader.i32(),
    "n_audio_head": reader.i32(),
    "n_audio_layer": reader.i32(),
    "n_text_ctx": reader.i32(),
    "n_text_state": reader.i32(),
    "n_text_head": reader.i32(),
    "n_text_layer": reader.i32(),
    "n_mels": reader.i32(),
    "ftype": reader.i32(),
  }
  if hparams["n_audio_layer"] != 4 or hparams["n_text_layer"] != 4:
    raise ValueError("only Whisper tiny is supported by this parity normalizer")

  n_mel = reader.i32()
  n_fft = reader.i32()
  mel_bytes = reader.read(n_mel * n_fft * 4)
  tensors = [TensorRecord("mel_filters", (n_fft, n_mel), DTYPE_F32, mel_bytes)]

  vocab_size = reader.i32()
  for _ in range(vocab_size):
    token_len = reader.u32()
    reader.read(token_len)

  while reader.offset < len(reader.data):
    if len(reader.data) - reader.offset < 12:
      raise ValueError("trailing partial tensor header")
    n_dims = reader.i32()
    name_len = reader.i32()
    dtype = reader.i32()
    if n_dims <= 0 or n_dims > 4 or name_len <= 0:
      raise ValueError("invalid tensor header")
    dims = tuple(reader.i32() for _ in range(n_dims))
    name = reader.read(name_len).decode("utf-8")
    data = reader.read(dtype_bytes(dtype, dims))
    mapped_name = canonical_name(name)
    tensors.append(TensorRecord(mapped_name, canonical_dims(mapped_name, dims), dtype, data))

  return hparams, tensors


def write_string(out: bytearray, value: str) -> None:
  encoded = value.encode("utf-8")
  out.extend(struct.pack("<Q", len(encoded)))
  out.extend(encoded)


def write_kv_string(out: bytearray, key: str, value: str) -> None:
  write_string(out, key)
  out.extend(struct.pack("<I", GGUF_TYPE_STRING))
  write_string(out, value)


def write_kv_u32(out: bytearray, key: str, value: int) -> None:
  write_string(out, key)
  out.extend(struct.pack("<I", GGUF_TYPE_UINT32))
  out.extend(struct.pack("<I", value))


def pad(out: bytearray, alignment: int = GGUF_ALIGNMENT) -> None:
  padding = (-len(out)) % alignment
  out.extend(b"\0" * padding)


def write_gguf(path: Path, hparams: dict[str, int], tensors: list[TensorRecord]) -> None:
  kv_count = 4
  out = bytearray()
  out.extend(b"GGUF")
  out.extend(struct.pack("<IQQ", GGUF_VERSION, len(tensors), kv_count))
  write_kv_string(out, "general.architecture", "whisper")
  write_kv_u32(out, "general.alignment", GGUF_ALIGNMENT)
  write_kv_u32(out, "whisper.n_mels", hparams["n_mels"])
  write_kv_u32(out, "whisper.n_vocab", hparams["n_vocab"])

  tensor_offset = 0
  for tensor in tensors:
    write_string(out, tensor.name)
    out.extend(struct.pack("<I", len(tensor.dims)))
    for dim in tensor.dims:
      out.extend(struct.pack("<Q", dim))
    out.extend(struct.pack("<IQ", tensor.dtype, tensor_offset))
    tensor_offset += dtype_bytes(tensor.dtype, tensor.dims)
    tensor_offset += (-tensor_offset) % GGUF_ALIGNMENT

  pad(out)
  for tensor in tensors:
    out.extend(tensor.data)
    pad(out)

  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(out)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  hparams, tensors = read_source(args.source)
  write_gguf(args.output, hparams, tensors)
  args.manifest.parent.mkdir(parents=True, exist_ok=True)
  args.manifest.write_text(
      "{\n"
      f"  \"source_model\": \"{args.source}\",\n"
      f"  \"source_sha256\": \"{sha256_file(args.source)}\",\n"
      f"  \"normalized_model\": \"{args.output}\",\n"
      f"  \"normalized_sha256\": \"{sha256_file(args.output)}\",\n"
      "  \"normalizer\": \"tools/bench/whisper_normalize_model.py\"\n"
      "}\n",
      encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
