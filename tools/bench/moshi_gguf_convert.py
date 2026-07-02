#!/usr/bin/env python3
"""Convert raw moshi.cpp GGUF caches into emel-enriched moshi GGUFs.

Raw moshi.cpp GGUF caches (e.g. Codes4Fun/personaplex-7b-v1-q4_k-GGUF) carry
zero metadata: no `general.architecture`, no hparams, no tokenizer, and any
tensor name >= 64 chars is replaced by an 8-hex CRC of the canonical name
(moshi.cpp loader.h `tensor_name`). emel refuses such files.

This converter produces the emel moshi GGUF convention:
  - `general.architecture = "moshi"`, `moshi.component = lm|mimi|voice`
  - `moshi.lm.*` / `moshi.mimi.*` / `moshi.voice.*` hparams (from the model
    config JSON for `lm`, from the pinned moshi.cpp code preset for `mimi`)
  - CRC-mangled tensor names restored to their canonical long names
  - for `lm`: the SentencePiece text tokenizer embedded as standard
    `tokenizer.ggml.*` keys so emel's generic vocab loader consumes it

Tensor payload bytes are streamed through unchanged (same order, same
alignment, same relative offsets).

stdlib-only by design, matching tools/bench/whisper_normalize_model.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# tensor dtype codes (ggml enumeration)
GGML_DTYPE_F32 = 0
GGML_DTYPE_F16 = 1
GGML_DTYPE_Q8_0 = 8
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34  # fp16 scale + 32 int8

# weight classes safe to quantize in a mimi component: the transformer and
# RVQ projections (uniform row length, k % 32 == 0, consumed as mat-vecs by
# the emel quantized kernels). Convs, norms, biases, scales, and codebooks
# stay in their float dtypes.
MIMI_QUANTIZABLE_SUFFIXES = (
    "self_attn.in_projs.0.weight",
    "self_attn.out_projs.0.weight",
    "linear1.weight",
    "linear2.weight",
)
# Measured on the real model: quantizing the RVQ projections costs no
# additional code-match quality (the flips come from transformer-latent
# perturbation), so they quantize along with the transformer projections.
MIMI_QUANTIZABLE_NAMES = (
    "mimi.quantizer.rvq_first.input_proj.weight",
    "mimi.quantizer.rvq_first.output_proj.weight",
    "mimi.quantizer.rvq_rest.input_proj.weight",
    "mimi.quantizer.rvq_rest.output_proj.weight",
)

# moshi.cpp mangles tensor names >= GGML_MAX_NAME when caching GGUFs.
GGML_MAX_NAME = 64

MIMI_PRESET = {
    "sample_rate": 24000,
    "frame_rate": 12.5,
    "card": 2048,
    "dim": 512,
    "semantic_n_q": 1,
    "transformer_num_layers": 8,
    "transformer_num_heads": 8,
    "transformer_context": 250,
    "transformer_max_period": 10000,
}

SUPPORTED_LM_VALUES = {
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
}

VOICE_FORMAT = "personaplex_prompt_v1"


@dataclass(frozen=True)
class TensorInfo:
  name: str
  dims: tuple[int, ...]
  dtype: int
  offset: int


class Reader:
  def __init__(self, data: bytes) -> None:
    self.data = data
    self.offset = 0

  def read(self, size: int) -> bytes:
    if self.offset + size > len(self.data):
      raise ValueError("unexpected EOF while parsing GGUF header")
    out = self.data[self.offset:self.offset + size]
    self.offset += size
    return out

  def u32(self) -> int:
    return struct.unpack("<I", self.read(4))[0]

  def u64(self) -> int:
    return struct.unpack("<Q", self.read(8))[0]

  def string(self) -> str:
    return self.read(self.u64()).decode("utf-8")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def mangle_name(name: str) -> str:
  """Replicate moshi.cpp loader.h tensor_name for names >= 64 chars.

  The upstream loop writes hex[(crc >> 4) & 0xf] then immediately overwrites
  it with hex[crc & 0xf] before shifting by 8, so only the low nibble of each
  byte survives and bytes 4..7 of the 32-bit CRC are always zero.
  """
  crc = zlib.crc32(name.encode("utf-8"))
  hexd = "0123456789abcdef"
  return "".join(hexd[(crc >> (8 * i)) & 0xf] for i in range(8))


def transformer_layer_candidates(base: str, num_layers: int,
                                 num_weights: int) -> list[str]:
  names: list[str] = []
  for layer in range(num_layers):
    prefix = f"{base}layers.{layer}."
    names.append(prefix + "self_attn.in_proj_weight")
    names.append(prefix + "self_attn.out_proj.weight")
    names.append(prefix + "layer_scale_1.scale")
    names.append(prefix + "layer_scale_2.scale")
    names.append(prefix + "gating.linear_in.weight")
    names.append(prefix + "gating.linear_out.weight")
    names.append(prefix + "linear1.weight")
    names.append(prefix + "linear2.weight")
    for norm in ("norm1", "norm2"):
      names.append(prefix + norm + ".alpha")
      names.append(prefix + norm + ".weight")
      names.append(prefix + norm + ".bias")
    for weight_index in range(max(1, num_weights)):
      names.append(prefix + f"self_attn.in_projs.{weight_index}.weight")
      names.append(prefix + f"self_attn.out_projs.{weight_index}.weight")
      names.append(prefix + f"gating.{weight_index}.linear_in.weight")
      names.append(prefix + f"gating.{weight_index}.linear_out.weight")
  return names


def unmangle_names(infos: list[TensorInfo],
                   candidates: list[str]) -> list[TensorInfo]:
  reverse = {mangle_name(name): name for name in candidates
             if len(name) >= GGML_MAX_NAME}
  restored: list[TensorInfo] = []
  for info in infos:
    name = info.name
    if len(name) == 8 and all(c in "0123456789abcdef" for c in name):
      if name not in reverse:
        raise ValueError(
            f"cannot restore CRC-mangled tensor name {name!r}; "
            "the schema candidate table does not cover it")
      name = reverse[name]
    restored.append(TensorInfo(name, info.dims, info.dtype, info.offset))
  return restored


def read_raw_gguf(path: Path) -> tuple[list[TensorInfo], int]:
  header = path.open("rb").read(16 * 1024 * 1024)
  reader = Reader(header)
  if reader.read(4) != GGUF_MAGIC:
    raise ValueError(f"{path} is not a GGUF file")
  version = reader.u32()
  if version != GGUF_VERSION:
    raise ValueError(f"{path}: unsupported GGUF version {version}")
  n_tensors = reader.u64()
  n_kv = reader.u64()
  if n_kv != 0:
    raise ValueError(
        f"{path} already carries {n_kv} metadata keys; expected a raw "
        "moshi.cpp GGUF cache (re-enriching is not supported)")

  infos: list[TensorInfo] = []
  for _ in range(n_tensors):
    name = reader.string()
    n_dims = reader.u32()
    if n_dims <= 0 or n_dims > 4:
      raise ValueError(f"{path}: invalid tensor rank {n_dims}")
    dims = tuple(reader.u64() for _ in range(n_dims))
    dtype = reader.u32()
    offset = reader.u64()
    infos.append(TensorInfo(name, dims, dtype, offset))

  data_start = reader.offset
  data_start += (-data_start) % GGUF_ALIGNMENT
  return infos, data_start


def detect_component(infos: list[TensorInfo]) -> str:
  prefixes = {info.name.split(".", 1)[0] for info in infos
              if not (len(info.name) == 8 and
                      all(c in "0123456789abcdef" for c in info.name))}
  for component in ("lm", "mimi", "voice"):
    if prefixes == {component}:
      return component
  raise ValueError(
      f"cannot determine moshi component from tensor prefixes {sorted(prefixes)}; "
      "expected all tensors under exactly one of lm./mimi./voice.")


def find_info(infos: list[TensorInfo], name: str) -> TensorInfo:
  for info in infos:
    if info.name == name:
      return info
  raise ValueError(f"required tensor {name!r} is missing from the source GGUF")


def count_prefixed(infos: list[TensorInfo], prefix: str) -> int:
  return sum(1 for info in infos if info.name.startswith(prefix))


def require_dims(infos: list[TensorInfo], name: str,
                 dims: tuple[int, ...]) -> None:
  info = find_info(infos, name)
  if info.dims != dims:
    raise ValueError(
        f"tensor {name!r} has dims {info.dims}, expected {dims}; "
        "the config JSON does not match the weights")


def load_lm_config(path: Path) -> dict:
  config = json.loads(path.read_text(encoding="utf-8"))
  required = [
      "card", "n_q", "dep_q", "text_card", "existing_text_padding_id", "dim",
      "num_heads", "num_layers", "hidden_scale", "causal", "context",
      "max_period", "gating", "norm", "positional_embedding", "delays",
      "cross_attention",
  ]
  missing = [key for key in required if key not in config]
  if missing:
    raise ValueError(f"{path} is missing required config keys: {missing}")

  for key in ("gating", "norm", "positional_embedding"):
    if config[key] != SUPPORTED_LM_VALUES[key]:
      raise ValueError(
          f"{path}: unsupported {key}={config[key]!r} "
          f"(only {SUPPORTED_LM_VALUES[key]!r} is supported)")
  if config.get("depformer_weights_per_step_schedule"):
    raise ValueError(
        f"{path}: depformer_weights_per_step_schedule is not supported")

  if config["dep_q"] > 0:
    dep_required = [
        "depformer_dim", "depformer_num_heads", "depformer_num_layers",
        "depformer_dim_feedforward", "depformer_context",
        "depformer_max_period", "depformer_gating", "depformer_pos_emb",
        "depformer_multi_linear", "depformer_weights_per_step",
    ]
    missing = [key for key in dep_required if key not in config]
    if missing:
      raise ValueError(f"{path} is missing depformer config keys: {missing}")
    for key in ("depformer_gating", "depformer_pos_emb"):
      if config[key] != SUPPORTED_LM_VALUES[key]:
        raise ValueError(
            f"{path}: unsupported {key}={config[key]!r} "
            f"(only {SUPPORTED_LM_VALUES[key]!r} is supported)")
    if not config["depformer_multi_linear"]:
      raise ValueError(f"{path}: depformer_multi_linear=false is unsupported")

  delays = config["delays"]
  if len(delays) != config["n_q"] + 1:
    raise ValueError(
        f"{path}: delays has {len(delays)} entries, expected n_q+1 = "
        f"{config['n_q'] + 1}")
  return config


def cross_check_lm(infos: list[TensorInfo], config: dict) -> None:
  dim = config["dim"]
  text_card = config["text_card"]
  card = config["card"]
  require_dims(infos, "lm.text_emb.weight", (dim, text_card + 1))
  require_dims(infos, "lm.text_linear.weight", (dim, text_card))
  require_dims(infos, "lm.emb.0.weight", (dim, card + 1))
  find_info(infos, "lm.out_norm.alpha")

  emb_count = count_prefixed(infos, "lm.emb.")
  if emb_count != config["n_q"]:
    raise ValueError(
        f"weights carry {emb_count} lm.emb.* tensors, config n_q={config['n_q']}")
  layer_count = count_prefixed(infos, "lm.transformer.layers.")
  if count_prefixed(infos, f"lm.transformer.layers.{config['num_layers'] - 1}.") == 0:
    raise ValueError(
        f"weights are missing lm.transformer.layers.{config['num_layers'] - 1}.* "
        f"(config num_layers={config['num_layers']}, "
        f"{layer_count} layer tensors present)")
  if config["dep_q"] > 0:
    if count_prefixed(infos, "lm.depformer_in.") != config["dep_q"]:
      raise ValueError("lm.depformer_in.* count does not match config dep_q")
    if count_prefixed(infos, "lm.linears.") != config["dep_q"]:
      raise ValueError("lm.linears.* count does not match config dep_q")
    depformer_last = config["depformer_num_layers"] - 1
    if count_prefixed(infos, f"lm.depformer.layers.{depformer_last}.") == 0:
      raise ValueError(
          "weights are missing the last depformer layer per config "
          f"depformer_num_layers={config['depformer_num_layers']}")


def cross_check_mimi(infos: list[TensorInfo], n_q: int, preset: dict) -> dict:
  first = find_info(
      infos, "mimi.quantizer.rvq_first.vq.layers.0._codebook.embedding")
  if len(first.dims) != 2 or first.dims[1] != preset["card"]:
    raise ValueError(
        f"semantic codebook dims {first.dims} do not match preset card "
        f"{preset['card']}")
  rest_count = sum(
      1 for info in infos
      if info.name.startswith("mimi.quantizer.rvq_rest.vq.layers.") and
      info.name.endswith("._codebook.embedding"))
  if rest_count < n_q - preset["semantic_n_q"]:
    raise ValueError(
        f"weights carry {rest_count} acoustic codebooks, need at least "
        f"{n_q - preset['semantic_n_q']} for n_q={n_q}")
  for family in ("mimi.encoder.", "mimi.encoder_transformer.",
                 "mimi.downsample.", "mimi.upsample.",
                 "mimi.decoder_transformer.", "mimi.decoder."):
    if count_prefixed(infos, family) == 0:
      raise ValueError(f"weights are missing the {family}* family")
  return dict(preset, n_q=n_q, codebook_dim=first.dims[0])


def cross_check_voice(infos: list[TensorInfo]) -> None:
  find_info(infos, "voice.embeddings")
  find_info(infos, "voice.cache")


# --- minimal SentencePiece ModelProto wire parser (stdlib only) -------------

SP_PIECE_TYPE_NORMAL = 1


def _varint(data: bytes, offset: int) -> tuple[int, int]:
  result = 0
  shift = 0
  while True:
    byte = data[offset]
    offset += 1
    result |= (byte & 0x7F) << shift
    if not byte & 0x80:
      return result, offset
    shift += 7
    if shift > 63:
      raise ValueError("varint overflow in SentencePiece model")


def _fields(data: bytes):
  offset = 0
  while offset < len(data):
    tag, offset = _varint(data, offset)
    field, wire = tag >> 3, tag & 7
    if wire == 0:
      value, offset = _varint(data, offset)
    elif wire == 1:
      value = data[offset:offset + 8]
      offset += 8
    elif wire == 2:
      size, offset = _varint(data, offset)
      value = data[offset:offset + size]
      offset += size
    elif wire == 5:
      value = data[offset:offset + 4]
      offset += 4
    else:
      raise ValueError(f"unsupported protobuf wire type {wire}")
    yield field, wire, value


def parse_sentencepiece_model(path: Path) -> dict:
  data = path.read_bytes()
  pieces: list[tuple[str, float, int]] = []
  trainer: dict[str, int] = {}
  for field, wire, value in _fields(data):
    if field == 1 and wire == 2:
      piece = ""
      score = 0.0
      ptype = SP_PIECE_TYPE_NORMAL
      for pfield, pwire, pvalue in _fields(value):
        if pfield == 1 and pwire == 2:
          piece = pvalue.decode("utf-8")
        elif pfield == 2 and pwire == 5:
          score = struct.unpack("<f", pvalue)[0]
        elif pfield == 3 and pwire == 0:
          ptype = pvalue
      pieces.append((piece, score, ptype))
    elif field == 2 and wire == 2:
      ids = {3: "model_type", 40: "unk_id", 41: "bos_id", 42: "eos_id",
             43: "pad_id"}
      for tfield, twire, tvalue in _fields(value):
        if tfield in ids and twire == 0:
          trainer[ids[tfield]] = tvalue

  if not pieces:
    raise ValueError(f"{path} contains no SentencePiece pieces")
  if trainer.get("model_type", 1) != 1:
    raise ValueError(
        f"{path}: only unigram SentencePiece models are supported "
        f"(model_type={trainer.get('model_type')})")
  return {"pieces": pieces, "trainer": trainer}


# --- GGUF writing ------------------------------------------------------------


def write_string(out: bytearray, value: str) -> None:
  encoded = value.encode("utf-8")
  out.extend(struct.pack("<Q", len(encoded)))
  out.extend(encoded)


class KvWriter:
  def __init__(self) -> None:
    self.buffer = bytearray()
    self.count = 0

  def string(self, key: str, value: str) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<I", GGUF_TYPE_STRING))
    write_string(self.buffer, value)
    self.count += 1

  def u32(self, key: str, value: int) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<II", GGUF_TYPE_UINT32, value))
    self.count += 1

  def i32(self, key: str, value: int) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<Ii", GGUF_TYPE_INT32, value))
    self.count += 1

  def f32(self, key: str, value: float) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<If", GGUF_TYPE_FLOAT32, value))
    self.count += 1

  def boolean(self, key: str, value: bool) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<IB", GGUF_TYPE_BOOL, 1 if value else 0))
    self.count += 1

  def i32_array(self, key: str, values: list[int]) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_INT32,
                                   len(values)))
    for value in values:
      self.buffer.extend(struct.pack("<i", value))
    self.count += 1

  def f32_array(self, key: str, values: list[float]) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32,
                                   len(values)))
    for value in values:
      self.buffer.extend(struct.pack("<f", value))
    self.count += 1

  def string_array(self, key: str, values: list[str]) -> None:
    write_string(self.buffer, key)
    self.buffer.extend(struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING,
                                   len(values)))
    for value in values:
      write_string(self.buffer, value)
    self.count += 1


def append_lm_kv(kv: KvWriter, config: dict) -> None:
  dim_feedforward = round(config["hidden_scale"] * config["dim"])
  kv.u32("moshi.lm.card", config["card"])
  kv.u32("moshi.lm.n_q", config["n_q"])
  kv.u32("moshi.lm.dep_q", config["dep_q"])
  kv.u32("moshi.lm.text_card", config["text_card"])
  kv.u32("moshi.lm.existing_text_padding_id",
         config["existing_text_padding_id"])
  kv.u32("moshi.lm.dim", config["dim"])
  kv.u32("moshi.lm.num_layers", config["num_layers"])
  kv.u32("moshi.lm.num_heads", config["num_heads"])
  kv.u32("moshi.lm.context", config["context"])
  kv.u32("moshi.lm.max_period", config["max_period"])
  kv.u32("moshi.lm.dim_feedforward", dim_feedforward)
  kv.string("moshi.lm.gating", config["gating"])
  kv.string("moshi.lm.norm", config["norm"])
  kv.string("moshi.lm.positional_embedding", config["positional_embedding"])
  kv.boolean("moshi.lm.causal", config["causal"])
  kv.boolean("moshi.lm.cross_attention", config["cross_attention"])
  kv.boolean("moshi.lm.demux_second_stream",
             config.get("demux_second_stream", False))
  kv.i32_array("moshi.lm.delays", list(config["delays"]))
  kv.u32("moshi.lm.extra_heads.num_heads",
         config.get("extra_heads_num_heads", 0))
  if "model_type" in config:
    kv.string("moshi.lm.model_type", config["model_type"])
  if config["dep_q"] > 0:
    kv.u32("moshi.lm.depformer.dim", config["depformer_dim"])
    kv.u32("moshi.lm.depformer.num_heads", config["depformer_num_heads"])
    kv.u32("moshi.lm.depformer.num_layers", config["depformer_num_layers"])
    kv.u32("moshi.lm.depformer.dim_feedforward",
           config["depformer_dim_feedforward"])
    kv.u32("moshi.lm.depformer.context", config["depformer_context"])
    kv.u32("moshi.lm.depformer.max_period", config["depformer_max_period"])
    kv.string("moshi.lm.depformer.gating", config["depformer_gating"])
    kv.string("moshi.lm.depformer.pos_emb", config["depformer_pos_emb"])
    kv.boolean("moshi.lm.depformer.multi_linear",
               config["depformer_multi_linear"])
    kv.boolean("moshi.lm.depformer.weights_per_step",
               config["depformer_weights_per_step"])
    kv.u32("moshi.lm.depformer.low_rank_embeddings",
           config.get("depformer_low_rank_embeddings") or 0)


def append_tokenizer_kv(kv: KvWriter, tokenizer: dict) -> None:
  pieces = tokenizer["pieces"]
  trainer = tokenizer["trainer"]
  kv.string("tokenizer.ggml.model", "llama")
  kv.string_array("tokenizer.ggml.tokens", [piece for piece, _, _ in pieces])
  kv.f32_array("tokenizer.ggml.scores", [score for _, score, _ in pieces])
  kv.i32_array("tokenizer.ggml.token_type", [ptype for _, _, ptype in pieces])
  for key, gguf_key in (("bos_id", "tokenizer.ggml.bos_token_id"),
                        ("eos_id", "tokenizer.ggml.eos_token_id"),
                        ("unk_id", "tokenizer.ggml.unknown_token_id"),
                        ("pad_id", "tokenizer.ggml.padding_token_id")):
    value = trainer.get(key, -1)
    if value >= 0:
      kv.u32(gguf_key, value)
  kv.boolean("tokenizer.ggml.add_bos_token", False)
  kv.boolean("tokenizer.ggml.add_eos_token", False)


def append_mimi_kv(kv: KvWriter, mimi: dict) -> None:
  kv.u32("moshi.mimi.sample_rate", mimi["sample_rate"])
  kv.f32("moshi.mimi.frame_rate", mimi["frame_rate"])
  kv.u32("moshi.mimi.n_q", mimi["n_q"])
  kv.u32("moshi.mimi.card", mimi["card"])
  kv.u32("moshi.mimi.dim", mimi["dim"])
  kv.u32("moshi.mimi.semantic_n_q", mimi["semantic_n_q"])
  kv.u32("moshi.mimi.codebook_dim", mimi["codebook_dim"])
  kv.u32("moshi.mimi.transformer.num_layers", mimi["transformer_num_layers"])
  kv.u32("moshi.mimi.transformer.num_heads", mimi["transformer_num_heads"])
  kv.u32("moshi.mimi.transformer.context", mimi["transformer_context"])
  kv.u32("moshi.mimi.transformer.max_period",
         mimi["transformer_max_period"])


def tensor_data_bytes(info: TensorInfo) -> int:
  elements = 1
  for dim in info.dims:
    elements *= dim
  if info.dtype == GGML_DTYPE_F32:
    return elements * 4
  if info.dtype == GGML_DTYPE_F16:
    return elements * 2
  if info.dtype == GGML_DTYPE_Q8_0:
    return elements // Q8_0_BLOCK * Q8_0_BLOCK_BYTES
  raise ValueError(f"unsupported dtype {info.dtype} for {info.name}")


def is_mimi_quantizable(info: TensorInfo) -> bool:
  # effective row length: conv1x1 projections store [1, in, out], linear
  # projections store [in, out]; blocks must not straddle rows. 1-D tensors
  # (biases hit by the suffix match on other models) are never quantized.
  if len(info.dims) < 2:
    return False
  row = info.dims[0] if info.dims[0] > 1 else info.dims[1]
  return (info.name in MIMI_QUANTIZABLE_NAMES or
          info.name.endswith(MIMI_QUANTIZABLE_SUFFIXES)) and \
      row % Q8_0_BLOCK == 0 and \
      info.dtype in (GGML_DTYPE_F32, GGML_DTYPE_F16)


def quantize_q8_0(values) -> bytes:
  """Row-blocked q8_0 exactly matching the emel/ggml reference encoder:
  per 32-value block, d = amax / 127 (stored fp16), q = clamp(round-half-
  away(x / d), -127, 127)."""
  import numpy as np
  blocks = values.astype(np.float32).reshape(-1, Q8_0_BLOCK)
  amax = np.abs(blocks).max(axis=1)
  d = amax / 127.0
  inv_d = np.where(d != 0.0, 1.0 / np.where(d == 0.0, 1.0, d), 0.0)
  scaled = blocks * inv_d[:, None]
  quants = np.clip(
      np.floor(np.abs(scaled) + 0.5) * np.sign(scaled), -127, 127
  ).astype(np.int8)
  out = np.zeros((blocks.shape[0], Q8_0_BLOCK_BYTES), dtype=np.uint8)
  out[:, 0:2] = d.astype(np.float16)[:, None].view(np.uint8)
  out[:, 2:] = quants.view(np.uint8)
  return out.tobytes()


def quantize_mimi_tensors(
    source: Path, infos: list[TensorInfo], data_start: int,
    quantize: str) -> tuple[list[TensorInfo], list[bytes], int]:
  """Returns (new infos with rebuilt offsets, per-tensor payloads, count)."""
  if quantize != "q8_0":
    raise ValueError(f"unsupported --quantize type: {quantize}")
  import numpy as np
  data = source.read_bytes()
  payloads: list[bytes] = []
  new_infos: list[TensorInfo] = []
  offset = 0
  quantized = 0
  for info in infos:
    raw = data[data_start + info.offset:
               data_start + info.offset + tensor_data_bytes(info)]
    dims = info.dims
    if is_mimi_quantizable(info):
      dt = np.float32 if info.dtype == GGML_DTYPE_F32 else np.float16
      payload = quantize_q8_0(np.frombuffer(raw, dtype=dt))
      dtype = GGML_DTYPE_Q8_0
      quantized += 1
      if dims[0] == 1:
        # conv1x1 projections store [1, in, out]; drop the leading unit dim
        # so the quantized row length (ne0) is the real contraction length
        dims = dims[1:]
    else:
      payload = raw
      dtype = info.dtype
    payloads.append(payload)
    new_infos.append(
        TensorInfo(name=info.name, dims=dims, dtype=dtype, offset=offset))
    offset += len(payload)
    pad = (-offset) % GGUF_ALIGNMENT
    offset += pad
    payloads.append(b"\0" * pad)
  return new_infos, payloads, quantized


def write_rewritten(output: Path, infos: list[TensorInfo],
                    payloads: list[bytes], kv: KvWriter) -> None:
  out = bytearray()
  out.extend(GGUF_MAGIC)
  out.extend(struct.pack("<IQQ", GGUF_VERSION, len(infos), kv.count))
  out.extend(kv.buffer)
  for info in infos:
    write_string(out, info.name)
    out.extend(struct.pack("<I", len(info.dims)))
    for dim in info.dims:
      out.extend(struct.pack("<Q", dim))
    out.extend(struct.pack("<IQ", info.dtype, info.offset))
  out.extend(b"\0" * ((-len(out)) % GGUF_ALIGNMENT))
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("wb") as dst:
    dst.write(out)
    for payload in payloads:
      dst.write(payload)


def write_enriched(source: Path, output: Path, infos: list[TensorInfo],
                   data_start: int, kv: KvWriter) -> None:
  out = bytearray()
  out.extend(GGUF_MAGIC)
  out.extend(struct.pack("<IQQ", GGUF_VERSION, len(infos), kv.count))
  out.extend(kv.buffer)
  for info in infos:
    write_string(out, info.name)
    out.extend(struct.pack("<I", len(info.dims)))
    for dim in info.dims:
      out.extend(struct.pack("<Q", dim))
    out.extend(struct.pack("<IQ", info.dtype, info.offset))
  out.extend(b"\0" * ((-len(out)) % GGUF_ALIGNMENT))

  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("wb") as dst, source.open("rb") as src:
    dst.write(out)
    src.seek(data_start)
    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
      dst.write(chunk)


def convert(source: Path, output: Path, config_path: Path | None,
            tokenizer_path: Path | None,
            mimi_preset_path: Path | None = None,
            quantize: str | None = None) -> dict:
  infos, data_start = read_raw_gguf(source)
  component = detect_component(infos)
  if quantize is not None and component != "mimi":
    raise ValueError("--quantize currently supports mimi components only")

  kv = KvWriter()
  kv.string("general.architecture", "moshi")
  kv.u32("general.alignment", GGUF_ALIGNMENT)
  kv.string("moshi.component", component)

  if component == "lm":
    if config_path is None:
      raise ValueError("--config is required for lm components")
    if tokenizer_path is None:
      raise ValueError("--tokenizer is required for lm components")
    config = load_lm_config(config_path)
    num_weights = config["dep_q"] if config.get("depformer_multi_linear") else 1
    candidates = transformer_layer_candidates(
        "lm.transformer.", config["num_layers"], 1)
    candidates += transformer_layer_candidates(
        "lm.depformer.", config.get("depformer_num_layers", 0), num_weights)
    infos = unmangle_names(infos, candidates)
    cross_check_lm(infos, config)
    append_lm_kv(kv, config)
    append_tokenizer_kv(kv, parse_sentencepiece_model(tokenizer_path))
  elif component == "mimi":
    preset = dict(MIMI_PRESET)
    if mimi_preset_path is not None:
      preset.update(json.loads(mimi_preset_path.read_text(encoding="utf-8")))
    n_q = 16
    if config_path is not None:
      n_q = json.loads(config_path.read_text(encoding="utf-8"))["n_q"]
    layers = preset["transformer_num_layers"]
    candidates = transformer_layer_candidates(
        "mimi.encoder_transformer.transformer.", layers, 1)
    candidates += transformer_layer_candidates(
        "mimi.decoder_transformer.transformer.", layers, 1)
    infos = unmangle_names(infos, candidates)
    append_mimi_kv(kv, cross_check_mimi(infos, n_q, preset))
    if quantize is not None:
      kv.string("moshi.mimi.quantization", quantize)
  else:
    infos = unmangle_names(infos, [])
    cross_check_voice(infos)
    kv.string("moshi.voice.format", VOICE_FORMAT)

  quantized_count = 0
  if quantize is not None:
    infos, payloads, quantized_count = quantize_mimi_tensors(
        source, infos, data_start, quantize)
    write_rewritten(output, infos, payloads, kv)
  else:
    write_enriched(source, output, infos, data_start, kv)
  manifest = {
      "component": component,
      "source_model": str(source),
      "source_sha256": sha256_file(source),
      "enriched_model": str(output),
      "enriched_sha256": sha256_file(output),
      "converter": "tools/bench/moshi_gguf_convert.py",
  }
  if quantize is not None:
    manifest["quantization"] = quantize
    manifest["quantized_tensors"] = quantized_count
  return manifest


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True,
                      help="raw moshi.cpp GGUF cache")
  parser.add_argument("--output", type=Path, required=True,
                      help="enriched emel moshi GGUF to write")
  parser.add_argument("--config", type=Path, default=None,
                      help="moshi config JSON (personaplex-config.json); "
                           "required for lm components")
  parser.add_argument("--tokenizer", type=Path, default=None,
                      help="SentencePiece .model file; required for lm "
                           "components")
  parser.add_argument("--mimi-preset", type=Path, default=None,
                      help="optional JSON overriding the built-in mimi "
                           "preset (used by tiny test fixtures)")
  parser.add_argument("--manifest", type=Path, default=None,
                      help="optional provenance manifest JSON to write")
  parser.add_argument("--quantize", choices=("q8_0",), default=None,
                      help="quantize the mimi projection weight classes "
                           "(transformer + RVQ projections) to the given "
                           "block format; requires numpy")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  manifest = convert(args.source, args.output, args.config, args.tokenizer,
                     args.mimi_preset, args.quantize)
  if args.manifest is not None:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
  print(json.dumps(manifest, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
