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
  - optionally for `lm`: selected Q4_K mat-vec weights rewritten to EMEL's
    packed aarch64 CPU layout, exposed in GGUF as tensor dtype metadata

Tensor payload bytes are streamed through unchanged unless an explicit rewrite
option is selected.

stdlib-only by design, matching tools/bench/whisper_normalize_model.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
GGML_DTYPE_Q4_0 = 2
GGML_DTYPE_Q4_1 = 3
GGML_DTYPE_Q5_0 = 6
GGML_DTYPE_Q5_1 = 7
GGML_DTYPE_Q8_0 = 8
GGML_DTYPE_Q2_K = 10
GGML_DTYPE_Q3_K = 11
GGML_DTYPE_Q4_K = 12
GGML_DTYPE_Q5_K = 13
GGML_DTYPE_Q6_K = 14
GGML_DTYPE_Q8_K = 15
EMEL_DTYPE_Q4_K_X8_BL8 = 42
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34  # fp16 scale + 32 int8
QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q4_K_X8_ROWS = 8
Q4_K_X8_BLOCK_BYTES = 1152

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

# Mirrors emel::model::data::moshi_lm_hparams::k_max_delays: the runtime
# stores n_q + 1 delay slots in a fixed array, so n_q must stay below it.
MAX_DELAY_SLOTS = 64

# Mirrors the codec's fixed array caps (k_max_transformer_layers and
# k_max_quantizer_levels in src/emel/speech/codec/mimi/detail.hpp).
MIMI_MAX_TRANSFORMER_LAYERS = 16
MIMI_MAX_RVQ_SPLIT_LEVELS = 32
# Mirrors the codec's 2^16 geometry/hparam extent cap
# (k_max_conv_geometry_extent) and the fixed 1920-sample frame the stride
# chain (4*5*6*8 encoder, 2 downsample) reduces to one token.
MIMI_MAX_EXTENT = 1 << 16
MIMI_FRAME_SAMPLES = 1920

# Fixed mimi_v0_1 SEANet module topology (module index, kind, stride),
# mirroring k_encoder_topology/k_decoder_topology in the codec.
MIMI_SEANET_ENCODER = (
    (0, "conv", 1), (1, "resnet", 1), (3, "conv", 4), (4, "resnet", 1),
    (6, "conv", 5), (7, "resnet", 1), (9, "conv", 6), (10, "resnet", 1),
    (12, "conv", 8), (14, "conv", 1),
)
MIMI_SEANET_DECODER = (
    (0, "conv", 1), (2, "convtr", 8), (3, "resnet", 1), (5, "convtr", 6),
    (6, "resnet", 1), (8, "convtr", 5), (9, "resnet", 1), (11, "convtr", 4),
    (12, "resnet", 1), (14, "conv", 1),
)

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


@dataclass(frozen=True)
class TensorPayloadPlan:
  source: TensorInfo
  mode: str
  padding: int


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
  # The moshi.cpp mangling keeps only four CRC nibbles, so collisions in the
  # candidate table are plausible and would silently restore a tensor to the
  # wrong canonical name while still passing shape checks. Fail loudly.
  reverse: dict[str, str] = {}
  for candidate in candidates:
    if len(candidate) < GGML_MAX_NAME:
      continue
    key = mangle_name(candidate)
    if key in reverse and reverse[key] != candidate:
      raise ValueError(
          f"CRC-mangled tensor name collision: {reverse[key]!r} and "
          f"{candidate!r} both mangle to {key!r}; the candidate table must "
          "disambiguate them before conversion can proceed")
    reverse[key] = candidate
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


def require_float_dtype(info: TensorInfo) -> None:
  # The codec bind consumes vectors, codebooks, and conv taps through prepare
  # helpers that accept only f32/f16.
  if info.dtype not in (GGML_DTYPE_F32, GGML_DTYPE_F16):
    raise ValueError(
        f"tensor {info.name!r} has dtype {info.dtype}; the runtime bind "
        "consumes only f32/f16 here")


def require_elements(infos: list[TensorInfo], name: str,
                     elements: int) -> None:
  # The runtime bind's prepare helpers accept any dims layout with the
  # expected total element count, so these probes compare element counts.
  info = find_info(infos, name)
  if math.prod(info.dims) != elements:
    raise ValueError(
        f"tensor {name!r} has dims {info.dims} ({math.prod(info.dims)} "
        f"elements), the runtime bind consumes {elements}")


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
    schedule = config.get("depformer_weights_per_step_schedule")
    if schedule is not None:
      if not isinstance(schedule, list):
        raise ValueError(
            f"{path}: depformer_weights_per_step_schedule must be a list")
      if len(schedule) != config["dep_q"]:
        raise ValueError(
            f"{path}: depformer_weights_per_step_schedule has "
            f"{len(schedule)} entries, expected dep_q={config['dep_q']}")
      if not all(isinstance(value, int) for value in schedule):
        raise ValueError(
            f"{path}: depformer_weights_per_step_schedule must contain only "
            "integers")
      for index, value in enumerate(schedule):
        if value < 0 or value >= config["dep_q"]:
          raise ValueError(
              f"{path}: depformer_weights_per_step_schedule[{index}]="
              f"{value} must be within [0, dep_q={config['dep_q']})")

  delays = config["delays"]
  if len(delays) != config["n_q"] + 1:
    raise ValueError(
        f"{path}: delays has {len(delays)} entries, expected n_q+1 = "
        f"{config['n_q'] + 1}")

  # The C++ hparam loader rejects inconsistent LM metadata before any tensor
  # work (load_lm_hparams in src/emel/model/moshi/detail.cpp); mirror those
  # consistency gates so a malformed config fails conversion instead of
  # emitting an enriched artifact the maintained loader immediately rejects.
  for key in ("card", "n_q", "text_card", "dim", "num_heads", "num_layers",
              "context", "max_period"):
    if config[key] <= 0:
      raise ValueError(f"{path}: {key} must be positive, got {config[key]}")
  if round(config["hidden_scale"] * config["dim"]) <= 0:
    raise ValueError(
        f"{path}: hidden_scale={config['hidden_scale']} yields a non-positive "
        f"dim_feedforward for dim={config['dim']}")
  if not 0 <= config["dep_q"] <= config["n_q"]:
    raise ValueError(
        f"{path}: dep_q={config['dep_q']} must be within [0, n_q="
        f"{config['n_q']}]")
  if config["n_q"] >= MAX_DELAY_SLOTS:
    raise ValueError(
        f"{path}: n_q={config['n_q']} exceeds the runtime delay-slot bound "
        f"of {MAX_DELAY_SLOTS - 1}")
  if config["dim"] % config["num_heads"] != 0:
    raise ValueError(
        f"{path}: dim={config['dim']} is not divisible by num_heads="
        f"{config['num_heads']}")
  if not 0 <= config["existing_text_padding_id"] < config["text_card"]:
    raise ValueError(
        f"{path}: existing_text_padding_id="
        f"{config['existing_text_padding_id']} must be within [0, text_card="
        f"{config['text_card']})")
  if config["dep_q"] > 0:
    for key in ("depformer_dim", "depformer_num_heads",
                "depformer_num_layers", "depformer_dim_feedforward",
                "depformer_context", "depformer_max_period"):
      if config[key] <= 0:
        raise ValueError(f"{path}: {key} must be positive, got {config[key]}")
    if config["depformer_dim"] % config["depformer_num_heads"] != 0:
      raise ValueError(
          f"{path}: depformer_dim={config['depformer_dim']} is not divisible "
          f"by depformer_num_heads={config['depformer_num_heads']}")
  normalize_lm_inference_config(config, path)
  return config


def normalize_lm_inference_config(config: dict, path: Path) -> None:
  """Normalize explicit LMGen inference knobs into emitted GGUF metadata."""
  if not config.get("depformer_weights_per_step", False):
    return

  inference = config.get("inference", {})

  def get_value(key: str):
    flat_key = f"inference_{key}"
    if flat_key in config:
      return config[flat_key]
    return inference.get(key)

  inference_dep_q = get_value("dep_q")
  if inference_dep_q is None:
    raise ValueError(
        f"{path}: depformer_weights_per_step=true requires "
        "inference_dep_q metadata")

  prompt_tokens = get_value("prompt_tokens")
  if prompt_tokens is None:
    raise ValueError(
        f"{path}: depformer_weights_per_step=true requires "
        "inference_prompt_tokens metadata")

  pre_silence = get_value("pre_text_silence_frames")
  if pre_silence is None:
    raise ValueError(
        f"{path}: depformer_weights_per_step=true requires "
        "inference_pre_text_silence_frames metadata")

  post_silence = get_value("post_text_silence_frames")
  if post_silence is None:
    raise ValueError(
        f"{path}: depformer_weights_per_step=true requires "
        "inference_post_text_silence_frames metadata")

  if not isinstance(inference_dep_q, int):
    raise ValueError(f"{path}: inference_dep_q must be an integer")
  if inference_dep_q <= 0 or inference_dep_q > config["dep_q"]:
    raise ValueError(
        f"{path}: inference_dep_q={inference_dep_q} must be within "
        f"[1, dep_q={config['dep_q']}]")
  if not isinstance(pre_silence, int) or pre_silence < 0:
    raise ValueError(
        f"{path}: inference_pre_text_silence_frames must be a non-negative "
        "integer")
  if not isinstance(post_silence, int) or post_silence < 0:
    raise ValueError(
        f"{path}: inference_post_text_silence_frames must be a non-negative "
        "integer")
  if not isinstance(prompt_tokens, list):
    raise ValueError(f"{path}: inference_prompt_tokens must be a list")
  if len(prompt_tokens) != config["n_q"] + 1:
    raise ValueError(
        f"{path}: inference_prompt_tokens has {len(prompt_tokens)} entries, "
        f"expected n_q+1 = {config['n_q'] + 1}")
  if any(not isinstance(token, int) for token in prompt_tokens):
    raise ValueError(
        f"{path}: inference_prompt_tokens must contain only integers")
  if not 0 <= prompt_tokens[0] < config["text_card"]:
    raise ValueError(
        f"{path}: inference_prompt_tokens[0]={prompt_tokens[0]} must be "
        f"within [0, text_card={config['text_card']})")
  for index, token in enumerate(prompt_tokens[1:], start=1):
    if not 0 <= token < config["card"]:
      raise ValueError(
          f"{path}: inference_prompt_tokens[{index}]={token} must be within "
          f"[0, card={config['card']})")

  config["inference_dep_q"] = inference_dep_q
  config["inference_prompt_tokens"] = list(prompt_tokens)
  config["inference_pre_text_silence_frames"] = pre_silence
  config["inference_post_text_silence_frames"] = post_silence


def resolve_mimi_n_q(config_path: Path | None) -> int:
  if config_path is None:
    return 16
  config = json.loads(config_path.read_text(encoding="utf-8"))
  if "n_q" not in config:
    raise ValueError(f"{config_path}: mimi conversion config requires n_q")
  if not isinstance(config["n_q"], int):
    raise ValueError(f"{config_path}: n_q must be an integer")
  if config.get("depformer_weights_per_step", False):
    normalize_lm_inference_config(config, config_path)
    return config["inference_dep_q"]
  return config["n_q"]


def depformer_weight_count(config: dict) -> int:
  schedule = config.get("depformer_weights_per_step_schedule")
  if schedule:
    return max(schedule) + 1
  return config["dep_q"] if config.get("depformer_multi_linear") else 1


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
  names = {info.name for info in infos}

  def require_block(prefix: str, layer: int, with_gating: bool) -> None:
    # Mirrors the C++ has_lm_transformer_block/has_depformer_block contract:
    # both norms, a fused or pre-split attention input projection, and (for
    # the main transformer) a fused or per-step gating input weight.
    base = f"{prefix}.layers.{layer}."
    missing = []
    for required in ("norm1.alpha", "norm2.alpha"):
      if base + required not in names:
        missing.append(required)
    if (base + "self_attn.in_proj_weight" not in names and
        base + "self_attn.in_projs.0.weight" not in names):
      missing.append("self_attn.in_proj_weight|self_attn.in_projs.0.weight")
    if with_gating and (base + "gating.linear_in.weight" not in names and
                        base + "gating.0.linear_in.weight" not in names):
      missing.append("gating.linear_in.weight|gating.0.linear_in.weight")
    if missing:
      raise ValueError(
          f"weights are missing {base}{{{', '.join(missing)}}} required by "
          "the runtime block contract")

  layer_count = count_prefixed(infos, "lm.transformer.layers.")
  # The C++ contract iterates every configured block and its required block
  # tensors; a missing intermediate layer or block tensor must fail
  # conversion instead of emitting an unusable artifact.
  for layer in range(config["num_layers"]):
    if count_prefixed(infos, f"lm.transformer.layers.{layer}.") == 0:
      raise ValueError(
          f"weights are missing lm.transformer.layers.{layer}.* "
          f"(config num_layers={config['num_layers']}, "
          f"{layer_count} layer tensors present)")
    require_block("lm.transformer", layer, with_gating=True)
  if config["dep_q"] > 0:
    expected_depformer_weights = depformer_weight_count(config)
    if count_prefixed(infos, "lm.depformer_in.") != expected_depformer_weights:
      raise ValueError(
          "lm.depformer_in.* count does not match depformer weight count "
          f"{expected_depformer_weights}")
    if count_prefixed(infos, "lm.linears.") != config["dep_q"]:
      raise ValueError("lm.linears.* count does not match config dep_q")
    # The C++ contract also requires the depformer text-embedding family for
    # any dep_q > 0 and at least dep_q - 1 audio-embedding tensors past the
    # first step (validate_lm_contract); a raw LM missing them converted
    # successfully and was then rejected at runtime.
    if count_prefixed(infos, "lm.depformer_text_emb.") == 0:
      raise ValueError(
          "weights are missing the lm.depformer_text_emb.* family required "
          "when dep_q > 0")
    if config["dep_q"] > 1:
      depformer_emb_count = count_prefixed(infos, "lm.depformer_emb.")
      if depformer_emb_count < config["dep_q"] - 1:
        raise ValueError(
            f"weights carry {depformer_emb_count} lm.depformer_emb.* "
            f"tensors, the runtime contract needs at least dep_q-1 = "
            f"{config['dep_q'] - 1}")
    for layer in range(config["depformer_num_layers"]):
      if count_prefixed(infos, f"lm.depformer.layers.{layer}.") == 0:
        raise ValueError(
            f"weights are missing lm.depformer.layers.{layer}.* per config "
            f"depformer_num_layers={config['depformer_num_layers']}")
      require_block("lm.depformer", layer, with_gating=False)


def check_mimi_hparams(n_q: int, preset: dict) -> None:
  # Mirrors the C++ load_mimi_hparams gate and the runtime contract caps
  # (src/emel/model/moshi/detail.cpp) so a preset or config override the
  # maintained loader rejects fails conversion instead of publishing an
  # unusable enriched artifact.
  for key in ("sample_rate", "card", "dim", "semantic_n_q",
              "transformer_num_layers", "transformer_num_heads",
              "transformer_context", "transformer_max_period"):
    if preset[key] <= 0:
      raise ValueError(
          f"mimi preset {key} must be positive, got {preset[key]}")
  if n_q <= 0:
    raise ValueError(f"mimi n_q must be positive, got {n_q}")
  # The codec stores transformer layers in fixed 16-entry arrays and each RVQ
  # split in fixed 32-level arrays; larger declared counts cannot initialize.
  if preset["transformer_num_layers"] > MIMI_MAX_TRANSFORMER_LAYERS:
    raise ValueError(
        f"mimi preset transformer_num_layers={preset['transformer_num_layers']} "
        f"exceeds the codec cap of {MIMI_MAX_TRANSFORMER_LAYERS}")
  if preset["semantic_n_q"] > MIMI_MAX_RVQ_SPLIT_LEVELS:
    raise ValueError(
        f"mimi preset semantic_n_q={preset['semantic_n_q']} exceeds the "
        f"per-split codec cap of {MIMI_MAX_RVQ_SPLIT_LEVELS}")
  if n_q - preset["semantic_n_q"] > MIMI_MAX_RVQ_SPLIT_LEVELS:
    raise ValueError(
        f"mimi acoustic level count {n_q - preset['semantic_n_q']} exceeds "
        f"the per-split codec cap of {MIMI_MAX_RVQ_SPLIT_LEVELS}")
  # The codec caps hparam extents at 2^16 to keep its arena sizing
  # representable.
  for key in ("card", "dim"):
    if preset[key] > MIMI_MAX_EXTENT:
      raise ValueError(
          f"mimi preset {key}={preset[key]} exceeds the codec extent cap of "
          f"{MIMI_MAX_EXTENT}")
  # The fixed stride chain (encoder 4*5*6*8, downsample 2) reduces one frame
  # to one token only for exactly 1920 samples; the codec truncates
  # sample_rate / frame_rate and rejects any other frame length.
  if int(preset["sample_rate"] / preset["frame_rate"]) != MIMI_FRAME_SAMPLES:
    raise ValueError(
        f"mimi preset sample_rate/frame_rate = "
        f"{preset['sample_rate']}/{preset['frame_rate']} does not truncate to "
        f"the fixed {MIMI_FRAME_SAMPLES}-sample frame the stride chain "
        "requires")
  frame_rate = float(preset["frame_rate"])
  if not math.isfinite(frame_rate) or frame_rate <= 0.0:
    raise ValueError(
        f"mimi preset frame_rate must be finite and positive, got "
        f"{preset['frame_rate']}")
  if preset["semantic_n_q"] >= n_q:
    raise ValueError(
        f"mimi preset semantic_n_q={preset['semantic_n_q']} must be below "
        f"n_q={n_q}")
  dim = preset["dim"]
  heads = preset["transformer_num_heads"]
  if dim % heads != 0 or (dim // heads) % 2 != 0:
    raise ValueError(
        f"mimi preset dim={dim} must split into an even head size across "
        f"num_heads={heads} (the codec rotary kernel halves head_dim)")


def resolve_conv_geometry(info: TensorInfo, in_channels: int) -> tuple[int, int]:
  # Mirrors the codec's resolve_conv_geometry: taps from dim 0, out channels
  # from the element count against the running channel chain.
  if not 1 <= len(info.dims) <= 3:
    raise ValueError(f"conv tensor {info.name!r} has rank {len(info.dims)}")
  require_float_dtype(info)
  taps = info.dims[0]
  total = math.prod(info.dims)
  divisor = taps * in_channels
  if taps <= 0 or divisor <= 0 or total % divisor != 0 or total // divisor <= 0:
    raise ValueError(
        f"conv tensor {info.name!r} dims {info.dims} do not resolve against "
        f"{in_channels} input channels")
  out_channels = total // divisor
  # The codec caps every conv geometry extent at 2^16.
  if taps > MIMI_MAX_EXTENT or out_channels > MIMI_MAX_EXTENT:
    raise ValueError(
        f"conv tensor {info.name!r} resolves to taps={taps}, "
        f"out_channels={out_channels}, past the codec extent cap of "
        f"{MIMI_MAX_EXTENT}")
  return taps, out_channels


def check_seanet_geometry(infos: list[TensorInfo], family: str,
                          topology: tuple, in_channels: int) -> int:
  # Mirrors plan_seanet's channel-chain walk: every module's weight must
  # resolve against the running channel count, resnet blocks must return to
  # their input width through a k1 conv, and strided kernels must span their
  # stride, or the codec bind rejects the converted artifact.
  channels = in_channels
  for index, kind, stride in topology:
    base = f"mimi.{family}.model.{index}."
    if kind == "resnet":
      taps1, half = resolve_conv_geometry(
          find_info(infos, base + "block.1.conv.conv.weight"), channels)
      taps3, out = resolve_conv_geometry(
          find_info(infos, base + "block.3.conv.conv.weight"), half)
      if taps3 != 1 or out != channels:
        raise ValueError(
            f"resnet block {base!r} does not return to {channels} channels "
            f"through a k1 conv (taps={taps3}, out={out})")
      continue
    name = base + ("convtr.convtr.weight" if kind == "convtr"
                   else "conv.conv.weight")
    taps, out = resolve_conv_geometry(find_info(infos, name), channels)
    if taps < stride:
      raise ValueError(
          f"conv tensor {name!r} has {taps} taps, below its stride {stride}")
    channels = out
  return channels


def cross_check_mimi(infos: list[TensorInfo], n_q: int, preset: dict,
                     quantize_projections: bool = False) -> dict:
  check_mimi_hparams(n_q, preset)
  first = find_info(
      infos, "mimi.quantizer.rvq_first.vq.layers.0._codebook.embedding")
  if len(first.dims) != 2 or first.dims[1] != preset["card"]:
    raise ValueError(
        f"semantic codebook dims {first.dims} do not match preset card "
        f"{preset['card']}")
  if first.dims[0] <= 0:
    raise ValueError(
        f"semantic codebook dims {first.dims} derive a non-positive "
        "codebook_dim; the maintained loader requires it positive")
  if first.dims[0] > MIMI_MAX_EXTENT:
    raise ValueError(
        f"semantic codebook dims {first.dims} derive codebook_dim past the "
        f"codec extent cap of {MIMI_MAX_EXTENT}")
  # The C++ contract and bind consume every semantic level 0..semantic_n_q-1
  # with the full codebook shape, mirroring the acoustic loop below.
  for level in range(preset["semantic_n_q"]):
    name = f"mimi.quantizer.rvq_first.vq.layers.{level}._codebook.embedding"
    require_dims(infos, name, (first.dims[0], preset["card"]))
    require_float_dtype(find_info(infos, name))
  # The C++ contract requires every acoustic level with the full codebook
  # shape; probe each expected level instead of counting, so a missing
  # intermediate or wrong-shaped codebook fails conversion.
  for level in range(n_q - preset["semantic_n_q"]):
    name = f"mimi.quantizer.rvq_rest.vq.layers.{level}._codebook.embedding"
    require_dims(infos, name, (first.dims[0], preset["card"]))
    require_float_dtype(find_info(infos, name))
  for family in ("mimi.encoder.", "mimi.encoder_transformer.",
                 "mimi.downsample.", "mimi.upsample.",
                 "mimi.decoder_transformer.", "mimi.decoder."):
    if count_prefixed(infos, family) == 0:
      raise ValueError(f"weights are missing the {family}* family")
  # The C++ codec bind consumes every configured transformer layer's norms,
  # layer scales, attention projections, and MLP linears (bind_transformer;
  # plan_transformer additionally pins the in_proj/linear1 shapes). Probe the
  # full per-layer tensor set here so a truncated family or a missing layer
  # tensor fails conversion instead of failing codec bind at runtime.
  dim = preset["dim"]
  for family in ("mimi.encoder_transformer.", "mimi.decoder_transformer."):
    family_mlp = 0
    for layer in range(preset["transformer_num_layers"]):
      base = f"{family}transformer.layers.{layer}."
      require_dims(infos, base + "self_attn.in_projs.0.weight", (dim, 3 * dim))
      linear1 = find_info(infos, base + "linear1.weight")
      if (len(linear1.dims) != 2 or linear1.dims[0] != dim or
          linear1.dims[1] <= 0):
        raise ValueError(
            f"tensor {linear1.name!r} has dims {linear1.dims}, expected "
            f"({dim}, mlp_dim); the preset does not match the weights")
      mlp_dim = linear1.dims[1]
      # The codec stores one MLP width per transformer family and binds
      # every layer against it, so mixed widths must reject here.
      if layer == 0:
        family_mlp = mlp_dim
      elif mlp_dim != family_mlp:
        raise ValueError(
            f"tensor {linear1.name!r} has MLP width {mlp_dim} but the "
            f"family's first layer uses {family_mlp}; the codec binds one "
            "width per family")
      require_elements(infos, base + "self_attn.out_projs.0.weight",
                       dim * dim)
      require_elements(infos, base + "linear2.weight", mlp_dim * dim)
      for vector in ("norm1.weight", "norm1.bias", "norm2.weight",
                     "norm2.bias", "layer_scale_1.scale",
                     "layer_scale_2.scale"):
        require_elements(infos, base + vector, dim)
        # norms, biases, and layer scales bind through prepare_vector
        # (f32/f16 only).
        require_float_dtype(find_info(infos, base + vector))
  # bind_rvq_split consumes the input/output 1x1 projections for both splits
  # before any encode or decode can run; require them (by element count,
  # matching prepare_linear/prepare_raw_q8_0) alongside the codebooks.
  codebook_dim = first.dims[0]
  for split in ("rvq_first", "rvq_rest"):
    for proj in ("input_proj", "output_proj"):
      require_elements(infos, f"mimi.quantizer.{split}.{proj}.weight",
                       dim * codebook_dim)
  # The codec plan selects the q8-vs-float class from the first transformer
  # in_proj (and the first RVQ input projection) and the bind requires every
  # projection in that family to match; a mixed-class artifact converts here
  # and then fails codec bind, so enforce class uniformity.
  proj_q8 = (find_info(
      infos, "mimi.encoder_transformer.transformer.layers.0.self_attn."
             "in_projs.0.weight").dtype == GGML_DTYPE_Q8_0)
  for family in ("mimi.encoder_transformer.", "mimi.decoder_transformer."):
    for layer in range(preset["transformer_num_layers"]):
      base = f"{family}transformer.layers.{layer}."
      for proj in ("self_attn.in_projs.0.weight",
                   "self_attn.out_projs.0.weight", "linear1.weight",
                   "linear2.weight"):
        info = find_info(infos, base + proj)
        if proj_q8:
          if info.dtype != GGML_DTYPE_Q8_0:
            raise ValueError(
                f"tensor {info.name!r} has dtype {info.dtype} but the "
                "transformer projection class is q8_0; the runtime bind "
                "requires one uniform class")
        elif info.dtype not in (GGML_DTYPE_F32, GGML_DTYPE_F16):
          raise ValueError(
              f"tensor {info.name!r} has dtype {info.dtype}; the runtime "
              "bind consumes only f32/f16 (or q8_0) projections")
  rvq_q8 = (find_info(infos, "mimi.quantizer.rvq_first.input_proj.weight")
            .dtype == GGML_DTYPE_Q8_0)
  for split in ("rvq_first", "rvq_rest"):
    for proj in ("input_proj", "output_proj"):
      info = find_info(infos, f"mimi.quantizer.{split}.{proj}.weight")
      if rvq_q8:
        if info.dtype != GGML_DTYPE_Q8_0:
          raise ValueError(
              f"tensor {info.name!r} has dtype {info.dtype} but the RVQ "
              "projection class is q8_0; the runtime bind requires one "
              "uniform class")
      elif info.dtype not in (GGML_DTYPE_F32, GGML_DTYPE_F16):
        raise ValueError(
            f"tensor {info.name!r} has dtype {info.dtype}; the runtime bind "
            "consumes only f32/f16 (or q8_0) projections")
  # The codec planner walks the fixed mimi_v0_1 SEANet module topology and
  # resolves every weight against the running channel/stride chain
  # (plan_seanet / resolve_conv_geometry / bind_conv); mirror the geometry so
  # a raw GGUF with every listed name but a malformed shape fails conversion
  # instead of failing codec initialization. plan_codec also selects the f16
  # conv operand class from the first encoder conv and bind_conv requires
  # every non-transposed SEANet/downsample conv to carry raw f16 taps.
  channels = check_seanet_geometry(infos, "encoder", MIMI_SEANET_ENCODER, 1)
  if channels != dim:
    raise ValueError(
        f"encoder SEANet chain ends at {channels} channels, expected dim={dim}")
  channels = check_seanet_geometry(infos, "decoder", MIMI_SEANET_DECODER, dim)
  if channels != 1:
    raise ValueError(
        f"decoder SEANet chain ends at {channels} channels, expected 1")
  down = find_info(infos, "mimi.downsample.conv.conv.conv.weight")
  down_taps, down_out = resolve_conv_geometry(down, dim)
  if down_out != dim or down_taps < 2:
    raise ValueError(
        f"downsample conv dims {down.dims} do not resolve to a stride-2 "
        f"dim->dim conv")
  up = find_info(infos, "mimi.upsample.convtr.convtr.convtr.weight")
  require_float_dtype(up)
  if (len(up.dims) < 1 or up.dims[0] < 2 or up.dims[0] > MIMI_MAX_EXTENT or
      math.prod(up.dims) != up.dims[0] * dim):
    raise ValueError(
        f"upsample convtr dims {up.dims} do not resolve to a depthwise "
        f"stride-2 conv over dim={dim}")
  conv_f16 = (find_info(infos, "mimi.encoder.model.0.conv.conv.weight").dtype
              == GGML_DTYPE_F16)
  if conv_f16 and not rvq_q8 and not quantize_projections:
    # bind_rvq_split prepares raw f16 projection copies whenever the f16 conv
    # class is selected and the split is not q8, so f32 RVQ projections in an
    # f16-class artifact fail initialization. A pending --quantize q8_0 run
    # converts these projections to q8_0 (bind accepts that class without
    # the raw-f16 copies), so the source-side f16 requirement only applies
    # to the non-quantized path.
    for split in ("rvq_first", "rvq_rest"):
      for proj in ("input_proj", "output_proj"):
        info = find_info(infos, f"mimi.quantizer.{split}.{proj}.weight")
        if info.dtype != GGML_DTYPE_F16:
          raise ValueError(
              f"tensor {info.name!r} has dtype {info.dtype} but the f16 conv "
              "class requires raw f16 RVQ projections")
  if conv_f16:
    non_transposed = ["mimi.downsample.conv.conv.conv.weight"]
    for family, topology in (("encoder", MIMI_SEANET_ENCODER),
                             ("decoder", MIMI_SEANET_DECODER)):
      for index, kind, _stride in topology:
        base = f"mimi.{family}.model.{index}."
        if kind == "resnet":
          non_transposed.append(base + "block.1.conv.conv.weight")
          non_transposed.append(base + "block.3.conv.conv.weight")
        elif kind == "conv":
          non_transposed.append(base + "conv.conv.weight")
    for name in non_transposed:
      info = find_info(infos, name)
      if info.dtype != GGML_DTYPE_F16:
        raise ValueError(
            f"tensor {name!r} has dtype {info.dtype} but the first encoder "
            "conv selected the f16 conv class; the runtime bind requires "
            "raw f16 taps on every non-transposed conv")
  return dict(preset, n_q=n_q, codebook_dim=codebook_dim)


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
  if config.get("depformer_weights_per_step", False):
    kv.u32("moshi.lm.inference.dep_q", config["inference_dep_q"])
    kv.u32("moshi.lm.inference.pre_text_silence_frames",
           config["inference_pre_text_silence_frames"])
    kv.u32("moshi.lm.inference.post_text_silence_frames",
           config["inference_post_text_silence_frames"])
    kv.i32_array("moshi.lm.inference.prompt_tokens",
                 list(config["inference_prompt_tokens"]))
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
    if config.get("depformer_weights_per_step_schedule") is not None:
      kv.i32_array("moshi.lm.depformer.weights_per_step_schedule",
                   list(config["depformer_weights_per_step_schedule"]))
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
  if info.dtype == GGML_DTYPE_Q4_0:
    return elements // 32 * 18
  if info.dtype == GGML_DTYPE_Q4_1:
    return elements // 32 * 20
  if info.dtype == GGML_DTYPE_Q5_0:
    return elements // 32 * 22
  if info.dtype == GGML_DTYPE_Q5_1:
    return elements // 32 * 24
  if info.dtype == GGML_DTYPE_Q8_0:
    return elements // Q8_0_BLOCK * Q8_0_BLOCK_BYTES
  if info.dtype == GGML_DTYPE_Q2_K:
    return elements // QK_K * 84
  if info.dtype == GGML_DTYPE_Q3_K:
    return elements // QK_K * 110
  if info.dtype == GGML_DTYPE_Q4_K:
    return elements // QK_K * Q4_K_BLOCK_BYTES
  if info.dtype == GGML_DTYPE_Q5_K:
    return elements // QK_K * 176
  if info.dtype == GGML_DTYPE_Q6_K:
    return elements // QK_K * 210
  if info.dtype == GGML_DTYPE_Q8_K:
    return elements // QK_K * 292
  if info.dtype == EMEL_DTYPE_Q4_K_X8_BL8:
    if len(info.dims) < 2:
      raise ValueError(
          f"packed q4_k_x8_bl8 tensor {info.name!r} has dims {info.dims}")
    cols, rows = info.dims[0], info.dims[1]
    groups = (rows + Q4_K_X8_ROWS - 1) // Q4_K_X8_ROWS
    return groups * (cols // QK_K) * Q4_K_X8_BLOCK_BYTES
  raise ValueError(f"unsupported dtype {info.dtype} for {info.name}")


def get_scale_min_k4(index: int, scales: bytes) -> tuple[int, int]:
  if index < 4:
    return scales[index] & 63, scales[index + 4] & 63
  scale = (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4)
  minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4)
  return scale, minimum


def write_q4_kx8_scale_chunk(out: bytearray, base: int, scales: list[int],
                             mins: list[int]) -> None:
  out[base + 0] = (scales[0] & 63) + ((scales[4] & 48) << 2)
  out[base + 1] = (scales[1] & 63) + ((scales[5] & 48) << 2)
  out[base + 2] = (scales[2] & 63) + ((scales[6] & 48) << 2)
  out[base + 3] = (scales[3] & 63) + ((scales[7] & 48) << 2)
  out[base + 4] = (mins[0] & 63) + ((mins[4] & 48) << 2)
  out[base + 5] = (mins[1] & 63) + ((mins[5] & 48) << 2)
  out[base + 6] = (mins[2] & 63) + ((mins[6] & 48) << 2)
  out[base + 7] = (mins[3] & 63) + ((mins[7] & 48) << 2)
  out[base + 8] = (scales[4] & 15) + ((mins[4] & 15) << 4)
  out[base + 9] = (scales[5] & 15) + ((mins[5] & 15) << 4)
  out[base + 10] = (scales[6] & 15) + ((mins[6] & 15) << 4)
  out[base + 11] = (scales[7] & 15) + ((mins[7] & 15) << 4)


def pack_q4_k_rows_x8_bl8(raw: bytes, rows: int, cols: int) -> bytes:
  if cols <= 0 or rows <= 0 or cols % QK_K != 0:
    raise ValueError(
        f"cannot pack q4_k matrix with rows={rows}, cols={cols}")
  block_count = cols // QK_K
  expected = rows * block_count * Q4_K_BLOCK_BYTES
  if len(raw) != expected:
    raise ValueError(
        f"q4_k payload has {len(raw)} bytes, expected {expected}")

  zero_block = bytes(Q4_K_BLOCK_BYTES)
  out = bytearray()
  group_count = (rows + Q4_K_X8_ROWS - 1) // Q4_K_X8_ROWS
  for group in range(group_count):
    row_base = group * Q4_K_X8_ROWS
    for block in range(block_count):
      blocks: list[bytes] = []
      for lane in range(Q4_K_X8_ROWS):
        logical_row = row_base + lane
        if logical_row < rows:
          offset = (logical_row * block_count + block) * Q4_K_BLOCK_BYTES
          blocks.append(raw[offset:offset + Q4_K_BLOCK_BYTES])
        else:
          blocks.append(zero_block)

      for lane in range(Q4_K_X8_ROWS):
        out.extend(blocks[lane][0:2])
      for lane in range(Q4_K_X8_ROWS):
        out.extend(blocks[lane][2:4])

      packed_scales = bytearray(12 * Q4_K_X8_ROWS)
      for scale_index in range(4):
        scales: list[int] = []
        mins: list[int] = []
        for lane in range(Q4_K_X8_ROWS):
          scales_bytes = blocks[lane][4:4 + 12]
          scales.append(scales_bytes[scale_index] & 63)
          mins.append(scales_bytes[scale_index + 4] & 63)
        write_q4_kx8_scale_chunk(packed_scales, scale_index * 12, scales,
                                 mins)
      for scale_index in range(4):
        scales = []
        mins = []
        for lane in range(Q4_K_X8_ROWS):
          scales_bytes = blocks[lane][4:4 + 12]
          scales.append(((scales_bytes[scale_index] & 192) >> 2) |
                        (scales_bytes[scale_index + 8] & 15))
          mins.append(((scales_bytes[scale_index + 4] & 192) >> 2) |
                      ((scales_bytes[scale_index + 8] & 240) >> 4))
        write_q4_kx8_scale_chunk(packed_scales, 48 + scale_index * 12,
                                 scales, mins)
      out.extend(packed_scales)

      packed_qs = bytearray((QK_K // 2) * Q4_K_X8_ROWS)
      interleave = 8
      end = (QK_K * 4) // interleave
      for index in range(end):
        src_row = index % Q4_K_X8_ROWS
        src_offset = (index // Q4_K_X8_ROWS) * interleave
        dst_offset = index * interleave
        qs = blocks[src_row][16:16 + QK_K // 2]
        packed_qs[dst_offset:dst_offset + interleave] = \
            qs[src_offset:src_offset + interleave]
      out.extend(packed_qs)

  return bytes(out)


LM_Q4_K_PACKABLE_PATTERNS = (
    re.compile(r"^lm\.text_linear\.weight$"),
    re.compile(r"^lm\.depformer_in\.\d+\.weight$"),
    re.compile(r"^lm\.linears\.\d+\.weight$"),
    re.compile(
        r"^lm\.transformer\.layers\.\d+\.self_attn\.in_proj_weight$"),
    re.compile(
        r"^lm\.transformer\.layers\.\d+\.self_attn\.in_projs\.0\.weight$"),
    re.compile(
        r"^lm\.transformer\.layers\.\d+\.self_attn\.out_projs\.0\.weight$"),
    re.compile(
        r"^lm\.transformer\.layers\.\d+\.gating\.linear_in\.weight$"),
    re.compile(
        r"^lm\.transformer\.layers\.\d+\.gating\.linear_out\.weight$"),
    re.compile(
        r"^lm\.depformer\.layers\.\d+\.self_attn\.in_proj_weight$"),
    re.compile(
        r"^lm\.depformer\.layers\.\d+\.self_attn\.in_projs\.\d+\.weight$"),
    re.compile(
        r"^lm\.depformer\.layers\.\d+\.self_attn\.out_projs\.\d+\.weight$"),
    re.compile(
        r"^lm\.depformer\.layers\.\d+\.gating\.\d+\.linear_in\.weight$"),
    re.compile(
        r"^lm\.depformer\.layers\.\d+\.gating\.\d+\.linear_out\.weight$"),
)


def is_lm_q4_k_packable(info: TensorInfo) -> bool:
  return (info.dtype == GGML_DTYPE_Q4_K and len(info.dims) == 2 and
          info.dims[0] % QK_K == 0 and info.dims[1] > 0 and
          any(pattern.match(info.name) for pattern in
              LM_Q4_K_PACKABLE_PATTERNS))


def plan_lm_q4_k_packing(
    infos: list[TensorInfo],
    pack_lm_q4_k: str) -> tuple[list[TensorInfo], list[TensorPayloadPlan], int]:
  if pack_lm_q4_k != "q4_k_x8_bl8":
    raise ValueError(f"unsupported --pack-lm-q4-k type: {pack_lm_q4_k}")

  new_infos: list[TensorInfo] = []
  plans: list[TensorPayloadPlan] = []
  offset = 0
  packed = 0
  for info in infos:
    dtype = info.dtype
    mode = "raw"
    if is_lm_q4_k_packable(info):
      dtype = EMEL_DTYPE_Q4_K_X8_BL8
      mode = "q4_k_x8_bl8"
      packed += 1
    new_info = TensorInfo(
        name=info.name, dims=info.dims, dtype=dtype, offset=offset)
    size = tensor_data_bytes(new_info)
    offset += size
    pad = (-offset) % GGUF_ALIGNMENT
    offset += pad
    new_infos.append(new_info)
    plans.append(TensorPayloadPlan(source=info, mode=mode, padding=pad))

  if packed == 0:
    raise ValueError(
        "--pack-lm-q4-k selected but no matching LM Q4_K tensors were found")
  return new_infos, plans, packed


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
      if (len(info.dims) >= 2 and
          (info.name in MIMI_QUANTIZABLE_NAMES or
           info.name.endswith(MIMI_QUANTIZABLE_SUFFIXES))):
        # The runtime selects the q8 bind path from the first converted
        # projection and then requires every projection in the family to be
        # q8_0; leaving one float would publish a mixed-class artifact the
        # codec bind rejects.
        raise ValueError(
            f"tensor {info.name!r} matches the mimi q8_0 projection set but "
            f"cannot be quantized (dims {info.dims}, dtype {info.dtype}); "
            "refusing a partial q8_0 conversion")
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


def copy_exact(src, dst, size: int) -> None:
  remaining = size
  while remaining > 0:
    chunk = src.read(min(remaining, 8 * 1024 * 1024))
    if not chunk:
      raise ValueError("unexpected EOF while copying GGUF tensor payload")
    dst.write(chunk)
    remaining -= len(chunk)


def write_rewritten_from_source(output: Path, infos: list[TensorInfo],
                                plans: list[TensorPayloadPlan], source: Path,
                                data_start: int, kv: KvWriter) -> None:
  if len(infos) != len(plans):
    raise ValueError("tensor payload plan count does not match tensor infos")
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
    for info, plan in zip(infos, plans):
      source_size = tensor_data_bytes(plan.source)
      src.seek(data_start + plan.source.offset)
      if plan.mode == "raw":
        copy_exact(src, dst, source_size)
      elif plan.mode == "q4_k_x8_bl8":
        raw = src.read(source_size)
        if len(raw) != source_size:
          raise ValueError(
              f"unexpected EOF while reading {plan.source.name!r}")
        packed = pack_q4_k_rows_x8_bl8(
            raw, rows=plan.source.dims[1], cols=plan.source.dims[0])
        expected = tensor_data_bytes(info)
        if len(packed) != expected:
          raise ValueError(
              f"packed {plan.source.name!r} to {len(packed)} bytes, "
              f"expected {expected}")
        dst.write(packed)
      else:
        raise ValueError(f"unknown tensor payload rewrite mode {plan.mode!r}")
      if plan.padding:
        dst.write(b"\0" * plan.padding)


def convert(source: Path, output: Path, config_path: Path | None,
            tokenizer_path: Path | None,
            mimi_preset_path: Path | None = None,
            quantize: str | None = None,
            pack_lm_q4_k: str | None = None) -> dict:
  infos, data_start = read_raw_gguf(source)
  component = detect_component(infos)
  if quantize is not None and component != "mimi":
    raise ValueError("--quantize currently supports mimi components only")
  if pack_lm_q4_k is not None and component != "lm":
    raise ValueError("--pack-lm-q4-k supports lm components only")
  if quantize is not None and pack_lm_q4_k is not None:
    raise ValueError("--quantize and --pack-lm-q4-k are component-specific")

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
    num_weights = depformer_weight_count(config)
    candidates = transformer_layer_candidates(
        "lm.transformer.", config["num_layers"], 1)
    candidates += transformer_layer_candidates(
        "lm.depformer.", config.get("depformer_num_layers", 0), num_weights)
    infos = unmangle_names(infos, candidates)
    cross_check_lm(infos, config)
    append_lm_kv(kv, config)
    append_tokenizer_kv(kv, parse_sentencepiece_model(tokenizer_path))
    if pack_lm_q4_k is not None:
      kv.string("moshi.lm.tensor_packing", pack_lm_q4_k)
  elif component == "mimi":
    preset = dict(MIMI_PRESET)
    if mimi_preset_path is not None:
      preset.update(json.loads(mimi_preset_path.read_text(encoding="utf-8")))
    n_q = resolve_mimi_n_q(config_path)
    layers = preset["transformer_num_layers"]
    candidates = transformer_layer_candidates(
        "mimi.encoder_transformer.transformer.", layers, 1)
    candidates += transformer_layer_candidates(
        "mimi.decoder_transformer.transformer.", layers, 1)
    infos = unmangle_names(infos, candidates)
    append_mimi_kv(kv, cross_check_mimi(infos, n_q, preset,
                                        quantize_projections=quantize
                                        is not None))
    if quantize is not None:
      kv.string("moshi.mimi.quantization", quantize)
  else:
    infos = unmangle_names(infos, [])
    cross_check_voice(infos)
    kv.string("moshi.voice.format", VOICE_FORMAT)

  quantized_count = 0
  packed_count = 0
  if quantize is not None:
    infos, payloads, quantized_count = quantize_mimi_tensors(
        source, infos, data_start, quantize)
    write_rewritten(output, infos, payloads, kv)
  elif pack_lm_q4_k is not None:
    infos, plans, packed_count = plan_lm_q4_k_packing(infos, pack_lm_q4_k)
    write_rewritten_from_source(output, infos, plans, source, data_start, kv)
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
  if pack_lm_q4_k is not None:
    manifest["lm_tensor_packing"] = pack_lm_q4_k
    manifest["packed_tensors"] = packed_count
  return manifest


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True,
                      help="raw moshi.cpp GGUF cache")
  parser.add_argument("--output", type=Path, required=True,
                      help="enriched emel moshi GGUF to write")
  parser.add_argument("--config", type=Path, default=None,
                      help="moshi config JSON (personaplex-config.json); "
                           "required for lm components; optional for mimi "
                           "components to declare active inference n_q")
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
  parser.add_argument("--pack-lm-q4-k", choices=("q4_k_x8_bl8",),
                      default=None,
                      help="rewrite selected LM Q4_K mat-vec/logit weights "
                           "to EMEL packed CPU tensor dtype metadata")
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  manifest = convert(args.source, args.output, args.config, args.tokenizer,
                     args.mimi_preset, args.quantize, args.pack_lm_q4_k)
  if args.manifest is not None:
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
  print(json.dumps(manifest, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
