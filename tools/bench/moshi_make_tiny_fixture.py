#!/usr/bin/env python3
"""Generate the tiny synthetic moshi GGUF test fixtures.

Produces, under --output-dir (tests/models by default):
  - moshi-tiny-config.json      tiny moshi config (converter input)
  - moshi-tokenizer-tiny.model  tiny SentencePiece unigram model (64 pieces)
  - moshi-tiny-lm-raw.gguf      raw moshi.cpp-style LM cache (zero metadata,
                                exercises the loader-rejection path and the
                                converter round-trip)
  - moshi-tiny-lm.gguf          enriched LM fixture (via moshi_gguf_convert)
  - mimi-tiny.gguf              enriched Mimi codec fixture
  - moshi-tiny-voice.gguf       enriched PersonaPlex voice-prompt fixture

All payloads are deterministic (fixed LCG seed); regenerating the fixtures is
bit-identical. Raw intermediates for mimi/voice are kept only in a temp dir.

stdlib-only, shares the GGUF writer conventions of moshi_gguf_convert.py.
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

import moshi_gguf_convert as converter

GGUF_TYPE_F32 = 0
GGUF_TYPE_I32 = 26

TINY_CONFIG = {
    "card": 32,
    "n_q": 4,
    "dep_q": 4,
    "delays": [0, 0, 1, 1, 1],
    "dim": 32,
    "text_card": 64,
    "existing_text_padding_id": 3,
    "num_heads": 2,
    "num_layers": 2,
    "hidden_scale": 4.125,
    "causal": True,
    "context": 64,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 16,
    "depformer_num_heads": 2,
    "depformer_num_layers": 2,
    "depformer_dim_feedforward": 48,
    "depformer_multi_linear": True,
    "depformer_context": 4,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "cross_attention": False,
    "model_type": "personaplex",
    "tokenizer_name": "moshi-tokenizer-tiny.model",
}

TINY_MIMI_PRESET = {
    "sample_rate": 24000,
    "frame_rate": 12.5,
    "card": 32,
    "dim": 16,
    "semantic_n_q": 1,
    "transformer_num_layers": 2,
    "transformer_num_heads": 2,
    "transformer_context": 8,
    "transformer_max_period": 10000,
}

TINY_MIMI_CODEBOOK_DIM = 8


class Lcg:
  def __init__(self, seed: int) -> None:
    self.state = seed & 0xFFFFFFFFFFFFFFFF

  def next_float(self) -> float:
    self.state = (self.state * 6364136223846793005 + 1442695040888963407) \
        & 0xFFFFFFFFFFFFFFFF
    return ((self.state >> 33) / float(1 << 31)) - 1.0


def f32_data(rng: Lcg, dims: tuple[int, ...]) -> bytes:
  count = 1
  for dim in dims:
    count *= dim
  return struct.pack(f"<{count}f",
                     *(rng.next_float() * 0.1 for _ in range(count)))


def i32_data(values: list[int]) -> bytes:
  return struct.pack(f"<{len(values)}i", *values)


def write_raw_gguf(path: Path,
                   tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> None:
  """Write a moshi.cpp-style raw cache: zero KV, names >=64 chars mangled."""
  out = bytearray()
  out.extend(converter.GGUF_MAGIC)
  out.extend(struct.pack("<IQQ", converter.GGUF_VERSION, len(tensors), 0))
  offset = 0
  offsets = []
  for name, dims, dtype, data in tensors:
    stored = name
    if len(stored) >= converter.GGML_MAX_NAME:
      stored = converter.mangle_name(stored)
    converter.write_string(out, stored)
    out.extend(struct.pack("<I", len(dims)))
    for dim in dims:
      out.extend(struct.pack("<Q", dim))
    out.extend(struct.pack("<IQ", dtype, offset))
    offsets.append(offset)
    offset += len(data)
    offset += (-offset) % converter.GGUF_ALIGNMENT
  out.extend(b"\0" * ((-len(out)) % converter.GGUF_ALIGNMENT))
  for _, _, _, data in tensors:
    out.extend(data)
    out.extend(b"\0" * ((-len(out)) % converter.GGUF_ALIGNMENT))
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(out)


def lm_tensors(rng: Lcg) -> list[tuple[str, tuple[int, ...], int, bytes]]:
  cfg = TINY_CONFIG
  dim = cfg["dim"]
  dep_dim = cfg["depformer_dim"]
  text_card = cfg["text_card"]
  card = cfg["card"]
  gating_hidden = 2 * round(cfg["hidden_scale"] * dim) * 2 // 3
  dep_gating_hidden = 2 * cfg["depformer_dim_feedforward"] * 2 // 3

  def t(name, *dims):
    dims = tuple(dims)
    return (name, dims, GGUF_TYPE_F32, f32_data(rng, dims))

  tensors = [
      t("lm.text_emb.weight", dim, text_card + 1),
      t("lm.text_linear.weight", dim, text_card),
      t("lm.out_norm.alpha", dim),
      t("lm.depformer_text_emb.weight", dep_dim, text_card + 1),
  ]
  for i in range(cfg["n_q"]):
    tensors.append(t(f"lm.emb.{i}.weight", dim, card + 1))
  for layer in range(cfg["num_layers"]):
    base = f"lm.transformer.layers.{layer}."
    tensors += [
        t(base + "norm1.alpha", dim),
        t(base + "norm2.alpha", dim),
        t(base + "self_attn.in_projs.0.weight", dim, 3 * dim),
        t(base + "self_attn.out_projs.0.weight", dim, dim),
        t(base + "gating.linear_in.weight", dim, gating_hidden),
        t(base + "gating.linear_out.weight", gating_hidden // 2, dim),
    ]
  for i in range(cfg["dep_q"]):
    tensors.append(t(f"lm.depformer_in.{i}.weight", dim, dep_dim))
    tensors.append(t(f"lm.linears.{i}.weight", dep_dim, card))
  for i in range(cfg["dep_q"] - 1):
    tensors.append(t(f"lm.depformer_emb.{i}.weight", dep_dim, card + 1))
  for layer in range(cfg["depformer_num_layers"]):
    base = f"lm.depformer.layers.{layer}."
    tensors += [
        t(base + "norm1.alpha", dep_dim),
        t(base + "norm2.alpha", dep_dim),
    ]
    for j in range(cfg["dep_q"]):
      tensors += [
          t(base + f"self_attn.in_projs.{j}.weight", dep_dim, 3 * dep_dim),
          t(base + f"self_attn.out_projs.{j}.weight", dep_dim, dep_dim),
          t(base + f"gating.{j}.linear_in.weight", dep_dim, dep_gating_hidden),
          t(base + f"gating.{j}.linear_out.weight", dep_gating_hidden // 2,
            dep_dim),
      ]
  return tensors


def mimi_tensors(rng: Lcg) -> list[tuple[str, tuple[int, ...], int, bytes]]:
  preset = TINY_MIMI_PRESET
  dim = preset["dim"]
  card = preset["card"]
  cb_dim = TINY_MIMI_CODEBOOK_DIM
  n_q = TINY_CONFIG["n_q"]

  def t(name, *dims):
    dims = tuple(dims)
    return (name, dims, GGUF_TYPE_F32, f32_data(rng, dims))

  tensors = [
      t("mimi.quantizer.rvq_first.input_proj.weight", 1, dim, cb_dim),
      t("mimi.quantizer.rvq_first.output_proj.weight", 1, cb_dim, dim),
      t("mimi.quantizer.rvq_first.vq.layers.0._codebook.embedding", cb_dim,
        card),
      t("mimi.quantizer.rvq_rest.input_proj.weight", 1, dim, cb_dim),
      t("mimi.quantizer.rvq_rest.output_proj.weight", 1, cb_dim, dim),
      t("mimi.encoder.model.0.conv.conv.weight", 7, 1, dim),
      t("mimi.encoder.model.0.conv.conv.bias", dim),
      t("mimi.encoder.model.14.conv.conv.weight", 3, dim, dim),
      t("mimi.encoder.model.14.conv.conv.bias", dim),
      t("mimi.decoder.model.0.conv.conv.weight", 3, dim, dim),
      t("mimi.decoder.model.0.conv.conv.bias", dim),
      t("mimi.decoder.model.2.convtr.convtr.weight", 4, dim, dim),
      t("mimi.decoder.model.2.convtr.convtr.bias", dim),
      t("mimi.decoder.model.14.conv.conv.weight", 3, dim, 1),
      t("mimi.decoder.model.14.conv.conv.bias", 1),
      t("mimi.downsample.conv.conv.conv.weight", 2, dim, dim),
      t("mimi.upsample.convtr.convtr.convtr.weight", 2, dim, dim),
  ]
  for i in range(n_q - preset["semantic_n_q"]):
    tensors.append(
        t(f"mimi.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding", cb_dim,
          card))
  for transformer in ("encoder_transformer", "decoder_transformer"):
    for layer in range(preset["transformer_num_layers"]):
      base = f"mimi.{transformer}.transformer.layers.{layer}."
      tensors += [
          t(base + "norm1.weight", dim),
          t(base + "norm1.bias", dim),
          t(base + "norm2.weight", dim),
          t(base + "norm2.bias", dim),
          t(base + "linear1.weight", dim, 2 * dim),
          t(base + "linear2.weight", 2 * dim, dim),
          t(base + "self_attn.in_projs.0.weight", dim, 3 * dim),
          t(base + "self_attn.out_projs.0.weight", dim, dim),
          t(base + "layer_scale_1.scale", dim),
          t(base + "layer_scale_2.scale", dim),
      ]
  return tensors


def voice_tensors(rng: Lcg) -> list[tuple[str, tuple[int, ...], int, bytes]]:
  dim = TINY_CONFIG["dim"]
  steps = 5
  streams = TINY_CONFIG["n_q"] + 1
  cache = [step for step in range(4 * steps)]
  return [
      ("voice.embeddings", (dim, 1, 1, steps), GGUF_TYPE_F32,
       f32_data(rng, (dim, 1, 1, steps))),
      ("voice.cache", (4, streams), GGUF_TYPE_I32,
       i32_data(cache[:4 * streams])),
  ]


# --- tiny SentencePiece unigram model ---------------------------------------


def _proto_varint(value: int) -> bytes:
  out = bytearray()
  while True:
    byte = value & 0x7F
    value >>= 7
    if value:
      out.append(byte | 0x80)
    else:
      out.append(byte)
      return bytes(out)


def _proto_field(field: int, wire: int, payload: bytes) -> bytes:
  return _proto_varint((field << 3) | wire) + payload


def _proto_len(field: int, payload: bytes) -> bytes:
  return _proto_field(field, 2, _proto_varint(len(payload)) + payload)


def make_tokenizer_model(path: Path) -> None:
  vocab: list[tuple[str, float, int]] = [
      ("<unk>", 0.0, 2),
      ("<s>", 0.0, 3),
      ("</s>", 0.0, 3),
      ("<pad>", 0.0, 3),
  ]
  words = [
      "▁hello", "▁world", "▁the", "▁a", "▁to",
      "▁and", "▁of", "▁in", "▁is", "▁you",
      "▁it", "▁that", "hello", "world", "ing", "ed", "er", "es",
      "▁s", "▁t", "an", "on", "en", "re", "he", "at", "or", "nd",
      "st", "ll", "le", "ha", "th", "▁.", ".", ",", "!", "?", "'",
      "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
      "o", "p", "q", "r", "s", "t", "u",
  ]
  for index, word in enumerate(words):
    vocab.append((word, -float(index + 1), 1))
  if len(vocab) != TINY_CONFIG["text_card"]:
    raise AssertionError(
        f"tiny tokenizer has {len(vocab)} pieces, expected "
        f"{TINY_CONFIG['text_card']}")

  out = bytearray()
  for piece, score, ptype in vocab:
    body = _proto_len(1, piece.encode("utf-8"))
    body += _proto_field(2, 5, struct.pack("<f", score))
    body += _proto_field(3, 0, _proto_varint(ptype))
    out += _proto_len(1, body)
  trainer = _proto_field(3, 0, _proto_varint(1))  # model_type = UNIGRAM
  trainer += _proto_field(40, 0, _proto_varint(0))  # unk_id
  trainer += _proto_field(41, 0, _proto_varint(1))  # bos_id
  trainer += _proto_field(42, 0, _proto_varint(2))  # eos_id
  trainer += _proto_field(43, 0, _proto_varint(3))  # pad_id
  out += _proto_len(2, trainer)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(bytes(out))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path,
                      default=Path("tests/models"))
  args = parser.parse_args()
  out_dir = args.output_dir

  config_path = out_dir / "moshi-tiny-config.json"
  config_path.parent.mkdir(parents=True, exist_ok=True)
  config_path.write_text(json.dumps(TINY_CONFIG, indent=2) + "\n",
                         encoding="utf-8")

  tokenizer_path = out_dir / "moshi-tokenizer-tiny.model"
  make_tokenizer_model(tokenizer_path)

  lm_raw = out_dir / "moshi-tiny-lm-raw.gguf"
  write_raw_gguf(lm_raw, lm_tensors(Lcg(0x4D4F5348)))
  converter.convert(lm_raw, out_dir / "moshi-tiny-lm.gguf", config_path,
                    tokenizer_path)

  with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    mimi_preset_path = tmp_dir / "mimi-tiny-preset.json"
    mimi_preset_path.write_text(json.dumps(TINY_MIMI_PRESET) + "\n",
                                encoding="utf-8")
    mimi_raw = tmp_dir / "mimi-tiny-raw.gguf"
    write_raw_gguf(mimi_raw, mimi_tensors(Lcg(0x4D494D49)))
    converter.convert(mimi_raw, out_dir / "mimi-tiny.gguf", config_path, None,
                      mimi_preset_path)

    voice_raw = tmp_dir / "moshi-tiny-voice-raw.gguf"
    write_raw_gguf(voice_raw, voice_tensors(Lcg(0x564F4943)))
    converter.convert(voice_raw, out_dir / "moshi-tiny-voice.gguf", None, None)

  for name in ("moshi-tiny-config.json", "moshi-tokenizer-tiny.model",
               "moshi-tiny-lm-raw.gguf", "moshi-tiny-lm.gguf",
               "mimi-tiny.gguf", "moshi-tiny-voice.gguf"):
    path = out_dir / name
    print(f"{converter.sha256_file(path)}  {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
