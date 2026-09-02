#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/text/tokenizer/needle/errors.hpp"

namespace emel::text::tokenizer::needle::detail {

namespace constants {

// `.cact` RAW tokenizer header: u32 n_pieces, u32 pad/eos/bos/unk id,
// u8 add_dummy_prefix, u8 byte_fallback, u16 pad (`export.py` _TK_HDR
// "<IIIIIBBH"); each record: f32 score, u8 type, u16 surface_len
// (_TK_REC "<fBH") followed by surface_len UTF-8 bytes.
inline constexpr size_t header_bytes = 24u;
inline constexpr size_t record_bytes = 7u;

// Piece type codes in the blob (`export.py` TK_*).
inline constexpr uint8_t piece_normal = 0u;
inline constexpr uint8_t piece_unknown = 1u;
inline constexpr uint8_t piece_control = 2u;
inline constexpr uint8_t piece_user_defined = 3u;
inline constexpr uint8_t piece_byte = 4u;

// Shared text/tokenizer vocab-entry type codes (GGUF token-type convention
// consumed by the SPM preprocessor, encoder, and detokenizer).
inline constexpr int32_t vocab_type_normal = 1;
inline constexpr int32_t vocab_type_unknown = 2;
inline constexpr int32_t vocab_type_control = 3;
inline constexpr int32_t vocab_type_user_defined = 4;
inline constexpr int32_t vocab_type_byte = 6;

} // namespace constants

inline emel::error::type cast_tokenizer_error(const error err) noexcept {
  return emel::error::cast(err);
}

inline uint32_t read_u32_le(const uint8_t *bytes) noexcept {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8u) |
         (static_cast<uint32_t>(bytes[2]) << 16u) |
         (static_cast<uint32_t>(bytes[3]) << 24u);
}

inline uint16_t read_u16_le(const uint8_t *bytes) noexcept {
  return static_cast<uint16_t>(
      static_cast<uint16_t>(bytes[0]) |
      static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8u));
}

inline float read_f32_le(const uint8_t *bytes) noexcept {
  return std::bit_cast<float>(read_u32_le(bytes));
}

// Maps a blob piece type to the shared vocab-entry type code. Returns a
// negative value for unknown piece types; the caller folds that into the
// single accumulated error code.
inline int32_t compute_vocab_type(const uint8_t piece_type) noexcept {
  constexpr std::array<int32_t, 5> mapping = {
      constants::vocab_type_normal,  constants::vocab_type_unknown,
      constants::vocab_type_control, constants::vocab_type_user_defined,
      constants::vocab_type_byte,
  };
  if (piece_type >= mapping.size()) {
    return -1;
  }
  return mapping[piece_type];
}

// Parses the RAW SentencePiece-BPE dump into the caller-owned shared vocab.
// Bulk data-plane iteration over a bounded piece count; every rejection is
// folded into a single error code consumed by the owning machine's guards.
// The populated vocab drives the existing SPM preprocessor + SPM encoder
// machines unchanged (piece scores select BPE merges, `<0xXX>` byte pieces
// provide byte fallback, user-defined chat markers partition as specials).
inline emel::error::type
parse_tokenizer_blob(const std::span<const uint8_t> blob,
                     emel::model::data::vocab &vocab_out) noexcept {
  vocab_out = {};

  if (blob.size() < constants::header_bytes) {
    return cast_tokenizer_error(error::parse_failed);
  }

  const uint32_t n_pieces = read_u32_le(blob.data());
  const uint32_t pad_id = read_u32_le(blob.data() + 4u);
  const uint32_t eos_id = read_u32_le(blob.data() + 8u);
  const uint32_t bos_id = read_u32_le(blob.data() + 12u);
  const uint32_t unk_id = read_u32_le(blob.data() + 16u);
  const uint8_t add_dummy_prefix = blob[20u];
  const uint8_t byte_fallback = blob[21u];

  if (n_pieces == 0u ||
      n_pieces > static_cast<uint32_t>(emel::model::data::k_max_vocab_tokens)) {
    return cast_tokenizer_error(error::capacity);
  }
  if (pad_id >= n_pieces || eos_id >= n_pieces || bos_id >= n_pieces ||
      unk_id >= n_pieces) {
    return cast_tokenizer_error(error::model_invalid);
  }
  // The blob format is a self-contained SentencePiece dump; byte_fallback is
  // always written as 1 by the exporter and the shared SPM encoder always
  // resolves `<0xXX>` pieces, so a cleared flag marks a foreign blob.
  if (byte_fallback == 0u) {
    return cast_tokenizer_error(error::model_invalid);
  }

  size_t offset = constants::header_bytes;
  uint32_t bytes_used = 0u;
  for (uint32_t piece = 0u; piece < n_pieces; ++piece) {
    if (blob.size() - offset < constants::record_bytes) {
      return cast_tokenizer_error(error::parse_failed);
    }
    const float score = read_f32_le(blob.data() + offset);
    const uint8_t piece_type = blob[offset + 4u];
    const uint16_t surface_len = read_u16_le(blob.data() + offset + 5u);
    offset += constants::record_bytes;

    if (!std::isfinite(score) || compute_vocab_type(piece_type) < 0) {
      return cast_tokenizer_error(error::model_invalid);
    }
    if (blob.size() - offset < surface_len) {
      return cast_tokenizer_error(error::parse_failed);
    }
    if (surface_len >
        static_cast<uint32_t>(emel::model::data::k_max_vocab_bytes) -
            bytes_used) {
      return cast_tokenizer_error(error::capacity);
    }
    bytes_used += surface_len;
    offset += surface_len;
  }

  if (offset != blob.size()) {
    return cast_tokenizer_error(error::parse_failed);
  }

  offset = constants::header_bytes;
  uint32_t published_bytes = 0u;
  for (uint32_t piece = 0u; piece < n_pieces; ++piece) {
    const float score = read_f32_le(blob.data() + offset);
    const uint8_t piece_type = blob[offset + 4u];
    const uint16_t surface_len = read_u16_le(blob.data() + offset + 5u);
    offset += constants::record_bytes;

    std::memcpy(vocab_out.token_storage.data() + published_bytes,
                blob.data() + offset, surface_len);
    vocab_out.entries[piece].text_offset = published_bytes;
    vocab_out.entries[piece].text_length = surface_len;
    vocab_out.entries[piece].score = score;
    vocab_out.entries[piece].type = compute_vocab_type(piece_type);
    published_bytes += surface_len;
    offset += surface_len;
  }

  vocab_out.n_tokens = n_pieces;

  vocab_out.token_bytes_used = published_bytes;
  vocab_out.tokenizer_model_id = emel::model::data::tokenizer_model::SPM;
  vocab_out.tokenizer_pre_id = emel::model::data::tokenizer_pre::NEEDLE;
  vocab_out.pad_id = static_cast<int32_t>(pad_id);
  vocab_out.eos_id = static_cast<int32_t>(eos_id);
  vocab_out.bos_id = static_cast<int32_t>(bos_id);
  vocab_out.unk_id = static_cast<int32_t>(unk_id);
  // RefTokenizer applies the dummy prefix once to the complete prompt before
  // identifying user-defined markers. The shared tokenizer recognizes this
  // model-owned profile and preserves that global state across raw fragments.
  vocab_out.add_bos = false;
  vocab_out.add_eos = false;
  vocab_out.add_space_prefix = add_dummy_prefix != 0u;
  vocab_out.escape_whitespaces = true;
  return cast_tokenizer_error(error::none);
}

} // namespace emel::text::tokenizer::needle::detail
