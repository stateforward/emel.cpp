#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "emel/encoder/context.hpp"
#include "emel/encoder/events.hpp"
#include "emel/encoder/types.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::encoder::detail {

template <size_t N>
inline std::string_view string_view_from_array(const std::array<char, N> &data) {
  size_t len = 0;
  while (len < N && data[len] != '\0') {
    ++len;
  }
  return std::string_view(data.data(), len);
}

inline size_t utf8_len(const char byte) {
  return emel::text::unicode_len_utf8(byte);
}

inline bool is_chinese_char(const uint32_t cpt) {
  return emel::text::unicode_cpt_is_han(cpt);
}

inline std::string cpt_to_utf8(const uint32_t cpt) {
  return emel::text::unicode_cpt_to_utf8(cpt);
}

inline std::string_view token_text(const emel::model::data::vocab &vocab,
                                   const int32_t id) {
  if (id < 0 || static_cast<uint32_t>(id) >= vocab.n_tokens) {
    return {};
  }
  const auto &entry = vocab.entries[static_cast<uint32_t>(id)];
  if (entry.text_length == 0) {
    return {};
  }
  return std::string_view(vocab.token_storage.data() + entry.text_offset,
                          entry.text_length);
}

inline bool is_token_type(const emel::model::data::vocab &vocab,
                          const int32_t id,
                          const int32_t type) {
  if (id < 0 || static_cast<uint32_t>(id) >= vocab.n_tokens) {
    return false;
  }
  return vocab.entries[static_cast<uint32_t>(id)].type == type;
}

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline uint32_t hash_bytes(const uint32_t seed, const std::string_view data) {
  uint32_t hash = seed;
  for (const unsigned char byte : data) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return hash == 0 ? 1u : hash;
}

inline uint32_t hash_sv(const std::string_view data) {
  return hash_bytes(k_fnv_offset, data);
}

inline uint32_t hash_concat(const std::string_view left,
                            const std::string_view right) {
  return hash_bytes(hash_bytes(k_fnv_offset, left), right);
}

inline uint32_t hash_pair(const std::string_view left,
                          const std::string_view right) {
  const uint32_t h1 = hash_sv(left);
  const uint32_t h2 = hash_sv(right);
  const uint32_t combined = h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  return combined == 0 ? 1u : combined;
}

inline std::string_view merge_text(const emel::model::data::vocab &vocab,
                                   const int32_t idx) {
  if (idx < 0 || static_cast<uint32_t>(idx) >= vocab.n_merges) {
    return {};
  }
  const uint32_t offset = vocab.merge_offsets[static_cast<uint32_t>(idx)];
  const uint32_t length = vocab.merge_lengths[static_cast<uint32_t>(idx)];
  if (offset + length > vocab.merge_storage.size()) {
    return {};
  }
  return std::string_view(vocab.merge_storage.data() + offset, length);
}

inline bool merge_match(const std::string_view merge,
                        const std::string_view left,
                        const std::string_view right) {
  if (merge.empty()) {
    return false;
  }
  const size_t pos = merge.find(' ');
  if (pos == std::string_view::npos) {
    return false;
  }
  if (merge.size() != left.size() + right.size() + 1) {
    return false;
  }
  if (merge.substr(0, pos) != left) {
    return false;
  }
  return merge.substr(pos + 1) == right;
}

inline bool insert_token_map(TokenMap &map,
                             const emel::model::data::vocab &vocab,
                             const std::string_view text,
                             const int32_t id) {
  if (text.empty()) {
    return true;
  }
  const uint32_t hash = hash_sv(text);
  const uint32_t mask = k_token_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    if (map.hashes[slot] == 0) {
      map.hashes[slot] = hash;
      map.values[slot] = id;
      map.count += 1;
      return true;
    }
    if (map.hashes[slot] == hash) {
      const int32_t existing = map.values[slot];
      const std::string_view existing_text = token_text(vocab, existing);
      if (existing_text == text) {
        map.values[slot] = id;
        return true;
      }
    }
    slot = (slot + 1) & mask;
  }
  return false;
}

inline bool insert_merge_map(MergeMap &map,
                             const std::string_view left,
                             const std::string_view right,
                             const int32_t rank,
                             const emel::model::data::vocab &vocab) {
  if (left.empty() || right.empty()) {
    return false;
  }
  const uint32_t hash = hash_pair(left, right);
  const uint32_t mask = k_merge_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_merge_hash_size; ++probes) {
    if (map.hashes[slot] == 0) {
      map.hashes[slot] = hash;
      map.values[slot] = rank;
      map.count += 1;
      return true;
    }
    if (map.hashes[slot] == hash) {
      const int32_t existing = map.values[slot];
      const std::string_view merge = merge_text(vocab, existing);
      if (merge_match(merge, left, right)) {
        return true;
      }
    }
    slot = (slot + 1) & mask;
  }
  return false;
}

inline int32_t lookup_token(const action::context &ctx,
                            const std::string_view text) {
  if (text.empty()) {
    return k_token_null;
  }
  const uint32_t hash = hash_sv(text);
  const uint32_t mask = k_token_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const uint32_t entry = ctx.token_to_id.hashes[slot];
    if (entry == 0) {
      return k_token_null;
    }
    if (entry == hash) {
      const int32_t id = ctx.token_to_id.values[slot];
      if (token_text(*ctx.vocab, id) == text) {
        return id;
      }
    }
    slot = (slot + 1) & mask;
  }
  return k_token_null;
}

inline int32_t lookup_token_concat(const action::context &ctx,
                                   const std::string_view left,
                                   const std::string_view right) {
  const uint32_t hash = hash_concat(left, right);
  const uint32_t mask = k_token_hash_size - 1;
  const size_t combined_len = left.size() + right.size();
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_token_hash_size; ++probes) {
    const uint32_t entry = ctx.token_to_id.hashes[slot];
    if (entry == 0) {
      return k_token_null;
    }
    if (entry == hash) {
      const int32_t id = ctx.token_to_id.values[slot];
      const std::string_view token = token_text(*ctx.vocab, id);
      if (token.size() != combined_len) {
        slot = (slot + 1) & mask;
        continue;
      }
      if (!left.empty() && std::memcmp(token.data(), left.data(), left.size()) != 0) {
        slot = (slot + 1) & mask;
        continue;
      }
      if (!right.empty() &&
          std::memcmp(token.data() + left.size(), right.data(), right.size()) != 0) {
        slot = (slot + 1) & mask;
        continue;
      }
      return id;
    }
    slot = (slot + 1) & mask;
  }
  return k_token_null;
}

inline int32_t lookup_merge_rank(const action::context &ctx,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view left,
                                 const std::string_view right) {
  if (left.empty() || right.empty()) {
    return k_token_null;
  }
  const uint32_t hash = hash_pair(left, right);
  const uint32_t mask = k_merge_hash_size - 1;
  uint32_t slot = hash & mask;
  for (uint32_t probes = 0; probes < k_merge_hash_size; ++probes) {
    const uint32_t entry = ctx.bpe_ranks.hashes[slot];
    if (entry == 0) {
      return k_token_null;
    }
    if (entry == hash) {
      const int32_t rank = ctx.bpe_ranks.values[slot];
      const std::string_view merge = merge_text(vocab, rank);
      if (merge_match(merge, left, right)) {
        return rank;
      }
    }
    slot = (slot + 1) & mask;
  }
  return k_token_null;
}

inline bool push_token(const event::encode &ev, const int32_t token, int32_t &count) {
  if (token < 0 || ev.token_capacity <= 0 || ev.token_ids == nullptr) {
    return false;
  }
  if (count >= ev.token_capacity) {
    return false;
  }
  ev.token_ids[count++] = token;
  return true;
}

inline const std::array<uint32_t, 256> &byte_to_codepoint_table() {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> map = {};
    std::array<bool, 256> used = {};
    for (size_t idx = 0; idx < 256; ++idx) {
      used[idx] = false;
      map[idx] = 0;
    }
    for (uint32_t c = 33; c <= 126; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }
    for (uint32_t c = 161; c <= 172; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }
    for (uint32_t c = 174; c <= 255; ++c) {
      const uint8_t idx = static_cast<uint8_t>(c);
      used[idx] = true;
      map[idx] = c;
    }
    uint32_t n = 0;
    for (int ch = 0; ch < 256; ++ch) {
      if (!used[static_cast<size_t>(ch)]) {
        map[static_cast<size_t>(ch)] = 256u + n;
        n += 1;
      }
    }
    return map;
  }();
  return table;
}

inline uint8_t encode_cpt_utf8(const uint32_t cpt, char out[4]) {
  if (cpt <= 0x7F) {
    out[0] = static_cast<char>(cpt);
    return 1;
  }
  if (cpt <= 0x7FF) {
    out[0] = static_cast<char>(0xC0 | ((cpt >> 6) & 0x1F));
    out[1] = static_cast<char>(0x80 | (cpt & 0x3F));
    return 2;
  }
  if (cpt <= 0xFFFF) {
    out[0] = static_cast<char>(0xE0 | ((cpt >> 12) & 0x0F));
    out[1] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
    out[2] = static_cast<char>(0x80 | (cpt & 0x3F));
    return 3;
  }
  out[0] = static_cast<char>(0xF0 | ((cpt >> 18) & 0x07));
  out[1] = static_cast<char>(0x80 | ((cpt >> 12) & 0x3F));
  out[2] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
  out[3] = static_cast<char>(0x80 | (cpt & 0x3F));
  return 4;
}

inline const std::array<std::string, 256> &byte_to_utf8_table() {
  static const std::array<std::string, 256> table = [] {
    std::array<std::string, 256> map = {};
    const auto &codepoints = byte_to_codepoint_table();
    for (size_t idx = 0; idx < map.size(); ++idx) {
      map[idx] = cpt_to_utf8(codepoints[idx]);
    }
    return map;
  }();
  return table;
}

inline int32_t byte_to_token(const action::context &ctx,
                             const emel::model::data::vocab &vocab,
                             const uint8_t byte,
                             const emel::model::data::TokenizerModel model) {
  (void)vocab;
  if (model == emel::model::data::TokenizerModel::NONE) {
    return k_token_null;
  }

  if (model == emel::model::data::TokenizerModel::SPM ||
      model == emel::model::data::TokenizerModel::UGM ||
      model == emel::model::data::TokenizerModel::PLAMO2) {
    char hex[7] = {};
    static const char *digits = "0123456789ABCDEF";
    hex[0] = '<';
    hex[1] = '0';
    hex[2] = 'x';
    hex[3] = digits[(byte >> 4) & 0x0F];
    hex[4] = digits[byte & 0x0F];
    hex[5] = '>';
    hex[6] = '\0';
    const int32_t hex_token = lookup_token(ctx, std::string_view(hex, 6));
    if (hex_token != k_token_null) {
      return hex_token;
    }
    const char raw = static_cast<char>(byte);
    return lookup_token(ctx, std::string_view(&raw, 1));
  }

  if (model == emel::model::data::TokenizerModel::BPE ||
      model == emel::model::data::TokenizerModel::WPM ||
      model == emel::model::data::TokenizerModel::RWKV) {
    const uint32_t cpt = byte_to_codepoint_table()[byte];
    char utf8[4] = {};
    const uint8_t len = encode_cpt_utf8(cpt, utf8);
    return lookup_token(ctx, std::string_view(utf8, len));
  }

  const char raw = static_cast<char>(byte);
  return lookup_token(ctx, std::string_view(&raw, 1));
}

inline bool ensure_tables(action::context &ctx) {
  if (ctx.vocab == nullptr) {
    return false;
  }
  if (ctx.tables_ready) {
    return true;
  }

  ctx.token_to_id.clear();
  ctx.bpe_ranks.clear();
  ctx.max_token_len = 0;

  const emel::model::data::vocab &vocab = *ctx.vocab;
  for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
    const std::string_view text = token_text(vocab, static_cast<int32_t>(id));
    if (!insert_token_map(ctx.token_to_id, vocab, text, static_cast<int32_t>(id))) {
      return false;
    }
    if (text.size() > static_cast<size_t>(ctx.max_token_len)) {
      ctx.max_token_len = static_cast<int32_t>(text.size());
    }
  }

  for (uint32_t idx = 0; idx < vocab.n_merges; ++idx) {
    const std::string_view merge = merge_text(vocab, static_cast<int32_t>(idx));
    if (merge.empty()) {
      continue;
    }
    const size_t pos = merge.find(' ');
    if (pos == std::string_view::npos) {
      continue;
    }
    const std::string_view left = merge.substr(0, pos);
    const std::string_view right = merge.substr(pos + 1);
    insert_merge_map(ctx.bpe_ranks, left, right, static_cast<int32_t>(idx), vocab);
  }

  ctx.ugm_ready = vocab.precompiled_charsmap_size > 0;
  ctx.tables_ready = true;
  return true;
}

inline void split_whitespace(const std::string_view text,
                             std::vector<std::string_view> &parts) {
  parts.clear();
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (std::isspace(c) != 0) {
      parts.emplace_back(text.substr(start, i - start));
      start = i + 1;
    }
  }
  parts.emplace_back(text.substr(start));
}

inline bool build_symbols(const std::string_view text,
                          EncodeScratch &scratch,
                          encode_result &result) {
  scratch.symbol_count = 0;
  size_t offset = 0;
  while (offset < text.size()) {
    if (scratch.symbol_count >= scratch.offsets.size()) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return false;
    }
    const size_t len = std::min(text.size() - offset, utf8_len(text[offset]));
    scratch.offsets[scratch.symbol_count] = static_cast<uint32_t>(offset);
    scratch.lengths[scratch.symbol_count] = static_cast<uint32_t>(len);
    scratch.prev[scratch.symbol_count] = static_cast<int32_t>(scratch.symbol_count) - 1;
    scratch.next[scratch.symbol_count] =
        (offset + len < text.size())
            ? static_cast<int32_t>(scratch.symbol_count) + 1
            : -1;
    scratch.symbol_count += 1;
    offset += len;
  }
  if (scratch.symbol_count > 0) {
    scratch.prev[0] = -1;
  }
  return true;
}

inline void merge_symbols(EncodeScratch &scratch,
                          const int32_t left,
                          const int32_t right) {
  scratch.lengths[static_cast<size_t>(left)] += scratch.lengths[static_cast<size_t>(right)];
  const int32_t right_next = scratch.next[static_cast<size_t>(right)];
  scratch.next[static_cast<size_t>(left)] = right_next;
  if (right_next >= 0) {
    scratch.prev[static_cast<size_t>(right_next)] = left;
  }
  scratch.lengths[static_cast<size_t>(right)] = 0;
}

inline bool encode_bytes(const event::encode &ev,
                         action::context &ctx,
                         const emel::model::data::vocab &vocab,
                         const emel::model::data::TokenizerModel model,
                         encode_result &result) {
  (void)vocab;
  int32_t count = 0;
  for (const unsigned char c : ev.text) {
    const int32_t token = byte_to_token(ctx, vocab, c, model);
    if (token == k_token_null || !push_token(ev, token, count)) {
      result.error = EMEL_ERR_BACKEND;
      return false;
    }
  }
  result.token_count = count;
  result.error = EMEL_OK;
  return true;
}

}  // namespace emel::encoder::detail
