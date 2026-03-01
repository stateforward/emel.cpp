#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

#include "emel/text/encoders/spm/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"

namespace emel::text::encoders::spm::detail {

using emel::text::encoders::detail::encode_result;
using emel::text::encoders::detail::k_token_null;

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline int32_t select_i32(const bool choose_true,
                          const int32_t true_value,
                          const int32_t false_value) noexcept {
  const int32_t mask = -static_cast<int32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint32_t select_u32(const bool choose_true,
                           const uint32_t true_value,
                           const uint32_t false_value) noexcept {
  const uint32_t mask = static_cast<uint32_t>(0) - static_cast<uint32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline std::string_view spm_token_text(const emel::model::data::vocab &vocab,
                                       const int32_t id) noexcept {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[idx];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(
    vocab.token_storage.data() + static_cast<size_t>(offset), static_cast<size_t>(length));
}

inline std::string_view spm_merge_text(const emel::model::data::vocab &vocab,
                                       const int32_t idx) noexcept {
  const bool valid_idx = idx >= 0 && static_cast<uint32_t>(idx) < vocab.n_merges;
  const uint32_t merge_idx = select_u32(valid_idx, static_cast<uint32_t>(idx), 0u);
  const uint32_t raw_offset = vocab.merge_offsets[merge_idx];
  const uint32_t raw_length = vocab.merge_lengths[merge_idx];
  const size_t merge_end = static_cast<size_t>(raw_offset) + static_cast<size_t>(raw_length);
  const bool bounded = valid_idx && merge_end <= vocab.merge_storage.size();
  const uint32_t offset = select_u32(bounded, raw_offset, 0u);
  const uint32_t length = select_u32(bounded, raw_length, 0u);
  return std::string_view(
    vocab.merge_storage.data() + static_cast<size_t>(offset), static_cast<size_t>(length));
}

inline bool spm_merge_match(const std::string_view merge,
                            const std::string_view left,
                            const std::string_view right) noexcept {
  const size_t pos = merge.find(' ');
  const bool has_space = pos != std::string_view::npos;
  const size_t left_len = select_size(has_space, pos, static_cast<size_t>(0));
  const size_t right_start = select_size(has_space, pos + static_cast<size_t>(1), merge.size());
  const size_t right_len = merge.size() - right_start;
  const std::string_view left_view(merge.data(), left_len);
  const std::string_view right_view(merge.data() + right_start, right_len);
  const size_t expected_size = left.size() + right.size() + static_cast<size_t>(1);
  const bool size_ok = merge.size() == expected_size;
  return has_space && size_ok && left_view == left && right_view == right;
}

inline uint32_t spm_hash_bytes(const uint32_t seed, const std::string_view text) noexcept {
  uint32_t hash = seed;
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return select_u32(hash != 0u, hash, 1u);
}

inline uint32_t spm_hash_sv(const std::string_view text) noexcept {
  return spm_hash_bytes(k_fnv_offset, text);
}

inline uint32_t spm_hash_concat(const std::string_view left,
                                const std::string_view right) noexcept {
  return spm_hash_bytes(spm_hash_bytes(k_fnv_offset, left), right);
}

inline uint32_t spm_hash_pair(const std::string_view left, const std::string_view right) noexcept {
  const uint32_t h1 = spm_hash_sv(left);
  const uint32_t h2 = spm_hash_sv(right);
  const uint32_t mixed = h1 ^ (h2 + 0x9e3779b9u + (h1 << 6u) + (h1 >> 2u));
  return select_u32(mixed != 0u, mixed, 1u);
}

inline bool spm_insert_token_map(emel::text::encoders::detail::token_map &map,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view text,
                                 const int32_t id) noexcept {
  const bool text_empty = text.empty();
  bool done = text_empty;
  bool success = text_empty;
  const uint32_t hash = spm_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_id = map.values[slot];
    const std::string_view existing_text = spm_token_text(vocab, existing_id);
    const bool same_text = hash_match && existing_text == text;
    const bool claim_slot = empty_slot || same_text;

    map.hashes[slot] = select_u32(claim_slot, hash, slot_hash);
    map.values[slot] = select_i32(claim_slot, id, existing_id);
    map.count += static_cast<uint32_t>(claim_slot && empty_slot);

    success = success || claim_slot;
    done = done || claim_slot;
    slot = (slot + 1u) & mask;
  }

  return success;
}

inline bool spm_insert_merge_map(emel::text::encoders::detail::merge_map &map,
                                 const std::string_view left,
                                 const std::string_view right,
                                 const int32_t rank,
                                 const emel::model::data::vocab &vocab) noexcept {
  const bool active = !left.empty() && !right.empty();
  bool done = !active;
  bool success = false;
  const uint32_t hash = spm_hash_pair(left, right);
  const uint32_t mask = emel::text::encoders::detail::k_merge_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_merge_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_rank = map.values[slot];
    const std::string_view merge = spm_merge_text(vocab, existing_rank);
    const bool same_pair = hash_match && spm_merge_match(merge, left, right);
    const bool claim_slot = empty_slot || same_pair;

    map.hashes[slot] = select_u32(claim_slot, hash, slot_hash);
    map.values[slot] = select_i32(claim_slot, rank, existing_rank);
    map.count += static_cast<uint32_t>(claim_slot && empty_slot);

    success = success || claim_slot;
    done = done || claim_slot;
    slot = (slot + 1u) & mask;
  }

  return success;
}

inline int32_t spm_lookup_token(const emel::text::encoders::spm::action::context &ctx,
                                const std::string_view text) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const bool active = has_vocab && !text.empty();
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = spm_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.token_to_id.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const bool exact = hash_match && spm_token_text(*ctx.vocab, id) == text;

    resolved = select_i32(exact, id, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline int32_t spm_lookup_token_concat(const emel::text::encoders::spm::action::context &ctx,
                                       const std::string_view left,
                                       const std::string_view right) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const bool active = has_vocab && (!left.empty() || !right.empty());
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = spm_hash_concat(left, right);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  const size_t combined_len = left.size() + right.size();
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.token_to_id.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const std::string_view token = spm_token_text(*ctx.vocab, id);
    const bool size_match = token.size() == combined_len;
    const bool left_match = size_match && token.substr(0, left.size()) == left;
    const bool right_match = size_match && token.substr(left.size()) == right;
    const bool exact = hash_match && left_match && right_match;

    resolved = select_i32(exact, id, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline bool spm_push_token(const event::encode &ev, const int32_t token, int32_t &count) noexcept {
  int32_t sink = 0;
  const bool has_buffer = !ev.token_ids.empty();
  int32_t *base_ptrs[2] = {&sink, ev.token_ids.data()};
  int32_t *base = base_ptrs[static_cast<size_t>(has_buffer)];
  const bool non_negative_count = count >= 0;
  const int32_t safe_count = select_i32(non_negative_count, count, 0);
  const size_t count_index = static_cast<size_t>(safe_count);
  const bool has_space = has_buffer && non_negative_count && count_index < ev.token_ids.size();
  const bool write = token >= 0 && has_space;
  const size_t target_index = count_index * static_cast<size_t>(write);
  int32_t *target = base + target_index;
  *target = select_i32(write, token, *target);
  count += static_cast<int32_t>(write);
  return write;
}

inline bool spm_build_symbols(const std::string_view text,
                              emel::text::encoders::detail::encode_scratch &scratch,
                              encode_result &result) noexcept {
  scratch.symbol_count = 0;
  size_t offset = 0;
  bool ok = true;

  for (; ok && offset < text.size();) {
    const bool has_capacity = scratch.symbol_count < scratch.offsets.size();
    const size_t len_raw = emel::text::encoders::detail::utf8_len(text[offset]);
    const size_t remaining = text.size() - offset;
    const size_t len = select_size(len_raw <= remaining, len_raw, remaining);

    for (bool write = has_capacity; write; write = false) {
      const size_t idx = static_cast<size_t>(scratch.symbol_count);
      scratch.offsets[idx] = static_cast<uint32_t>(offset);
      scratch.lengths[idx] = static_cast<uint32_t>(len);
      scratch.prev[idx] = static_cast<int32_t>(scratch.symbol_count) - 1;
      const bool has_next = offset + len < text.size();
      scratch.next[idx] = select_i32(has_next, static_cast<int32_t>(scratch.symbol_count) + 1, -1);
      scratch.symbol_count += 1;
      offset += len;
    }

    ok = ok && has_capacity;
  }

  for (bool patch_head = scratch.symbol_count > 0; patch_head; patch_head = false) {
    scratch.prev[0] = -1;
  }

  const std::array<int32_t, 2> errors{EMEL_ERR_INVALID_ARGUMENT, EMEL_OK};
  result.error = errors[static_cast<size_t>(ok)];
  return ok;
}

inline void spm_merge_symbols(emel::text::encoders::detail::encode_scratch &scratch,
                              const int32_t left,
                              const int32_t right) noexcept {
  scratch.lengths[static_cast<size_t>(left)] += scratch.lengths[static_cast<size_t>(right)];
  const int32_t right_next = scratch.next[static_cast<size_t>(right)];
  scratch.next[static_cast<size_t>(left)] = right_next;
  for (bool patch_next = right_next >= 0; patch_next; patch_next = false) {
    scratch.prev[static_cast<size_t>(right_next)] = left;
  }
  scratch.lengths[static_cast<size_t>(right)] = 0;
}

inline bool spm_tables_ready(const emel::text::encoders::spm::action::context &ctx,
                             const emel::model::data::vocab &vocab) noexcept {
  return ctx.tables_ready && ctx.vocab == &vocab;
}

inline bool ensure_spm_tables(emel::text::encoders::spm::action::context &ctx) noexcept {
  const bool has_vocab = ctx.vocab != nullptr;
  const bool already_ready = has_vocab && ctx.tables_ready;
  bool ok = has_vocab;

  for (bool rebuild = has_vocab && !ctx.tables_ready; rebuild; rebuild = false) {
    ctx.token_to_id.clear();
    ctx.bpe_ranks.clear();
    ctx.max_token_len = 0;

    const emel::model::data::vocab &vocab = *ctx.vocab;
    for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
      const std::string_view text = spm_token_text(vocab, static_cast<int32_t>(id));
      const bool inserted = spm_insert_token_map(
        ctx.token_to_id, vocab, text, static_cast<int32_t>(id));
      ok = ok && inserted;
      const int32_t text_len = static_cast<int32_t>(text.size());
      const bool longer = text_len > ctx.max_token_len;
      ctx.max_token_len = select_i32(longer, text_len, ctx.max_token_len);
    }

    for (uint32_t idx = 0; idx < vocab.n_merges; ++idx) {
      const std::string_view merge = spm_merge_text(vocab, static_cast<int32_t>(idx));
      const size_t split = merge.find(' ');
      const bool has_pair = !merge.empty() && split != std::string_view::npos;
      for (bool insert_pair = has_pair; insert_pair; insert_pair = false) {
        const std::string_view left(merge.data(), split);
        const size_t right_start = split + static_cast<size_t>(1);
        const std::string_view right(merge.data() + right_start, merge.size() - right_start);
        spm_insert_merge_map(ctx.bpe_ranks, left, right, static_cast<int32_t>(idx), vocab);
      }
    }

    ctx.ugm_ready = vocab.precompiled_charsmap_size > 0;
    ctx.tables_ready = ok;
  }

  return has_vocab && (already_ready || ctx.tables_ready);
}

inline bool spm_emit_space_marker(emel::text::encoders::detail::encode_scratch &scratch,
                                  size_t &out_len,
                                  const bool escape_spaces) noexcept {
  constexpr std::array<char, 3> marker = {'\xE2', '\x96', '\x81'};
  const size_t marker_len = select_size(escape_spaces, static_cast<size_t>(3), static_cast<size_t>(1));
  const bool has_capacity = out_len + marker_len <= scratch.buffer.size();

  for (bool write = has_capacity; write; write = false) {
    for (size_t i = 0; i < marker_len; ++i) {
      const char plain = ' ';
      const int32_t escaped_i32 = static_cast<int32_t>(marker[i]);
      const int32_t plain_i32 = static_cast<int32_t>(plain);
      scratch.buffer[out_len + i] = static_cast<char>(select_i32(escape_spaces, escaped_i32, plain_i32));
    }
    out_len += marker_len;
  }

  return has_capacity;
}

inline bool spm_emit_char(emel::text::encoders::detail::encode_scratch &scratch,
                          size_t &out_len,
                          const char value) noexcept {
  const bool has_capacity = out_len + 1u <= scratch.buffer.size();
  for (bool write = has_capacity; write; write = false) {
    scratch.buffer[out_len] = value;
    out_len += 1u;
  }
  return has_capacity;
}

inline int32_t spm_byte_to_token(const emel::text::encoders::spm::action::context &ctx,
                                 const uint8_t byte) noexcept {
  constexpr std::array<char, 16> digits = {
    '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
  };

  std::array<char, 6> hex = {};
  hex[0] = '<';
  hex[1] = '0';
  hex[2] = 'x';
  hex[3] = digits[(byte >> 4u) & 0x0Fu];
  hex[4] = digits[byte & 0x0Fu];
  hex[5] = '>';

  const int32_t hex_token = spm_lookup_token(ctx, std::string_view(hex.data(), hex.size()));
  const char raw = static_cast<char>(byte);
  const int32_t raw_token = spm_lookup_token(ctx, std::string_view(&raw, 1));
  return select_i32(hex_token != k_token_null, hex_token, raw_token);
}

inline encode_result encode_spm(const event::encode &ev,
                                emel::text::encoders::spm::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  result.token_count = 0;

  for (bool empty_text = ev.text.empty(); empty_text; empty_text = false) {
    result.error = EMEL_OK;
    return result;
  }

  const bool tables_ready = spm_tables_ready(ctx, vocab);
  for (bool tables_missing = !tables_ready; tables_missing; tables_missing = false) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  size_t out_len = 0;
  const bool add_prefix = vocab.add_space_prefix && !vocab.treat_whitespace_as_suffix;
  const bool add_suffix = vocab.add_space_prefix && vocab.treat_whitespace_as_suffix;
  const bool escape_spaces = vocab.escape_whitespaces;
  bool prefix_inserted = false;

  for (const char c : ev.text) {
    const bool prefix_now = add_prefix && !prefix_inserted && c != ' ';
    bool prefix_ok = true;
    for (bool emit_prefix = prefix_now; emit_prefix; emit_prefix = false) {
      prefix_ok = spm_emit_space_marker(ctx.scratch, out_len, escape_spaces);
    }
    for (bool prefix_fail = prefix_now && !prefix_ok; prefix_fail; prefix_fail = false) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }
    prefix_inserted = prefix_inserted || prefix_now;

    const bool is_space = c == ' ';
    bool space_ok = true;
    for (bool emit_space = is_space; emit_space; emit_space = false) {
      space_ok = spm_emit_space_marker(ctx.scratch, out_len, escape_spaces);
    }
    for (bool space_fail = is_space && !space_ok; space_fail; space_fail = false) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }

    bool char_ok = true;
    for (bool emit_char = !is_space; emit_char; emit_char = false) {
      char_ok = spm_emit_char(ctx.scratch, out_len, c);
    }
    for (bool char_fail = !is_space && !char_ok; char_fail; char_fail = false) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }
  }

  bool suffix_ok = true;
  for (bool emit_suffix = add_suffix; emit_suffix; emit_suffix = false) {
    suffix_ok = spm_emit_space_marker(ctx.scratch, out_len, escape_spaces);
  }
  for (bool suffix_fail = add_suffix && !suffix_ok; suffix_fail; suffix_fail = false) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  const std::string_view escaped(ctx.scratch.buffer.data(), out_len);
  const bool symbols_ok = spm_build_symbols(escaped, ctx.scratch, result);
  for (bool build_fail = !symbols_ok; build_fail; build_fail = false) {
    return result;
  }

  for (bool keep_merging = ctx.scratch.symbol_count > 1; keep_merging;) {
    float best_score = -std::numeric_limits<float>::infinity();
    int32_t best_left = -1;
    int32_t best_right = -1;

    for (int32_t left = 0; left != -1; left = ctx.scratch.next[static_cast<size_t>(left)]) {
      const int32_t right = ctx.scratch.next[static_cast<size_t>(left)];
      for (bool has_right = right >= 0; has_right; has_right = false) {
        const size_t left_off = ctx.scratch.offsets[static_cast<size_t>(left)];
        const size_t left_len = ctx.scratch.lengths[static_cast<size_t>(left)];
        const size_t right_off = ctx.scratch.offsets[static_cast<size_t>(right)];
        const size_t right_len = ctx.scratch.lengths[static_cast<size_t>(right)];
        const std::string_view left_view = escaped.substr(left_off, left_len);
        const std::string_view right_view = escaped.substr(right_off, right_len);
        const int32_t token = spm_lookup_token_concat(ctx, left_view, right_view);

        for (bool has_token = token != k_token_null; has_token; has_token = false) {
          const float score = vocab.entries[static_cast<uint32_t>(token)].score;
          const bool better = score > best_score;
          const bool tie = score == best_score;
          const bool left_pref = best_left < 0 || left < best_left;
          const bool choose = better || (tie && left_pref);
          best_score = std::array<float, 2>{best_score, score}[static_cast<size_t>(choose)];
          best_left = select_i32(choose, left, best_left);
          best_right = select_i32(choose, right, best_right);
        }
      }
    }

    const bool has_best = best_left >= 0 && best_right >= 0;
    for (bool merge_once = has_best; merge_once; merge_once = false) {
      spm_merge_symbols(ctx.scratch, best_left, best_right);
    }
    keep_merging = has_best;
  }

  int32_t count = 0;
  for (int32_t idx = 0; idx != -1; idx = ctx.scratch.next[static_cast<size_t>(idx)]) {
    const bool has_symbol = ctx.scratch.lengths[static_cast<size_t>(idx)] != 0u;
    for (bool emit_symbol = has_symbol; emit_symbol; emit_symbol = false) {
      const size_t offset = ctx.scratch.offsets[static_cast<size_t>(idx)];
      const size_t length = ctx.scratch.lengths[static_cast<size_t>(idx)];
      const std::string_view symbol = escaped.substr(offset, length);
      const int32_t token = spm_lookup_token(ctx, symbol);

      bool direct_ok = true;
      for (bool emit_direct = token != k_token_null; emit_direct; emit_direct = false) {
        direct_ok = spm_push_token(ev, token, count);
      }
      for (bool direct_fail = (token != k_token_null) && !direct_ok;
           direct_fail;
           direct_fail = false) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }

      for (bool emit_bytes = token == k_token_null; emit_bytes; emit_bytes = false) {
        for (const unsigned char c : symbol) {
          const int32_t byte_token = spm_byte_to_token(ctx, c);
          const bool byte_valid = byte_token != k_token_null;
          bool byte_ok = false;
          for (bool push_byte = byte_valid; push_byte; push_byte = false) {
            byte_ok = spm_push_token(ev, byte_token, count);
          }
          for (bool byte_fail = !byte_valid || !byte_ok; byte_fail; byte_fail = false) {
            result.error = EMEL_ERR_BACKEND;
            return result;
          }
        }
      }
    }
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::text::encoders::spm::detail
