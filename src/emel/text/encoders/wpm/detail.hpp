#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "emel/text/encoders/wpm/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::text::encoders::wpm::detail {

using emel::text::encoders::detail::encode_result;
using emel::text::encoders::detail::k_token_null;

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

constexpr uint32_t k_fnv_offset = 2166136261u;
constexpr uint32_t k_fnv_prime = 16777619u;

inline uint32_t wpm_hash_bytes(const uint32_t seed, const std::string_view text) noexcept {
  uint32_t hash = seed;
  for (const unsigned char byte : text) {
    hash ^= byte;
    hash *= k_fnv_prime;
  }
  return select_u32(hash != 0u, hash, 1u);
}

inline uint32_t wpm_hash_sv(const std::string_view text) noexcept {
  return wpm_hash_bytes(k_fnv_offset, text);
}

inline std::string_view wpm_token_text(const emel::model::data::vocab &vocab,
                                       const int32_t id) noexcept {
  const bool valid_id = id >= 0 && static_cast<uint32_t>(id) < vocab.n_tokens;
  const uint32_t idx = select_u32(valid_id, static_cast<uint32_t>(id), 0u);
  const auto &entry = vocab.entries[idx];
  const bool has_text = valid_id && entry.text_length > 0u;
  const uint32_t offset = select_u32(has_text, entry.text_offset, 0u);
  const uint32_t length = select_u32(has_text, entry.text_length, 0u);
  return std::string_view(
    vocab.token_storage.data() + static_cast<size_t>(offset),
    static_cast<size_t>(length));
}

inline bool wpm_insert_token_map(emel::text::encoders::detail::token_map &map,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view text,
                                 const int32_t id) noexcept {
  const bool text_empty = text.empty();
  bool done = text_empty;
  bool success = text_empty;
  const uint32_t hash = wpm_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = map.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t existing_id = map.values[slot];
    const std::string_view existing_text = wpm_token_text(vocab, existing_id);
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

inline bool ensure_wpm_tables(emel::text::encoders::action::context &ctx,
                              const emel::model::data::vocab &vocab) noexcept {
  const bool already_ready = ctx.tables_ready && ctx.vocab == &vocab;
  bool ok = true;

  for (bool rebuild = !already_ready; rebuild; rebuild = false) {
    ctx.vocab = &vocab;
    ctx.tables_ready = false;
    ctx.token_to_id.clear();
    ctx.max_token_len = 0;

    for (uint32_t id = 0; id < vocab.n_tokens; ++id) {
      const std::string_view text = wpm_token_text(vocab, static_cast<int32_t>(id));
      const bool inserted = wpm_insert_token_map(
        ctx.token_to_id, vocab, text, static_cast<int32_t>(id));
      ok = ok && inserted;
      const int32_t text_len = static_cast<int32_t>(text.size());
      const bool longer = text_len > ctx.max_token_len;
      ctx.max_token_len = select_i32(longer, text_len, ctx.max_token_len);
    }

    ctx.tables_ready = ok;
  }

  return already_ready || ctx.tables_ready;
}

inline int32_t wpm_lookup_token(const emel::text::encoders::action::context &ctx,
                                const emel::model::data::vocab &vocab,
                                const std::string_view text) noexcept {
  const bool active = !text.empty();
  bool done = !active;
  int32_t resolved = k_token_null;
  const uint32_t hash = wpm_hash_sv(text);
  const uint32_t mask = emel::text::encoders::detail::k_token_hash_size - 1u;
  uint32_t slot = hash & mask;

  for (uint32_t probes = 0;
       probes < emel::text::encoders::detail::k_token_hash_size && !done;
       ++probes) {
    const uint32_t slot_hash = ctx.token_to_id.hashes[slot];
    const bool empty_slot = slot_hash == 0u;
    const bool hash_match = slot_hash == hash;
    const int32_t id = ctx.token_to_id.values[slot];
    const bool exact = hash_match && wpm_token_text(vocab, id) == text;

    resolved = select_i32(exact, id, resolved);
    done = done || empty_slot || exact;
    slot = (slot + 1u) & mask;
  }

  return resolved;
}

inline bool wpm_push_token(const event::encode &ev, const int32_t token, int32_t &count) noexcept {
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

inline std::vector<std::string> wpm_preprocess(const std::string_view text) {
  const std::string utf8_text(text);
  const std::vector<uint32_t> cpts =
    emel::text::unicode_cpts_normalize_nfd(
      emel::text::unicode_cpts_from_utf8(utf8_text));
  std::vector<std::string> words(1, "");
  for (const uint32_t cpt : cpts) {
    const auto flags = emel::text::unicode_cpt_flags_from_cpt(cpt);

    for (bool is_whitespace = flags.is_whitespace; is_whitespace; is_whitespace = false) {
      for (bool start_new_word = !words.back().empty(); start_new_word; start_new_word = false) {
        words.emplace_back();
      }
    }

    const bool invalid = cpt == 0u || cpt == 0xFFFDu || flags.is_control;
    const bool emit = !flags.is_whitespace && !invalid;
    for (bool process = emit; process; process = false) {
      const std::string s =
        emel::text::unicode_cpt_to_utf8(emel::text::unicode_tolower(cpt));
      const bool split_token =
        flags.is_punctuation || (cpt < 0x7Fu && flags.is_symbol) ||
        emel::text::encoders::detail::is_chinese_char(cpt)) {
      for (bool split = split_token; split; split = false) {
        for (bool start_new_word = !words.back().empty(); start_new_word; start_new_word = false) {
          words.emplace_back();
        }
        words.back() = s;
        words.emplace_back();
      }
      for (bool append = !split_token; append; append = false) {
        words.back() += s;
      }
    }
  }
  for (bool trim_tail = !words.empty() && words.back().empty(); trim_tail; trim_tail = false) {
    words.pop_back();
  }
  return words;
}

inline encode_result encode_wpm(const event::encode &ev,
                                emel::text::encoders::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  result.token_count = 0;

  for (bool empty_text = ev.text.empty(); empty_text; empty_text = false) {
    result.error = EMEL_OK;
    return result;
  }

  const bool tables_ready = ctx.tables_ready && ctx.vocab == &vocab;
  for (bool missing_tables = !tables_ready; missing_tables; missing_tables = false) {
    result.error = EMEL_ERR_INVALID_ARGUMENT;
    return result;
  }

  int32_t count = 0;
  const std::vector<std::string> words = wpm_preprocess(ev.text);
  const char *prefix = "\xE2\x96\x81";
  constexpr size_t prefix_len = 3;

  for (const std::string &word : words) {
    for (bool process_word = !word.empty(); process_word; process_word = false) {
      const int32_t word_token_start = count;
      const size_t word_len = word.size();
      const bool has_capacity = prefix_len + word_len <= ctx.scratch.buffer.size();
      for (bool overflow = !has_capacity; overflow; overflow = false) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      std::memcpy(ctx.scratch.buffer.data(), prefix, prefix_len);
      std::memcpy(ctx.scratch.buffer.data() + prefix_len, word.data(), word_len);
      const std::string_view word_view(ctx.scratch.buffer.data(),
                                       prefix_len + word_len);
      const int32_t n = static_cast<int32_t>(word_view.size());
      for (int32_t i = 0; i < n; ++i) {
        bool found = false;
        int32_t matched_end = i;
        const int32_t end = std::min(n, i + ctx.max_token_len + 1);
        for (int32_t j = end; j > i; --j) {
          const std::string_view piece = word_view.substr(
            static_cast<size_t>(i),
            static_cast<size_t>(j - i));
          const int32_t token = wpm_lookup_token(ctx, vocab, piece);
          for (bool hit = token != k_token_null && !found; hit; hit = false) {
            const bool pushed = wpm_push_token(ev, token, count);
            for (bool push_fail = !pushed; push_fail; push_fail = false) {
              result.error = EMEL_ERR_INVALID_ARGUMENT;
              return result;
            }
            found = true;
            matched_end = j;
          }
        }
        i = select_i32(found, matched_end - 1, i);
        for (bool rollback = !found; rollback; rollback = false) {
          count = word_token_start;
          i = n;
        }
      }

      for (bool needs_unk = count == word_token_start; needs_unk; needs_unk = false) {
        int32_t unk = vocab.unk_id;
        for (bool resolve_unk = unk == k_token_null; resolve_unk; resolve_unk = false) {
          unk = wpm_lookup_token(ctx, vocab, "<unk>");
        }
        for (bool have_unk = unk != k_token_null; have_unk; have_unk = false) {
          const bool pushed = wpm_push_token(ev, unk, count);
          for (bool push_fail = !pushed; push_fail; push_fail = false) {
            result.error = EMEL_ERR_INVALID_ARGUMENT;
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

}  // namespace emel::text::encoders::wpm::detail
