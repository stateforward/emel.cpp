#pragma once

#include <algorithm>
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

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
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

inline void ensure_wpm_tables_rebuild_none(emel::text::encoders::action::context &,
                                           const emel::model::data::vocab &,
                                           bool &) noexcept {}

inline void ensure_wpm_tables_rebuild_some(emel::text::encoders::action::context &ctx,
                                           const emel::model::data::vocab &vocab,
                                           bool &ok) noexcept {
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

inline bool ensure_wpm_tables(emel::text::encoders::action::context &ctx,
                              const emel::model::data::vocab &vocab) noexcept {
  const bool already_ready = ctx.tables_ready && ctx.vocab == &vocab;
  bool ok = true;
  using rebuild_handler_t = void (*)(emel::text::encoders::action::context &,
                                     const emel::model::data::vocab &,
                                     bool &) noexcept;
  const rebuild_handler_t rebuild_handlers[2] = {
      ensure_wpm_tables_rebuild_none,
      ensure_wpm_tables_rebuild_some,
  };
  rebuild_handlers[static_cast<size_t>(!already_ready)](ctx, vocab, ok);
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

inline void wpm_preprocess_start_new_word_none(std::vector<std::string> &) {}

inline void wpm_preprocess_start_new_word_some(std::vector<std::string> &words) {
  words.emplace_back();
}

inline void wpm_preprocess_start_new_word_if_needed(std::vector<std::string> &words) {
  using start_handler_t = void (*)(std::vector<std::string> &);
  const start_handler_t start_handlers[2] = {
      wpm_preprocess_start_new_word_none,
      wpm_preprocess_start_new_word_some,
  };
  start_handlers[static_cast<size_t>(!words.back().empty())](words);
}

inline void wpm_preprocess_whitespace_none(std::vector<std::string> &) {}

inline void wpm_preprocess_whitespace_some(std::vector<std::string> &words) {
  wpm_preprocess_start_new_word_if_needed(words);
}

inline void wpm_preprocess_split_none(std::vector<std::string> &, const std::string &) {}

inline void wpm_preprocess_split_some(std::vector<std::string> &words,
                                      const std::string &token) {
  wpm_preprocess_start_new_word_if_needed(words);
  words.back() = token;
  words.emplace_back();
}

inline void wpm_preprocess_append_none(std::vector<std::string> &, const std::string &) {}

inline void wpm_preprocess_append_some(std::vector<std::string> &words,
                                       const std::string &token) {
  words.back() += token;
}

inline void wpm_preprocess_emit_none(std::vector<std::string> &,
                                     uint32_t,
                                     emel::text::unicode_cpt_flags) {}

inline void wpm_preprocess_emit_some(std::vector<std::string> &words,
                                     const uint32_t cpt,
                                     const emel::text::unicode_cpt_flags flags) {
  const std::string token =
      emel::text::unicode_cpt_to_utf8(emel::text::unicode_tolower(cpt));
  const bool split_token =
      flags.is_punctuation || (cpt < 0x7Fu && flags.is_symbol) ||
      emel::text::encoders::detail::is_chinese_char(cpt);

  using split_handler_t = void (*)(std::vector<std::string> &, const std::string &);
  const split_handler_t split_handlers[2] = {
      wpm_preprocess_split_none,
      wpm_preprocess_split_some,
  };
  split_handlers[static_cast<size_t>(split_token)](words, token);

  using append_handler_t = void (*)(std::vector<std::string> &, const std::string &);
  const append_handler_t append_handlers[2] = {
      wpm_preprocess_append_none,
      wpm_preprocess_append_some,
  };
  append_handlers[static_cast<size_t>(!split_token)](words, token);
}

inline void wpm_preprocess_trim_tail_none(std::vector<std::string> &) {}

inline void wpm_preprocess_trim_tail_some(std::vector<std::string> &words) {
  words.pop_back();
}

inline std::vector<std::string> wpm_preprocess(const std::string_view text) {
  const std::string utf8_text(text);
  const std::vector<uint32_t> cpts =
    emel::text::unicode_cpts_normalize_nfd(
      emel::text::unicode_cpts_from_utf8(utf8_text));
  std::vector<std::string> words(1, "");
  for (const uint32_t cpt : cpts) {
    const auto flags = emel::text::unicode_cpt_flags_from_cpt(cpt);
    using whitespace_handler_t = void (*)(std::vector<std::string> &);
    const whitespace_handler_t whitespace_handlers[2] = {
        wpm_preprocess_whitespace_none,
        wpm_preprocess_whitespace_some,
    };
    whitespace_handlers[static_cast<size_t>(flags.is_whitespace)](words);

    const bool invalid = cpt == 0u || cpt == 0xFFFDu || flags.is_control;
    const bool emit = !flags.is_whitespace && !invalid;
    using emit_handler_t = void (*)(std::vector<std::string> &,
                                    uint32_t,
                                    emel::text::unicode_cpt_flags);
    const emit_handler_t emit_handlers[2] = {
        wpm_preprocess_emit_none,
        wpm_preprocess_emit_some,
    };
    emit_handlers[static_cast<size_t>(emit)](words, cpt, flags);
  }
  using trim_tail_handler_t = void (*)(std::vector<std::string> &);
  const trim_tail_handler_t trim_tail_handlers[2] = {
      wpm_preprocess_trim_tail_none,
      wpm_preprocess_trim_tail_some,
  };
  trim_tail_handlers[static_cast<size_t>(!words.empty() && words.back().empty())](words);
  return words;
}

inline constexpr size_t k_wpm_continuation_prefix_len = 2u;
inline constexpr char k_wpm_continuation_prefix[] = "##";
inline constexpr size_t k_wpm_word_start_prefix_len = 3u;
inline constexpr char k_wpm_word_start_prefix[] = "\xE2\x96\x81";

inline void wpm_copy_word_none(emel::text::encoders::action::context &,
                               const std::string &) noexcept {}

inline void wpm_copy_word_some(emel::text::encoders::action::context &ctx,
                               const std::string &word) noexcept {
  std::memcpy(ctx.scratch.buffer.data(), word.data(), word.size());
}

inline void wpm_copy_continuation_piece_none(
    emel::text::encoders::action::context &,
    const std::string_view) noexcept {}

inline void wpm_copy_continuation_piece_some(
    emel::text::encoders::action::context &ctx,
    const std::string_view piece) noexcept {
  std::memcpy(ctx.scratch.buffer_alt.data(),
              k_wpm_continuation_prefix,
              k_wpm_continuation_prefix_len);
  std::memcpy(ctx.scratch.buffer_alt.data() + k_wpm_continuation_prefix_len,
              piece.data(),
              piece.size());
}

inline void wpm_copy_word_start_piece_none(
    emel::text::encoders::action::context &,
    const std::string_view) noexcept {}

inline void wpm_copy_word_start_piece_some(
    emel::text::encoders::action::context &ctx,
    const std::string_view piece) noexcept {
  std::memcpy(ctx.scratch.buffer_alt.data(),
              k_wpm_word_start_prefix,
              k_wpm_word_start_prefix_len);
  std::memcpy(ctx.scratch.buffer_alt.data() + k_wpm_word_start_prefix_len,
              piece.data(),
              piece.size());
}

inline void wpm_push_candidate_none(const event::encode &,
                                    const int32_t,
                                    int32_t &,
                                    bool &pushed) noexcept {
  pushed = true;
}

inline void wpm_push_candidate_some(const event::encode &ev,
                                    const int32_t token,
                                    int32_t &count,
                                    bool &pushed) noexcept {
  pushed = wpm_push_token(ev, token, count);
}

inline void wpm_resolve_unk_none(const emel::text::encoders::action::context &,
                                 const emel::model::data::vocab &,
                                 int32_t &) noexcept {}

inline void wpm_resolve_unk_some(const emel::text::encoders::action::context &ctx,
                                 const emel::model::data::vocab &vocab,
                                 int32_t &unk) noexcept {
  unk = wpm_lookup_token(ctx, vocab, "<unk>");
}

inline int32_t wpm_lookup_candidate_none(const emel::text::encoders::action::context &,
                                         const emel::model::data::vocab &,
                                         const std::string_view) noexcept {
  return k_token_null;
}

inline int32_t wpm_lookup_candidate_some(const emel::text::encoders::action::context &ctx,
                                         const emel::model::data::vocab &vocab,
                                         const std::string_view piece) noexcept {
  return wpm_lookup_token(ctx, vocab, piece);
}

inline bool encode_wpm_process_word_none(const event::encode &,
                                         emel::text::encoders::action::context &,
                                         const emel::model::data::vocab &,
                                         const std::string &,
                                         int32_t &,
                                         encode_result &) {
  return true;
}

inline bool encode_wpm_process_word_some(const event::encode &ev,
                                         emel::text::encoders::action::context &ctx,
                                         const emel::model::data::vocab &vocab,
                                         const std::string &word,
                                         int32_t &count,
                                         encode_result &result) {
  const int32_t word_token_start = count;
  const size_t word_len = word.size();
  const bool has_capacity = word_len <= ctx.scratch.buffer.size();
  using copy_handler_t = void (*)(emel::text::encoders::action::context &,
                                  const std::string &) noexcept;
  const copy_handler_t copy_handlers[2] = {
      wpm_copy_word_none,
      wpm_copy_word_some,
  };
  copy_handlers[static_cast<size_t>(has_capacity)](ctx, word);

  result.error = select_i32(!has_capacity, emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument), result.error);
  bool ok = has_capacity;
  const size_t word_view_len = select_size(has_capacity, word_len, 0u);
  const std::string_view word_view(ctx.scratch.buffer.data(), word_view_len);
  const int32_t n = static_cast<int32_t>(word_view.size());
  int32_t cursor = 0;

  for (int32_t step = 0; step < n; ++step) {
    const bool step_active = ok && cursor < n;
    const int32_t i = select_i32(step_active, cursor, 0);
    bool found = false;
    int32_t matched_end = i;
    const bool continuation = step_active && i > 0;
    const int32_t prefix_len = select_i32(
      continuation,
      static_cast<int32_t>(k_wpm_continuation_prefix_len),
      0);
    const int32_t max_piece_len = std::max(1, ctx.max_token_len - prefix_len);
    const int32_t end = select_i32(step_active, std::min(n, i + max_piece_len), i);
    bool scan_active = step_active;
    for (int32_t j = end; j > i; --j) {
      const bool scan_step_active = scan_active;
      const size_t piece_offset = static_cast<size_t>(select_i32(scan_step_active, i, 0));
      const size_t piece_size = static_cast<size_t>(select_i32(
        scan_step_active,
        j - i,
        0));
      const std::string_view raw_piece(word_view.data() + piece_offset, piece_size);
      const bool continuation_piece = scan_step_active && i > 0;

      for (int32_t candidate_index = 0; candidate_index < 2; ++candidate_index) {
        const bool candidate_scan_active = scan_active;
        const bool prefixed_candidate =
            candidate_scan_active && candidate_index == 0;
        const size_t prefix_len = select_size(
            continuation_piece,
            k_wpm_continuation_prefix_len,
            k_wpm_word_start_prefix_len);
        const size_t candidate_size =
            piece_size + (prefix_len * static_cast<size_t>(prefixed_candidate));
        const bool candidate_fits =
            !prefixed_candidate || candidate_size <= ctx.scratch.buffer_alt.size();

        using prefixed_handler_t = void (*)(emel::text::encoders::action::context &,
                                            const std::string_view) noexcept;
        const prefixed_handler_t prefixed_handlers[2][2] = {
            {
                wpm_copy_word_start_piece_none,
                wpm_copy_word_start_piece_some,
            },
            {
                wpm_copy_continuation_piece_none,
                wpm_copy_continuation_piece_some,
            },
        };
        prefixed_handlers[static_cast<size_t>(continuation_piece)]
                         [static_cast<size_t>(prefixed_candidate && candidate_fits)](
                             ctx, raw_piece);

        const char *candidate_bases[2] = {raw_piece.data(), ctx.scratch.buffer_alt.data()};
        const char *candidate_data =
            candidate_bases[static_cast<size_t>(prefixed_candidate)];
        const size_t safe_candidate_size = select_size(candidate_fits, candidate_size, 0u);
        const std::string_view piece(candidate_data, safe_candidate_size);

        using lookup_handler_t = int32_t (*)(const emel::text::encoders::action::context &,
                                             const emel::model::data::vocab &,
                                             const std::string_view) noexcept;
        const lookup_handler_t lookup_handlers[2] = {
            wpm_lookup_candidate_none,
            wpm_lookup_candidate_some,
        };
        const int32_t token =
            lookup_handlers[static_cast<size_t>(candidate_scan_active && candidate_fits)](
                ctx, vocab, piece);
        const bool hit = token != k_token_null;

        bool pushed = true;
        using push_handler_t = void (*)(const event::encode &,
                                        int32_t,
                                        int32_t &,
                                        bool &) noexcept;
        const push_handler_t push_handlers[2] = {
            wpm_push_candidate_none,
            wpm_push_candidate_some,
        };
        push_handlers[static_cast<size_t>(candidate_scan_active && hit)](
            ev, token, count, pushed);

        const bool found_step = candidate_scan_active && hit;
        const bool push_fail = found_step && !pushed;
        result.error = select_i32(
            push_fail,
            emel::text::encoders::error::to_emel(
                emel::text::encoders::error::code::invalid_argument),
            result.error);
        ok = ok && !push_fail;
        found = found || found_step;
        matched_end = select_i32(found_step, j, matched_end);
        scan_active = scan_active && !push_fail && !found_step;
      }
    }

    const bool advance_cursor = step_active && found;
    cursor = select_i32(advance_cursor, matched_end, cursor);
    const bool rollback = step_active && !found;
    count = select_i32(rollback, word_token_start, count);
    cursor = select_i32(rollback, n, cursor);
  }

  const bool needs_unk = ok && count == word_token_start;
  int32_t unk = vocab.unk_id;
  using resolve_handler_t = void (*)(const emel::text::encoders::action::context &,
                                     const emel::model::data::vocab &,
                                     int32_t &) noexcept;
  const resolve_handler_t resolve_handlers[2] = {
      wpm_resolve_unk_none,
      wpm_resolve_unk_some,
  };
  resolve_handlers[static_cast<size_t>(needs_unk && unk == k_token_null)](ctx, vocab, unk);

  const bool have_unk = needs_unk && unk != k_token_null;
  bool pushed_unk = true;
  using push_handler_t = void (*)(const event::encode &,
                                  int32_t,
                                  int32_t &,
                                  bool &) noexcept;
  const push_handler_t push_handlers[2] = {
      wpm_push_candidate_none,
      wpm_push_candidate_some,
  };
  push_handlers[static_cast<size_t>(have_unk)](ev, unk, count, pushed_unk);
  const bool push_fail_unk = have_unk && !pushed_unk;
  result.error = select_i32(push_fail_unk, emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument), result.error);
  ok = ok && !push_fail_unk;
  return ok;
}

inline encode_result encode_wpm_ready_tables(const event::encode &ev,
                                             emel::text::encoders::action::context &ctx,
                                             const emel::model::data::vocab &vocab) {
  encode_result result{};
  result.token_count = 0;
  result.error = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);

  int32_t count = 0;
  const std::vector<std::string> words = wpm_preprocess(ev.text);
  bool ok = true;

  for (const std::string &word : words) {
    using process_word_handler_t = bool (*)(const event::encode &,
                                            emel::text::encoders::action::context &,
                                            const emel::model::data::vocab &,
                                            const std::string &,
                                            int32_t &,
                                            encode_result &);
    const process_word_handler_t process_word_handlers[2] = {
        encode_wpm_process_word_none,
        encode_wpm_process_word_some,
    };
    const bool processed_ok = process_word_handlers[static_cast<size_t>(!word.empty())](
        ev, ctx, vocab, word, count, result);
    ok = ok && processed_ok;
  }

  const bool success = ok && result.error == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  result.token_count = select_i32(success, count, 0);
  return result;
}

inline encode_result encode_wpm_missing_tables(const event::encode &,
                                               emel::text::encoders::action::context &,
                                               const emel::model::data::vocab &) {
  encode_result result{};
  result.token_count = 0;
  result.error = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
  return result;
}

inline encode_result encode_wpm_empty(const event::encode &,
                                      emel::text::encoders::action::context &,
                                      const emel::model::data::vocab &) {
  encode_result result{};
  result.token_count = 0;
  result.error = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  return result;
}

}  // namespace emel::text::encoders::wpm::detail
