#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "emel/text/encoders/plamo2/context.hpp"
#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::text::encoders::plamo2::detail {

using emel::text::encoders::detail::encode_result;

inline int32_t select_i32(const bool choose_true, const int32_t true_value,
                          const int32_t false_value) noexcept {
  const int32_t mask = -static_cast<int32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline int64_t select_i64(const bool choose_true, const int64_t true_value,
                          const int64_t false_value) noexcept {
  const int64_t mask = -static_cast<int64_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint32_t select_u32(const bool choose_true, const uint32_t true_value,
                           const uint32_t false_value) noexcept {
  const uint32_t mask = static_cast<uint32_t>(0) - static_cast<uint32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline size_t select_size(const bool choose_true, const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint8_t select_u8(const bool choose_true, const uint8_t true_value,
                         const uint8_t false_value) noexcept {
  const uint8_t mask = static_cast<uint8_t>(0) - static_cast<uint8_t>(choose_true);
  return static_cast<uint8_t>((false_value & static_cast<uint8_t>(~mask)) |
                              (true_value & mask));
}

template <typename It>
inline int32_t iterator_second_or_i32_none(const It &, const int32_t fallback) noexcept {
  return fallback;
}

template <typename It>
inline int32_t iterator_second_or_i32_some(const It &it, const int32_t) noexcept {
  return it->second;
}

template <typename It>
inline int32_t iterator_second_or_i32(const It &it, const int32_t fallback,
                                      const bool has_value) noexcept {
  using load_handler_t = int32_t (*)(const It &, int32_t) noexcept;
  const load_handler_t load_handlers[2] = {
      iterator_second_or_i32_none<It>,
      iterator_second_or_i32_some<It>,
  };
  return load_handlers[static_cast<size_t>(has_value)](it, fallback);
}

template <typename It>
inline float iterator_second_or_f32_none(const It &, const float fallback) noexcept {
  return fallback;
}

template <typename It>
inline float iterator_second_or_f32_some(const It &it, const float) noexcept {
  return it->second;
}

template <typename It>
inline float iterator_second_or_f32(const It &it, const float fallback,
                                    const bool has_value) noexcept {
  using load_handler_t = float (*)(const It &, float) noexcept;
  const load_handler_t load_handlers[2] = {
      iterator_second_or_f32_none<It>,
      iterator_second_or_f32_some<It>,
  };
  return load_handlers[static_cast<size_t>(has_value)](it, fallback);
}

inline int32_t hex_nibble_value(const char ch, bool &valid) noexcept;

inline void parse_plamo2_byte_token_unsized(const std::string_view,
                                            bool &,
                                            uint8_t &) noexcept {}

inline void parse_plamo2_byte_token_sized(const std::string_view text,
                                          bool &parse_ok,
                                          uint8_t &byte_value) noexcept {
  const bool prefix_ok = text[0] == '<' && text[1] == '0' &&
                         (text[2] == 'x' || text[2] == 'X') && text[5] == '>';
  bool hi_valid = false;
  bool lo_valid = false;
  const int32_t hi = hex_nibble_value(text[3], hi_valid);
  const int32_t lo = hex_nibble_value(text[4], lo_valid);
  parse_ok = prefix_ok && hi_valid && lo_valid;
  const int32_t byte_i32 = (hi << 4) | lo;
  byte_value = static_cast<uint8_t>(select_i32(parse_ok, byte_i32, 0));
}

inline std::string_view plamo2_token_text(const emel::model::data::vocab &vocab,
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

inline int32_t hex_nibble_value(const char ch, bool &valid) noexcept {
  const uint8_t value = static_cast<uint8_t>(ch);
  const uint8_t lower = static_cast<uint8_t>(value | static_cast<uint8_t>(0x20u));
  const bool is_digit = value >= static_cast<uint8_t>('0') && value <= static_cast<uint8_t>('9');
  const bool is_hex = lower >= static_cast<uint8_t>('a') && lower <= static_cast<uint8_t>('f');
  valid = is_digit || is_hex;
  const int32_t digit_value = static_cast<int32_t>(value) - static_cast<int32_t>('0');
  const int32_t alpha_value = static_cast<int32_t>(lower) - static_cast<int32_t>('a') + 10;
  return select_i32(is_hex, alpha_value, digit_value);
}

inline void parse_plamo2_byte_token(const std::string_view text,
                                    bool &parse_ok,
                                    uint8_t &byte_value) noexcept {
  parse_ok = false;
  byte_value = 0;
  using parse_handler_t = void (*)(std::string_view, bool &, uint8_t &) noexcept;
  const parse_handler_t parse_handlers[2] = {
      parse_plamo2_byte_token_unsized,
      parse_plamo2_byte_token_sized,
  };
  parse_handlers[static_cast<size_t>(text.size() == 6u)](text, parse_ok, byte_value);
}

inline bool plamo2_push_token(const event::encode &ev, const int32_t token,
                              int32_t &count) noexcept {
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

inline void plamo2_push_token_none(const event::encode &, const int32_t, int32_t &,
                                   bool &pushed) noexcept {
  pushed = true;
}

inline void plamo2_push_token_some(const event::encode &ev, const int32_t token,
                                   int32_t &count, bool &pushed) noexcept {
  pushed = plamo2_push_token(ev, token, count);
}

inline void plamo2_collect_suffixes_none(std::unordered_map<std::string, float> &,
                                         const std::string_view,
                                         const float) {}

inline void plamo2_collect_suffixes_some(std::unordered_map<std::string, float> &suffix_to_score,
                                         const std::string_view text,
                                         const float score) {
  suffix_to_score[std::string(text)] = score;
  const std::vector<uint32_t> cpts = emel::text::unicode_cpts_from_utf8(std::string(text));
  for (size_t i = 1; i < cpts.size(); ++i) {
    std::string suffix;
    for (size_t j = i; j < cpts.size(); ++j) {
      suffix += emel::text::unicode_cpt_to_utf8(cpts[j]);
    }
    suffix_to_score.emplace(suffix, std::numeric_limits<float>::quiet_NaN());
  }
}

inline void plamo2_assign_byte_token_none(std::array<int32_t, 256> &, const uint8_t,
                                          const int32_t) noexcept {}

inline void plamo2_assign_byte_token_some(std::array<int32_t, 256> &byte_tokens,
                                          const uint8_t byte_value,
                                          const int32_t token_id) noexcept {
  byte_tokens[static_cast<size_t>(byte_value)] = token_id;
}

inline void plamo2_collect_vocab_token_none(
    emel::text::encoders::plamo2::action::context &, std::unordered_map<std::string, float> &,
    std::unordered_map<std::string, int32_t> &, const emel::model::data::vocab &,
    const uint32_t, const std::string_view) {}

inline void plamo2_collect_vocab_token_some(
    emel::text::encoders::plamo2::action::context &ctx,
    std::unordered_map<std::string, float> &suffix_to_score,
    std::unordered_map<std::string, int32_t> &token_to_id,
    const emel::model::data::vocab &vocab, const uint32_t token_id,
    const std::string_view text) {
  token_to_id[std::string(text)] = static_cast<int32_t>(token_id);

  const auto &entry = vocab.entries[token_id];
  const bool is_byte = entry.type == 6;

  bool byte_parse_ok = false;
  uint8_t byte_value = 0;
  parse_plamo2_byte_token(text, byte_parse_ok, byte_value);

  using assign_byte_handler_t = void (*)(std::array<int32_t, 256> &, uint8_t, int32_t) noexcept;
  const assign_byte_handler_t assign_byte_handlers[2] = {
      plamo2_assign_byte_token_none,
      plamo2_assign_byte_token_some,
  };
  assign_byte_handlers[static_cast<size_t>(is_byte && byte_parse_ok)](
      ctx.byte_tokens, byte_value, static_cast<int32_t>(token_id));

  using collect_suffix_handler_t =
      void (*)(std::unordered_map<std::string, float> &, std::string_view, float);
  const collect_suffix_handler_t collect_suffix_handlers[2] = {
      plamo2_collect_suffixes_none,
      plamo2_collect_suffixes_some,
  };
  collect_suffix_handlers[static_cast<size_t>(!is_byte)](suffix_to_score, text, entry.score);
}

inline int32_t plamo2_count_suffix_pieces_empty(
    emel::text::encoders::plamo2::action::context &, std::unordered_map<std::string, int32_t> &,
    const std::unordered_map<std::string, float> &, const std::string &, const int32_t) {
  return 1;
}

inline int32_t plamo2_count_suffix_pieces_non_empty(
    emel::text::encoders::plamo2::action::context &ctx,
    std::unordered_map<std::string, int32_t> &suffix_to_id,
    const std::unordered_map<std::string, float> &suffix_to_score,
    const std::string &suffix, const int32_t num_pieces) {
  const std::vector<uint32_t> cpts = emel::text::unicode_cpts_from_utf8(suffix);
  std::string remaining;
  for (size_t i = 1; i < cpts.size(); ++i) {
    remaining += emel::text::unicode_cpt_to_utf8(cpts[i]);
  }
  const int64_t piece_code =
      (static_cast<int64_t>(cpts[0]) << 32) | static_cast<uint32_t>(suffix_to_id[remaining]);
  ctx.suffix_map[piece_code] = num_pieces;

  int32_t pieces_for_suffix = 1;
  for (int32_t piece_len = static_cast<int32_t>(cpts.size()); piece_len > 0; --piece_len) {
    std::string piece;
    for (int32_t i = 0; i < piece_len; ++i) {
      piece += emel::text::unicode_cpt_to_utf8(cpts[static_cast<size_t>(i)]);
    }
    const bool has_piece = suffix_to_score.find(piece) != suffix_to_score.end();
    pieces_for_suffix += static_cast<int32_t>(has_piece);
  }
  return pieces_for_suffix;
}

inline void plamo2_emit_piece_none(emel::text::encoders::plamo2::action::context &,
                                   std::unordered_map<std::string, int32_t> &,
                                   const std::unordered_map<std::string, int32_t> &,
                                   const std::string &, const int32_t, const float, int32_t &,
                                   const int32_t) {}

inline void plamo2_emit_piece_some(
    emel::text::encoders::plamo2::action::context &ctx,
    std::unordered_map<std::string, int32_t> &suffix_to_id,
    const std::unordered_map<std::string, int32_t> &token_to_id, const std::string &piece,
    const int32_t piece_len, const float score, int32_t &table_idx,
    const int32_t k_invalid_score) {
  const auto token_it = token_to_id.find(piece);
  const bool has_token = token_it != token_to_id.end();
  const int32_t token_id = iterator_second_or_i32(token_it, -1, has_token);
  auto &row = ctx.table[static_cast<size_t>(table_idx)];
  row.piece_length = piece_len;
  row.token_id = token_id;
  const int32_t rounded = static_cast<int32_t>(std::round(score * 1e4f));
  row.score = select_i32(std::isfinite(score), rounded, k_invalid_score);
  row.piece_id = suffix_to_id[piece];
  table_idx += 1;
}

inline bool plamo2_finalize_tables_none(
    emel::text::encoders::plamo2::action::context &, std::unordered_map<std::string, float> &,
    std::unordered_map<std::string, int32_t> &) {
  return false;
}

inline bool plamo2_finalize_tables_some(
    emel::text::encoders::plamo2::action::context &ctx,
    std::unordered_map<std::string, float> &suffix_to_score,
    std::unordered_map<std::string, int32_t> &token_to_id) {
  std::vector<std::string> suffixes;
  suffixes.reserve(suffix_to_score.size() + 1);
  for (const auto &pair : suffix_to_score) {
    suffixes.push_back(pair.first);
  }
  suffixes.emplace_back();

  std::sort(suffixes.begin(), suffixes.end(), [](const std::string &a, const std::string &b) {
    const std::string rev_a(a.rbegin(), a.rend());
    const std::string rev_b(b.rbegin(), b.rend());
    return rev_a < rev_b;
  });

  std::unordered_map<std::string, int32_t> suffix_to_id;
  int32_t num_pieces = 0;
  for (const auto &suffix : suffixes) {
    suffix_to_id[suffix] = num_pieces;
    using count_suffix_handler_t = int32_t (*)(
        emel::text::encoders::plamo2::action::context &, std::unordered_map<std::string, int32_t> &,
        const std::unordered_map<std::string, float> &, const std::string &, int32_t);
    const count_suffix_handler_t count_suffix_handlers[2] = {
        plamo2_count_suffix_pieces_empty,
        plamo2_count_suffix_pieces_non_empty,
    };
    const int32_t piece_increase = count_suffix_handlers[static_cast<size_t>(!suffix.empty())](
        ctx, suffix_to_id, suffix_to_score, suffix, num_pieces);
    num_pieces += piece_increase;
  }

  ctx.table.resize(static_cast<size_t>(num_pieces));
  int32_t table_idx = 0;
  constexpr int32_t k_invalid_score = -20000000;
  constexpr int32_t k_unknown_score = -10000000;

  for (const auto &suffix : suffixes) {
    const std::vector<uint32_t> cpts = emel::text::unicode_cpts_from_utf8(suffix);
    for (int32_t piece_len = static_cast<int32_t>(cpts.size()); piece_len > 0; --piece_len) {
      std::string piece;
      for (int32_t i = 0; i < piece_len; ++i) {
        piece += emel::text::unicode_cpt_to_utf8(cpts[static_cast<size_t>(i)]);
      }
      const auto score_it = suffix_to_score.find(piece);
      const bool has_score = score_it != suffix_to_score.end();
      const float score = iterator_second_or_f32(score_it, 0.0f, has_score);
      using emit_piece_handler_t = void (*)(
          emel::text::encoders::plamo2::action::context &,
          std::unordered_map<std::string, int32_t> &,
          const std::unordered_map<std::string, int32_t> &, const std::string &, int32_t, float,
          int32_t &, int32_t);
      const emit_piece_handler_t emit_piece_handlers[2] = {
          plamo2_emit_piece_none,
          plamo2_emit_piece_some,
      };
      emit_piece_handlers[static_cast<size_t>(has_score)](
          ctx, suffix_to_id, token_to_id, piece, piece_len, score, table_idx, k_invalid_score);
    }
    auto &row = ctx.table[static_cast<size_t>(table_idx)];
    row.piece_length = 1;
    row.token_id = -1;
    row.score = k_unknown_score;
    row.piece_id = 0;
    table_idx += 1;
  }

  return true;
}

inline bool rebuild_plamo2_tables(emel::text::encoders::plamo2::action::context &ctx,
                                  const emel::model::data::vocab &vocab) {
  ctx.plamo2_vocab = &vocab;
  ctx.plamo2_tables_ready = false;
  ctx.byte_tokens.fill(0);
  ctx.suffix_map.clear();
  ctx.table.clear();

  std::unordered_map<std::string, float> suffix_to_score;
  std::unordered_map<std::string, int32_t> token_to_id;

  for (uint32_t token_id = 0; token_id < vocab.n_tokens; ++token_id) {
    const std::string_view text = plamo2_token_text(vocab, static_cast<int32_t>(token_id));
    using collect_token_handler_t = void (*)(
        emel::text::encoders::plamo2::action::context &, std::unordered_map<std::string, float> &,
        std::unordered_map<std::string, int32_t> &, const emel::model::data::vocab &, uint32_t,
        std::string_view);
    const collect_token_handler_t collect_token_handlers[2] = {
        plamo2_collect_vocab_token_none,
        plamo2_collect_vocab_token_some,
    };
    collect_token_handlers[static_cast<size_t>(!text.empty())](
        ctx, suffix_to_score, token_to_id, vocab, token_id, text);
  }

  bool byte_tokens_complete = true;
  for (size_t i = 0; i < ctx.byte_tokens.size(); ++i) {
    byte_tokens_complete = byte_tokens_complete && ctx.byte_tokens[i] != 0;
  }
  using finalize_tables_handler_t =
      bool (*)(emel::text::encoders::plamo2::action::context &,
               std::unordered_map<std::string, float> &,
               std::unordered_map<std::string, int32_t> &);
  const finalize_tables_handler_t finalize_tables_handlers[2] = {
      plamo2_finalize_tables_none,
      plamo2_finalize_tables_some,
  };
  const bool built = finalize_tables_handlers[static_cast<size_t>(byte_tokens_complete)](
      ctx, suffix_to_score, token_to_id);
  ctx.plamo2_tables_ready = built;
  return built;
}

inline bool keep_plamo2_tables(emel::text::encoders::plamo2::action::context &,
                               const emel::model::data::vocab &) {
  return true;
}

inline bool ensure_plamo2_tables(emel::text::encoders::plamo2::action::context &ctx,
                                 const emel::model::data::vocab &vocab) {
  const bool already_ready = ctx.plamo2_tables_ready && ctx.plamo2_vocab == &vocab;
  using ensure_tables_handler_t = bool (*)(emel::text::encoders::plamo2::action::context &,
                                           const emel::model::data::vocab &);
  const ensure_tables_handler_t ensure_tables_handlers[2] = {
      rebuild_plamo2_tables,
      keep_plamo2_tables,
  };
  return ensure_tables_handlers[static_cast<size_t>(already_ready)](ctx, vocab);
}

inline bool ensure_plamo2_tables_none(emel::text::encoders::plamo2::action::context &,
                                      const emel::model::data::vocab &) {
  return true;
}

inline bool ensure_plamo2_tables_some(emel::text::encoders::plamo2::action::context &ctx,
                                      const emel::model::data::vocab &vocab) {
  return ensure_plamo2_tables(ctx, vocab);
}

struct decode_result {
  int32_t data_len = 0;
  int32_t error = EMEL_OK;
};

inline void plamo2_decode_unicode_none(std::vector<uint32_t> &, const std::string_view) {}

inline void plamo2_decode_unicode_some(std::vector<uint32_t> &unicode_data,
                                       const std::string_view text) {
  unicode_data = emel::text::unicode_cpts_from_utf8(std::string(text));
}

inline decode_result decode_plamo2_input(const event::encode &ev,
                                         emel::text::encoders::plamo2::action::context &ctx,
                                         const int32_t prior_error) {
  decode_result result{};
  result.error = prior_error;
  std::vector<uint32_t> unicode_data;
  const bool decode_active = !ev.text.empty() && result.error == EMEL_OK;
  using decode_unicode_handler_t = void (*)(std::vector<uint32_t> &, std::string_view);
  const decode_unicode_handler_t decode_unicode_handlers[2] = {
      plamo2_decode_unicode_none,
      plamo2_decode_unicode_some,
  };
  decode_unicode_handlers[static_cast<size_t>(decode_active)](unicode_data, ev.text);

  const bool has_bom = decode_active && !unicode_data.empty() && unicode_data[0] == 0xFEFF;
  const size_t bom_offset = static_cast<size_t>(has_bom);
  const size_t decoded_len = unicode_data.size() - bom_offset;
  const bool too_long = decode_active && decoded_len > ctx.cpts.size();
  result.error = select_i32(result.error == EMEL_OK && too_long, EMEL_ERR_INVALID_ARGUMENT,
                            result.error);
  const bool copy_active = result.error == EMEL_OK;
  const size_t data_len = decoded_len * static_cast<size_t>(copy_active);
  for (size_t i = 0; i < data_len; ++i) {
    ctx.cpts[i] = unicode_data[bom_offset + i];
  }
  result.data_len = static_cast<int32_t>(data_len);
  return result;
}

inline void prepare_plamo2_dp(emel::text::encoders::plamo2::action::context &ctx,
                              const int32_t data_len_i32) {
  const int32_t safe_data_len_i32 = select_i32(data_len_i32 > 0, data_len_i32, 0);
  const size_t data_len = static_cast<size_t>(safe_data_len_i32);
  constexpr int64_t k_big = static_cast<int64_t>(1) << 60;
  for (size_t i = 0; i <= data_len; ++i) {
    ctx.scores[i] = k_big;
    ctx.paths[i] = {};
  }
  ctx.scores[data_len] = 0;
}

inline void run_plamo2_dp(emel::text::encoders::plamo2::action::context &ctx,
                          const int32_t data_len_i32) {
  constexpr int32_t k_invalid_score = -20000000;
  constexpr int32_t k_unknown_score = -10000000;
  int32_t suffix_id = 0;
  for (int32_t i = data_len_i32 - 1; i >= 0; --i) {
    const uint32_t c = ctx.cpts[static_cast<size_t>(i)];

    for (size_t p = static_cast<size_t>(suffix_id); p < ctx.table.size();) {
      const int64_t piece_code =
          (static_cast<int64_t>(c) << 32) | static_cast<uint32_t>(ctx.table[p].piece_id);
      const auto it = ctx.suffix_map.find(piece_code);
      const bool found = it != ctx.suffix_map.end();
      suffix_id = iterator_second_or_i32(it, 0, found);
      const bool stop = suffix_id > 0 || ctx.table[p].score == k_unknown_score;
      const size_t jump = select_size(stop, ctx.table.size() - p, static_cast<size_t>(1));
      p += jump;
    }

    for (size_t p = static_cast<size_t>(suffix_id); p < ctx.table.size();) {
      const int32_t score = ctx.table[p].score;
      const bool valid_score = score > k_invalid_score;
      const int32_t piece_length = ctx.table[p].piece_length;
      const bool valid_piece_length = piece_length > 0 && i + piece_length <= data_len_i32;
      const int32_t safe_piece_length = select_i32(valid_piece_length, piece_length, 0);
      const size_t i_idx = static_cast<size_t>(i);
      const size_t next_idx = static_cast<size_t>(i + safe_piece_length);
      const int64_t candidate_score = ctx.scores[next_idx] - static_cast<int64_t>(score);
      const bool better = valid_score && valid_piece_length && candidate_score < ctx.scores[i_idx];
      const int32_t utf8_extra = static_cast<int32_t>(c >= 0x80) +
                                 static_cast<int32_t>(c >= 0x800) +
                                 static_cast<int32_t>(c >= 0x10000);
      const int32_t add_unknown = static_cast<int32_t>(score == k_unknown_score) * utf8_extra;
      const int32_t next_num_tokens = ctx.paths[next_idx].num_tokens + 1 + add_unknown;
      ctx.scores[i_idx] = select_i64(better, candidate_score, ctx.scores[i_idx]);
      ctx.paths[i_idx].token_length = select_i32(better, piece_length, ctx.paths[i_idx].token_length);
      ctx.paths[i_idx].token_id = select_i32(better, ctx.table[p].token_id, ctx.paths[i_idx].token_id);
      ctx.paths[i_idx].num_tokens = select_i32(better, next_num_tokens, ctx.paths[i_idx].num_tokens);

      const bool stop_unknown = score == k_unknown_score;
      const size_t jump = select_size(stop_unknown, ctx.table.size() - p, static_cast<size_t>(1));
      p += jump;
    }
  }
}

inline encode_result emit_plamo2_tokens(const event::encode &ev,
                                        emel::text::encoders::plamo2::action::context &ctx,
                                        const int32_t data_len_i32,
                                        const int32_t prior_error) {
  encode_result result{};
  int32_t count = 0;
  int32_t pos = 0;
  bool loop_failed = prior_error != EMEL_OK;
  int32_t loop_error = EMEL_OK;
  using push_handler_t = void (*)(const event::encode &, int32_t, int32_t &, bool &) noexcept;
  const push_handler_t push_handlers[2] = {
      plamo2_push_token_none,
      plamo2_push_token_some,
  };
  while (pos < data_len_i32) {
    const auto &path = ctx.paths[static_cast<size_t>(pos)];
    const bool step_active = !loop_failed;
    const bool invalid_path = step_active && path.token_length <= 0;
    loop_error = select_i32(invalid_path, EMEL_ERR_BACKEND, loop_error);
    loop_failed = loop_failed || invalid_path;

    const bool direct_token = path.token_id >= 0;
    bool direct_push_ok = true;
    push_handlers[static_cast<size_t>(step_active && !loop_failed && direct_token)](ev,
                                                                    path.token_id, count,
                                                                    direct_push_ok);
    const bool direct_fail = step_active && !loop_failed && direct_token && !direct_push_ok;
    loop_error = select_i32(direct_fail, EMEL_ERR_INVALID_ARGUMENT, loop_error);
    loop_failed = loop_failed || direct_fail;

    const bool emit_bytes = step_active && !loop_failed && !direct_token;
    const int32_t safe_pos = select_i32(emit_bytes, pos, 0);
    const uint32_t c = ctx.cpts[static_cast<size_t>(safe_pos)];
    const int32_t s = 1 + static_cast<int32_t>(c >= 0x80) + static_cast<int32_t>(c >= 0x800) +
                      static_cast<int32_t>(c >= 0x10000);
    const int32_t emit_byte_count = s * static_cast<int32_t>(emit_bytes);
    for (int32_t i = 0; i < emit_byte_count; ++i) {
      const bool byte_step_active = !loop_failed;
      const uint8_t single_prefix = static_cast<uint8_t>(c);
      const uint8_t lead_prefix = static_cast<uint8_t>((0xF00 >> s) & 0xFF);
      uint8_t prefix = 0x80;
      prefix = select_u8(i == 0, lead_prefix, prefix);
      prefix = select_u8(s == 1, single_prefix, prefix);
      const uint8_t payload = static_cast<uint8_t>((c >> ((s - i - 1) * 6)) & 0x3F);
      const uint8_t b = static_cast<uint8_t>(prefix | payload);
      const int32_t byte_token = ctx.byte_tokens[static_cast<size_t>(b)];
      const bool byte_valid = byte_token > 0;
      bool byte_push_ok = true;
      push_handlers[static_cast<size_t>(byte_step_active && byte_valid)](
          ev, byte_token, count, byte_push_ok);
      const bool byte_fail = byte_step_active && (!byte_valid || !byte_push_ok);
      loop_error = select_i32(byte_fail, EMEL_ERR_INVALID_ARGUMENT, loop_error);
      loop_failed = loop_failed || byte_fail;
    }

    const int32_t step = select_i32(path.token_length > 0, path.token_length, 1);
    pos += step;
  }

  result.error = select_i32(prior_error == EMEL_OK && loop_failed, loop_error, prior_error);
  result.token_count = count * static_cast<int32_t>(result.error == EMEL_OK);
  return result;
}

inline encode_result encode_plamo2(const event::encode &ev,
                                   emel::text::encoders::plamo2::action::context &ctx,
                                   const emel::model::data::vocab &vocab) {
  encode_result result{};
  const bool has_text = !ev.text.empty();
  using ensure_tables_handler_t = bool (*)(emel::text::encoders::plamo2::action::context &,
                                           const emel::model::data::vocab &);
  const ensure_tables_handler_t ensure_tables_handlers[2] = {
      ensure_plamo2_tables_none,
      ensure_plamo2_tables_some,
  };
  const bool tables_ready = ensure_tables_handlers[static_cast<size_t>(has_text)](ctx, vocab);
  const int32_t tables_error = select_i32(has_text && !tables_ready, EMEL_ERR_MODEL_INVALID, EMEL_OK);
  const decode_result decoded = decode_plamo2_input(ev, ctx, tables_error);
  const bool run_dp = decoded.error == EMEL_OK && decoded.data_len > 0;
  const int32_t dp_len = decoded.data_len * static_cast<int32_t>(run_dp);
  prepare_plamo2_dp(ctx, dp_len);
  run_plamo2_dp(ctx, dp_len);
  result = emit_plamo2_tokens(ev, ctx, dp_len, decoded.error);
  return result;
}

}  // namespace emel::text::encoders::plamo2::detail
