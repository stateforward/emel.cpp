#pragma once

#include <algorithm>
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

inline uint8_t select_u8(const bool choose_true,
                         const uint8_t true_value,
                         const uint8_t false_value) noexcept {
  const uint8_t mask = static_cast<uint8_t>(0) - static_cast<uint8_t>(choose_true);
  return static_cast<uint8_t>((false_value & static_cast<uint8_t>(~mask)) | (true_value & mask));
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
  for (bool sized = text.size() == 6u; sized; sized = false) {
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
}

inline bool plamo2_push_token(const event::encode &ev, const int32_t token, int32_t &count) noexcept {
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

inline bool ensure_plamo2_tables(emel::text::encoders::plamo2::action::context &ctx,
                                 const emel::model::data::vocab &vocab) {
  for (bool already_ready = ctx.plamo2_tables_ready && ctx.plamo2_vocab == &vocab;
       already_ready;
       already_ready = false) {
    return true;
  }
  ctx.plamo2_vocab = &vocab;
  ctx.plamo2_tables_ready = false;
  ctx.byte_tokens.fill(0);
  ctx.suffix_map.clear();
  ctx.table.clear();

  std::unordered_map<std::string, float> suffix_to_score;
  std::unordered_map<std::string, int32_t> token_to_id;

  for (uint32_t token_id = 0; token_id < vocab.n_tokens; ++token_id) {
    const std::string_view text = plamo2_token_text(vocab, static_cast<int32_t>(token_id));
    for (bool has_text = !text.empty(); has_text; has_text = false) {
      token_to_id[std::string(text)] = static_cast<int32_t>(token_id);

      const auto &entry = vocab.entries[token_id];
      const bool is_byte = entry.type == 6;

      bool byte_parse_ok = false;
      uint8_t byte_value = 0;
      parse_plamo2_byte_token(text, byte_parse_ok, byte_value);

      for (bool apply_byte = is_byte && byte_parse_ok; apply_byte; apply_byte = false) {
        ctx.byte_tokens[static_cast<size_t>(byte_value)] = static_cast<int32_t>(token_id);
      }

      for (bool apply_suffix = !is_byte; apply_suffix; apply_suffix = false) {
        suffix_to_score[std::string(text)] = entry.score;
        const std::vector<uint32_t> cpts = emel::text::unicode_cpts_from_utf8(std::string(text));
        for (size_t i = 1; i < cpts.size(); ++i) {
          std::string suffix;
          for (size_t j = i; j < cpts.size(); ++j) {
            suffix += emel::text::unicode_cpt_to_utf8(cpts[j]);
          }
          const bool missing = suffix_to_score.find(suffix) == suffix_to_score.end();
          for (bool insert_missing = missing; insert_missing; insert_missing = false) {
            suffix_to_score[suffix] = std::numeric_limits<float>::quiet_NaN();
          }
        }
      }
    }
  }

  bool byte_tokens_complete = true;
  for (size_t i = 0; i < ctx.byte_tokens.size(); ++i) {
    byte_tokens_complete = byte_tokens_complete && ctx.byte_tokens[i] != 0;
  }
  for (bool missing_byte = !byte_tokens_complete; missing_byte; missing_byte = false) {
    return false;
  }

  std::vector<std::string> suffixes;
  suffixes.reserve(suffix_to_score.size() + 1);
  for (const auto &pair : suffix_to_score) {
    suffixes.push_back(pair.first);
  }
  suffixes.emplace_back();

  std::sort(suffixes.begin(), suffixes.end(),
            [](const std::string &a, const std::string &b) {
              const std::string rev_a(a.rbegin(), a.rend());
              const std::string rev_b(b.rbegin(), b.rend());
              return rev_a < rev_b;
            });

  std::unordered_map<std::string, int32_t> suffix_to_id;
  int32_t num_pieces = 0;
  for (const auto &suffix : suffixes) {
    suffix_to_id[suffix] = num_pieces;
    for (bool non_empty_suffix = !suffix.empty(); non_empty_suffix; non_empty_suffix = false) {
      const std::vector<uint32_t> cpts =
          emel::text::unicode_cpts_from_utf8(suffix);
      std::string remaining;
      for (size_t i = 1; i < cpts.size(); ++i) {
        remaining += emel::text::unicode_cpt_to_utf8(cpts[i]);
      }
      const int64_t piece_code =
          (static_cast<int64_t>(cpts[0]) << 32) |
          static_cast<uint32_t>(suffix_to_id[remaining]);
      ctx.suffix_map[piece_code] = num_pieces;

      int32_t pieces_for_suffix = 1;
      for (int32_t piece_len = static_cast<int32_t>(cpts.size()); piece_len > 0;
           --piece_len) {
        std::string piece;
        for (int32_t i = 0; i < piece_len; ++i) {
          piece += emel::text::unicode_cpt_to_utf8(cpts[static_cast<size_t>(i)]);
        }
        const bool has_piece = suffix_to_score.find(piece) != suffix_to_score.end();
        pieces_for_suffix += static_cast<int32_t>(has_piece);
      }
      num_pieces += pieces_for_suffix;
    }
    for (bool empty_suffix = suffix.empty(); empty_suffix; empty_suffix = false) {
      num_pieces += 1;
    }
  }

  ctx.table.resize(static_cast<size_t>(num_pieces));
  int32_t table_idx = 0;
  constexpr int32_t k_invalid_score = -20000000;
  constexpr int32_t k_unknown_score = -10000000;

  for (const auto &suffix : suffixes) {
    const std::vector<uint32_t> cpts =
        emel::text::unicode_cpts_from_utf8(suffix);
    for (int32_t piece_len = static_cast<int32_t>(cpts.size()); piece_len > 0;
         --piece_len) {
      std::string piece;
      for (int32_t i = 0; i < piece_len; ++i) {
        piece += emel::text::unicode_cpt_to_utf8(cpts[static_cast<size_t>(i)]);
      }
      auto score_it = suffix_to_score.find(piece);
      const bool has_score = score_it != suffix_to_score.end();
      for (bool emit_piece = has_score; emit_piece; emit_piece = false) {
        auto token_it = token_to_id.find(piece);
        const bool has_token = token_it != token_to_id.end();
        ctx.table[static_cast<size_t>(table_idx)].piece_length = piece_len;
        ctx.table[static_cast<size_t>(table_idx)].token_id =
            select_i32(has_token, token_it->second, -1);
        const float score = score_it->second;
        const int32_t rounded = static_cast<int32_t>(std::round(score * 1e4f));
        ctx.table[static_cast<size_t>(table_idx)].score =
            select_i32(std::isfinite(score), rounded, k_invalid_score);
        ctx.table[static_cast<size_t>(table_idx)].piece_id =
            suffix_to_id[piece];
        table_idx += 1;
      }
    }
    ctx.table[static_cast<size_t>(table_idx)].piece_length = 1;
    ctx.table[static_cast<size_t>(table_idx)].token_id = -1;
    ctx.table[static_cast<size_t>(table_idx)].score = k_unknown_score;
    ctx.table[static_cast<size_t>(table_idx)].piece_id = 0;
    table_idx += 1;
  }

  ctx.plamo2_tables_ready = true;
  return true;
}

inline encode_result encode_plamo2(const event::encode &ev,
                                   emel::text::encoders::plamo2::action::context &ctx,
                                   const emel::model::data::vocab &vocab) {
  encode_result result{};
  for (bool has_text = !ev.text.empty(); has_text; has_text = false) {
    const bool tables_ready = ensure_plamo2_tables(ctx, vocab);
    for (bool table_error = !tables_ready; table_error; table_error = false) {
      result.error = EMEL_ERR_MODEL_INVALID;
      return result;
    }

    std::vector<uint32_t> unicode_data =
        emel::text::unicode_cpts_from_utf8(std::string(ev.text));
    const bool has_bom = !unicode_data.empty() && unicode_data[0] == 0xFEFF;
    for (bool drop_bom = has_bom; drop_bom; drop_bom = false) {
      unicode_data.erase(unicode_data.begin());
    }
    for (bool no_data = unicode_data.empty(); no_data; no_data = false) {
      result.error = EMEL_OK;
      return result;
    }
    for (bool too_long = unicode_data.size() > ctx.cpts.size(); too_long; too_long = false) {
      result.error = EMEL_ERR_INVALID_ARGUMENT;
      return result;
    }

    const size_t data_len = unicode_data.size();
    constexpr int64_t k_big = static_cast<int64_t>(1) << 60;
    for (size_t i = 0; i <= data_len; ++i) {
      ctx.scores[i] = k_big;
      ctx.paths[i] = {};
    }
    ctx.scores[data_len] = 0;

    constexpr int32_t k_invalid_score = -20000000;
    constexpr int32_t k_unknown_score = -10000000;

    int32_t suffix_id = 0;
    const int32_t data_len_i32 = static_cast<int32_t>(data_len);
    for (int32_t i = data_len_i32 - 1; i >= 0; --i) {
      const uint32_t c = unicode_data[static_cast<size_t>(i)];

      for (size_t p = static_cast<size_t>(suffix_id); p < ctx.table.size(); ++p) {
        const int64_t piece_code =
            (static_cast<int64_t>(c) << 32) |
            static_cast<uint32_t>(ctx.table[p].piece_id);
        const auto it = ctx.suffix_map.find(piece_code);
        const bool found = it != ctx.suffix_map.end();
        suffix_id = select_i32(found, it->second, 0);
        const bool stop = suffix_id > 0 || ctx.table[p].score == k_unknown_score;
        for (bool stop_scan = stop; stop_scan; stop_scan = false) {
          p = ctx.table.size();
        }
      }

      for (size_t p = static_cast<size_t>(suffix_id); p < ctx.table.size(); ++p) {
        const int32_t score = ctx.table[p].score;
        for (bool valid_score = score > k_invalid_score; valid_score; valid_score = false) {
          const int32_t piece_length = ctx.table[p].piece_length;
          const bool valid_piece_length =
              piece_length > 0 && i + piece_length <= data_len_i32;
          for (bool valid_piece = valid_piece_length; valid_piece; valid_piece = false) {
            const int64_t s = ctx.scores[static_cast<size_t>(i + piece_length)] - score;
            const bool better = s < ctx.scores[static_cast<size_t>(i)];
            for (bool update_best = better; update_best; update_best = false) {
              ctx.scores[static_cast<size_t>(i)] = s;
              ctx.paths[static_cast<size_t>(i)].token_length = piece_length;
              ctx.paths[static_cast<size_t>(i)].token_id = ctx.table[p].token_id;
              ctx.paths[static_cast<size_t>(i)].num_tokens =
                  ctx.paths[static_cast<size_t>(i + piece_length)].num_tokens + 1;
              const int32_t utf8_extra =
                  static_cast<int32_t>(c >= 0x80) +
                  static_cast<int32_t>(c >= 0x800) +
                  static_cast<int32_t>(c >= 0x10000);
              const int32_t add_unknown =
                  static_cast<int32_t>(score == k_unknown_score) * utf8_extra;
              ctx.paths[static_cast<size_t>(i)].num_tokens += add_unknown;
            }
          }
        }
        for (bool stop_unknown = score == k_unknown_score; stop_unknown; stop_unknown = false) {
          p = ctx.table.size();
        }
      }
    }

    int32_t count = 0;
    int32_t pos = 0;
    while (pos < data_len_i32) {
      const auto &path = ctx.paths[static_cast<size_t>(pos)];
      for (bool invalid_path = path.token_length <= 0; invalid_path; invalid_path = false) {
        result.error = EMEL_ERR_BACKEND;
        return result;
      }
      const bool direct_token = path.token_id >= 0;
      bool direct_push_ok = true;
      for (bool emit_direct = direct_token; emit_direct; emit_direct = false) {
        direct_push_ok = plamo2_push_token(ev, path.token_id, count);
      }
      for (bool direct_fail = direct_token && !direct_push_ok; direct_fail; direct_fail = false) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      for (bool emit_bytes = !direct_token; emit_bytes; emit_bytes = false) {
        const uint32_t c = unicode_data[static_cast<size_t>(pos)];
        const int32_t s = 1 + static_cast<int32_t>(c >= 0x80) +
                          static_cast<int32_t>(c >= 0x800) +
                          static_cast<int32_t>(c >= 0x10000);
        for (int32_t i = 0; i < s; ++i) {
          const uint8_t single_prefix = static_cast<uint8_t>(c);
          const uint8_t lead_prefix = static_cast<uint8_t>((0xF00 >> s) & 0xFF);
          uint8_t prefix = 0x80;
          prefix = select_u8(i == 0, lead_prefix, prefix);
          prefix = select_u8(s == 1, single_prefix, prefix);
          const uint8_t payload =
              static_cast<uint8_t>((c >> ((s - i - 1) * 6)) & 0x3F);
          const uint8_t b = static_cast<uint8_t>(prefix | payload);
          const int32_t byte_token = ctx.byte_tokens[static_cast<size_t>(b)];
          const bool byte_valid = byte_token > 0;
          bool byte_push_ok = false;
          for (bool emit_byte = byte_valid; emit_byte; emit_byte = false) {
            byte_push_ok = plamo2_push_token(ev, byte_token, count);
          }
          for (bool byte_fail = !byte_valid || !byte_push_ok; byte_fail; byte_fail = false) {
            result.error = EMEL_ERR_INVALID_ARGUMENT;
            return result;
          }
        }
      }
      pos += path.token_length;
    }

    result.token_count = count;
    result.error = EMEL_OK;
  }
  return result;
}

}  // namespace emel::text::encoders::plamo2::detail
