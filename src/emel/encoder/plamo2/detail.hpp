#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "emel/encoder/plamo2/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::encoder::plamo2::detail {

using emel::encoder::detail::encode_result;

inline bool ensure_plamo2_tables(emel::encoder::plamo2::action::context &ctx,
                                 const emel::model::data::vocab &vocab) {
  if (ctx.plamo2_tables_ready && ctx.plamo2_vocab == &vocab) {
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
    const std::string_view text = emel::encoder::detail::token_text(vocab, token_id);
    if (text.empty()) {
      continue;
    }
    token_to_id[std::string(text)] = static_cast<int32_t>(token_id);

    const auto &entry = vocab.entries[token_id];
    if (entry.type == 6) {
      if (text.size() == 6 && text.substr(0, 3) == "<0x" &&
          text.back() == '>') {
        const std::string hex = std::string(text.substr(3, 2));
        const int byte_val = std::stoi(hex, nullptr, 16);
        if (byte_val >= 0 && byte_val < 256) {
          ctx.byte_tokens[static_cast<size_t>(byte_val)] =
              static_cast<int32_t>(token_id);
        }
      }
      continue;
    }

    suffix_to_score[std::string(text)] = entry.score;
    const std::vector<uint32_t> cpts =
        emel::text::unicode_cpts_from_utf8(std::string(text));
    for (size_t i = 1; i < cpts.size(); ++i) {
      std::string suffix;
      for (size_t j = i; j < cpts.size(); ++j) {
        suffix += emel::text::unicode_cpt_to_utf8(cpts[j]);
      }
      if (suffix_to_score.find(suffix) == suffix_to_score.end()) {
        suffix_to_score[suffix] = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }

  for (size_t i = 0; i < ctx.byte_tokens.size(); ++i) {
    if (ctx.byte_tokens[i] == 0) {
      return false;
    }
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
    if (!suffix.empty()) {
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
        if (suffix_to_score.find(piece) != suffix_to_score.end()) {
          pieces_for_suffix += 1;
        }
      }
      num_pieces += pieces_for_suffix;
    } else {
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
      if (score_it == suffix_to_score.end()) {
        continue;
      }
      auto token_it = token_to_id.find(piece);
      ctx.table[static_cast<size_t>(table_idx)].piece_length = piece_len;
      ctx.table[static_cast<size_t>(table_idx)].token_id =
          (token_it != token_to_id.end()) ? token_it->second : -1;
      const float score = score_it->second;
      ctx.table[static_cast<size_t>(table_idx)].score =
          std::isfinite(score) ? static_cast<int32_t>(std::round(score * 1e4))
                               : k_invalid_score;
      ctx.table[static_cast<size_t>(table_idx)].piece_id =
          suffix_to_id[piece];
      table_idx += 1;
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
                                   emel::encoder::plamo2::action::context &ctx,
                                   const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);
  if (!ensure_plamo2_tables(ctx, vocab)) {
    result.error = EMEL_ERR_MODEL_INVALID;
    return result;
  }

  std::vector<uint32_t> unicode_data =
      emel::text::unicode_cpts_from_utf8(std::string(ev.text));
  if (!unicode_data.empty() && unicode_data[0] == 0xFEFF) {
    unicode_data.erase(unicode_data.begin());
  }
  if (unicode_data.empty()) {
    result.error = EMEL_OK;
    return result;
  }
  if (unicode_data.size() > ctx.cpts.size()) {
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
  for (int32_t i = static_cast<int32_t>(data_len) - 1; i >= 0; --i) {
    const uint32_t c = unicode_data[static_cast<size_t>(i)];

    for (size_t p = static_cast<size_t>(suffix_id); p < ctx.table.size(); ++p) {
      const int64_t piece_code =
          (static_cast<int64_t>(c) << 32) |
          static_cast<uint32_t>(ctx.table[p].piece_id);
      const auto it = ctx.suffix_map.find(piece_code);
      suffix_id = (it != ctx.suffix_map.end()) ? it->second : 0;
      if (suffix_id > 0 || ctx.table[p].score == k_unknown_score) {
        break;
      }
    }

    for (size_t p = static_cast<size_t>(suffix_id); p < ctx.table.size(); ++p) {
      const int32_t score = ctx.table[p].score;
      if (score > k_invalid_score) {
        const int32_t piece_length = ctx.table[p].piece_length;
        if (piece_length <= 0 || i + piece_length > static_cast<int32_t>(data_len)) {
          continue;
        }
        const int64_t s = ctx.scores[static_cast<size_t>(i + piece_length)] - score;
        if (s < ctx.scores[static_cast<size_t>(i)]) {
          ctx.scores[static_cast<size_t>(i)] = s;
          ctx.paths[static_cast<size_t>(i)].token_length = piece_length;
          ctx.paths[static_cast<size_t>(i)].token_id = ctx.table[p].token_id;
          ctx.paths[static_cast<size_t>(i)].num_tokens =
              ctx.paths[static_cast<size_t>(i + piece_length)].num_tokens + 1;
          if (score == k_unknown_score) {
            ctx.paths[static_cast<size_t>(i)].num_tokens +=
                (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
          }
        }
      }
      if (score == k_unknown_score) {
        break;
      }
    }
  }

  int32_t count = 0;
  int32_t pos = 0;
  while (pos < static_cast<int32_t>(data_len)) {
    const auto &path = ctx.paths[static_cast<size_t>(pos)];
    if (path.token_length <= 0) {
      result.error = EMEL_ERR_BACKEND;
      return result;
    }
    if (path.token_id >= 0) {
      if (!emel::encoder::detail::push_token(ev, path.token_id, count)) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
    } else {
      const uint32_t c = unicode_data[static_cast<size_t>(pos)];
      const int32_t s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
      for (int32_t i = 0; i < s; ++i) {
        uint8_t b = 0;
        if (s == 1) {
          b = static_cast<uint8_t>(c);
        } else if (i == 0) {
          b = static_cast<uint8_t>((0xF00 >> s) & 0xFF);
        } else {
          b = 0x80;
        }
        b = static_cast<uint8_t>(b | ((c >> ((s - i - 1) * 6)) & 0x3F));
        const int32_t byte_token = ctx.byte_tokens[b];
        if (byte_token <= 0 ||
            !emel::encoder::detail::push_token(ev, byte_token, count)) {
          result.error = EMEL_ERR_INVALID_ARGUMENT;
          return result;
        }
      }
    }
    pos += path.token_length;
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::plamo2::detail
