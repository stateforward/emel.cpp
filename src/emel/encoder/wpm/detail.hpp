#pragma once

#include <cctype>
#include <cstring>

#include "emel/encoder/wpm/context.hpp"
#include "emel/encoder/detail.hpp"
#include "emel/encoder/events.hpp"
#include "emel/model/data.hpp"
#include "emel/text/unicode.hpp"

namespace emel::encoder::wpm::detail {

using emel::encoder::detail::encode_result;
using emel::encoder::detail::k_token_null;

inline std::vector<std::string> wpm_preprocess(const std::string &text) {
  const std::vector<uint32_t> cpts =
      emel::text::unicode_cpts_normalize_nfd(
          emel::text::unicode_cpts_from_utf8(text));
  std::vector<std::string> words(1, "");
  for (const uint32_t cpt : cpts) {
    const auto flags = emel::text::unicode_cpt_flags_from_cpt(cpt);
    if (flags.is_whitespace) {
      if (!words.back().empty()) {
        words.emplace_back();
      }
      continue;
    }
    if (cpt == 0 || cpt == 0xFFFD || flags.is_control) {
      continue;
    }
    const std::string s =
        emel::text::unicode_cpt_to_utf8(emel::text::unicode_tolower(cpt));
    if (flags.is_punctuation || (cpt < 0x7F && flags.is_symbol) ||
        emel::encoder::detail::is_chinese_char(cpt)) {
      if (!words.back().empty()) {
        words.emplace_back();
      }
      words.back() = s;
      words.emplace_back();
    } else {
      words.back() += s;
    }
  }
  if (!words.empty() && words.back().empty()) {
    words.pop_back();
  }
  return words;
}

inline encode_result encode_wpm(const event::encode &ev,
                                emel::encoder::action::context &ctx,
                                const emel::model::data::vocab &vocab) {
  encode_result result{};
  if (ev.text.empty()) {
    return result;
  }
  emel::encoder::detail::ensure_tables(ctx);

  int32_t count = 0;
  const char *prefix = "\xE2\x96\x81";
  constexpr size_t prefix_len = 3;

  size_t word_start = 0;
  for (size_t i = 0; i <= ev.text.size(); ++i) {
    const bool is_end = (i == ev.text.size());
    const bool is_space = !is_end && std::isspace(static_cast<unsigned char>(ev.text[i])) != 0;
    if (!is_end && !is_space) {
      continue;
    }
    const size_t word_len = i - word_start;
    if (word_len > 0) {
      const int32_t word_token_start = count;
      if (prefix_len + word_len > ctx.scratch.buffer.size()) {
        result.error = EMEL_ERR_INVALID_ARGUMENT;
        return result;
      }
      std::memcpy(ctx.scratch.buffer.data(), prefix, prefix_len);
      std::memcpy(ctx.scratch.buffer.data() + prefix_len,
                  ev.text.data() + word_start, word_len);
      const std::string_view word(ctx.scratch.buffer.data(), prefix_len + word_len);
      bool word_ok = true;
      for (size_t pos = 0; pos < word.size();) {
        bool found = false;
        const size_t max_len = std::min(word.size() - pos,
                                        static_cast<size_t>(ctx.max_token_len));
        for (size_t len = max_len; len > 0; --len) {
          const std::string_view piece = word.substr(pos, len);
          const int32_t token = emel::encoder::detail::lookup_token(ctx, piece);
          if (token != k_token_null) {
            if (!emel::encoder::detail::push_token(ev, token, count)) {
              result.error = EMEL_ERR_INVALID_ARGUMENT;
              return result;
            }
            pos += len;
            found = true;
            break;
          }
        }
        if (!found) {
          word_ok = false;
          break;
        }
      }
      if (!word_ok) {
        count = word_token_start;
        int32_t unk = vocab.unk_id;
        if (unk == k_token_null) {
          unk = emel::encoder::detail::lookup_token(ctx, "<unk>");
        }
        if (unk == k_token_null) {
          continue;
        }
        if (!emel::encoder::detail::push_token(ev, unk, count)) {
          result.error = EMEL_ERR_INVALID_ARGUMENT;
          return result;
        }
      }
    }
    word_start = i + 1;
  }

  result.token_count = count;
  result.error = EMEL_OK;
  return result;
}

}  // namespace emel::encoder::wpm::detail
