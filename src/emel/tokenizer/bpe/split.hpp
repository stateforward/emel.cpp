#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "emel/text/unicode.hpp"
#include "emel/tokenizer/bpe/regex.hpp"

namespace emel::tokenizer::bpe::detail {

constexpr size_t k_max_bpe_words = 1024;
constexpr size_t k_max_bpe_bytes = 65536;
constexpr size_t k_max_bpe_cpts = k_max_bpe_bytes;

struct split_view {
  const std::string_view * words = nullptr;
  size_t count = 0;
};

struct split_scratch {
  std::array<uint32_t, k_max_bpe_cpts> cpts = {};
  std::array<size_t, k_max_bpe_words> offsets_a = {};
  std::array<size_t, k_max_bpe_words> offsets_b = {};
  std::array<char, k_max_bpe_bytes> encoded = {};
  std::array<std::string_view, k_max_bpe_words> words = {};
  size_t encoded_size = 0;
  size_t cpt_count = 0;
  size_t offset_count = 0;
  size_t word_count = 0;

  void reset() noexcept {
    encoded_size = 0;
    cpt_count = 0;
    offset_count = 0;
    word_count = 0;
  }
};

inline constexpr uint32_t bpe_out_of_range = 0xFFFFFFFFu;

inline constexpr std::array<uint16_t, 256> bpe_byte_to_unicode_map() {
  std::array<uint16_t, 256> table = {};
  bool keep[256] = {};
  for (int ch = 0x21; ch <= 0x7E; ++ch) {
    keep[ch] = true;
  }
  for (int ch = 0xA1; ch <= 0xAC; ++ch) {
    keep[ch] = true;
  }
  for (int ch = 0xAE; ch <= 0xFF; ++ch) {
    keep[ch] = true;
  }
  uint16_t n = 0;
  for (int ch = 0; ch < 256; ++ch) {
    if (keep[ch]) {
      table[ch] = static_cast<uint16_t>(ch);
    } else {
      table[ch] = static_cast<uint16_t>(256 + n);
      n += 1;
    }
  }
  return table;
}

inline constexpr std::array<uint16_t, 256> k_bpe_byte_to_unicode =
    bpe_byte_to_unicode_map();

inline size_t encode_utf8(uint32_t cpt, char * out, size_t capacity) {
  if (out == nullptr || capacity == 0) {
    return 0;
  }
  if (cpt <= 0x7F) {
    if (capacity < 1) {
      return 0;
    }
    out[0] = static_cast<char>(cpt);
    return 1;
  }
  if (cpt <= 0x7FF) {
    if (capacity < 2) {
      return 0;
    }
    out[0] = static_cast<char>(0xC0 | ((cpt >> 6) & 0x1F));
    out[1] = static_cast<char>(0x80 | (cpt & 0x3F));
    return 2;
  }
  if (cpt <= 0xFFFF) {
    if (capacity < 3) {
      return 0;
    }
    out[0] = static_cast<char>(0xE0 | ((cpt >> 12) & 0x0F));
    out[1] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
    out[2] = static_cast<char>(0x80 | (cpt & 0x3F));
    return 3;
  }
  if (cpt <= 0x10FFFF) {
    if (capacity < 4) {
      return 0;
    }
    out[0] = static_cast<char>(0xF0 | ((cpt >> 18) & 0x07));
    out[1] = static_cast<char>(0x80 | ((cpt >> 12) & 0x3F));
    out[2] = static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
    out[3] = static_cast<char>(0x80 | (cpt & 0x3F));
    return 4;
  }
  return 0;
}

inline bool decode_utf8_to_cpts(const std::string_view text,
                                split_scratch & scratch) {
  scratch.cpt_count = 0;
  size_t offset = 0;
  while (offset < text.size()) {
    if (scratch.cpt_count >= scratch.cpts.size()) {
      return false;
    }
    const uint8_t byte = static_cast<uint8_t>(text[offset]);
    uint32_t cpt = 0xFFFD;
    size_t len = 1;
    if ((byte & 0x80u) == 0) {
      cpt = byte;
      len = 1;
    } else if ((byte & 0xE0u) == 0xC0u && offset + 1 < text.size()) {
      const uint8_t b1 = static_cast<uint8_t>(text[offset + 1]);
      if ((b1 & 0xC0u) == 0x80u) {
        cpt = ((byte & 0x1Fu) << 6) | (b1 & 0x3Fu);
        len = 2;
      }
    } else if ((byte & 0xF0u) == 0xE0u && offset + 2 < text.size()) {
      const uint8_t b1 = static_cast<uint8_t>(text[offset + 1]);
      const uint8_t b2 = static_cast<uint8_t>(text[offset + 2]);
      if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u) {
        cpt = ((byte & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
        len = 3;
      }
    } else if ((byte & 0xF8u) == 0xF0u && offset + 3 < text.size()) {
      const uint8_t b1 = static_cast<uint8_t>(text[offset + 1]);
      const uint8_t b2 = static_cast<uint8_t>(text[offset + 2]);
      const uint8_t b3 = static_cast<uint8_t>(text[offset + 3]);
      if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u &&
          (b3 & 0xC0u) == 0x80u) {
        cpt = ((byte & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) |
              ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
        len = 4;
      }
    }
    scratch.cpts[scratch.cpt_count++] = cpt;
    offset += len;
  }
  return true;
}

inline bool push_offset(size_t value, size_t * out, size_t capacity,
                        size_t & out_count) {
  if (value == 0) {
    return true;
  }
  if (out_count >= capacity) {
    return false;
  }
  out[out_count++] = value;
  return true;
}

inline bool split_gpt2(const uint32_t * cpts, size_t cpt_count,
                       const size_t * offsets_in, size_t offsets_in_count,
                       size_t * offsets_out, size_t out_capacity,
                       size_t & out_count) {
  out_count = 0;
  size_t start = 0;
  for (size_t idx = 0; idx < offsets_in_count; ++idx) {
    const size_t offset_ini = start;
    const size_t offset_end = start + offsets_in[idx];
    if (offset_end > cpt_count) {
      return false;
    }
    start = offset_end;

    auto get_cpt = [&](const size_t pos) -> uint32_t {
      return (offset_ini <= pos && pos < offset_end) ? cpts[pos]
                                                     : bpe_out_of_range;
    };
    auto get_flags = [&](const size_t pos) -> emel::text::unicode_cpt_flags {
      return (offset_ini <= pos && pos < offset_end)
                 ? emel::text::unicode_cpt_flags_from_cpt(cpts[pos])
                 : emel::text::unicode_cpt_flags{};
    };

    size_t prev_end = offset_ini;
    auto add_token = [&](const size_t end) -> bool {
      if (end < prev_end || end > offset_end) {
        return false;
      }
      const size_t len = end - prev_end;
      prev_end = end;
      return push_offset(len, offsets_out, out_capacity, out_count);
    };

    for (size_t pos = offset_ini; pos < offset_end;) {
      const uint32_t cpt = get_cpt(pos);
      const auto flags = get_flags(pos);

      if (cpt == '\'' && pos + 1 < offset_end) {
        const uint32_t cpt_next = get_cpt(pos + 1);
        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' ||
            cpt_next == 'd') {
          if (!add_token(pos + 2)) {
            return false;
          }
          pos += 2;
          continue;
        }
        if (pos + 2 < offset_end) {
          const uint32_t cpt_next_next = get_cpt(pos + 2);
          if ((cpt_next == 'r' && cpt_next_next == 'e') ||
              (cpt_next == 'v' && cpt_next_next == 'e') ||
              (cpt_next == 'l' && cpt_next_next == 'l')) {
            if (!add_token(pos + 3)) {
              return false;
            }
            pos += 3;
            continue;
          }
        }
      }

      auto flags2 = (cpt == ' ') ? get_flags(pos + 1) : flags;
      if (flags2.is_letter) {
        pos += (cpt == ' ');
        while (get_flags(pos).is_letter) {
          pos++;
        }
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }
      if (flags2.is_number) {
        pos += (cpt == ' ');
        while (get_flags(pos).is_number) {
          pos++;
        }
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }
      if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) &&
          flags2.as_uint()) {
        pos += (cpt == ' ');
        while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) &&
               flags2.as_uint()) {
          flags2 = get_flags(++pos);
        }
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      size_t num_whitespaces = 0;
      while (get_flags(pos + num_whitespaces).is_whitespace) {
        num_whitespaces++;
      }

      if (num_whitespaces > 1 &&
          get_cpt(pos + num_whitespaces) != bpe_out_of_range) {
        pos += num_whitespaces - 1;
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      if (num_whitespaces > 0) {
        pos += num_whitespaces;
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      pos += 1;
      if (!add_token(pos)) {
        return false;
      }
    }
  }
  return true;
}

inline bool split_llama3(const uint32_t * cpts, size_t cpt_count,
                         const size_t * offsets_in, size_t offsets_in_count,
                         size_t * offsets_out, size_t out_capacity,
                         size_t & out_count) {
  out_count = 0;
  size_t start = 0;
  for (size_t idx = 0; idx < offsets_in_count; ++idx) {
    const size_t offset_ini = start;
    const size_t offset_end = start + offsets_in[idx];
    if (offset_end > cpt_count) {
      return false;
    }
    start = offset_end;

    auto get_cpt = [&](const size_t pos) -> uint32_t {
      return (offset_ini <= pos && pos < offset_end) ? cpts[pos]
                                                     : bpe_out_of_range;
    };
    auto get_flags = [&](const size_t pos) -> emel::text::unicode_cpt_flags {
      return (offset_ini <= pos && pos < offset_end)
                 ? emel::text::unicode_cpt_flags_from_cpt(cpts[pos])
                 : emel::text::unicode_cpt_flags{};
    };

    size_t prev_end = offset_ini;
    auto add_token = [&](const size_t end) -> bool {
      if (end < prev_end || end > offset_end) {
        return false;
      }
      const size_t len = end - prev_end;
      prev_end = end;
      return push_offset(len, offsets_out, out_capacity, out_count);
    };

    for (size_t pos = offset_ini; pos < offset_end;) {
      const uint32_t cpt = get_cpt(pos);
      const auto flags = get_flags(pos);

      if (cpt == '\'' && pos + 1 < offset_end) {
        const uint32_t cpt_next = emel::text::unicode_tolower(get_cpt(pos + 1));
        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' ||
            cpt_next == 'd') {
          if (!add_token(pos + 2)) {
            return false;
          }
          pos += 2;
          continue;
        }
        if (pos + 2 < offset_end) {
          const uint32_t cpt_next_next =
              emel::text::unicode_tolower(get_cpt(pos + 2));
          if ((cpt_next == 'r' && cpt_next_next == 'e') ||
              (cpt_next == 'v' && cpt_next_next == 'e') ||
              (cpt_next == 'l' && cpt_next_next == 'l')) {
            if (!add_token(pos + 3)) {
              return false;
            }
            pos += 3;
            continue;
          }
        }
      }

      if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
        if (flags.is_letter || get_flags(pos + 1).is_letter) {
          pos++;
          while (get_flags(pos).is_letter) {
            pos++;
          }
          if (!add_token(pos)) {
            return false;
          }
          continue;
        }
      }

      if (flags.is_number) {
        size_t ini = pos;
        while (get_flags(pos).is_number) {
          pos++;
        }
        const size_t len = pos - ini;
        if (len > 3) {
          pos = ini + 3;
        }
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      if (!(flags.is_whitespace | flags.is_letter | flags.is_number) &&
          flags.as_uint()) {
        if (cpt != ' ' && (cpt == '\r' || cpt == '\n')) {
          pos += 1;
        } else if (cpt == ' ') {
          pos += 1;
        }
        while (!(get_flags(pos).is_whitespace | get_flags(pos).is_letter |
                 get_flags(pos).is_number) &&
               get_flags(pos).as_uint()) {
          pos++;
        }
        while (get_cpt(pos) == '\r' || get_cpt(pos) == '\n') {
          pos++;
        }
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      if (cpt == '\r' || cpt == '\n') {
        while (get_cpt(pos) == '\r' || get_cpt(pos) == '\n') {
          pos++;
        }
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      size_t num_whitespaces = 0;
      while (get_flags(pos + num_whitespaces).is_whitespace) {
        num_whitespaces++;
      }

      if (num_whitespaces > 1 &&
          get_cpt(pos + num_whitespaces) != bpe_out_of_range) {
        pos += num_whitespaces - 1;
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      if (num_whitespaces > 0) {
        pos += num_whitespaces;
        if (!add_token(pos)) {
          return false;
        }
        continue;
      }

      pos += 1;
      if (!add_token(pos)) {
        return false;
      }
    }
  }
  return true;
}

inline bool encode_bpe_segment(const uint32_t * cpts, size_t start,
                               size_t len, split_scratch & scratch) {
  size_t segment_offset = scratch.encoded_size;
  for (size_t idx = 0; idx < len; ++idx) {
    char utf8[4];
    const size_t utf8_len = encode_utf8(cpts[start + idx], utf8, sizeof(utf8));
    if (utf8_len == 0) {
      return false;
    }
    for (size_t j = 0; j < utf8_len; ++j) {
      const uint8_t byte = static_cast<uint8_t>(utf8[j]);
      const uint32_t mapped = k_bpe_byte_to_unicode[byte];
      char encoded[4];
      const size_t encoded_len = encode_utf8(mapped, encoded, sizeof(encoded));
      if (encoded_len == 0) {
        return false;
      }
      if (scratch.encoded_size + encoded_len > scratch.encoded.size()) {
        return false;
      }
      for (size_t k = 0; k < encoded_len; ++k) {
        scratch.encoded[scratch.encoded_size++] = encoded[k];
      }
    }
  }
  const size_t encoded_len = scratch.encoded_size - segment_offset;
  if (encoded_len == 0) {
    return true;
  }
  if (scratch.word_count >= scratch.words.size()) {
    return false;
  }
  scratch.words[scratch.word_count++] =
      std::string_view(scratch.encoded.data() + segment_offset, encoded_len);
  return true;
}

inline bool split_and_encode_fallback(const std::string_view text,
                                      const regex_list & regex,
                                      split_scratch & scratch) {
  std::vector<std::string> regex_exprs;
  regex_exprs.reserve(regex.count);
  for (size_t idx = 0; idx < regex.count; ++idx) {
    regex_exprs.emplace_back(regex.exprs[idx]);
  }
  const std::string raw_text(text);
  const auto words = emel::text::unicode_regex_split(raw_text, regex_exprs);
  scratch.word_count = 0;
  for (const std::string & word : words) {
    if (word.empty()) {
      continue;
    }
    if (scratch.word_count >= scratch.words.size()) {
      return false;
    }
    if (scratch.encoded_size + word.size() > scratch.encoded.size()) {
      return false;
    }
    const size_t offset = scratch.encoded_size;
    for (const char c : word) {
      scratch.encoded[scratch.encoded_size++] = c;
    }
    scratch.words[scratch.word_count++] =
        std::string_view(scratch.encoded.data() + offset, word.size());
  }
  return true;
}

inline bool split_and_encode_append(const std::string_view text,
                                    const emel::model::data::vocab & vocab,
                                    split_scratch & scratch,
                                    split_view & view) {
  view.words = scratch.words.data();
  view.count = 0;
  scratch.word_count = 0;
  if (text.empty()) {
    return true;
  }

  const regex_list regex = regex_for(vocab);
  if (regex.count != 1) {
    const bool ok = split_and_encode_fallback(text, regex, scratch);
    view.count = scratch.word_count;
    return ok;
  }

  static constexpr std::string_view k_gpt2_regex =
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
  static constexpr std::string_view k_llama3_regex =
      "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
      "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+"
      "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  const std::string_view expr = regex.exprs[0];
  if (expr != k_gpt2_regex && expr != k_llama3_regex) {
    const bool ok = split_and_encode_fallback(text, regex, scratch);
    view.count = scratch.word_count;
    return ok;
  }

  if (!decode_utf8_to_cpts(text, scratch)) {
    return false;
  }

  scratch.offsets_a[0] = scratch.cpt_count;
  scratch.offset_count = 1;
  size_t out_count = 0;
  bool ok = false;
  if (expr == k_gpt2_regex) {
    ok = split_gpt2(scratch.cpts.data(), scratch.cpt_count,
                    scratch.offsets_a.data(), scratch.offset_count,
                    scratch.offsets_b.data(), scratch.offsets_b.size(),
                    out_count);
  } else {
    ok = split_llama3(scratch.cpts.data(), scratch.cpt_count,
                      scratch.offsets_a.data(), scratch.offset_count,
                      scratch.offsets_b.data(), scratch.offsets_b.size(),
                      out_count);
  }
  if (!ok) {
    return false;
  }

  size_t start = 0;
  for (size_t idx = 0; idx < out_count; ++idx) {
    const size_t len = scratch.offsets_b[idx];
    if (len == 0) {
      continue;
    }
    if (!encode_bpe_segment(scratch.cpts.data(), start, len, scratch)) {
      return false;
    }
    start += len;
  }

  view.count = scratch.word_count;
  return true;
}

}  // namespace emel::tokenizer::bpe::detail
