#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"

namespace emel::gbnf::rule_parser::detail {

inline constexpr int32_t error_code(const emel::gbnf::rule_parser::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

struct rule_builder {
  std::array<emel::gbnf::element, emel::gbnf::k_max_gbnf_rule_elements> elements = {};
  uint32_t size = 0;

  bool push(const emel::gbnf::element elem) noexcept {
        {
      const size_t emel_branch_1 = static_cast<size_t>(size < elements.size());
      for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 1u; emel_case_1 = 2u) {
                elements[size++] = elem;
                return true;
      }
      for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 0u; emel_case_1 = 2u) {
                return false;
      }
    }
    return false;
  }

  bool append(const emel::gbnf::element *src, uint32_t count) noexcept {
        {
      const size_t emel_branch_2 = static_cast<size_t>(count == 0);
      for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 1u; emel_case_2 = 2u) {
                return true;
      }
      for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 0u; emel_case_2 = 2u) {

      }
    }
        {
      const size_t emel_branch_3 = static_cast<size_t>(size + count <= elements.size());
      for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 1u; emel_case_3 = 2u) {
                std::memcpy(elements.data() + size, src, sizeof(emel::gbnf::element) * count);
                size += count;
                return true;
      }
      for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 0u; emel_case_3 = 2u) {
                return false;
      }
    }
    return false;
  }

  bool resize(uint32_t new_size) noexcept {
        {
      const size_t emel_branch_4 = static_cast<size_t>(new_size <= size);
      for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 1u; emel_case_4 = 2u) {
                size = new_size;
                return true;
      }
      for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 0u; emel_case_4 = 2u) {
                return false;
      }
    }
    return false;
  }
};

struct symbol_table {
  struct entry {
    std::string_view name = {};
    uint32_t id = 0;
    uint32_t hash = 0;
    bool occupied = false;
  };

  std::vector<entry> entries = {};
  std::vector<uint32_t> touched_slots = {};
  uint32_t count = 0;

  symbol_table() {
    entries.resize(emel::gbnf::k_gbnf_symbol_table_slots);
    touched_slots.reserve(emel::gbnf::k_max_gbnf_symbols);
  }

  static uint32_t hash_name(const std::string_view name) noexcept {
    constexpr uint32_t k_fnv_offset = 2166136261u;
    constexpr uint32_t k_fnv_prime = 16777619u;
    uint32_t hash = k_fnv_offset;
    for (const unsigned char byte : name) {
      hash ^= byte;
      hash *= k_fnv_prime;
    }
    const std::array<uint32_t, 2> hash_candidates = {hash, 1u};
    return hash_candidates[static_cast<size_t>(hash == 0)];
  }

  void clear() noexcept {
    for (const uint32_t slot : touched_slots) {
      entries[slot] = {};
    }
    touched_slots.clear();
    count = 0;
  }

  bool find(const std::string_view name, const uint32_t hash, uint32_t &id) const noexcept {
    const uint32_t slot_count = static_cast<uint32_t>(entries.size());
    const uint32_t mask = slot_count - 1u;
    uint32_t slot = hash & mask;
    for (uint32_t probes = 0; probes < slot_count; ++probes) {
      const auto &entry = entries[slot];
            {
        const size_t emel_branch_5 = static_cast<size_t>(entry.occupied);
        for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 1u; emel_case_5 = 2u) {

        }
        for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 0u; emel_case_5 = 2u) {
                    return false;
        }
      }
            {
        const size_t emel_branch_6 = static_cast<size_t>(entry.hash == hash && entry.name == name);
        for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 1u; emel_case_6 = 2u) {
                    id = entry.id;
                    return true;
        }
        for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 0u; emel_case_6 = 2u) {

        }
      }
      slot = (slot + 1u) & mask;
    }
    return false;
  }

  bool insert(const std::string_view name, const uint32_t hash, const uint32_t id) noexcept {
    const uint32_t slot_count = static_cast<uint32_t>(entries.size());
    const uint32_t mask = slot_count - 1u;
    uint32_t slot = hash & mask;
    for (uint32_t probes = 0; probes < slot_count; ++probes) {
      auto &entry = entries[slot];
            {
        const size_t emel_branch_7 = static_cast<size_t>(entry.occupied);
        for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 1u; emel_case_7 = 2u) {

        }
        for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 0u; emel_case_7 = 2u) {
                    entry.name = name;
                    entry.id = id;
                    entry.hash = hash;
                    entry.occupied = true;
                    touched_slots.push_back(slot);
                    count += 1;
                    return true;
        }
      }
            {
        const size_t emel_branch_8 = static_cast<size_t>(entry.hash == hash && entry.name == name);
        for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 1u; emel_case_8 = 2u) {
                    entry.id = id;
                    return true;
        }
        for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 0u; emel_case_8 = 2u) {

        }
      }
      slot = (slot + 1u) & mask;
    }
    return false;
  }
};

inline constexpr uint32_t k_max_group_nesting_depth = 32;

inline bool is_digit_char(const char c) noexcept {
  return '0' <= c && c <= '9';
}

inline bool is_word_char(const char c) noexcept {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

inline bool parse_uint64(const char *src,
                         const char *end,
                         uint64_t &value_out,
                         const char **next_out) noexcept {
    {
    const size_t emel_branch_9 = static_cast<size_t>(src < end && is_digit_char(*src));
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 1u; emel_case_9 = 2u) {

    }
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 0u; emel_case_9 = 2u) {
            return false;
    }
  }
  uint64_t value = 0;
  const uint64_t max_div_10 = std::numeric_limits<uint64_t>::max() / 10u;
  for (; src < end && is_digit_char(*src); ++src) {
    const uint64_t digit = static_cast<uint64_t>(*src - '0');
        {
      const size_t emel_branch_10 = static_cast<size_t>(
        value > max_div_10 ||
        (value == max_div_10 &&
         digit > (std::numeric_limits<uint64_t>::max() % 10u)));
      for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 1u; emel_case_10 = 2u) {
                return false;
      }
      for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 0u; emel_case_10 = 2u) {

      }
    }
    value = value * 10u + digit;
  }
  value_out = value;
  *next_out = src;
  return true;
}

inline const char *parse_name(const char *src, const char *end) noexcept {
  const char *pos = src;
  while (pos < end && is_word_char(*pos)) {
    pos++;
  }
  const size_t has_name = static_cast<size_t>(pos != src);
  const char *results[2] = {nullptr, pos};
  return results[has_name];
}

inline std::pair<uint32_t, const char *> parse_hex(const char *src,
                                                   const char *end,
                                                   const int size) noexcept {
    {
    const size_t emel_branch_11 = static_cast<size_t>(src + size <= end);
    for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 1u; emel_case_11 = 2u) {

    }
    for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 0u; emel_case_11 = 2u) {
            return std::make_pair(0, nullptr);
    }
  }
  const char *pos = src;
  const char *limit = src + size;
  uint32_t value = 0;
  constexpr std::array<uint32_t, 22> k_hex_values = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15,
      10, 11, 12, 13, 14, 15};
  constexpr std::string_view k_hex_digits = "0123456789abcdefABCDEF";
  for (; pos < limit; ++pos) {
    value <<= 4;
    const char c = *pos;
    const size_t digit_index = k_hex_digits.find(c);
        {
      const size_t emel_branch_12 = static_cast<size_t>(digit_index != std::string_view::npos);
      for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 1u; emel_case_12 = 2u) {
                value += k_hex_values[digit_index];
      }
      for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 0u; emel_case_12 = 2u) {
                return std::make_pair(0, nullptr);
      }
    }
  }
  return std::make_pair(value, pos);
}

inline std::pair<uint32_t, const char *> decode_utf8(const char *src,
                                                     const char *end) noexcept {
    {
    const size_t emel_branch_13 = static_cast<size_t>(src < end);
    for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 1u; emel_case_13 = 2u) {

    }
    for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 0u; emel_case_13 = 2u) {
            return std::make_pair(0, nullptr);
    }
  }
  static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 2, 2, 3, 4};
  const uint8_t first_byte = static_cast<uint8_t>(*src);
  const uint8_t highbits = first_byte >> 4;
  const int len = lookup[highbits];
    {
    const size_t emel_branch_14 = static_cast<size_t>(src + len <= end);
    for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 1u; emel_case_14 = 2u) {

    }
    for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 0u; emel_case_14 = 2u) {
            return std::make_pair(0, nullptr);
    }
  }
  const uint8_t mask = static_cast<uint8_t>((1u << (8 - len)) - 1u);
  uint32_t value = first_byte & mask;
  for (int i = 1; i < len; ++i) {
    const uint8_t byte = static_cast<uint8_t>(src[i]);
    value = (value << 6) + (byte & 0x3F);
  }
  return std::make_pair(value, src + len);
}

inline std::pair<uint32_t, const char *> parse_char(const char *src,
                                                    const char *end) noexcept {
    {
    const size_t emel_branch_15 = static_cast<size_t>(src < end);
    for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 1u; emel_case_15 = 2u) {

    }
    for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 0u; emel_case_15 = 2u) {
            return std::make_pair(0, nullptr);
    }
  }
    {
    const size_t emel_branch_16 = static_cast<size_t>(*src == '\\');
    for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 1u; emel_case_16 = 2u) {
      {
        const size_t emel_branch_17 = static_cast<size_t>(src + 1 < end);
        for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 1u; emel_case_17 = 2u) {

        }
        for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 0u; emel_case_17 = 2u) {
          return std::make_pair(0, nullptr);
        }
      }
      const char escaped = src[1];
      const size_t is_hex2 = static_cast<size_t>(escaped == 'x');
      const size_t is_hex4 = static_cast<size_t>(escaped == 'u');
      const size_t is_hex8 = static_cast<size_t>(escaped == 'U');
      const size_t is_tab = static_cast<size_t>(escaped == 't');
      const size_t is_cr = static_cast<size_t>(escaped == 'r');
      const size_t is_lf = static_cast<size_t>(escaped == 'n');
      const size_t is_literal = static_cast<size_t>(escaped == '\\' || escaped == '"' ||
                                                    escaped == '[' || escaped == ']');
      {
        const size_t emel_branch_hex2 = is_hex2;
        for (size_t emel_case_hex2 = emel_branch_hex2; emel_case_hex2 == 1u;
             emel_case_hex2 = 2u) {
          return parse_hex(src + 2, end, 2);
        }
        for (size_t emel_case_hex2 = emel_branch_hex2; emel_case_hex2 == 0u;
             emel_case_hex2 = 2u) {

        }
      }
      {
        const size_t emel_branch_hex4 = is_hex4;
        for (size_t emel_case_hex4 = emel_branch_hex4; emel_case_hex4 == 1u;
             emel_case_hex4 = 2u) {
          return parse_hex(src + 2, end, 4);
        }
        for (size_t emel_case_hex4 = emel_branch_hex4; emel_case_hex4 == 0u;
             emel_case_hex4 = 2u) {

        }
      }
      {
        const size_t emel_branch_hex8 = is_hex8;
        for (size_t emel_case_hex8 = emel_branch_hex8; emel_case_hex8 == 1u;
             emel_case_hex8 = 2u) {
          return parse_hex(src + 2, end, 8);
        }
        for (size_t emel_case_hex8 = emel_branch_hex8; emel_case_hex8 == 0u;
             emel_case_hex8 = 2u) {

        }
      }
      {
        const size_t emel_branch_tab = is_tab;
        for (size_t emel_case_tab = emel_branch_tab; emel_case_tab == 1u;
             emel_case_tab = 2u) {
          return std::make_pair(static_cast<uint32_t>('\t'), src + 2);
        }
        for (size_t emel_case_tab = emel_branch_tab; emel_case_tab == 0u;
             emel_case_tab = 2u) {

        }
      }
      {
        const size_t emel_branch_cr = is_cr;
        for (size_t emel_case_cr = emel_branch_cr; emel_case_cr == 1u;
             emel_case_cr = 2u) {
          return std::make_pair(static_cast<uint32_t>('\r'), src + 2);
        }
        for (size_t emel_case_cr = emel_branch_cr; emel_case_cr == 0u;
             emel_case_cr = 2u) {

        }
      }
      {
        const size_t emel_branch_lf = is_lf;
        for (size_t emel_case_lf = emel_branch_lf; emel_case_lf == 1u;
             emel_case_lf = 2u) {
          return std::make_pair(static_cast<uint32_t>('\n'), src + 2);
        }
        for (size_t emel_case_lf = emel_branch_lf; emel_case_lf == 0u;
             emel_case_lf = 2u) {

        }
      }
      {
        const size_t emel_branch_literal = is_literal;
        for (size_t emel_case_literal = emel_branch_literal; emel_case_literal == 1u;
             emel_case_literal = 2u) {
          return std::make_pair(static_cast<uint32_t>(escaped), src + 2);
        }
        for (size_t emel_case_literal = emel_branch_literal; emel_case_literal == 0u;
             emel_case_literal = 2u) {

        }
      }
      return std::make_pair(0u, nullptr);
    }
    for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 0u; emel_case_16 = 2u) {

    }
  }
  return decode_utf8(src, end);
}

} // namespace emel::gbnf::rule_parser::detail
