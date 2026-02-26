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
    if (size >= elements.size()) {
      return false;
    }
    elements[size++] = elem;
    return true;
  }

  bool append(const emel::gbnf::element *src, uint32_t count) noexcept {
    if (count == 0) {
      return true;
    }
    if (size + count > elements.size()) {
      return false;
    }
    std::memcpy(elements.data() + size, src, sizeof(emel::gbnf::element) * count);
    size += count;
    return true;
  }

  bool resize(uint32_t new_size) noexcept {
    if (new_size > size) {
      return false;
    }
    size = new_size;
    return true;
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
    return hash == 0 ? 1u : hash;
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
      if (!entry.occupied) {
        return false;
      }
      if (entry.hash == hash && entry.name == name) {
        id = entry.id;
        return true;
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
      if (!entry.occupied) {
        entry.name = name;
        entry.id = id;
        entry.hash = hash;
        entry.occupied = true;
        touched_slots.push_back(slot);
        count += 1;
        return true;
      }
      if (entry.hash == hash && entry.name == name) {
        entry.id = id;
        return true;
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
  if (src >= end || !is_digit_char(*src)) {
    return false;
  }
  uint64_t value = 0;
  const uint64_t max_div_10 = std::numeric_limits<uint64_t>::max() / 10u;
  for (; src < end && is_digit_char(*src); ++src) {
    const uint64_t digit = static_cast<uint64_t>(*src - '0');
    if (value > max_div_10 ||
        (value == max_div_10 &&
         digit > (std::numeric_limits<uint64_t>::max() % 10u))) {
      return false;
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
  if (pos == src) {
    return nullptr;
  }
  return pos;
}

inline std::pair<uint32_t, const char *> parse_hex(const char *src,
                                                   const char *end,
                                                   const int size) noexcept {
  if (src + size > end) {
    return std::make_pair(0, nullptr);
  }
  const char *pos = src;
  const char *limit = src + size;
  uint32_t value = 0;
  for (; pos < limit; ++pos) {
    value <<= 4;
    const char c = *pos;
    if ('a' <= c && c <= 'f') {
      value += static_cast<uint32_t>(c - 'a' + 10);
    } else if ('A' <= c && c <= 'F') {
      value += static_cast<uint32_t>(c - 'A' + 10);
    } else if ('0' <= c && c <= '9') {
      value += static_cast<uint32_t>(c - '0');
    } else {
      return std::make_pair(0, nullptr);
    }
  }
  return std::make_pair(value, pos);
}

inline std::pair<uint32_t, const char *> decode_utf8(const char *src,
                                                     const char *end) noexcept {
  if (src >= end) {
    return std::make_pair(0, nullptr);
  }
  static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 2, 2, 3, 4};
  const uint8_t first_byte = static_cast<uint8_t>(*src);
  const uint8_t highbits = first_byte >> 4;
  const int len = lookup[highbits];
  if (src + len > end) {
    return std::make_pair(0, nullptr);
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
  if (src >= end) {
    return std::make_pair(0, nullptr);
  }
  if (*src == '\\') {
    if (src + 1 >= end) {
      return std::make_pair(0, nullptr);
    }
    switch (src[1]) {
    case 'x':
      return parse_hex(src + 2, end, 2);
    case 'u':
      return parse_hex(src + 2, end, 4);
    case 'U':
      return parse_hex(src + 2, end, 8);
    case 't':
      return std::make_pair(static_cast<uint32_t>('\t'), src + 2);
    case 'r':
      return std::make_pair(static_cast<uint32_t>('\r'), src + 2);
    case 'n':
      return std::make_pair(static_cast<uint32_t>('\n'), src + 2);
    case '\\':
    case '"':
    case '[':
    case ']':
      return std::make_pair(static_cast<uint32_t>(src[1]), src + 2);
    default:
      return std::make_pair(0, nullptr);
    }
  }
  return decode_utf8(src, end);
}

} // namespace emel::gbnf::rule_parser::detail
