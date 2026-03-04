#pragma once

#include <charconv>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"

namespace emel::gbnf::rule_parser::detail {

inline constexpr int32_t
error_code(const emel::gbnf::rule_parser::error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

inline uint32_t select_u32(const bool choose_true, const uint32_t true_value,
                           const uint32_t false_value) noexcept {
  const uint32_t mask =
      static_cast<uint32_t>(0) - static_cast<uint32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint64_t select_u64(const bool choose_true, const uint64_t true_value,
                           const uint64_t false_value) noexcept {
  const uint64_t mask =
      static_cast<uint64_t>(0) - static_cast<uint64_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline size_t select_size(const bool choose_true, const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uintptr_t select_uptr(const bool choose_true, const uintptr_t true_value,
                             const uintptr_t false_value) noexcept {
  const uintptr_t mask =
      static_cast<uintptr_t>(0) - static_cast<uintptr_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline bool select_bool(const bool choose_true, const bool true_value,
                        const bool false_value) noexcept {
  return select_u32(choose_true, static_cast<uint32_t>(true_value),
                    static_cast<uint32_t>(false_value)) != 0u;
}

struct rule_builder {
  std::array<emel::gbnf::element, emel::gbnf::k_max_gbnf_rule_elements>
      elements = {};
  uint32_t size = 0;

  bool push(const emel::gbnf::element elem) noexcept {
    const bool can_write = size < elements.size();
    const uint32_t write_index = select_u32(can_write, size, 0u);
    const size_t copy_bytes =
        sizeof(emel::gbnf::element) * static_cast<size_t>(can_write);
    std::memcpy(elements.data() + write_index, &elem, copy_bytes);
    size += static_cast<uint32_t>(can_write);
    return can_write;
  }

  bool append(const emel::gbnf::element *src, uint32_t count) noexcept {
    const bool has_count = count != 0u;
    const bool has_room = size + count <= elements.size();
    const bool do_copy = has_count && has_room;
    const uint32_t copy_count = count * static_cast<uint32_t>(do_copy);
    const uint32_t write_index = select_u32(do_copy, size, 0u);

    for (uint32_t i = 0; i < copy_count; ++i) {
      elements[write_index + i] = src[i];
    }

    size += copy_count;
    return !has_count || has_room;
  }

  bool resize(uint32_t new_size) noexcept {
    const bool can_resize = new_size <= size;
    size = select_u32(can_resize, new_size, size);
    return can_resize;
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
    return select_u32(hash == 0u, 1u, hash);
  }

  void clear() noexcept {
    for (const uint32_t slot : touched_slots) {
      entries[slot] = {};
    }
    touched_slots.clear();
    count = 0;
  }

  bool find(const std::string_view name, const uint32_t hash,
            uint32_t &id) const noexcept {
    const uint32_t slot_count = static_cast<uint32_t>(entries.size());
    const uint32_t mask = slot_count - 1u;
    uint32_t slot = hash & mask;
    bool found = false;
    uint32_t probe_limit = slot_count;

    for (uint32_t probes = 0; probes < probe_limit; ++probes) {
      const auto &entry = entries[slot];
      const bool occupied = entry.occupied;
      const bool match = occupied && entry.hash == hash && entry.name == name;
      id = select_u32(match, entry.id, id);
      found = found || match;
      const bool stop_step = !occupied || match;
      probe_limit = select_u32(stop_step, probes + 1u, probe_limit);
      slot = (slot + 1u) & mask;
    }

    return found;
  }

  bool insert(const std::string_view name, const uint32_t hash,
              const uint32_t id) noexcept {
    const uint32_t slot_count = static_cast<uint32_t>(entries.size());
    const uint32_t mask = slot_count - 1u;
    uint32_t slot = hash & mask;
    uint32_t probe_limit = slot_count;
    bool success = false;
    bool inserted_new = false;
    uint32_t inserted_slot = 0;

    for (uint32_t probes = 0; probes < probe_limit; ++probes) {
      auto &entry = entries[slot];
      const bool occupied = entry.occupied;
      const bool empty_slot = !occupied;
      const bool same_slot =
          occupied && entry.hash == hash && entry.name == name;
      const bool claim_empty = empty_slot;
      const bool claim_existing = same_slot;
      const bool claim = claim_empty || claim_existing;

      const size_t name_bytes =
          sizeof(entry.name) * static_cast<size_t>(claim_empty);
      std::memcpy(&entry.name, &name, name_bytes);
      entry.id = select_u32(claim, id, entry.id);
      entry.hash = select_u32(claim_empty, hash, entry.hash);
      entry.occupied = entry.occupied || claim_empty;

      inserted_slot = select_u32(claim_empty, slot, inserted_slot);
      inserted_new = inserted_new || claim_empty;
      success = success || claim;
      probe_limit = select_u32(claim, probes + 1u, probe_limit);
      slot = (slot + 1u) & mask;
    }

    const size_t prior_touched_size = touched_slots.size();
    touched_slots.push_back(inserted_slot);
    touched_slots.resize(prior_touched_size +
                         static_cast<size_t>(inserted_new));
    count += static_cast<uint32_t>(inserted_new);

    return success;
  }
};

inline constexpr uint32_t k_max_group_nesting_depth = 32;

inline bool is_digit_char(const char c) noexcept {
  return '0' <= c && c <= '9';
}

inline bool is_word_char(const char c) noexcept {
  return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' ||
         is_digit_char(c);
}

inline bool parse_uint64(const char *src, const char *end, uint64_t &value_out,
                         const char **next_out) noexcept {
  const bool ordered = src <= end;
  static constexpr char k_zero = '\0';
  const uintptr_t begin_addr = select_uptr(ordered,
                                           reinterpret_cast<uintptr_t>(src),
                                           reinterpret_cast<uintptr_t>(&k_zero));
  const uintptr_t end_addr = select_uptr(ordered,
                                         reinterpret_cast<uintptr_t>(end),
                                         reinterpret_cast<uintptr_t>(&k_zero));
  const char *begin = reinterpret_cast<const char *>(begin_addr);
  const char *safe_end = reinterpret_cast<const char *>(end_addr);

  uint64_t parsed = 0;
  const auto result = std::from_chars(begin, safe_end, parsed, 10);
  const bool has_digit = result.ptr != begin;
  const bool no_error = result.ec == std::errc{};
  const bool ok = ordered && has_digit && no_error;
  const char *next = result.ptr;

  const size_t out_bytes = sizeof(uint64_t) * static_cast<size_t>(ok);
  std::memcpy(&value_out, &parsed, out_bytes);
  const size_t next_bytes = sizeof(next) * static_cast<size_t>(ok);
  std::memcpy(next_out, &next, next_bytes);
  return ok;
}

inline const char *parse_name(const char *src, const char *end) noexcept {
  const char *pos = src;
  while (pos < end && is_word_char(*pos)) {
    pos++;
  }

  const bool has_name = pos != src;
  const uintptr_t pos_addr = reinterpret_cast<uintptr_t>(pos);
  const uintptr_t out_addr = select_uptr(has_name, pos_addr, 0u);
  return reinterpret_cast<const char *>(out_addr);
}

inline std::pair<uint32_t, const char *>
parse_hex(const char *src, const char *end, const int size) noexcept {
  const bool src_le_end = src <= end;
  const ptrdiff_t distance = (end - src) * static_cast<ptrdiff_t>(src_le_end);
  const bool in_range = src_le_end && distance >= static_cast<ptrdiff_t>(size);
  const int parse_len = size * static_cast<int>(in_range);

  const char *limit = src + parse_len;
  uint32_t value = 0;
  bool valid = in_range;

  constexpr std::array<uint32_t, 22> k_hex_values = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
      11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15};
  constexpr std::string_view k_hex_digits = "0123456789abcdefABCDEF";

  for (const char *pos = src; pos < limit; ++pos) {
    const size_t digit_index = k_hex_digits.find(*pos);
    const bool is_hex_digit = digit_index != std::string_view::npos;
    const bool apply_digit = valid && is_hex_digit;
    const size_t safe_index = select_size(apply_digit, digit_index, 0u);
    const uint32_t shifted = value << 4;
    const uint32_t parsed = shifted + k_hex_values[safe_index];
    value = select_u32(apply_digit, parsed, value);
    valid = apply_digit;
  }

  const bool success = valid;
  const uint32_t out_value = select_u32(success, value, 0u);
  const uintptr_t out_next_addr =
      select_uptr(success, reinterpret_cast<uintptr_t>(limit), 0u);
  return std::make_pair(out_value,
                        reinterpret_cast<const char *>(out_next_addr));
}

inline std::pair<uint32_t, const char *> decode_utf8(const char *src,
                                                     const char *end) noexcept {
  static constexpr char k_zero = '\0';
  const bool has_src = src < end;
  const uintptr_t first_addr =
      select_uptr(has_src, reinterpret_cast<uintptr_t>(src),
                  reinterpret_cast<uintptr_t>(&k_zero));
  const uint8_t first_byte =
      static_cast<uint8_t>(*reinterpret_cast<const char *>(first_addr));

  static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  const uint8_t highbits = first_byte >> 4;
  const int len = lookup[highbits];

  const bool src_le_end = src <= end;
  const ptrdiff_t distance = (end - src) * static_cast<ptrdiff_t>(src_le_end);
  const bool has_bytes = has_src && distance >= static_cast<ptrdiff_t>(len);
  const int decode_len = len * static_cast<int>(has_bytes);

  const uint8_t mask = static_cast<uint8_t>((1u << (8 - len)) - 1u);
  uint32_t value = first_byte & mask;
  for (int i = 1; i < decode_len; ++i) {
    const uint8_t byte = static_cast<uint8_t>(src[i]);
    value = (value << 6) + (byte & 0x3Fu);
  }

  const uint32_t out_value = select_u32(has_bytes, value, 0u);
  const uintptr_t next_addr =
      select_uptr(has_bytes, reinterpret_cast<uintptr_t>(src + decode_len), 0u);
  return std::make_pair(out_value, reinterpret_cast<const char *>(next_addr));
}

inline std::pair<uint32_t, const char *> parse_char(const char *src,
                                                    const char *end) noexcept {
  static constexpr char k_zero = '\0';
  const bool has_src = src < end;
  const uintptr_t first_addr =
      select_uptr(has_src, reinterpret_cast<uintptr_t>(src),
                  reinterpret_cast<uintptr_t>(&k_zero));
  const char first = *reinterpret_cast<const char *>(first_addr);
  const bool is_escape = has_src && first == '\\';

  const bool src_le_end = src <= end;
  const ptrdiff_t distance = (end - src) * static_cast<ptrdiff_t>(src_le_end);
  const bool has_escape_next = is_escape && distance >= 2;

  const uintptr_t escaped_addr =
      select_uptr(has_escape_next, reinterpret_cast<uintptr_t>(src + 1),
                  reinterpret_cast<uintptr_t>(&k_zero));
  const char escaped = *reinterpret_cast<const char *>(escaped_addr);

  const uintptr_t hex_src_addr =
      select_uptr(has_escape_next, reinterpret_cast<uintptr_t>(src + 2),
                  reinterpret_cast<uintptr_t>(src));
  const char *hex_src = reinterpret_cast<const char *>(hex_src_addr);

  const auto hex2 = parse_hex(hex_src, end, 2);
  const auto hex4 = parse_hex(hex_src, end, 4);
  const auto hex8 = parse_hex(hex_src, end, 8);

  const size_t is_hex2 = static_cast<size_t>(escaped == 'x');
  const size_t is_hex4 = static_cast<size_t>(escaped == 'u');
  const size_t is_hex8 = static_cast<size_t>(escaped == 'U');
  const size_t is_tab = static_cast<size_t>(escaped == 't');
  const size_t is_cr = static_cast<size_t>(escaped == 'r');
  const size_t is_lf = static_cast<size_t>(escaped == 'n');
  const size_t is_literal = static_cast<size_t>(
      escaped == '\\' || escaped == '"' || escaped == '[' || escaped == ']');

  const bool hex2_ok = is_hex2 != 0u && hex2.second != nullptr;
  const bool hex4_ok = is_hex4 != 0u && hex4.second != nullptr;
  const bool hex8_ok = is_hex8 != 0u && hex8.second != nullptr;
  const bool direct_kind = (is_tab | is_cr | is_lf | is_literal) != 0u;
  const bool direct_ok = has_escape_next && direct_kind;
  const bool escape_ok = hex2_ok || hex4_ok || hex8_ok || direct_ok;

  const uint32_t escape_value =
      static_cast<uint32_t>(is_hex2) * hex2.first +
      static_cast<uint32_t>(is_hex4) * hex4.first +
      static_cast<uint32_t>(is_hex8) * hex8.first +
      static_cast<uint32_t>(is_tab) * static_cast<uint32_t>('\t') +
      static_cast<uint32_t>(is_cr) * static_cast<uint32_t>('\r') +
      static_cast<uint32_t>(is_lf) * static_cast<uint32_t>('\n') +
      static_cast<uint32_t>(is_literal) *
          static_cast<uint32_t>(static_cast<unsigned char>(escaped));

  const uintptr_t escape_next_addr =
      static_cast<uintptr_t>(is_hex2) *
          reinterpret_cast<uintptr_t>(hex2.second) +
      static_cast<uintptr_t>(is_hex4) *
          reinterpret_cast<uintptr_t>(hex4.second) +
      static_cast<uintptr_t>(is_hex8) *
          reinterpret_cast<uintptr_t>(hex8.second) +
      static_cast<uintptr_t>(direct_kind) *
          select_uptr(has_escape_next, reinterpret_cast<uintptr_t>(src + 2),
                      0u);

  const auto utf8 = decode_utf8(src, end);
  const bool utf8_ok = utf8.second != nullptr;

  const uint32_t selected_value =
      select_u32(is_escape, escape_value, utf8.first);
  const uintptr_t selected_next = select_uptr(
      is_escape, escape_next_addr, reinterpret_cast<uintptr_t>(utf8.second));
  const bool selected_ok = select_bool(is_escape, escape_ok, utf8_ok);

  const uint32_t out_value = select_u32(selected_ok, selected_value, 0u);
  const uintptr_t out_next = select_uptr(selected_ok, selected_next, 0u);
  return std::make_pair(out_value, reinterpret_cast<const char *>(out_next));
}

} // namespace emel::gbnf::rule_parser::detail
