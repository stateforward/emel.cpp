#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

#include "emel/emel.h"
#include "emel/gbnf/parser/context.hpp"
#include "emel/gbnf/types.hpp"

namespace emel::gbnf::parser::detail {

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

  std::array<entry, emel::gbnf::k_gbnf_symbol_table_slots> entries = {};
  uint32_t count = 0;

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
    for (auto &entry : entries) {
      entry = {};
    }
    count = 0;
  }

  bool find(const std::string_view name, const uint32_t hash, uint32_t &id) const noexcept {
    const uint32_t mask = static_cast<uint32_t>(entries.size() - 1);
    uint32_t slot = hash & mask;
    for (uint32_t probes = 0; probes < entries.size(); ++probes) {
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
    const uint32_t mask = static_cast<uint32_t>(entries.size() - 1);
    uint32_t slot = hash & mask;
    for (uint32_t probes = 0; probes < entries.size(); ++probes) {
      auto &entry = entries[slot];
      if (!entry.occupied) {
        entry.name = name;
        entry.id = id;
        entry.hash = hash;
        entry.occupied = true;
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

/**
 * encapsulates the recursive descent grammar parser from llama.cpp.
 * translated to use bounded storage and length-aware parsing.
 */
struct recursive_descent_parser {
  action::context &ctx;
  emel::gbnf::grammar *grammar = nullptr;
  symbol_table symbols = {};
  uint32_t next_symbol_id = 0;

  explicit recursive_descent_parser(action::context &c,
                                    emel::gbnf::grammar *out) noexcept
      : ctx(c), grammar(out) {
    symbols.clear();
  }

  static bool is_digit_char(char c) noexcept { return '0' <= c && c <= '9'; }

  static bool is_word_char(char c) noexcept {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' ||
           is_digit_char(c);
  }

  static bool parse_uint64(const char *src,
                           const char *end,
                           uint64_t &value_out,
                           const char **next_out) noexcept {
    if (src >= end || !is_digit_char(*src)) {
      return false;
    }
    uint64_t value = 0;
    const uint64_t max_div_10 =
        std::numeric_limits<uint64_t>::max() / 10u;
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

  static const char *parse_space(const char *src,
                                 const char *end,
                                 bool newline_ok) noexcept {
    const char *pos = src;
    while (pos < end &&
           (*pos == ' ' || *pos == '\t' || *pos == '#' ||
            (newline_ok && (*pos == '\r' || *pos == '\n')))) {
      if (*pos == '#') {
        while (pos < end && *pos != '\r' && *pos != '\n') {
          pos++;
        }
      } else {
        pos++;
      }
    }
    return pos;
  }

  static const char *parse_name(const char *src, const char *end) noexcept {
    const char *pos = src;
    while (pos < end && is_word_char(*pos)) {
      pos++;
    }
    if (pos == src) {
      return nullptr;
    }
    return pos;
  }

  static std::pair<uint32_t, const char *> parse_hex(const char *src,
                                                     const char *end,
                                                     int size) noexcept {
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

  static std::pair<uint32_t, const char *> decode_utf8(const char *src,
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

  static std::pair<uint32_t, const char *> parse_char(const char *src,
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

  static std::pair<uint32_t, const char *> parse_token(const char *src,
                                                       const char *end) noexcept {
    const char *pos = src;
    if (pos >= end || *pos != '<') {
      return std::make_pair(0, nullptr);
    }
    pos++;
    if (pos >= end || *pos != '[') {
      return std::make_pair(0, nullptr);
    }
    pos++;

    uint64_t token_id = 0;
    const char *int_end = nullptr;
    if (!parse_uint64(pos, end, token_id, &int_end)) {
      return std::make_pair(0, nullptr);
    }
    if (token_id > std::numeric_limits<uint32_t>::max()) {
      return std::make_pair(0, nullptr);
    }
    pos = int_end;
    if (pos >= end || *pos != ']') {
      return std::make_pair(0, nullptr);
    }
    pos++;
    if (pos >= end || *pos != '>') {
      return std::make_pair(0, nullptr);
    }
    pos++;
    return std::make_pair(static_cast<uint32_t>(token_id), pos);
  }

  uint32_t get_symbol_id(const char *src, size_t len) noexcept {
    if (len == 0 || src == nullptr) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return 0;
    }
    const std::string_view name(src, len);
    const uint32_t hash = symbol_table::hash_name(name);
    uint32_t id = 0;
    if (symbols.find(name, hash, id)) {
      return id;
    }
    if (next_symbol_id >= emel::gbnf::k_max_gbnf_rules ||
        symbols.count >= emel::gbnf::k_max_gbnf_symbols) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return 0;
    }
    id = next_symbol_id++;
    if (!symbols.insert(name, hash, id)) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return 0;
    }
    return id;
  }

  uint32_t generate_symbol_id(const std::string_view &) noexcept {
    if (next_symbol_id >= emel::gbnf::k_max_gbnf_rules) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return 0;
    }
    return next_symbol_id++;
  }

  bool add_rule(uint32_t rule_id, const rule_builder &rule) noexcept {
    if (grammar == nullptr) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    if (rule_id >= emel::gbnf::k_max_gbnf_rules) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    if (grammar->rule_lengths[rule_id] != 0) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    if (grammar->element_count + rule.size > emel::gbnf::k_max_gbnf_elements) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    grammar->rule_offsets[rule_id] = grammar->element_count;
    grammar->rule_lengths[rule_id] = rule.size;
    if (rule_id + 1 > grammar->rule_count) {
      grammar->rule_count = rule_id + 1;
    }
    std::memcpy(grammar->elements.data() + grammar->element_count,
                rule.elements.data(),
                sizeof(emel::gbnf::element) * rule.size);
    grammar->element_count += rule.size;
    return true;
  }

  const char *parse_alternates(const char *src,
                               const char *end,
                               const std::string_view &rule_name,
                               uint32_t rule_id,
                               bool is_nested) noexcept {
    rule_builder current_rule{};
    const char *pos = parse_sequence(src, end, rule_name, current_rule, is_nested);
    if (!pos) {
      return nullptr;
    }

    while (pos < end && *pos == '|') {
      if (!current_rule.push({emel::gbnf::element_type::alt, 0})) {
        ctx.phase_error = EMEL_ERR_PARSE_FAILED;
        return nullptr;
      }
      pos = parse_space(pos + 1, end, true);
      pos = parse_sequence(pos, end, rule_name, current_rule, is_nested);
      if (!pos) {
        return nullptr;
      }
    }

    if (!current_rule.push({emel::gbnf::element_type::end, 0})) {
      ctx.phase_error = EMEL_ERR_PARSE_FAILED;
      return nullptr;
    }
    if (!add_rule(rule_id, current_rule)) {
      return nullptr;
    }
    return pos;
  }

  const char *parse_sequence(const char *src,
                             const char *end,
                             const std::string_view &rule_name,
                             rule_builder &current_rule,
                             bool is_nested) noexcept {
    uint32_t last_sym_start = current_rule.size;
    const char *pos = src;

    auto handle_repetitions = [&](uint64_t min_times,
                                  uint64_t max_times) -> bool {
      const bool no_max = max_times == UINT64_MAX;
      if (last_sym_start == current_rule.size) {
        return false;
      }
      if (!no_max && max_times < min_times) {
        return false;
      }

      const uint32_t prev_len = current_rule.size - last_sym_start;
      rule_builder prev_rule{};
      if (!prev_rule.append(current_rule.elements.data() + last_sym_start, prev_len)) {
        return false;
      }

      if (min_times == 0) {
        current_rule.resize(last_sym_start);
      } else {
        for (uint64_t i = 1; i < min_times; ++i) {
          if (!current_rule.append(prev_rule.elements.data(), prev_len)) {
            return false;
          }
        }
      }

      uint32_t last_rec_rule_id = 0;
      const uint64_t n_opt = no_max ? 1 : (max_times - min_times);
      rule_builder rec_rule{};

      for (uint64_t i = 0; i < n_opt; ++i) {
        rec_rule.size = 0;
        if (!rec_rule.append(prev_rule.elements.data(), prev_len)) {
          return false;
        }
        const uint32_t rec_rule_id = generate_symbol_id(rule_name);
        if (ctx.phase_error != EMEL_OK) {
          return false;
        }
        if (i > 0 || no_max) {
          if (!rec_rule.push({emel::gbnf::element_type::rule_ref,
                              no_max ? rec_rule_id : last_rec_rule_id})) {
            return false;
          }
        }
        if (!rec_rule.push({emel::gbnf::element_type::alt, 0}) ||
            !rec_rule.push({emel::gbnf::element_type::end, 0})) {
          return false;
        }
        if (!add_rule(rec_rule_id, rec_rule)) {
          return false;
        }
        last_rec_rule_id = rec_rule_id;
      }

      if (n_opt > 0) {
        if (!current_rule.push(
                {emel::gbnf::element_type::rule_ref, last_rec_rule_id})) {
          return false;
        }
      }
      return true;
    };

    while (pos < end) {
      const char c = *pos;
      if (c == '"') {
        pos++;
        last_sym_start = current_rule.size;
        while (pos < end && *pos != '"') {
          auto char_pair = parse_char(pos, end);
          if (!char_pair.second) {
            return nullptr;
          }
          pos = char_pair.second;
          if (!current_rule.push(
                  {emel::gbnf::element_type::character, char_pair.first})) {
            ctx.phase_error = EMEL_ERR_PARSE_FAILED;
            return nullptr;
          }
        }
        if (pos >= end || *pos != '"') {
          return nullptr;
        }
        pos = parse_space(pos + 1, end, is_nested);
      } else if (c == '[') {
        pos++;
        auto start_type = emel::gbnf::element_type::character;
        if (pos < end && *pos == '^') {
          pos++;
          start_type = emel::gbnf::element_type::char_not;
        }
        last_sym_start = current_rule.size;
        while (pos < end && *pos != ']') {
          auto char_pair = parse_char(pos, end);
          if (!char_pair.second) {
            return nullptr;
          }
          pos = char_pair.second;
          auto type = last_sym_start < current_rule.size
                          ? emel::gbnf::element_type::char_alt
                          : start_type;
          if (!current_rule.push({type, char_pair.first})) {
            ctx.phase_error = EMEL_ERR_PARSE_FAILED;
            return nullptr;
          }
          if (pos + 1 < end && pos[0] == '-' && pos[1] != ']') {
            auto end_pair = parse_char(pos + 1, end);
            if (!end_pair.second) {
              return nullptr;
            }
            pos = end_pair.second;
            if (!current_rule.push(
                    {emel::gbnf::element_type::char_rng_upper, end_pair.first})) {
              ctx.phase_error = EMEL_ERR_PARSE_FAILED;
              return nullptr;
            }
          }
        }
        if (pos >= end || *pos != ']') {
          return nullptr;
        }
        pos = parse_space(pos + 1, end, is_nested);
      } else if (c == '<' || c == '!') {
        auto type = emel::gbnf::element_type::token;
        if (c == '!') {
          type = emel::gbnf::element_type::token_not;
          pos++;
        }
        auto token_pair = parse_token(pos, end);
        if (!token_pair.second) {
          return nullptr;
        }
        last_sym_start = current_rule.size;
        if (!current_rule.push({type, token_pair.first})) {
          ctx.phase_error = EMEL_ERR_PARSE_FAILED;
          return nullptr;
        }
        pos = parse_space(token_pair.second, end, is_nested);
      } else if (is_word_char(c)) {
        const char *name_end = parse_name(pos, end);
        if (!name_end) {
          return nullptr;
        }
        const uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
        if (ctx.phase_error != EMEL_OK) {
          return nullptr;
        }
        pos = parse_space(name_end, end, is_nested);
        last_sym_start = current_rule.size;
        if (!current_rule.push(
                {emel::gbnf::element_type::rule_ref, ref_rule_id})) {
          ctx.phase_error = EMEL_ERR_PARSE_FAILED;
          return nullptr;
        }
      } else if (c == '(') {
        pos = parse_space(pos + 1, end, true);
        const uint32_t sub_rule_id = generate_symbol_id(rule_name);
        if (ctx.phase_error != EMEL_OK) {
          return nullptr;
        }
        pos = parse_alternates(pos, end, rule_name, sub_rule_id, true);
        if (!pos) {
          return nullptr;
        }
        last_sym_start = current_rule.size;
        if (!current_rule.push(
                {emel::gbnf::element_type::rule_ref, sub_rule_id})) {
          ctx.phase_error = EMEL_ERR_PARSE_FAILED;
          return nullptr;
        }
        if (pos >= end || *pos != ')') {
          return nullptr;
        }
        pos = parse_space(pos + 1, end, is_nested);
      } else if (c == '.') {
        last_sym_start = current_rule.size;
        if (!current_rule.push({emel::gbnf::element_type::char_any, 0})) {
          ctx.phase_error = EMEL_ERR_PARSE_FAILED;
          return nullptr;
        }
        pos = parse_space(pos + 1, end, is_nested);
      } else if (c == '*') {
        pos = parse_space(pos + 1, end, is_nested);
        if (!handle_repetitions(0, UINT64_MAX)) {
          return nullptr;
        }
      } else if (c == '+') {
        pos = parse_space(pos + 1, end, is_nested);
        if (!handle_repetitions(1, UINT64_MAX)) {
          return nullptr;
        }
      } else if (c == '?') {
        pos = parse_space(pos + 1, end, is_nested);
        if (!handle_repetitions(0, 1)) {
          return nullptr;
        }
      } else if (c == '{') {
        pos = parse_space(pos + 1, end, is_nested);
        if (pos >= end || !is_digit_char(*pos)) {
          return nullptr;
        }

        uint64_t min_times = 0;
        const char *int_end = nullptr;
        if (!parse_uint64(pos, end, min_times, &int_end)) {
          return nullptr;
        }
        pos = parse_space(int_end, end, is_nested);

        uint64_t max_times = UINT64_MAX;
        if (pos < end && *pos == '}') {
          max_times = min_times;
          pos = parse_space(pos + 1, end, is_nested);
        } else if (pos < end && *pos == ',') {
          pos = parse_space(pos + 1, end, is_nested);
          if (pos < end && is_digit_char(*pos)) {
            if (!parse_uint64(pos, end, max_times, &int_end)) {
              return nullptr;
            }
            pos = parse_space(int_end, end, is_nested);
          }
          if (pos >= end || *pos != '}') {
            return nullptr;
          }
          pos = parse_space(pos + 1, end, is_nested);
        } else {
          return nullptr;
        }
        constexpr uint64_t k_max_repetition_threshold = 2000;
        if (min_times > k_max_repetition_threshold ||
            (max_times != UINT64_MAX &&
             max_times > k_max_repetition_threshold)) {
          return nullptr;
        }
        if (!handle_repetitions(min_times, max_times)) {
          return nullptr;
        }
      } else {
        break;
      }
    }
    return pos;
  }

  const char *parse_rule(const char *src, const char *end) noexcept {
    const char *name_end = parse_name(src, end);
    if (!name_end) {
      return nullptr;
    }
    const char *pos = parse_space(name_end, end, false);
    const size_t name_len = static_cast<size_t>(name_end - src);
    const uint32_t rule_id = get_symbol_id(src, name_len);
    if (ctx.phase_error != EMEL_OK) {
      return nullptr;
    }
    const std::string_view name(src, name_len);

    if (pos + 2 >= end || pos[0] != ':' || pos[1] != ':' || pos[2] != '=') {
      return nullptr;
    }
    pos = parse_space(pos + 3, end, true);

    pos = parse_alternates(pos, end, name, rule_id, false);
    if (!pos) {
      return nullptr;
    }

    if (pos < end && *pos == '\r') {
      pos += (pos + 1 < end && pos[1] == '\n') ? 2 : 1;
    } else if (pos < end && *pos == '\n') {
      pos++;
    } else if (pos < end) {
      return nullptr;
    }
    return parse_space(pos, end, true);
  }

  bool parse(const std::string_view text) noexcept {
    if (grammar == nullptr) {
      ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
      return false;
    }
    const char *pos = parse_space(text.data(), text.data() + text.size(), true);
    const char *end = text.data() + text.size();
    while (pos < end) {
      pos = parse_rule(pos, end);
      if (!pos) {
        ctx.phase_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
    }
    return true;
  }
};

} // namespace emel::gbnf::parser::detail
