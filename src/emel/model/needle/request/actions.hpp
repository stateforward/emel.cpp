#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#include "emel/model/needle/request/context.hpp"
#include "emel/model/needle/request/events.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/tokenizer/errors.hpp"
#include "emel/text/tokenizer/needle/events.hpp"
#include "emel/text/tokenizer/needle/sm.hpp"

namespace emel::model::needle::request::action {

inline constexpr std::string_view k_im_start = "<|im_start|>";
inline constexpr std::string_view k_im_end = "<|im_end|>";
inline constexpr std::string_view k_tools_start = "<tools>";
inline constexpr std::string_view k_tools_end = "</tools>";
inline constexpr std::string_view k_think_start = "<think>";
inline constexpr std::string_view k_think_end = "</think>";
inline constexpr std::string_view k_tool_call_start = "<tool_call>";
inline constexpr std::string_view k_tool_call_end = "</tool_call>";

inline void on_tokenizer_load_done(
    const emel::text::tokenizer::needle::events::load_done &) noexcept {}
inline void on_tokenizer_load_error(
    const emel::text::tokenizer::needle::events::load_error &) noexcept {}

inline const emel::text::tokenizer::needle::event::load_done_fn
    k_tokenizer_load_done = emel::text::tokenizer::needle::event::load_done_fn::
        from<&on_tokenizer_load_done>();
inline const emel::text::tokenizer::needle::event::load_error_fn
    k_tokenizer_load_error = emel::text::tokenizer::needle::event::load_error_fn::
        from<&on_tokenizer_load_error>();

inline void reset_response_outputs(context &ctx) noexcept {
  ctx.generated_id_count = 0u;
  ctx.generated_text_size = 0u;
  ctx.normalized_envelope_size = 0u;
  ctx.prefill_nanoseconds = 0u;
  ctx.decode_nanoseconds = 0u;
  ctx.detokenize_pending.fill(0u);
}

inline void reset_outputs(context &ctx) noexcept {
  ctx.prompt_size = 0u;
  ctx.prompt_id_count = 0u;
  reset_response_outputs(ctx);
}

inline void copy_bytes(char *destination,
                       const std::string_view source) noexcept {
  for (size_t i = 0u; i < source.size(); ++i) destination[i] = source[i];
}

inline void append_bytes(std::vector<char> &destination, size_t &size,
                         const std::string_view source) noexcept {
  for (size_t i = 0u; i < source.size(); ++i) destination[size + i] = source[i];
  size += source.size();
}

inline uint32_t argmax(const std::span<const float> logits) noexcept {
  uint32_t best = 0u;
  for (uint32_t index = 1u; index < logits.size(); ++index)
    best = logits[index] > logits[best] ? index : best;
  return best;
}

inline bool append_json_string(std::vector<char> &output, size_t &size,
                               const std::string_view value) noexcept {
  static constexpr char k_hex[] = "0123456789abcdef";
  if (size + 2u > output.size()) return false;
  output[size++] = '"';
  for (const unsigned char c : value) {
    const char *escape = nullptr;
    switch (c) {
    case '"': escape = "\\\""; break;
    case '\\': escape = "\\\\"; break;
    case '\b': escape = "\\b"; break;
    case '\f': escape = "\\f"; break;
    case '\n': escape = "\\n"; break;
    case '\r': escape = "\\r"; break;
    case '\t': escape = "\\t"; break;
    default: break;
    }
    if (escape != nullptr) {
      if (size + 2u > output.size()) return false;
      output[size++] = escape[0];
      output[size++] = escape[1];
    } else if (c < 0x20u) {
      if (size + 6u > output.size()) return false;
      output[size++] = '\\'; output[size++] = 'u';
      output[size++] = '0'; output[size++] = '0';
      output[size++] = k_hex[c >> 4u]; output[size++] = k_hex[c & 0x0fu];
    } else {
      if (size == output.size()) return false;
      output[size++] = static_cast<char>(c);
    }
  }
  if (size == output.size()) return false;
  output[size++] = '"';
  return true;
}

inline bool append_json_literal(std::vector<char> &output, size_t &size,
                                const std::string_view value) noexcept {
  if (size + value.size() > output.size()) return false;
  append_bytes(output, size, value);
  return true;
}

inline bool is_json_space(const char value) noexcept {
  return value == ' ' || value == '\n' || value == '\r' || value == '\t';
}

struct json_cursor {
  std::string_view input;
  size_t offset = 0u;
};

inline void skip_json_space(json_cursor &cursor) noexcept {
  while (cursor.offset < cursor.input.size() &&
         is_json_space(cursor.input[cursor.offset]))
    ++cursor.offset;
}

inline bool consume_json(json_cursor &cursor, const char expected) noexcept {
  skip_json_space(cursor);
  if (cursor.offset == cursor.input.size() ||
      cursor.input[cursor.offset] != expected)
    return false;
  ++cursor.offset;
  return true;
}

struct json_string {
  std::string_view encoded = {};
  bool escaped = false;
};

inline bool is_json_hex(const char value) noexcept {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

inline uint32_t json_hex_value(const char value) noexcept {
  if (value >= '0' && value <= '9')
    return static_cast<uint32_t>(value - '0');
  if (value >= 'a' && value <= 'f')
    return static_cast<uint32_t>(value - 'a' + 10);
  return static_cast<uint32_t>(value - 'A' + 10);
}

inline bool parse_json_string(json_cursor &cursor, json_string &result) noexcept {
  skip_json_space(cursor);
  if (cursor.offset == cursor.input.size() ||
      cursor.input[cursor.offset] != '"')
    return false;
  const size_t begin = ++cursor.offset;
  bool escaped = false;
  while (cursor.offset < cursor.input.size()) {
    const unsigned char value =
        static_cast<unsigned char>(cursor.input[cursor.offset++]);
    if (value == '"') {
      result = {cursor.input.substr(begin, cursor.offset - begin - 1u), escaped};
      return true;
    }
    if (value < 0x20u) return false;
    if (value != '\\') continue;
    escaped = true;
    if (cursor.offset == cursor.input.size()) return false;
    const char escape = cursor.input[cursor.offset++];
    if (escape == 'u') {
      if (cursor.offset + 4u > cursor.input.size()) return false;
      for (size_t digit = 0u; digit < 4u; ++digit)
        if (!is_json_hex(cursor.input[cursor.offset + digit])) return false;
      cursor.offset += 4u;
    } else if (escape != '"' && escape != '\\' && escape != '/' &&
               escape != 'b' && escape != 'f' && escape != 'n' &&
               escape != 'r' && escape != 't') {
      return false;
    }
  }
  return false;
}

inline bool parse_json_string(json_cursor &cursor,
                              std::string_view &unescaped) noexcept {
  json_string value = {};
  if (!parse_json_string(cursor, value) || value.escaped) return false;
  unescaped = value.encoded;
  return true;
}

inline bool json_string_equals(const json_string &value,
                               const std::string_view expected) noexcept {
  size_t encoded_at = 0u;
  size_t expected_at = 0u;
  const auto matches_byte = [&](const unsigned char byte) noexcept {
    if (expected_at == expected.size() ||
        static_cast<unsigned char>(expected[expected_at]) != byte)
      return false;
    ++expected_at;
    return true;
  };
  while (encoded_at < value.encoded.size()) {
    const unsigned char byte =
        static_cast<unsigned char>(value.encoded[encoded_at++]);
    if (byte != '\\') {
      if (!matches_byte(byte)) return false;
      continue;
    }
    const char escape = value.encoded[encoded_at++];
    unsigned char decoded = 0u;
    switch (escape) {
    case '"': decoded = '"'; break;
    case '\\': decoded = '\\'; break;
    case '/': decoded = '/'; break;
    case 'b': decoded = '\b'; break;
    case 'f': decoded = '\f'; break;
    case 'n': decoded = '\n'; break;
    case 'r': decoded = '\r'; break;
    case 't': decoded = '\t'; break;
    case 'u': {
      uint32_t codepoint = 0u;
      for (size_t digit = 0u; digit < 4u; ++digit)
        codepoint = (codepoint << 4u) |
                    json_hex_value(value.encoded[encoded_at + digit]);
      encoded_at += 4u;
      if (codepoint > 0x7fu) return false;
      decoded = static_cast<unsigned char>(codepoint);
      break;
    }
    default: return false;
    }
    if (!matches_byte(decoded)) return false;
  }
  return expected_at == expected.size();
}

inline bool parse_exact_json_string(json_cursor &cursor,
                                    const std::string_view expected) noexcept {
  json_string value = {};
  return parse_json_string(cursor, value) && json_string_equals(value, expected);
}

inline bool is_route_domain(const std::string_view value) noexcept {
  return value == "agentic-coding" || value == "programming-qa" ||
         value == "math" || value == "research" || value == "writing" ||
         value == "extraction" || value == "chat" || value == "other";
}

inline bool is_route_effort(const std::string_view value) noexcept {
  return value == "low" || value == "medium" || value == "high" ||
         value == "xhigh";
}

inline bool parse_route_arguments(json_cursor &cursor) noexcept {
  if (!consume_json(cursor, '{')) return false;
  bool found_domain = false;
  bool found_effort = false;
  bool need_member = true;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == '}') {
      if (need_member) return false;
      ++cursor.offset;
      return found_domain && found_effort;
    }
    std::string_view key = {};
    std::string_view value = {};
    if (!parse_json_string(cursor, key) || !consume_json(cursor, ':') ||
        !parse_json_string(cursor, value))
      return false;
    if (key == "domain") {
      if (found_domain || !is_route_domain(value)) return false;
      found_domain = true;
    } else if (key == "effort") {
      if (found_effort || !is_route_effort(value)) return false;
      found_effort = true;
    } else {
      return false;
    }
    need_member = false;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ',') {
      ++cursor.offset;
      need_member = true;
      continue;
    }
    if (cursor.input[cursor.offset] != '}') return false;
  }
}

inline bool parse_route_call_object(json_cursor &cursor) noexcept {
  if (!consume_json(cursor, '{')) return false;
  bool found_name = false;
  bool found_arguments = false;
  bool need_member = true;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == '}') {
      if (need_member) return false;
      ++cursor.offset;
      return found_name && found_arguments;
    }
    std::string_view key = {};
    if (!parse_json_string(cursor, key) || !consume_json(cursor, ':'))
      return false;
    if (key == "name") {
      std::string_view name = {};
      if (found_name || !parse_json_string(cursor, name) || name != "route")
        return false;
      found_name = true;
    } else if (key == "arguments") {
      if (found_arguments || !parse_route_arguments(cursor)) return false;
      found_arguments = true;
    } else {
      return false;
    }
    need_member = false;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ',') {
      ++cursor.offset;
      need_member = true;
      continue;
    }
    if (cursor.input[cursor.offset] != '}') return false;
  }
}

inline bool validate_route_calls_json(const std::string_view calls) noexcept {
  json_cursor cursor{calls};
  if (!consume_json(cursor, '[')) return false;
  bool need_element = true;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ']') {
      if (need_element) return false;
      ++cursor.offset;
      skip_json_space(cursor);
      return cursor.offset == cursor.input.size();
    }
    if (!parse_route_call_object(cursor)) return false;
    need_element = false;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ',') {
      ++cursor.offset;
      need_element = true;
      continue;
    }
    if (cursor.input[cursor.offset] != ']') return false;
  }
}

inline bool consume_json_literal(json_cursor &cursor,
                                 const std::string_view literal) noexcept {
  if (cursor.input.substr(cursor.offset, literal.size()) != literal) return false;
  cursor.offset += literal.size();
  if (cursor.offset == cursor.input.size()) return true;
  const char next = cursor.input[cursor.offset];
  return is_json_space(next) || next == ',' || next == ']' || next == '}';
}

inline bool skip_json_number(json_cursor &cursor) noexcept {
  const size_t size = cursor.input.size();
  if (cursor.offset < size && cursor.input[cursor.offset] == '-') ++cursor.offset;
  if (cursor.offset == size) return false;
  if (cursor.input[cursor.offset] == '0') {
    ++cursor.offset;
    if (cursor.offset < size && cursor.input[cursor.offset] >= '0' &&
        cursor.input[cursor.offset] <= '9')
      return false;
  } else {
    if (cursor.input[cursor.offset] < '1' || cursor.input[cursor.offset] > '9')
      return false;
    do {
      ++cursor.offset;
    } while (cursor.offset < size && cursor.input[cursor.offset] >= '0' &&
             cursor.input[cursor.offset] <= '9');
  }
  if (cursor.offset < size && cursor.input[cursor.offset] == '.') {
    ++cursor.offset;
    if (cursor.offset == size || cursor.input[cursor.offset] < '0' ||
        cursor.input[cursor.offset] > '9')
      return false;
    do {
      ++cursor.offset;
    } while (cursor.offset < size && cursor.input[cursor.offset] >= '0' &&
             cursor.input[cursor.offset] <= '9');
  }
  if (cursor.offset < size &&
      (cursor.input[cursor.offset] == 'e' || cursor.input[cursor.offset] == 'E')) {
    ++cursor.offset;
    if (cursor.offset < size &&
        (cursor.input[cursor.offset] == '+' || cursor.input[cursor.offset] == '-'))
      ++cursor.offset;
    if (cursor.offset == size || cursor.input[cursor.offset] < '0' ||
        cursor.input[cursor.offset] > '9')
      return false;
    do {
      ++cursor.offset;
    } while (cursor.offset < size && cursor.input[cursor.offset] >= '0' &&
             cursor.input[cursor.offset] <= '9');
  }
  if (cursor.offset == size) return true;
  const char next = cursor.input[cursor.offset];
  return is_json_space(next) || next == ',' || next == ']' || next == '}';
}

inline bool skip_json_value(json_cursor &cursor, uint32_t depth = 0u) noexcept {
  if (depth > 32u) return false;
  skip_json_space(cursor);
  if (cursor.offset == cursor.input.size()) return false;
  const char value = cursor.input[cursor.offset];
  if (value == '"') {
    json_string ignored = {};
    return parse_json_string(cursor, ignored);
  }
  if (value == '{') {
    ++cursor.offset;
    skip_json_space(cursor);
    if (cursor.offset < cursor.input.size() && cursor.input[cursor.offset] == '}') {
      ++cursor.offset;
      return true;
    }
    for (;;) {
      json_string key = {};
      if (!parse_json_string(cursor, key) || !consume_json(cursor, ':') ||
          !skip_json_value(cursor, depth + 1u))
        return false;
      skip_json_space(cursor);
      if (cursor.offset == cursor.input.size()) return false;
      if (cursor.input[cursor.offset] == '}') {
        ++cursor.offset;
        return true;
      }
      if (cursor.input[cursor.offset++] != ',') return false;
      skip_json_space(cursor);
      if (cursor.offset == cursor.input.size() ||
          cursor.input[cursor.offset] == '}')
        return false;
    }
  }
  if (value == '[') {
    ++cursor.offset;
    skip_json_space(cursor);
    if (cursor.offset < cursor.input.size() && cursor.input[cursor.offset] == ']') {
      ++cursor.offset;
      return true;
    }
    for (;;) {
      if (!skip_json_value(cursor, depth + 1u)) return false;
      skip_json_space(cursor);
      if (cursor.offset == cursor.input.size()) return false;
      if (cursor.input[cursor.offset] == ']') {
        ++cursor.offset;
        return true;
      }
      if (cursor.input[cursor.offset++] != ',') return false;
      skip_json_space(cursor);
      if (cursor.offset == cursor.input.size() ||
          cursor.input[cursor.offset] == ']')
        return false;
    }
  }
  if (value == 't') return consume_json_literal(cursor, "true");
  if (value == 'f') return consume_json_literal(cursor, "false");
  if (value == 'n') return consume_json_literal(cursor, "null");
  return value == '-' || (value >= '0' && value <= '9')
             ? skip_json_number(cursor)
             : false;
}

template <size_t Size>
inline bool parse_exact_string_set(
    json_cursor &cursor,
    const std::array<std::string_view, Size> &expected) noexcept {
  static_assert(Size <= 32u);
  if (!consume_json(cursor, '[')) return false;
  uint32_t found = 0u;
  size_t count = 0u;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size() || cursor.input[cursor.offset] == ']')
      return false;
    json_string value = {};
    if (!parse_json_string(cursor, value)) return false;
    size_t index = 0u;
    while (index < Size && !json_string_equals(value, expected[index])) ++index;
    if (index == Size || (found & (1u << index)) != 0u) return false;
    found |= 1u << index;
    ++count;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ']') {
      ++cursor.offset;
      return count == Size;
    }
    if (cursor.input[cursor.offset++] != ',') return false;
  }
}

template <size_t Size>
inline bool parse_route_property_schema(
    json_cursor &cursor,
    const std::array<std::string_view, Size> &enum_values) noexcept {
  if (!consume_json(cursor, '{')) return false;
  bool found_type = false;
  bool found_enum = false;
  bool need_member = true;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == '}') {
      if (need_member) return false;
      ++cursor.offset;
      return found_type && found_enum;
    }
    json_string key = {};
    if (!parse_json_string(cursor, key) || !consume_json(cursor, ':')) return false;
    if (json_string_equals(key, "type")) {
      if (found_type || !parse_exact_json_string(cursor, "string")) return false;
      found_type = true;
    } else if (json_string_equals(key, "enum")) {
      if (found_enum || !parse_exact_string_set(cursor, enum_values)) return false;
      found_enum = true;
    } else {
      return false;
    }
    need_member = false;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ',') {
      ++cursor.offset;
      need_member = true;
    } else if (cursor.input[cursor.offset] != '}') {
      return false;
    }
  }
}

inline bool parse_route_properties(json_cursor &cursor) noexcept {
  static constexpr std::array<std::string_view, 8> k_domains = {
      "agentic-coding", "programming-qa", "math", "research",
      "writing",        "extraction",     "chat", "other"};
  static constexpr std::array<std::string_view, 4> k_efforts = {
      "low", "medium", "high", "xhigh"};
  if (!consume_json(cursor, '{')) return false;
  bool found_domain = false;
  bool found_effort = false;
  bool need_member = true;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == '}') {
      if (need_member) return false;
      ++cursor.offset;
      return found_domain && found_effort;
    }
    json_string key = {};
    if (!parse_json_string(cursor, key) || !consume_json(cursor, ':')) return false;
    if (json_string_equals(key, "domain")) {
      if (found_domain || !parse_route_property_schema(cursor, k_domains))
        return false;
      found_domain = true;
    } else if (json_string_equals(key, "effort")) {
      if (found_effort || !parse_route_property_schema(cursor, k_efforts))
        return false;
      found_effort = true;
    } else {
      return false;
    }
    need_member = false;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ',') {
      ++cursor.offset;
      need_member = true;
    } else if (cursor.input[cursor.offset] != '}') {
      return false;
    }
  }
}

inline bool parse_route_parameters(json_cursor &cursor) noexcept {
  static constexpr std::array<std::string_view, 2> k_required = {"domain",
                                                                 "effort"};
  if (!consume_json(cursor, '{')) return false;
  bool found_type = false;
  bool found_properties = false;
  bool found_required = false;
  bool need_member = true;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == '}') {
      if (need_member) return false;
      ++cursor.offset;
      return found_type && found_properties && found_required;
    }
    json_string key = {};
    if (!parse_json_string(cursor, key) || !consume_json(cursor, ':')) return false;
    if (json_string_equals(key, "type")) {
      if (found_type || !parse_exact_json_string(cursor, "object")) return false;
      found_type = true;
    } else if (json_string_equals(key, "properties")) {
      if (found_properties || !parse_route_properties(cursor)) return false;
      found_properties = true;
    } else if (json_string_equals(key, "required")) {
      if (found_required || !parse_exact_string_set(cursor, k_required))
        return false;
      found_required = true;
    } else {
      return false;
    }
    need_member = false;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ',') {
      ++cursor.offset;
      need_member = true;
    } else if (cursor.input[cursor.offset] != '}') {
      return false;
    }
  }
}

inline bool parse_route_tool(json_cursor &cursor) noexcept {
  if (!consume_json(cursor, '{')) return false;
  bool found_name = false;
  bool found_description = false;
  bool found_parameters = false;
  bool need_member = true;
  for (;;) {
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == '}') {
      if (need_member) return false;
      ++cursor.offset;
      return found_name && found_description && found_parameters;
    }
    json_string key = {};
    if (!parse_json_string(cursor, key) || !consume_json(cursor, ':')) return false;
    if (json_string_equals(key, "name")) {
      if (found_name || !parse_exact_json_string(cursor, "route")) return false;
      found_name = true;
    } else if (json_string_equals(key, "description")) {
      json_string description = {};
      if (found_description || !parse_json_string(cursor, description)) return false;
      found_description = true;
    } else if (json_string_equals(key, "parameters")) {
      if (found_parameters || !parse_route_parameters(cursor)) return false;
      found_parameters = true;
    } else if (!skip_json_value(cursor)) {
      return false;
    }
    need_member = false;
    skip_json_space(cursor);
    if (cursor.offset == cursor.input.size()) return false;
    if (cursor.input[cursor.offset] == ',') {
      ++cursor.offset;
      need_member = true;
    } else if (cursor.input[cursor.offset] != '}') {
      return false;
    }
  }
}

inline bool validate_tools_json(const std::string_view tools) noexcept {
  json_cursor cursor{tools};
  if (!consume_json(cursor, '[') || !parse_route_tool(cursor)) return false;
  if (!consume_json(cursor, ']')) return false;
  skip_json_space(cursor);
  return cursor.offset == cursor.input.size();
}

inline bool parse_route_call(const std::string_view generated,
                             std::string_view &reasoning,
                             std::string_view &calls) noexcept {
  reasoning = {};
  calls = {};
  const size_t think_begin = generated.find(k_think_start);
  const size_t think_end = generated.find(k_think_end);
  if (think_begin != std::string_view::npos &&
      think_end != std::string_view::npos && think_end > think_begin) {
    size_t begin = think_begin + k_think_start.size();
    if (begin < generated.size() && generated[begin] == '\n') ++begin;
    size_t end = think_end;
    if (end > begin && generated[end - 1u] == '\n') --end;
    reasoning = generated.substr(begin, end - begin);
  }
  const size_t call_begin = generated.find(k_tool_call_start);
  const size_t call_end = generated.find(k_tool_call_end);
  if (call_begin == std::string_view::npos || call_end == std::string_view::npos ||
      call_end <= call_begin + k_tool_call_start.size())
    return false;
  calls = generated.substr(call_begin + k_tool_call_start.size(),
                           call_end - call_begin - k_tool_call_start.size());
  return validate_route_calls_json(calls);
}
template <class runtime_event>
inline const auto &origin_event(const runtime_event &ev) noexcept {
  if constexpr (requires { ev.event_; })
    return ev.event_;
  else
    return ev;
}

struct effect_begin_configure {
  void operator()(const event::configure_run &ev, context &ctx) const noexcept {
    ctx.configured = false;
    ctx.assets_ready = false;
    ctx.reset_ready = false;
    reset_outputs(ctx);
    ev.ctx.err = emel::error::cast(error::none);
  }
};


inline bool normalize_generated_response(context &ctx,
                                         const std::string_view generated) noexcept {
  size_t &size = ctx.normalized_envelope_size;
  size = 0u;
  std::string_view reasoning = {};
  std::string_view calls = {};
  if (!parse_route_call(generated, reasoning, calls)) return false;
  auto literal = [&](const std::string_view text) noexcept {
    return append_json_literal(ctx.normalized_envelope, size, text);
  };
  return literal("{\"error\":null,\"error_code\":null,\"function_calls\":") &&
         literal(calls) && literal(",\"reason\":null,\"reasoning\":") &&
         append_json_string(ctx.normalized_envelope, size, reasoning) &&
         literal(",\"success\":true,\"type\":\"call\",\"validation\":{\"negation\":false,\"ungrounded\":[]}}");
}
struct effect_initialize_assets {
  void operator()(const event::configure_run &ev, context &ctx) const noexcept {
    const auto blob = std::span<const uint8_t>{ctx.bound.tokenizer_blob.data,
                                               static_cast<size_t>(ctx.bound.tokenizer_blob.nbytes)};
    emel::text::tokenizer::needle::sm loader{};
    const bool loaded = loader.process_event(emel::text::tokenizer::needle::event::load{
        blob, *ctx.vocab, k_tokenizer_load_done, k_tokenizer_load_error});
    int32_t tokenizer_error = emel::text::tokenizer::error_code(
        emel::text::tokenizer::error::none);
    emel::text::tokenizer::event::bind tokenizer_bind{};
    tokenizer_bind.vocab = ctx.vocab.get();
    tokenizer_bind.preprocessor_variant = emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
    tokenizer_bind.encoder_variant = emel::text::encoders::encoder_kind::spm;
    tokenizer_bind.error_out = &tokenizer_error;
    const bool tokenizer_bound = loaded && ctx.tokenizer->process_event(tokenizer_bind);
    int32_t detokenizer_error = emel::text::detokenizer::error_code(
        emel::text::detokenizer::error::none);
    const bool detokenizer_bound = tokenizer_bound && ctx.detokenizer->process_event(
        emel::text::detokenizer::event::bind{*ctx.vocab, detokenizer_error});
    ctx.assets_ready = detokenizer_bound && ctx.vocab->bos_id >= 0 && ctx.vocab->eos_id >= 0;
    ev.ctx.err = ctx.assets_ready ? emel::error::cast(error::none)
                                  : emel::error::cast(error::not_initialized);
  }
};

struct effect_store_configuration {
  void operator()(const event::configure_run &ev, context &ctx) const noexcept {
    copy_bytes(ctx.system_storage.data(), ev.request.system);
    copy_bytes(ctx.tools_storage.data(), ev.request.tools_json);
    ctx.system_size = ev.request.system.size();
    ctx.tools_size = ev.request.tools_json.size();
    ctx.configured = true;
    ctx.reset_ready = false;
  }
};

struct effect_begin_reset {
  void operator()(const event::reset_run &ev, context &ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    reset_outputs(ctx);
  }
};
struct effect_exec_reset {
  void operator()(const event::reset_run &ev, context &ctx) const noexcept {
    ctx.reset_ready = ctx.graph->process_event(
        needle::graph::event::init{.activation_quant = false});
    ev.ctx.err = ctx.reset_ready ? emel::error::cast(error::none)
                                 : emel::error::cast(error::graph_rejected);
  }
};

struct effect_begin_complete {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    reset_response_outputs(ctx);
    ev.ctx.timestamp_now = ctx.timestamp_now;
  }
};

struct effect_render_prompt {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &ctx) const noexcept {
    const auto &run = origin_event(ev);
    ctx.prompt_size = 0u;
    ctx.prompt_id_count = 0u;
    size_t required = k_im_start.size() + 5u + k_tools_start.size() +
                      ctx.tools_size + k_tools_end.size() + 1u +
                      run.request.query.size() + k_im_end.size() + 1u +
                      k_im_start.size() + 10u;
    if (ctx.system_size != 0u)
      required += k_im_start.size() + 7u + ctx.system_size + k_im_end.size() + 1u;
    if (required > ctx.prompt_storage.size()) {
      run.ctx.err = emel::error::cast(error::capacity_exceeded);
      return;
    }
    if (ctx.system_size != 0u) {
      append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_start);
      append_bytes(ctx.prompt_storage, ctx.prompt_size, "system\n");
      append_bytes(ctx.prompt_storage, ctx.prompt_size,
                   {ctx.system_storage.data(), ctx.system_size});
      append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_end);
      append_bytes(ctx.prompt_storage, ctx.prompt_size, "\n");
    }
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_start);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "user\n");
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_tools_start);
    append_bytes(ctx.prompt_storage, ctx.prompt_size,
                 {ctx.tools_storage.data(), ctx.tools_size});
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_tools_end);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "\n");
    append_bytes(ctx.prompt_storage, ctx.prompt_size, run.request.query);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_end);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "\n");
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_start);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "assistant\n");
  }
};

struct effect_tokenize_prompt {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &ctx) const noexcept {
    const auto &run = origin_event(ev);
    int32_t token_count = 0;
    int32_t tokenizer_error =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
    emel::text::tokenizer::event::tokenize tokenize{};
    tokenize.vocab = ctx.vocab.get();
    tokenize.text = {ctx.prompt_storage.data(), ctx.prompt_size};
    tokenize.add_special = false;
    tokenize.parse_special = true;
    tokenize.token_ids_out = ctx.prompt_ids.data() + 1u;
    tokenize.token_capacity = static_cast<int32_t>(ctx.prompt_ids.size() - 1u);
    tokenize.token_count_out = &token_count;
    tokenize.error_out = &tokenizer_error;
    const bool accepted = ctx.tokenizer->process_event(tokenize);
    if (!accepted || token_count < 0 ||
        static_cast<size_t>(token_count) + 1u + run.request.max_new_tokens >
            ctx.prompt_ids.size() ||
        static_cast<uint64_t>(token_count) + 1u + run.request.max_new_tokens >
            ctx.bound.geo.max_seq_len) {
      run.ctx.err = emel::error::cast(accepted ? error::capacity_exceeded
                                               : error::tokenizer_rejected);
      return;
    }
    ctx.prompt_ids[0] = ctx.vocab->bos_id;
    ctx.prompt_id_count = static_cast<size_t>(token_count) + 1u;
  }
};

struct effect_prefill {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    const uint64_t begin = ev.ctx.timestamp_now();
    const bool accepted = ctx.graph->process_event(needle::graph::event::prefill{
        {ctx.prompt_ids.data(), ctx.prompt_id_count}, ctx.logits});
    const uint64_t end = ev.ctx.timestamp_now();
    ctx.prefill_nanoseconds = end - begin;
    ev.ctx.err = accepted ? emel::error::cast(error::none)
                           : emel::error::cast(error::graph_rejected);
    if (accepted)
      ctx.generated_ids[0] =
          static_cast<int32_t>(argmax(std::span<const float>{ctx.logits}));
  }
};

struct effect_decode_token {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    const int32_t token = ctx.generated_ids[ctx.generated_id_count];
    ++ctx.generated_id_count;
    const uint64_t begin = ev.ctx.timestamp_now();
    const bool accepted = ctx.graph->process_event(
        needle::graph::event::decode{token, ctx.logits});
    const uint64_t end = ev.ctx.timestamp_now();
    ctx.decode_nanoseconds += end - begin;
    ev.ctx.err = accepted ? emel::error::cast(error::none)
                           : emel::error::cast(error::graph_rejected);
    if (accepted)
      ctx.generated_ids[ctx.generated_id_count] =
          static_cast<int32_t>(argmax(std::span<const float>{ctx.logits}));
  }
};

struct effect_finish_generation {
  void operator()(const event::complete_run &, context &ctx) const noexcept {
    if (ctx.generated_id_count < ctx.generated_ids.size() &&
        ctx.generated_ids[ctx.generated_id_count] != ctx.vocab->eos_id)
      ++ctx.generated_id_count;
  }
};

struct effect_detokenize_generation {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    size_t pending_length = 0u;
    for (size_t i = 0u; i < ctx.generated_id_count; ++i) {
      size_t piece_length = 0u;
      size_t next_pending = pending_length;
      int32_t detokenizer_error = emel::text::detokenizer::error_code(
          emel::text::detokenizer::error::none);
      const bool accepted = ctx.detokenizer->process_event(
          emel::text::detokenizer::event::detokenize{
              ctx.generated_ids[i], true, ctx.detokenize_pending.data(),
              pending_length, ctx.detokenize_pending.size(),
              ctx.detokenize_piece.data(), ctx.detokenize_piece.size(),
              piece_length, next_pending, detokenizer_error});
      if (!accepted ||
          ctx.generated_text_size + piece_length > ctx.generated_text.size()) {
        ev.ctx.err = emel::error::cast(accepted ? error::capacity_exceeded
                                                 : error::detokenizer_rejected);
        return;
      }
      for (size_t byte = 0u; byte < piece_length; ++byte) {
        const char value = ctx.detokenize_piece[byte];
        if (value == '\xE2' && byte + 2u < piece_length &&
            static_cast<unsigned char>(ctx.detokenize_piece[byte + 1u]) == 0x96u &&
            static_cast<unsigned char>(ctx.detokenize_piece[byte + 2u]) == 0x81u) {
          ctx.generated_text[ctx.generated_text_size++] = ' ';
          byte += 2u;
        } else {
          ctx.generated_text[ctx.generated_text_size++] = value;
        }
      }
      pending_length = next_pending;
    }
    if (pending_length != 0u)
      ev.ctx.err = emel::error::cast(error::detokenizer_rejected);
  }
};

struct effect_normalize_response {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    const std::string_view generated{ctx.generated_text.data(),
                                     ctx.generated_text_size};
    const bool ok = normalize_generated_response(ctx, generated);
    ev.ctx.err = ok ? emel::error::cast(error::none)
                    : emel::error::cast(error::response_invalid);
  }
};

struct effect_publish_configured {
  void operator()(const event::configure_run &ev, context &) const noexcept {
    ev.request.on_done(events::configured{ev.request});
  }
};

struct effect_publish_reset_done {
  void operator()(const event::reset_run &ev, context &) const noexcept {
    ev.request.on_done(events::reset_done{ev.request});
  }
};

struct effect_publish_completed {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    ev.request.on_done(events::completed{
        .request = ev.request,
        .normalized_envelope = {ctx.normalized_envelope.data(),
                                ctx.normalized_envelope_size},
        .generated_token_ids = {ctx.generated_ids.data(),
                                ctx.generated_id_count},
        .prompt_tokens = static_cast<uint32_t>(ctx.prompt_id_count),
        .generated_tokens = static_cast<uint32_t>(ctx.generated_id_count),
        .prefill_nanoseconds = ctx.prefill_nanoseconds,
        .decode_nanoseconds = ctx.decode_nanoseconds,
    });
  }
};

struct effect_publish_error {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &) const noexcept {
    ev.request.on_error(events::request_error{ev.ctx.err});
  }
};

struct effect_mark_invalid {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.err; })
      ev.event_.ctx.err = emel::error::cast(error::invalid_request);
    else
      ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_on_unexpected {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &) const noexcept {
    const auto &run = origin_event(ev);
    if constexpr (requires { run.ctx.err; })
      run.ctx.err = emel::error::cast(error::internal_error);
  }
};

} // namespace emel::model::needle::request::action
