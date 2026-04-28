#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace emel::speech::tokenizer::whisper::detail {

struct control_tokens {
  int32_t eot = 50257;
  int32_t sot = 50258;
  int32_t language_en = 50259;
  int32_t translate = 50358;
  int32_t transcribe = 50359;
  int32_t no_speech = 50362;
  int32_t notimestamps = 50363;
  int32_t timestamp_begin = 50364;
  int32_t space = 220;
};

inline constexpr control_tokens k_tiny_control_tokens{};

enum class language_role : uint8_t {
  english = 0u,
};

enum class task_role : uint8_t {
  transcribe = 0u,
};

enum class timestamp_mode : uint8_t {
  timestamp_tokens = 0u,
};

struct asr_decode_policy {
  control_tokens tokens = {};
  language_role language = language_role::english;
  task_role task = task_role::transcribe;
  timestamp_mode timestamps = timestamp_mode::timestamp_tokens;
  bool suppress_translate = true;
  std::array<int32_t, 3> prompt_tokens = {};
};

inline constexpr asr_decode_policy k_tiny_asr_decode_policy{
    .tokens = k_tiny_control_tokens,
    .language = language_role::english,
    .task = task_role::transcribe,
    .timestamps = timestamp_mode::timestamp_tokens,
    .suppress_translate = true,
    .prompt_tokens =
        {
            k_tiny_control_tokens.sot,
            k_tiny_control_tokens.language_en,
            k_tiny_control_tokens.transcribe,
        },
};

inline bool contains(std::string_view text, std::string_view needle) noexcept {
  return text.find(needle) != std::string_view::npos;
}

inline uint32_t write_i32(const int32_t id, char *out) noexcept {
  char digits[12] = {};
  int32_t value = id;
  uint32_t offset = 0;
  if (value < 0) {
    out[offset] = '-';
    ++offset;
    value = -value;
  }
  uint32_t digits_count = 0;
  do {
    digits[digits_count] = static_cast<char>('0' + (value % 10));
    value /= 10;
    ++digits_count;
  } while (value != 0);
  while (digits_count > 0) {
    --digits_count;
    out[offset] = digits[digits_count];
    ++offset;
  }
  return offset;
}

inline bool contains_token_role(std::string_view tokenizer_json,
                                std::string_view content,
                                const int32_t id) noexcept {
  const auto content_pos = tokenizer_json.find(content);
  if (content_pos == std::string_view::npos) {
    return false;
  }

  char id_buffer[16] = {};
  const uint32_t offset = write_i32(id, id_buffer);

  const std::string_view id_view{id_buffer, offset};
  const size_t search_begin = content_pos > 96u ? content_pos - 96u : 0u;
  const size_t search_end = content_pos + content.size() + 96u;
  const auto window =
      tokenizer_json.substr(search_begin, search_end - search_begin);
  return contains(window, "\"id\"") && contains(window, id_view);
}

inline bool is_json_space(const char value) noexcept {
  return value == ' ' || value == '\n' || value == '\r' || value == '\t';
}

inline size_t skip_json_space(std::string_view text,
                              size_t offset) noexcept {
  while (offset < text.size() && is_json_space(text[offset])) {
    ++offset;
  }
  return offset;
}

inline bool find_json_object_value(std::string_view json,
                                   std::string_view key,
                                   std::string_view &object_out) noexcept {
  size_t search = 0u;
  while (search < json.size()) {
    const size_t key_pos = json.find(key, search);
    if (key_pos == std::string_view::npos) {
      return false;
    }
    search = key_pos + key.size();

    size_t colon = skip_json_space(json, search);
    if (colon >= json.size() || json[colon] != ':') {
      continue;
    }
    size_t object_begin = skip_json_space(json, colon + 1u);
    if (object_begin >= json.size() || json[object_begin] != '{') {
      continue;
    }

    uint32_t depth = 1u;
    bool in_string = false;
    bool escaped = false;
    const size_t content_begin = object_begin + 1u;
    for (size_t offset = content_begin; offset < json.size(); ++offset) {
      const char value = json[offset];
      if (in_string) {
        if (escaped) {
          escaped = false;
        } else if (value == '\\') {
          escaped = true;
        } else if (value == '"') {
          in_string = false;
        }
        continue;
      }

      if (value == '"') {
        in_string = true;
      } else if (value == '{') {
        ++depth;
      } else if (value == '}') {
        --depth;
        if (depth == 0u) {
          object_out = json.substr(content_begin, offset - content_begin);
          return true;
        }
      }
    }
    return false;
  }
  return false;
}

inline std::string_view vocab_lookup_scope(std::string_view tokenizer_json)
    noexcept {
  std::string_view vocab_json = {};
  if (find_json_object_value(tokenizer_json, "\"vocab\"", vocab_json)) {
    return vocab_json;
  }
  return tokenizer_json;
}

inline bool find_next_json_object_entry(std::string_view object_json,
                                        size_t &search,
                                        std::string_view &key_out,
                                        size_t &value_begin_out) noexcept {
  while (search < object_json.size()) {
    const size_t key_begin = object_json.find('"', search);
    if (key_begin == std::string_view::npos) {
      return false;
    }

    bool escaped = false;
    size_t key_end = key_begin + 1u;
    while (key_end < object_json.size()) {
      const char value = object_json[key_end];
      if (escaped) {
        escaped = false;
      } else if (value == '\\') {
        escaped = true;
      } else if (value == '"') {
        break;
      }
      ++key_end;
    }
    if (key_end >= object_json.size()) {
      return false;
    }

    const size_t colon = skip_json_space(object_json, key_end + 1u);
    if (colon >= object_json.size() || object_json[colon] != ':') {
      search = key_end + 1u;
      continue;
    }

    key_out = object_json.substr(key_begin + 1u, key_end - key_begin - 1u);
    value_begin_out = skip_json_space(object_json, colon + 1u);
    search = value_begin_out;
    return true;
  }
  return false;
}

inline bool json_value_ends_at(std::string_view object_json,
                               size_t value_end) noexcept {
  value_end = skip_json_space(object_json, value_end);
  return value_end == object_json.size() || object_json[value_end] == ',' ||
         object_json[value_end] == '}';
}

inline bool find_vocab_token_text(std::string_view tokenizer_json,
                                  const int32_t id,
                                  std::string_view &token_text_out) noexcept {
  char id_buffer[16] = {};
  const uint32_t id_size = write_i32(id, id_buffer);
  const std::string_view id_view{id_buffer, id_size};
  const std::string_view vocab_json = vocab_lookup_scope(tokenizer_json);
  size_t search = 0u;
  while (search < vocab_json.size()) {
    std::string_view token_text = {};
    size_t value_begin = 0u;
    if (!find_next_json_object_entry(vocab_json, search, token_text,
                                     value_begin)) {
      return false;
    }
    const size_t value_end = value_begin + id_view.size();
    if (value_end <= vocab_json.size() &&
        vocab_json.substr(value_begin, id_view.size()) == id_view &&
        json_value_ends_at(vocab_json, value_end)) {
      token_text_out = token_text;
      return true;
    }
    search = value_begin < vocab_json.size() ? value_begin + 1u
                                             : vocab_json.size();
  }
  return false;
}

inline size_t find_max_vocab_token_bytes(std::string_view tokenizer_json)
    noexcept {
  const std::string_view vocab_json = vocab_lookup_scope(tokenizer_json);
  size_t search = 0u;
  size_t max_token_bytes = 0u;
  while (search < vocab_json.size()) {
    std::string_view token_text = {};
    size_t value_begin = 0u;
    if (!find_next_json_object_entry(vocab_json, search, token_text,
                                     value_begin)) {
      break;
    }
    if (token_text.size() > max_token_bytes) {
      max_token_bytes = token_text.size();
    }
    search = value_begin < vocab_json.size() ? value_begin + 1u
                                             : vocab_json.size();
  }
  return max_token_bytes;
}

inline size_t required_transcript_capacity(std::string_view tokenizer_json,
                                           const size_t token_count) noexcept {
  if (token_count == 0u) {
    return 0u;
  }
  size_t token_bytes = find_max_vocab_token_bytes(tokenizer_json);
  if (token_bytes == 0u) {
    token_bytes = tokenizer_json.size();
  }
  if (token_bytes == 0u) {
    token_bytes = 1u;
  }
  if (token_count > static_cast<size_t>(-1) / token_bytes) {
    return static_cast<size_t>(-1);
  }
  return token_count * token_bytes;
}

inline void append_decoded_piece(std::string_view piece, char *transcript,
                                 const uint64_t capacity,
                                 uint64_t &size) noexcept {
  for (size_t index = 0u; index < piece.size();) {
    char out = piece[index];
    uint64_t advance = 1u;
    if (index + 1u < piece.size() &&
        static_cast<unsigned char>(piece[index]) == 0xc4u &&
        static_cast<unsigned char>(piece[index + 1u]) == 0xa0u) {
      out = ' ';
      advance = 2u;
    }
    if (size >= capacity) {
      break;
    }
    transcript[size] = out;
    ++size;
    index += static_cast<size_t>(advance);
  }
}

inline uint64_t decode_token_ids(std::string_view tokenizer_json,
                                 std::span<const int32_t> token_ids,
                                 char *transcript,
                                 const uint64_t capacity) noexcept {
  uint64_t size = 0u;
  for (const int32_t token_id : token_ids) {
    if (token_id >= k_tiny_control_tokens.eot) {
      continue;
    }
    std::string_view piece = {};
    if (find_vocab_token_text(tokenizer_json, token_id, piece)) {
      append_decoded_piece(piece, transcript, capacity, size);
    }
  }
  const uint64_t compact_size = size < capacity ? size : capacity;
  uint64_t leading_spaces = 0u;
  while (leading_spaces < compact_size && transcript[leading_spaces] == ' ') {
    ++leading_spaces;
  }
  for (uint64_t index = leading_spaces; index < compact_size; ++index) {
    transcript[index - leading_spaces] = transcript[index];
  }
  return compact_size - leading_spaces;
}

inline bool
validate_tiny_control_tokens(std::string_view tokenizer_json) noexcept {
  const auto tokens = k_tiny_control_tokens;
  return contains_token_role(tokenizer_json, "<|endoftext|>", tokens.eot) &&
         contains_token_role(tokenizer_json, "<|startoftranscript|>",
                             tokens.sot) &&
         contains_token_role(tokenizer_json, "<|en|>", tokens.language_en) &&
         contains_token_role(tokenizer_json, "<|translate|>",
                             tokens.translate) &&
         contains_token_role(tokenizer_json, "<|transcribe|>",
                             tokens.transcribe) &&
         contains_token_role(tokenizer_json, "<|nocaptions|>",
                             tokens.no_speech) &&
         contains_token_role(tokenizer_json, "<|notimestamps|>",
                             tokens.notimestamps) &&
         contains_token_role(tokenizer_json, "<|0.00|>",
                             tokens.timestamp_begin);
}

} // namespace emel::speech::tokenizer::whisper::detail
