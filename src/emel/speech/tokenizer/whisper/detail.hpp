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

inline bool find_vocab_token_text(std::string_view tokenizer_json,
                                  const int32_t id,
                                  std::string_view &token_text_out) noexcept {
  char id_buffer[16] = {};
  const uint32_t id_size = write_i32(id, id_buffer);
  const std::string_view id_view{id_buffer, id_size};
  size_t search = 0u;
  while (search < tokenizer_json.size()) {
    const size_t id_pos = tokenizer_json.find(id_view, search);
    if (id_pos == std::string_view::npos) {
      return false;
    }
    search = id_pos + id_view.size();

    size_t colon = id_pos;
    while (colon > 0u && tokenizer_json[colon - 1u] == ' ') {
      --colon;
    }
    if (colon == 0u || tokenizer_json[colon - 1u] != ':') {
      continue;
    }
    const size_t value_end = id_pos + id_view.size();
    if (value_end < tokenizer_json.size() && tokenizer_json[value_end] != ',' &&
        tokenizer_json[value_end] != '\n' &&
        tokenizer_json[value_end] != '\r') {
      continue;
    }

    const size_t key_end = tokenizer_json.rfind('"', colon - 1u);
    if (key_end == std::string_view::npos || key_end == 0u) {
      continue;
    }
    const size_t key_begin = tokenizer_json.rfind('"', key_end - 1u);
    if (key_begin == std::string_view::npos) {
      continue;
    }
    token_text_out =
        tokenizer_json.substr(key_begin + 1u, key_end - key_begin - 1u);
    return true;
  }
  return false;
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
    if (size < capacity) {
      transcript[size] = out;
      ++size;
    }
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
  uint64_t leading_spaces = 0u;
  while (leading_spaces < size && transcript[leading_spaces] == ' ') {
    ++leading_spaces;
  }
  for (uint64_t index = leading_spaces; index < size; ++index) {
    transcript[index - leading_spaces] = transcript[index];
  }
  return size - leading_spaces;
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
