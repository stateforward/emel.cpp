#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/speech/tokenizer/whisper/detail.hpp"

namespace emel::speech::tokenizer::whisper {

using asr_decode_policy = detail::asr_decode_policy;
using control_tokens = detail::control_tokens;
using language_role = detail::language_role;
using task_role = detail::task_role;
using timestamp_mode = detail::timestamp_mode;

inline const asr_decode_policy &tiny_asr_decode_policy() noexcept {
  return detail::k_tiny_asr_decode_policy;
}

inline constexpr std::string_view tiny_tokenizer_sha256() noexcept {
  return "dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759";
}

inline bool
is_tiny_asr_decode_policy_supported(const asr_decode_policy &policy) noexcept {
  const auto tokens = detail::k_tiny_control_tokens;
  return policy.language == language_role::english &&
         policy.task == task_role::transcribe &&
         policy.timestamps == timestamp_mode::timestamp_tokens &&
         policy.suppress_translate && policy.tokens.eot == tokens.eot &&
         policy.tokens.sot == tokens.sot &&
         policy.tokens.language_en == tokens.language_en &&
         policy.tokens.translate == tokens.translate &&
         policy.tokens.transcribe == tokens.transcribe &&
         policy.tokens.no_speech == tokens.no_speech &&
         policy.tokens.notimestamps == tokens.notimestamps &&
         policy.tokens.timestamp_begin == tokens.timestamp_begin &&
         policy.tokens.space == tokens.space &&
         policy.prompt_tokens[0] == tokens.sot &&
         policy.prompt_tokens[1] == tokens.language_en &&
         policy.prompt_tokens[2] == tokens.transcribe;
}

inline std::string_view language_role_name(language_role) noexcept {
  return "english";
}

inline std::string_view task_role_name(task_role) noexcept {
  return "transcribe";
}

inline std::string_view timestamp_mode_name(timestamp_mode) noexcept {
  return "timestamp_tokens";
}

inline bool
validate_tiny_control_tokens(std::string_view tokenizer_json) noexcept {
  return detail::validate_tiny_control_tokens(tokenizer_json);
}

inline uint64_t decode_token_ids(std::string_view tokenizer_json,
                                 std::span<const int32_t> token_ids,
                                 char *transcript,
                                 const uint64_t capacity) noexcept {
  return detail::decode_token_ids(tokenizer_json, token_ids, transcript,
                                  capacity);
}

} // namespace emel::speech::tokenizer::whisper
