#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"

namespace emel::speech::tokenizer {

// Component decode-policy contract: variant-neutral ASR decode configuration
// bound at the variant boundary (e.g.
// tokenizer::whisper::tiny_asr_decode_policy). Pure data; the variant machines
// validate the bound values before use.
struct control_tokens {
  int32_t eot = 0;
  int32_t sot = 0;
  int32_t language_en = 0;
  int32_t translate = 0;
  int32_t transcribe = 0;
  int32_t no_speech = 0;
  int32_t notimestamps = 0;
  int32_t timestamp_begin = 0;
  int32_t space = 0;
};

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
  bool suppress_translate = false;
  std::array<int32_t, 3> prompt_tokens = {};
};

} // namespace emel::speech::tokenizer

namespace emel::speech::tokenizer::events {

struct detokenize_done;
struct detokenize_error;

} // namespace emel::speech::tokenizer::events

namespace emel::speech::tokenizer::event {

struct detokenize {
  detokenize(const std::string_view tokenizer_json_ref,
             std::span<const int32_t> token_ids_ref,
             std::span<char> transcript_ref,
             int32_t &transcript_size_out_ref) noexcept
      : tokenizer_json(tokenizer_json_ref), token_ids(token_ids_ref),
        transcript(transcript_ref),
        transcript_size_out(transcript_size_out_ref) {}

  std::string_view tokenizer_json = {};
  std::span<const int32_t> token_ids = {};
  std::span<char> transcript = {};
  int32_t &transcript_size_out;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::detokenize_done &)> on_done = {};
  emel::callback<void(const events::detokenize_error &)> on_error = {};
};

} // namespace emel::speech::tokenizer::event

namespace emel::speech::tokenizer::events {

struct detokenize_done {
  const event::detokenize *request = nullptr;
  int32_t transcript_size = 0;
};

struct detokenize_error {
  const event::detokenize *request = nullptr;
  emel::error::type err = {};
};

} // namespace emel::speech::tokenizer::events
