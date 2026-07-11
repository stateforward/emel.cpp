#pragma once

#include <cstdint>
#include <span>

namespace emel::speech::tokenizer::moshi::event {

struct initialize {
  int32_t &error_out;
};

struct tokenize {
  std::span<const int32_t> audio_tokens = {};
  std::span<int32_t> model_tokens_out = {};
  int32_t &error_out;
};

struct detokenize {
  int32_t text_token;
  std::span<const int32_t> audio_tokens;
  int32_t &text_token_out;
  std::span<int32_t> audio_tokens_out;
  bool &produced_out;
  int32_t &error_out;
};

struct restore_cache {
  std::span<const int32_t> column_major_cache;
  int64_t offset;
  int32_t &error_out;
};

struct advance {
  int32_t &error_out;
};

struct reset {
  int32_t &error_out;
};

struct detokenize_ctx {
  int64_t source_offset = 0;
};

struct detokenize_run {
  const detokenize &request;
  detokenize_ctx &ctx;
};

} // namespace emel::speech::tokenizer::moshi::event
