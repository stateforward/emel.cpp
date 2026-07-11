#pragma once

#include <cstdint>
#include <span>

namespace emel::speech::tokenizer::moshi {

struct dependencies {
  std::span<const int32_t> delays = {};
  std::span<int32_t> cache = {};
  int32_t codebooks = 0;
  int32_t generated_audio_codebooks = 0;
  int32_t delayed_audio_codebooks = 0;
  int32_t cache_rows = 0;
  int32_t maximum_delay = 0;
  int32_t initial_delay_frames = 0;
  int32_t text_initial_token = 0;
  int32_t audio_initial_token = 0;
  int32_t token_zero = 0;
  int32_t token_ungenerated = 0;
};

namespace action {

struct context {
  explicit context(const dependencies &deps) noexcept : config(deps) {}

  const dependencies config;
  int64_t offset = 0;
};

} // namespace action

} // namespace emel::speech::tokenizer::moshi
