#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/speech/tokenizer/moshi/context.hpp"
#include "emel/speech/tokenizer/moshi/events.hpp"

namespace emel::speech::tokenizer::moshi::detail {

inline int32_t compute_cache_row(const action::context &ctx,
                                 const int64_t offset) noexcept {
  const int64_t rows = ctx.config.cache_rows;
  const int64_t position = offset % rows;
  return static_cast<int32_t>(position + ((position < 0) * rows));
}

inline int32_t &cache_at(action::context &ctx, const int32_t row,
                         const int32_t codebook) noexcept {
  return ctx.config.cache[static_cast<size_t>(row) *
                              static_cast<size_t>(ctx.config.codebooks) +
                          static_cast<size_t>(codebook)];
}

inline int32_t cache_at(const action::context &ctx, const int32_t row,
                        const int32_t codebook) noexcept {
  return ctx.config.cache[static_cast<size_t>(row) *
                              static_cast<size_t>(ctx.config.codebooks) +
                          static_cast<size_t>(codebook)];
}

inline void compute_model_tokens(const event::tokenize &ev,
                                 const action::context &ctx) noexcept {
  const int32_t row = compute_cache_row(ctx, ctx.offset);
  const int32_t text_initial_lane = static_cast<int32_t>(
      ctx.offset <= static_cast<int64_t>(ctx.config.delays[0]));
  const int32_t text_cache_lane = 1 - text_initial_lane;
  ev.model_tokens_out[0] = text_initial_lane * ctx.config.text_initial_token +
                           text_cache_lane * cache_at(ctx, row, 0);

  for (int32_t codebook = 1; codebook < ctx.config.codebooks; ++codebook) {
    const int32_t initial_lane = static_cast<int32_t>(
        ctx.offset <= static_cast<int64_t>(ctx.config.delays[codebook]));
    const int32_t cache_lane = 1 - initial_lane;
    ev.model_tokens_out[static_cast<size_t>(codebook)] =
        initial_lane * ctx.config.audio_initial_token +
        cache_lane * cache_at(ctx, row, codebook);
  }
}

} // namespace emel::speech::tokenizer::moshi::detail
