#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/speech/tokenizer/moshi/context.hpp"
#include "emel/speech/tokenizer/moshi/detail.hpp"
#include "emel/speech/tokenizer/moshi/errors.hpp"
#include "emel/speech/tokenizer/moshi/events.hpp"

namespace emel::speech::tokenizer::moshi::action {

inline int32_t error_code(const error value) noexcept {
  return static_cast<int32_t>(emel::error::cast(value));
}

struct effect_initialize {
  void operator()(const event::initialize &ev, context &ctx) const noexcept {
    std::fill(ctx.config.cache.begin(), ctx.config.cache.end(),
              ctx.config.token_ungenerated);
    ctx.offset = 0;
    ev.error_out = error_code(error::none);
  }
};

struct effect_reset {
  void operator()(const event::reset &ev, context &ctx) const noexcept {
    std::fill(ctx.config.cache.begin(), ctx.config.cache.end(),
              ctx.config.token_ungenerated);
    ctx.offset = 0;
    ev.error_out = error_code(error::none);
  }
};

struct effect_restore_column_major_cache {
  void operator()(const event::restore_cache &ev, context &ctx) const noexcept {
    for (int32_t row = 0; row < ctx.config.cache_rows; ++row) {
      for (int32_t codebook = 0; codebook < ctx.config.codebooks; ++codebook) {
        const size_t source = static_cast<size_t>(row) +
                              static_cast<size_t>(codebook) *
                                  static_cast<size_t>(ctx.config.cache_rows);
        detail::cache_at(ctx, row, codebook) = ev.column_major_cache[source];
      }
    }
    ctx.offset = ev.offset;
    ev.error_out = error_code(error::none);
  }
};

struct effect_advance {
  void operator()(const event::advance &ev, context &ctx) const noexcept {
    ++ctx.offset;
    ev.error_out = error_code(error::none);
  }
};

struct effect_tokenize_full {
  void operator()(const event::tokenize &ev, context &ctx) const noexcept {
    for (int32_t codebook = 0; codebook < ctx.config.codebooks; ++codebook) {
      const int32_t row = detail::compute_cache_row(
          ctx, ctx.offset + ctx.config.delays[static_cast<size_t>(codebook)]);
      detail::cache_at(ctx, row, codebook) =
          ev.audio_tokens[static_cast<size_t>(codebook)];
    }
    detail::compute_model_tokens(ev, ctx);
    ev.error_out = error_code(error::none);
  }
};

struct effect_tokenize_tail {
  void operator()(const event::tokenize &ev, context &ctx) const noexcept {
    const int32_t first_codebook = ctx.config.delayed_audio_codebooks + 1;
    const int32_t needed = ctx.config.codebooks - first_codebook;
    for (int32_t tail = 0; tail < needed; ++tail) {
      const int32_t codebook = first_codebook + tail;
      const int32_t row = detail::compute_cache_row(
          ctx, ctx.offset + ctx.config.delays[static_cast<size_t>(codebook)]);
      detail::cache_at(ctx, row, codebook) =
          ev.audio_tokens[static_cast<size_t>(tail)];
    }
    detail::compute_model_tokens(ev, ctx);
    ev.error_out = error_code(error::none);
  }
};

struct effect_tokenize_empty {
  void operator()(const event::tokenize &ev,
                  const context &ctx) const noexcept {
    detail::compute_model_tokens(ev, ctx);
    ev.error_out = error_code(error::none);
  }
};

struct effect_begin_detokenize {
  void operator()(const event::detokenize_run &runtime_ev,
                  const context &ctx) const noexcept {
    runtime_ev.ctx.source_offset = ctx.offset;
    runtime_ev.request.text_token_out = ctx.config.token_zero;
    std::fill(runtime_ev.request.audio_tokens_out.begin(),
              runtime_ev.request.audio_tokens_out.end(), ctx.config.token_zero);
    runtime_ev.request.produced_out = false;
    runtime_ev.request.error_out = error_code(error::none);
  }
};

enum class cache_write_mode : uint8_t {
  replace,
  preserve_provided,
};

enum class audio_write_mode : uint8_t {
  zero,
  generated,
};

template <cache_write_mode cache_mode, audio_write_mode audio_mode>
struct effect_commit_generated {
  void operator()(const event::detokenize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.offset = runtime_ev.ctx.source_offset + 1;
    const int32_t row = detail::compute_cache_row(ctx, ctx.offset);
    detail::cache_at(ctx, row, 0) = runtime_ev.request.text_token;

    for (int32_t audio_codebook = 0;
         audio_codebook < ctx.config.generated_audio_codebooks;
         ++audio_codebook) {
      const int32_t codebook = audio_codebook + 1;
      int32_t token = ctx.config.token_zero;
      if constexpr (audio_mode == audio_write_mode::generated) {
        const int32_t masked = static_cast<int32_t>(
            ctx.config.initial_delay_frames > 0 &&
            runtime_ev.ctx.source_offset <
                static_cast<int64_t>(
                    ctx.config.delays[static_cast<size_t>(codebook)]) +
                    static_cast<int64_t>(ctx.config.initial_delay_frames));
        token = masked * ctx.config.token_zero +
                (1 - masked) *
                    runtime_ev.request
                        .audio_tokens[static_cast<size_t>(audio_codebook)];
      }

      if constexpr (cache_mode == cache_write_mode::preserve_provided) {
        const int32_t current = detail::cache_at(ctx, row, codebook);
        const int32_t missing =
            static_cast<int32_t>(current == ctx.config.token_ungenerated);
        token = missing * token + (1 - missing) * current;
      }
      detail::cache_at(ctx, row, codebook) = token;
    }
    runtime_ev.request.error_out = error_code(error::none);
  }
};

struct effect_collect_output {
  void operator()(const event::detokenize_run &runtime_ev,
                  const context &ctx) const noexcept {
    int32_t row = detail::compute_cache_row(
        ctx, ctx.offset - ctx.config.maximum_delay + ctx.config.delays[0]);
    runtime_ev.request.text_token_out = detail::cache_at(ctx, row, 0);
    for (int32_t audio_codebook = 0;
         audio_codebook < ctx.config.delayed_audio_codebooks;
         ++audio_codebook) {
      const int32_t codebook = audio_codebook + 1;
      row = detail::compute_cache_row(
          ctx, ctx.offset - ctx.config.maximum_delay +
                   ctx.config.delays[static_cast<size_t>(codebook)]);
      runtime_ev.request.audio_tokens_out[static_cast<size_t>(audio_codebook)] =
          detail::cache_at(ctx, row, codebook);
    }
  }
};

struct effect_publish_output {
  void operator()(const event::detokenize_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.produced_out = true;
    runtime_ev.request.error_out = error_code(error::none);
  }
};

struct effect_publish_no_output {
  void operator()(const event::detokenize_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.produced_out = false;
    runtime_ev.request.error_out = error_code(error::none);
  }
};

template <error error_value> struct effect_reject {
  template <class event_type>
  void operator()(const event_type &ev, const context &) const noexcept {
    ev.error_out = error_code(error_value);
  }
};

template <error error_value> struct effect_reject_detokenize {
  void operator()(const event::detokenize_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.error_out = error_code(error_value);
  }
};

struct effect_unexpected {
  template <class event_type>
  void operator()(const event_type &, const context &) const noexcept {}
};

} // namespace emel::speech::tokenizer::moshi::action
