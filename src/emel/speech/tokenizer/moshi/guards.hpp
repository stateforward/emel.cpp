#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/speech/tokenizer/moshi/context.hpp"
#include "emel/speech/tokenizer/moshi/events.hpp"

namespace emel::speech::tokenizer::moshi::guard {

struct guard_configuration_valid {
  bool operator()(const event::initialize &,
                  const action::context &ctx) const noexcept {
    const auto &config = ctx.config;
    if (config.codebooks <= 1 || config.generated_audio_codebooks <= 0 ||
        config.generated_audio_codebooks >= config.codebooks ||
        config.delayed_audio_codebooks <= 0 ||
        config.delayed_audio_codebooks > config.generated_audio_codebooks ||
        config.cache_rows <= 0 || config.maximum_delay < 0 ||
        config.initial_delay_frames < 0 || config.audio_initial_token <= 0 ||
        config.token_zero == config.token_ungenerated ||
        config.delays.size() < static_cast<size_t>(config.codebooks)) {
      return false;
    }
    const int32_t needed =
        config.codebooks - config.delayed_audio_codebooks - 1;
    if (needed < 0 || static_cast<int64_t>(config.cache_rows) <
                          static_cast<int64_t>(config.maximum_delay) + 2) {
      return false;
    }
    const uint64_t elements = static_cast<uint64_t>(config.cache_rows) *
                              static_cast<uint64_t>(config.codebooks);
    if (elements > config.cache.size()) {
      return false;
    }
    int32_t observed_maximum = 0;
    for (int32_t codebook = 0; codebook < config.codebooks; ++codebook) {
      const int32_t delay = config.delays[static_cast<size_t>(codebook)];
      if (delay < 0 || delay > config.maximum_delay) {
        return false;
      }
      if (delay > observed_maximum) {
        observed_maximum = delay;
      }
    }
    return observed_maximum == config.maximum_delay;
  }
};

struct guard_configuration_invalid {
  bool operator()(const event::initialize &ev,
                  const action::context &ctx) const noexcept {
    return !guard_configuration_valid{}(ev, ctx);
  }
};

struct guard_tokenize_output_valid {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return ev.model_tokens_out.size() ==
           static_cast<size_t>(ctx.config.codebooks);
  }
};

struct guard_tokenize_position_available {
  bool operator()(const event::tokenize &,
                  const action::context &ctx) const noexcept {
    return ctx.offset <= std::numeric_limits<int64_t>::max() -
                             static_cast<int64_t>(ctx.config.maximum_delay);
  }
};

struct guard_tokenize_full_shape {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return guard_tokenize_output_valid{}(ev, ctx) &&
           ev.audio_tokens.size() == static_cast<size_t>(ctx.config.codebooks);
  }
};

struct guard_tokenize_tail_shape {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    const int32_t needed =
        ctx.config.codebooks - ctx.config.delayed_audio_codebooks - 1;
    return guard_tokenize_output_valid{}(ev, ctx) && needed > 0 &&
           ev.audio_tokens.size() == static_cast<size_t>(needed);
  }
};

struct guard_tokenize_empty_shape {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    const int32_t needed =
        ctx.config.codebooks - ctx.config.delayed_audio_codebooks - 1;
    return guard_tokenize_output_valid{}(ev, ctx) && needed == 0 &&
           ev.audio_tokens.empty();
  }
};

struct guard_tokenize_tokens_valid {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    bool valid = true;
    for (const int32_t token : ev.audio_tokens) {
      if (token < 0 || token >= ctx.config.audio_initial_token) {
        valid = false;
      }
    }
    return valid;
  }
};

struct guard_tokenize_shape_valid {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return (guard_tokenize_full_shape{}(ev, ctx) ||
            guard_tokenize_tail_shape{}(ev, ctx) ||
            guard_tokenize_empty_shape{}(ev, ctx)) &&
           guard_tokenize_tokens_valid{}(ev, ctx);
  }
};

struct guard_tokenize_full {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return guard_tokenize_full_shape{}(ev, ctx) &&
           guard_tokenize_tokens_valid{}(ev, ctx) &&
           guard_tokenize_position_available{}(ev, ctx);
  }
};

struct guard_tokenize_tail {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return guard_tokenize_tail_shape{}(ev, ctx) &&
           guard_tokenize_tokens_valid{}(ev, ctx) &&
           guard_tokenize_position_available{}(ev, ctx);
  }
};

struct guard_tokenize_empty {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return guard_tokenize_empty_shape{}(ev, ctx) &&
           guard_tokenize_tokens_valid{}(ev, ctx) &&
           guard_tokenize_position_available{}(ev, ctx);
  }
};

struct guard_tokenize_invalid {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return !guard_tokenize_shape_valid{}(ev, ctx);
  }
};

struct guard_tokenize_position_overflow {
  bool operator()(const event::tokenize &ev,
                  const action::context &ctx) const noexcept {
    return guard_tokenize_shape_valid{}(ev, ctx) &&
           !guard_tokenize_position_available{}(ev, ctx);
  }
};

struct guard_restore_valid {
  bool operator()(const event::restore_cache &ev,
                  const action::context &ctx) const noexcept {
    const uint64_t elements = static_cast<uint64_t>(ctx.config.cache_rows) *
                              static_cast<uint64_t>(ctx.config.codebooks);
    return elements == ev.column_major_cache.size() && ev.offset >= 0 &&
           ev.offset <= std::numeric_limits<int64_t>::max() -
                            static_cast<int64_t>(ctx.config.maximum_delay);
  }
};

struct guard_advance_position_available {
  bool operator()(const event::advance &,
                  const action::context &ctx) const noexcept {
    return ctx.offset < std::numeric_limits<int64_t>::max();
  }
};

struct guard_advance_position_overflow {
  bool operator()(const event::advance &ev,
                  const action::context &ctx) const noexcept {
    return !guard_advance_position_available{}(ev, ctx);
  }
};

struct guard_restore_invalid {
  bool operator()(const event::restore_cache &ev,
                  const action::context &ctx) const noexcept {
    return !guard_restore_valid{}(ev, ctx);
  }
};

struct guard_detokenize_shape_valid {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return runtime_ev.request.audio_tokens.size() ==
               static_cast<size_t>(ctx.config.generated_audio_codebooks) &&
           runtime_ev.request.audio_tokens_out.size() ==
               static_cast<size_t>(ctx.config.delayed_audio_codebooks);
  }
};

struct guard_detokenize_shape_invalid {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_detokenize_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_position_available {
  bool operator()(const event::detokenize_run &,
                  const action::context &ctx) const noexcept {
    return ctx.offset < std::numeric_limits<int64_t>::max();
  }
};

struct guard_position_overflow {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_detokenize_shape_valid{}(runtime_ev, ctx) &&
           !guard_position_available{}(runtime_ev, ctx);
  }
};

struct guard_replace_generated_audio {
  bool operator()(const event::detokenize_run &,
                  const action::context &ctx) const noexcept {
    return ctx.config.initial_delay_frames > 0 &&
           ctx.offset < ctx.config.initial_delay_frames;
  }
};

struct guard_use_generated_audio {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_replace_generated_audio{}(runtime_ev, ctx);
  }
};

struct guard_detokenize_valid_replace {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_detokenize_shape_valid{}(runtime_ev, ctx) &&
           guard_position_available{}(runtime_ev, ctx) &&
           guard_replace_generated_audio{}(runtime_ev, ctx);
  }
};

struct guard_detokenize_valid_generated {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_detokenize_shape_valid{}(runtime_ev, ctx) &&
           guard_position_available{}(runtime_ev, ctx) &&
           guard_use_generated_audio{}(runtime_ev, ctx);
  }
};

struct guard_before_output_delay {
  bool operator()(const event::detokenize_run &,
                  const action::context &ctx) const noexcept {
    return ctx.offset <= ctx.config.maximum_delay;
  }
};

struct guard_past_output_delay {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_before_output_delay{}(runtime_ev, ctx);
  }
};

struct guard_output_incomplete {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    bool incomplete =
        runtime_ev.request.text_token_out == ctx.config.token_zero ||
        runtime_ev.request.text_token_out == ctx.config.token_ungenerated;
    for (int32_t codebook = 0; codebook < ctx.config.delayed_audio_codebooks;
         ++codebook) {
      const int32_t token =
          runtime_ev.request.audio_tokens_out[static_cast<size_t>(codebook)];
      if (token == ctx.config.token_zero ||
          token == ctx.config.token_ungenerated) {
        incomplete = true;
      }
    }
    return incomplete;
  }
};

struct guard_output_complete {
  bool operator()(const event::detokenize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_output_incomplete{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::tokenizer::moshi::guard
