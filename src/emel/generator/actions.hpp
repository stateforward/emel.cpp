#pragma once

#include "emel/generator/context.hpp"
#include "emel/generator/events.hpp"

namespace emel::generator::action {

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

struct begin_generate {
  void operator()(const event::generate & ev, context & ctx) const noexcept {
    ctx.tokens_generated = 0;
    ctx.max_tokens = ev.max_tokens;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct reject_invalid_generate {
  void operator()(const event::generate & ev, context & ctx) const noexcept {
    set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
  }
};

struct tokenize_prompt {
  void operator()(const event::generate &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct run_prefill {
  void operator()(const event::generate &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct run_decode_step {
  void operator()(const event::generate &, context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.tokens_generated += 1;
  }
};

struct dispatch_generation_done_to_owner {
  void operator()(const event::generate & ev, const context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    if (ev.dispatch_done) {
      ev.dispatch_done(events::generation_done{
        .request = &ev,
        .tokens_generated = ctx.tokens_generated,
      });
    }
  }
};

struct dispatch_generation_error_to_owner {
  void operator()(const event::generate & ev, const context & ctx) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = ctx.last_error;
    }
    if (ev.dispatch_error) {
      ev.dispatch_error(events::generation_error{
        .request = &ev,
        .err = ctx.last_error,
        .tokens_generated = ctx.tokens_generated,
      });
    }
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context & ctx) const noexcept {
    set_error(ctx, EMEL_ERR_BACKEND);
  }
};

inline constexpr begin_generate begin_generate{};
inline constexpr reject_invalid_generate reject_invalid_generate{};
inline constexpr tokenize_prompt tokenize_prompt{};
inline constexpr run_prefill run_prefill{};
inline constexpr run_decode_step run_decode_step{};
inline constexpr dispatch_generation_done_to_owner dispatch_generation_done_to_owner{};
inline constexpr dispatch_generation_error_to_owner dispatch_generation_error_to_owner{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::generator::action
