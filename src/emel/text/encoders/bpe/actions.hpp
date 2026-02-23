#pragma once

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/bpe/context.hpp"
#include "emel/text/encoders/bpe/detail.hpp"

namespace emel::text::encoders::bpe::action {

struct reject_invalid_encode {
  void operator()(const event::encode & ev, context & ctx) const {
    emel::text::encoders::action::reject_invalid_encode(ev, ctx);
  }
};

struct run_encode {
  void operator()(context & ctx) const noexcept {
    emel::text::encoders::action::run_encode(ctx);
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    emel::text::encoders::action::mark_done(ctx);
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    emel::text::encoders::action::ensure_last_error(ctx);
  }
};

struct on_unexpected {
  template <class event>
  void operator()(const event & ev, context & ctx) const noexcept {
    emel::text::encoders::action::on_unexpected(ev, ctx);
  }
};

struct begin_encode {
  void operator()(const event::encode & ev, context & ctx) const noexcept {
    ctx.token_count = 0;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (emel::text::encoders::action::detail::sync_vocab(ctx, ev.vocab)) {
    }
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    const auto result = emel::text::encoders::bpe::detail::encode_bpe(ev, ctx, *ctx.vocab);
    ctx.token_count = result.token_count;
    ctx.phase_error = result.error;
    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = result.token_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = result.error;
    }
    if (result.error != EMEL_OK) {
      ctx.last_error = result.error;
      emel::text::encoders::action::detail::dispatch_error(ev, result.error);
      return;
    }
    emel::text::encoders::action::detail::dispatch_done(ev, result.token_count);
  }
};

inline constexpr begin_encode begin_encode{};
inline constexpr reject_invalid_encode reject_invalid_encode{};
inline constexpr run_encode run_encode{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::bpe::action
