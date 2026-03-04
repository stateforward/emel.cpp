#pragma once

#include <array>
#include <cstddef>

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/fallback/context.hpp"
#include "emel/text/encoders/fallback/detail.hpp"

namespace emel::text::encoders::fallback::action {

struct begin_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    ev.emit_result_token_count = 0;
  }
};

struct begin_encode_sync_vocab {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    emel::text::encoders::action::sync_vocab(ev.event_, ctx);
    ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    ev.emit_result_token_count = 0;
  }
};

struct reject_invalid_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::reject_invalid_encode(ev.event_, ctx);
    ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    ev.emit_result_token_count = 0;
  }
};

struct prepare_tables {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::fallback::detail::ensure_fallback_tables(ctx, *ctx.vocab);
    const std::array<int32_t, 2> errors{emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok)};
    ev.event_.ctx.err = errors[static_cast<size_t>(ready)];
  }
};

struct run_encode_exec {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::fallback::detail::encode_fallback_exec(
      ev.event_.request, ctx, *ctx.vocab);
    ev.emit_result_token_count = result.token_count;
    ev.emit_result_error = result.error;
  }
};

struct apply_emit_result_ok {
  void operator()(const runtime::encode_runtime & ev, context &) const noexcept {
    ev.event_.ctx.token_count = ev.emit_result_token_count;
    ev.event_.ctx.err =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }
};

struct apply_emit_result_failed {
  void operator()(const runtime::encode_runtime & ev, context &) const noexcept {
    ev.event_.ctx.token_count = 0;
    ev.event_.ctx.err = ev.emit_result_error;
  }
};

struct mark_done {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::mark_done(ev.event_, ctx);
  }
};

struct ensure_last_error {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::ensure_last_error(ev.event_, ctx);
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context & ctx) const noexcept {
    emel::text::encoders::action::on_unexpected(ev, ctx);
  }
};

inline constexpr begin_encode begin_encode{};
inline constexpr begin_encode_sync_vocab begin_encode_sync_vocab{};
inline constexpr reject_invalid_encode reject_invalid_encode{};
inline constexpr prepare_tables prepare_tables{};
inline constexpr run_encode_exec run_encode_exec{};
inline constexpr apply_emit_result_ok apply_emit_result_ok{};
inline constexpr apply_emit_result_failed apply_emit_result_failed{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::fallback::action
