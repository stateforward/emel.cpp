#pragma once

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/plamo2/context.hpp"
#include "emel/text/encoders/plamo2/detail.hpp"

namespace emel::text::encoders::plamo2::action {

struct begin_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    ev.data_len = 0;
    ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    ev.emit_result_token_count = 0;
  }
};

struct begin_encode_sync_vocab {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev.event_, ctx);
    emel::text::encoders::action::sync_vocab(ev.event_, ctx);
    ctx.plamo2_tables_ready = false;
    ctx.plamo2_vocab = nullptr;
    ctx.byte_tokens.fill(0);
    ctx.suffix_map.clear();
    ctx.table.clear();
    ev.data_len = 0;
    ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    ev.emit_result_token_count = 0;
  }
};

struct reject_invalid_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::reject_invalid_encode(ev.event_, ctx);
    ev.data_len = 0;
    ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    ev.emit_result_token_count = 0;
  }
};

struct sync_tables {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::plamo2::detail::ensure_plamo2_tables(ctx, *ctx.vocab);
    ev.event_.ctx.err = emel::text::encoders::plamo2::detail::select_i32(
      ready, emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::model_invalid));
  }
};

struct decode_input {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::plamo2::detail::decode_plamo2_input(
      ev.event_.request, ctx, ev.event_.ctx.err);
    ev.data_len = result.data_len;
    ev.event_.ctx.err = result.error;
  }
};

struct prepare_dp {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::plamo2::detail::prepare_plamo2_dp(ctx, ev.data_len);
  }
};

struct run_dp {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::plamo2::detail::run_plamo2_dp(ctx, ev.data_len);
  }
};

struct emit_tokens {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::plamo2::detail::emit_plamo2_tokens(
      ev.event_.request, ctx, ev.data_len, ev.event_.ctx.err);
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
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.token_count; ev.event_.ctx.err; }) {
      ev.event_.ctx.token_count = 0;
      ev.event_.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
    } else if constexpr (requires { ev.ctx.token_count; ev.ctx.err; }) {
      ev.ctx.token_count = 0;
      ev.ctx.err = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument);
    } else if constexpr (requires { ev.request; }) {
      emel::text::encoders::action::detail::signal_unexpected_request(ev.request);
    }
  }
};

inline constexpr begin_encode begin_encode{};
inline constexpr begin_encode_sync_vocab begin_encode_sync_vocab{};
inline constexpr reject_invalid_encode reject_invalid_encode{};
inline constexpr sync_tables sync_tables{};
inline constexpr decode_input decode_input{};
inline constexpr prepare_dp prepare_dp{};
inline constexpr run_dp run_dp{};
inline constexpr emit_tokens emit_tokens{};
inline constexpr apply_emit_result_ok apply_emit_result_ok{};
inline constexpr apply_emit_result_failed apply_emit_result_failed{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::plamo2::action
