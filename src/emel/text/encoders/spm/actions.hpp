#pragma once

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/spm/context.hpp"
#include "emel/text/encoders/spm/detail.hpp"

namespace emel::text::encoders::spm::action {

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

struct run_encode {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::spm::detail::emit_spm(
      ev.event_.request, ctx, *ctx.vocab);
    ev.emit_result_token_count = result.token_count;
    ev.emit_result_error = result.error;
  }
};

struct set_emit_result_empty {
  void operator()(const runtime::encode_runtime & ev, context &) const noexcept {
    ev.emit_result_token_count = 0;
    ev.emit_result_error =
      emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }
};

struct run_prepare {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    ev.event_.ctx.err = emel::text::encoders::spm::detail::prepare_spm(
      ev.event_.request, ctx, *ctx.vocab);
  }
};

struct run_merge {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    ev.event_.ctx.err = emel::text::encoders::spm::detail::merge_spm(ctx, *ctx.vocab);
  }
};

struct sync_tables {
  void operator()(const runtime::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::spm::detail::ensure_spm_tables(ctx);
    ev.event_.ctx.err = emel::text::encoders::spm::detail::select_i32(
      ready, emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::invalid_argument));
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
inline constexpr run_prepare run_prepare{};
inline constexpr run_merge run_merge{};
inline constexpr run_encode run_encode{};
inline constexpr set_emit_result_empty set_emit_result_empty{};
inline constexpr sync_tables sync_tables{};
inline constexpr apply_emit_result_ok apply_emit_result_ok{};
inline constexpr apply_emit_result_failed apply_emit_result_failed{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::spm::action
