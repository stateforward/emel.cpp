#pragma once

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/spm/context.hpp"
#include "emel/text/encoders/spm/detail.hpp"

namespace emel::text::encoders::spm::action {

struct begin_encode {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev, ctx);
  }
};

struct begin_encode_sync_vocab {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev, ctx);
    emel::text::encoders::action::sync_vocab(ev, ctx);
  }
};

struct reject_invalid_encode {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::reject_invalid_encode(ev, ctx);
  }
};

struct run_encode {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::spm::detail::emit_spm(ev.request, ctx, *ctx.vocab);
    ev.ctx.token_count = result.token_count;
    ev.ctx.err = result.error;
  }
};

struct run_prepare {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::text::encoders::spm::detail::prepare_spm(ev.request, ctx, *ctx.vocab);
  }
};

struct run_merge {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::text::encoders::spm::detail::merge_spm(ctx, *ctx.vocab);
  }
};

struct sync_tables {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::spm::detail::ensure_spm_tables(ctx);
    ev.ctx.err = emel::text::encoders::spm::detail::select_i32(
      ready, EMEL_OK, EMEL_ERR_INVALID_ARGUMENT);
  }
};

struct mark_done {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::mark_done(ev, ctx);
  }
};

struct ensure_last_error {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::ensure_last_error(ev, ctx);
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
inline constexpr sync_tables sync_tables{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::spm::action
