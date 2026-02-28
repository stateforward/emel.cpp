#pragma once

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/plamo2/context.hpp"
#include "emel/text/encoders/plamo2/detail.hpp"

namespace emel::text::encoders::plamo2::action {

struct begin_encode {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev, ctx);
  }
};

struct begin_encode_sync_vocab {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::begin_encode(ev, ctx);
    emel::text::encoders::action::sync_vocab(ev, ctx);
    ctx.plamo2_tables_ready = false;
    ctx.plamo2_vocab = nullptr;
    ctx.byte_tokens.fill(0);
    ctx.suffix_map.clear();
    ctx.table.clear();
  }
};

struct reject_invalid_encode {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    emel::text::encoders::action::reject_invalid_encode(ev, ctx);
  }
};

struct run_encode {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::plamo2::detail::encode_plamo2(ev.request, ctx, *ctx.vocab);
    ev.ctx.token_count = result.token_count;
    ev.ctx.err = result.error;
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
inline constexpr run_encode run_encode{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::plamo2::action
