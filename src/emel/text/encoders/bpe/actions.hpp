#pragma once

#include <array>
#include <cstddef>

#include "emel/text/encoders/actions.hpp"
#include "emel/text/encoders/bpe/context.hpp"
#include "emel/text/encoders/bpe/detail.hpp"

namespace emel::text::encoders::bpe::action {

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

struct prepare_tables {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    const bool ready = emel::text::encoders::bpe::detail::ensure_bpe_tables(ctx);
    const std::array<int32_t, 2> errors{emel::text::encoders::error::to_emel(emel::text::encoders::error::code::backend), emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok)};
    ev.ctx.err = errors[static_cast<size_t>(ready)];
  }
};

struct run_encode_ignore_merges {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::bpe::detail::encode_bpe_ignore_merges(
      ev.request, ctx);
    ev.ctx.token_count = result.token_count;
    ev.ctx.err = result.error;
  }
};

struct run_encode_merge_path {
  void operator()(const event::encode_runtime & ev, context & ctx) const noexcept {
    const auto result = emel::text::encoders::bpe::detail::encode_bpe_merge_path(
      ev.request, ctx, *ctx.vocab);
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
inline constexpr prepare_tables prepare_tables{};
inline constexpr run_encode_ignore_merges run_encode_ignore_merges{};
inline constexpr run_encode_merge_path run_encode_merge_path{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::encoders::bpe::action
