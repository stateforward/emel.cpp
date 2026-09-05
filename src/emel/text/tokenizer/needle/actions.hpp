#pragma once

#include "emel/text/tokenizer/needle/context.hpp"
#include "emel/text/tokenizer/needle/detail.hpp"
#include "emel/text/tokenizer/needle/events.hpp"

namespace emel::text::tokenizer::needle::action {

struct effect_begin_load {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct effect_mark_load_invalid_request {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_exec_load {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.ctx.err = needle::detail::parse_tokenizer_blob(ev.request.blob,
                                                      ev.request.vocab_out);
  }
};

struct effect_publish_load_done {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.request.on_done(events::load_done{
        .request = ev.request,
        .vocab_out = ev.request.vocab_out,
    });
  }
};

struct effect_publish_load_error {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    ev.request.on_error(events::load_error{
        .request = ev.request,
        .err = ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.err; }) {
      ev.event_.ctx.err = emel::error::cast(error::internal_error);
    } else if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr effect_begin_load effect_begin_load{};
inline constexpr effect_mark_load_invalid_request
    effect_mark_load_invalid_request{};
inline constexpr effect_exec_load effect_exec_load{};
inline constexpr effect_publish_load_done effect_publish_load_done{};
inline constexpr effect_publish_load_error effect_publish_load_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::text::tokenizer::needle::action
