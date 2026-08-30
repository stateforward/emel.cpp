#pragma once

#include "emel/model/needle/context.hpp"
#include "emel/model/needle/detail.hpp"
#include "emel/model/needle/events.hpp"

namespace emel::model::needle::action {

struct effect_begin_bind {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};

struct effect_mark_bind_invalid_request {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_exec_bind {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.ctx.err = needle::detail::bind_contract(
        ev.request.geo, ev.request.tensors, ev.request.contract_out);
  }
};

struct effect_publish_bind_done {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.request.on_done(events::bind_done{
        .request = ev.request,
        .contract_out = ev.request.contract_out,
    });
  }
};

struct effect_publish_bind_error {
  void operator()(const event::bind_runtime &ev, context &) const noexcept {
    ev.request.on_error(events::bind_error{
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

inline constexpr effect_begin_bind effect_begin_bind{};
inline constexpr effect_mark_bind_invalid_request
    effect_mark_bind_invalid_request{};
inline constexpr effect_exec_bind effect_exec_bind{};
inline constexpr effect_publish_bind_done effect_publish_bind_done{};
inline constexpr effect_publish_bind_error effect_publish_bind_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::model::needle::action
