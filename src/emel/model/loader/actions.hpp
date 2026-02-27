#pragma once

#include "emel/model/loader/context.hpp"
#include "emel/model/loader/events.hpp"

namespace emel::model::loader::action {

namespace detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

}  // namespace detail

struct begin_load {
  void operator()(const event::load_runtime & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.bytes_total = 0;
    ev.ctx.bytes_done = 0;
    ev.ctx.used_mmap = false;
  }
};

struct mark_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct mark_internal_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::internal_error);
  }
};

struct run_parse {
  void operator()(const event::load_runtime & ev, context &) const noexcept {
    ev.ctx.err = ev.request.parse_model(ev.request);
  }
};

struct run_load_weights {
  void operator()(const event::load_runtime & ev, context &) const noexcept {
    ev.ctx.bytes_total = 0;
    ev.ctx.bytes_done = 0;
    ev.ctx.used_mmap = false;
    ev.ctx.err =
      ev.request.load_weights(ev.request, ev.ctx.bytes_total, ev.ctx.bytes_done, ev.ctx.used_mmap);
  }
};

struct run_map_layers {
  void operator()(const event::load_runtime & ev, context &) const noexcept {
    ev.ctx.err = ev.request.map_layers(ev.request);
  }
};

struct run_validate_structure {
  void operator()(const event::load_runtime & ev, context &) const noexcept {
    ev.ctx.err = ev.request.validate_structure(ev.request);
  }
};

struct run_validate_architecture {
  void operator()(const event::load_runtime & ev, context &) const noexcept {
    ev.ctx.err = ev.request.validate_architecture_impl(ev.request);
  }
};

struct publish_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
    runtime_ev.request.on_done(events::load_done{
      .request = runtime_ev.request,
      .bytes_total = runtime_ev.ctx.bytes_total,
      .bytes_done = runtime_ev.ctx.bytes_done,
      .used_mmap = runtime_ev.ctx.used_mmap,
    });
  }
};

struct publish_done_noop {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.ctx.err = emel::error::cast(error::none);
  }
};

struct publish_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    runtime_ev.request.on_error(events::load_error{
      .request = runtime_ev.request,
      .err = runtime_ev.ctx.err,
    });
  }
};

struct publish_error_noop {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev, context &) const noexcept {
    static_cast<void>(detail::unwrap_runtime_event(ev));
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr begin_load begin_load{};
inline constexpr mark_invalid_request mark_invalid_request{};
inline constexpr mark_internal_error mark_internal_error{};
inline constexpr run_parse run_parse{};
inline constexpr run_load_weights run_load_weights{};
inline constexpr run_map_layers run_map_layers{};
inline constexpr run_validate_structure run_validate_structure{};
inline constexpr run_validate_architecture run_validate_architecture{};
inline constexpr publish_done publish_done{};
inline constexpr publish_done_noop publish_done_noop{};
inline constexpr publish_error publish_error{};
inline constexpr publish_error_noop publish_error_noop{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::model::loader::action
