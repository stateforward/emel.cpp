#pragma once

#include <cstring>

#include "emel/text/jinja/formatter/detail.hpp"
#include "emel/text/jinja/formatter/events.hpp"
#include "emel/text/jinja/formatter/errors.hpp"

namespace emel::text::jinja::formatter::action {

namespace runtime_detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

}  // namespace runtime_detail

struct reject_invalid_render {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::mark_error(runtime_ev.ctx, error::invalid_request, false, 0);
  }
};

struct begin_render {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::reset_result(runtime_ev.ctx);
  }
};

struct mark_empty_output {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::mark_done(runtime_ev.ctx, 0, false);
  }
};

struct copy_source_text {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    const auto & request = runtime_ev.request;
    std::memcpy(&request.output, request.source_text.data(), request.source_text.size());
    detail::mark_done(runtime_ev.ctx, request.source_text.size(), false);
  }
};

struct mark_capacity_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::mark_error(runtime_ev.ctx, error::invalid_request, true, 0);
  }
};

struct dispatch_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::emit_done(runtime_ev.request, runtime_ev.ctx);
  }
};

struct dispatch_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = runtime_detail::unwrap_runtime_event(ev);
    detail::emit_error(runtime_ev.request, runtime_ev.ctx);
  }
};

struct on_unexpected {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & ev) const noexcept {
    (void)ev;
  }
};

inline constexpr reject_invalid_render reject_invalid_render{};
inline constexpr begin_render begin_render{};
inline constexpr mark_empty_output mark_empty_output{};
inline constexpr copy_source_text copy_source_text{};
inline constexpr mark_capacity_error mark_capacity_error{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::text::jinja::formatter::action
