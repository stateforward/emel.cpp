#pragma once

#include "emel/text/jinja/formatter/errors.hpp"
#include "emel/text/jinja/formatter/events.hpp"

namespace emel::text::jinja::formatter::guard {

namespace detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

inline bool valid_render_request(const emel::text::jinja::event::render & ev) noexcept {
  return ev.output_capacity > 0 &&
         (ev.source_text.empty() || ev.source_text.data() != nullptr);
}

inline bool callbacks_present(const emel::text::jinja::event::render & ev) noexcept {
  return static_cast<bool>(ev.dispatch_done) &&
         static_cast<bool>(ev.dispatch_error);
}

}  // namespace detail

inline bool valid_render_request(const emel::text::jinja::event::render & ev) noexcept {
  return detail::valid_render_request(ev);
}

struct valid_render {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return valid_render_request(runtime_ev.request) &&
           detail::callbacks_present(runtime_ev.request);
  }
};

struct invalid_render_with_callbacks {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return !valid_render_request(runtime_ev.request) &&
           detail::callbacks_present(runtime_ev.request);
  }
};

struct invalid_render_without_callbacks {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return !detail::callbacks_present(runtime_ev.request);
  }
};

struct source_empty {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.request.source_text.empty();
  }
};

struct source_non_empty {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return !source_empty{}(ev);
  }
};

struct source_fits {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.request.source_text.size() <= runtime_ev.request.output_capacity;
  }
};

struct source_overflow {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return !source_fits{}(ev);
  }
};

struct copy_ready {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return source_non_empty{}(ev) && source_fits{}(ev);
  }
};

struct request_ok {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = detail::unwrap_runtime_event(ev);
    return runtime_ev.ctx.err == error::none;
  }
};

struct request_failed {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return !request_ok{}(ev);
  }
};

}  // namespace emel::text::jinja::formatter::guard
