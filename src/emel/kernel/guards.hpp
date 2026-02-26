#pragma once

#include <type_traits>

#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/context.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel::guard {

namespace detail {

template <class dispatch_event_type>
constexpr decltype(auto) unwrap_dispatch_event(const dispatch_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

}  // namespace detail

struct valid_dispatch {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    using resolved_event_type = std::remove_cvref_t<decltype(dispatch_ev)>;
    if constexpr (std::is_same_v<resolved_event_type, event::dispatch_op>) {
      return dispatch_ev.request != nullptr &&
             dispatch_ev.dispatch_primary != nullptr &&
             dispatch_ev.dispatch_secondary != nullptr &&
             dispatch_ev.dispatch_tertiary != nullptr;
    }
    return true;
  }
};

struct phase_ok {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.err == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct phase_failed {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.err != static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct primary_done {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.err == static_cast<int32_t>(emel::error::cast(error::none)) &&
           dispatch_ev.ctx.primary_outcome == event::phase_outcome::done;
  }
};

struct primary_unsupported {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.primary_outcome == event::phase_outcome::unsupported;
  }
};

struct primary_failed {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.primary_outcome == event::phase_outcome::failed;
  }
};

struct secondary_done {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.err == static_cast<int32_t>(emel::error::cast(error::none)) &&
           dispatch_ev.ctx.secondary_outcome == event::phase_outcome::done;
  }
};

struct secondary_unsupported {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.secondary_outcome == event::phase_outcome::unsupported;
  }
};

struct secondary_failed {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.secondary_outcome == event::phase_outcome::failed;
  }
};

struct tertiary_done {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.err == static_cast<int32_t>(emel::error::cast(error::none)) &&
           dispatch_ev.ctx.tertiary_outcome == event::phase_outcome::done;
  }
};

struct tertiary_unsupported {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.tertiary_outcome == event::phase_outcome::unsupported;
  }
};

struct tertiary_failed {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.tertiary_outcome == event::phase_outcome::failed;
  }
};

}  // namespace emel::kernel::guard
