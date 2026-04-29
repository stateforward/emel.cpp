#pragma once

#include "emel/emel.h"
#include "emel/kernel/context.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/errors.hpp"
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
    return ::emel::kernel::detail::validate_dispatch_request(dispatch_ev.request);
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
    return dispatch_ev.ctx.primary_accepted;
  }
};

struct primary_unsupported {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return !dispatch_ev.ctx.primary_accepted;
  }
};

struct secondary_done {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return dispatch_ev.ctx.secondary_accepted;
  }
};

struct secondary_unsupported {
  template <class dispatch_event_type>
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    const auto & dispatch_ev = detail::unwrap_dispatch_event(ev);
    return !dispatch_ev.ctx.secondary_accepted;
  }
};

}  // namespace emel::kernel::guard
