#pragma once

#include "emel/emel.h"
#include "emel/memory/hybrid/detail.hpp"
#include "emel/memory/hybrid/events.hpp"

namespace emel::memory::hybrid::guard {

struct kv_accepted {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return hybrid::detail::unwrap_runtime_event(ev).ctx.kv_accepted;
  }
};

struct kv_rejected_with_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.kv_accepted &&
           runtime_ev.ctx.kv_error != static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct kv_rejected_without_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.kv_accepted &&
           runtime_ev.ctx.kv_error == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct kv_rejected_backend_family {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.kv_accepted &&
           hybrid::detail::is_backend_family_error(runtime_ev.ctx.kv_error);
  }
};

struct kv_rejected_out_of_memory {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.kv_accepted &&
           hybrid::detail::is_out_of_memory_error(runtime_ev.ctx.kv_error);
  }
};

struct kv_rejected_backend_or_none {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.kv_accepted &&
           hybrid::detail::is_backend_or_none_error(runtime_ev.ctx.kv_error);
  }
};

struct kv_rejected_non_backend_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return kv_rejected_with_error{}(ev) && !kv_rejected_backend_family{}(ev);
  }
};

struct recurrent_accepted {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return hybrid::detail::unwrap_runtime_event(ev).ctx.recurrent_accepted;
  }
};

struct recurrent_rejected_with_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.recurrent_accepted &&
           runtime_ev.ctx.recurrent_error != static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct recurrent_rejected_without_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.recurrent_accepted &&
           runtime_ev.ctx.recurrent_error == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct recurrent_rejected_any {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return !hybrid::detail::unwrap_runtime_event(ev).ctx.recurrent_accepted;
  }
};

struct recurrent_rejected_backend_family {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.recurrent_accepted &&
           hybrid::detail::is_backend_family_error(runtime_ev.ctx.recurrent_error);
  }
};

struct recurrent_rejected_out_of_memory {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.recurrent_accepted &&
           hybrid::detail::is_out_of_memory_error(runtime_ev.ctx.recurrent_error);
  }
};

struct recurrent_rejected_backend_or_none {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.recurrent_accepted &&
           hybrid::detail::is_backend_or_none_error(runtime_ev.ctx.recurrent_error);
  }
};

struct recurrent_rejected_non_backend_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return recurrent_rejected_with_error{}(ev) && !recurrent_rejected_backend_family{}(ev);
  }
};

struct rollback_accepted {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    return hybrid::detail::unwrap_runtime_event(ev).ctx.rollback_accepted;
  }
};

struct rollback_rejected_with_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.rollback_accepted &&
           runtime_ev.ctx.rollback_error != static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct rollback_rejected_without_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & ev) const noexcept {
    const auto & runtime_ev = hybrid::detail::unwrap_runtime_event(ev);
    return !runtime_ev.ctx.rollback_accepted &&
           runtime_ev.ctx.rollback_error == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

struct capture_request_valid {
  bool operator()(const event::capture_view_runtime & ev) const noexcept {
    return ev.has_snapshot_out;
  }
};

struct capture_request_invalid {
  bool operator()(const event::capture_view_runtime & ev) const noexcept {
    return !capture_request_valid{}(ev);
  }
};

}  // namespace emel::memory::hybrid::guard
