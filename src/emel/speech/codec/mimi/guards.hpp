#pragma once

#include "emel/speech/codec/mimi/context.hpp"
#include "emel/speech/codec/mimi/events.hpp"

namespace emel::speech::codec::mimi::guard {

template <class runtime_event_type> struct guard_stage_ok {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.stage_ok;
  }
};

template <class runtime_event_type> struct guard_stage_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.ctx.stage_ok;
  }
};

template <class runtime_event_type> struct guard_has_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

template <class runtime_event_type> struct guard_no_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out == nullptr;
  }
};

template <class runtime_event_type> struct guard_has_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class runtime_event_type> struct guard_no_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

template <class runtime_event_type> struct guard_no_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::codec::mimi::guard
