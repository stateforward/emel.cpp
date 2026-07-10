#pragma once

#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/events.hpp"

namespace emel::speech::generator::guard {

struct guard_generate_request_valid {
  bool operator()(const event::generate_run &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.request.text.empty() &&
           !runtime_ev.request.pcm_out.empty();
  }
};

struct guard_generate_request_invalid {
  bool operator()(const event::generate_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_generate_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_stream_request_valid {
  bool operator()(const event::stream_frame_run &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.request.pcm_in.empty() &&
           !runtime_ev.request.pcm_out.empty();
  }
};

struct guard_stream_request_invalid {
  bool operator()(const event::stream_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_stream_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_flush_request_valid {
  bool operator()(const event::flush_run &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.request.pcm_out.empty();
  }
};

struct guard_flush_request_invalid {
  bool operator()(const event::flush_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_flush_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_initialize_done_callback_present {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_initialize_done_callback_absent {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_initialize_done_callback_present{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_error_callback_present {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

template <class runtime_event_type> struct guard_error_callback_absent {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_error_callback_present<runtime_event_type>{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::generator::guard
