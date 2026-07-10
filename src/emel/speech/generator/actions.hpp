#pragma once

#include "emel/error/error.hpp"
#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/errors.hpp"
#include "emel/speech/generator/events.hpp"

namespace emel::speech::generator::action {

inline emel::error::type error_code(const error value) noexcept {
  return emel::error::cast(value);
}

struct effect_accept_initialize {
  void operator()(const event::initialize_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <error error_value> struct effect_fail_initialize {
  void operator()(const event::initialize_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.ctx.err = error_code(error_value);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_prepare_generate {
  void operator()(const event::generate_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <error error_value> struct effect_fail_generate {
  void operator()(const event::generate_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.ctx.err = error_code(error_value);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_prepare_stream_frame {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <error error_value> struct effect_fail_stream_frame {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    runtime_ev.ctx.err = error_code(error_value);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_prepare_flush {
  void operator()(const event::flush_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.complete_out = false;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <error error_value> struct effect_fail_flush {
  void operator()(const event::flush_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.complete_out = false;
    runtime_ev.ctx.err = error_code(error_value);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.on_done(events::initialize_done{runtime_ev.request});
  }
};

struct effect_emit_initialize_error {
  void operator()(const event::initialize_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.on_error(
        events::initialize_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

struct effect_emit_generation_error {
  void operator()(const event::generate_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.on_error(
        events::generation_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

struct effect_emit_stream_frame_error {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.on_error(
        events::stream_frame_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

struct effect_emit_flush_error {
  void operator()(const event::flush_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.request.on_error(
        events::flush_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

struct effect_reset {
  void operator()(const event::reset &ev, const context &) const noexcept {
    ev.error_out = error_code(error::none);
  }
};

template <error error_value> struct effect_reject {
  void operator()(const event::reset &ev, const context &) const noexcept {
    ev.error_out = error_code(error_value);
  }
};

struct effect_unexpected {
  template <class event_type>
  void operator()(const event_type &, const context &) const noexcept {}
};

} // namespace emel::speech::generator::action
