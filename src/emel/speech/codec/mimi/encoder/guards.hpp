#pragma once

#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/encoder/context.hpp"
#include "emel/speech/codec/mimi/encoder/events.hpp"

namespace emel::speech::codec::mimi::encoder::guard {

struct guard_runtime_bound {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &runtime = runtime_ev.request.runtime;
    return runtime.model != nullptr && runtime.frame_samples > 0 &&
           runtime.dim > 0 &&
           runtime_ev.request.streaming.arena.data() != nullptr;
  }
};

struct guard_runtime_unbound {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_runtime_bound{}(runtime_ev, ctx);
  }
};

struct guard_request_shape_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &request = runtime_ev.request;
    return request.pcm.size() ==
               static_cast<size_t>(request.runtime.frame_samples) &&
           request.latent_out.size() >=
               static_cast<size_t>(request.runtime.dim);
  }
};

struct guard_request_shape_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_request_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_buffer_capacity_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &request = runtime_ev.request;
    const auto *model = request.runtime.model;
    return model != nullptr &&
           request.frame.size() >=
               mimi::detail::required_frame_floats(*model) &&
           request.workspace.size() >=
               mimi::detail::required_workspace_floats(*model);
  }
};

struct guard_buffer_capacity_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_buffer_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_conv_f16 {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.runtime.conv_f16;
  }
};

struct guard_conv_f32 {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.request.runtime.conv_f16;
  }
};

struct guard_proj_q8 {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.runtime.proj_q8;
  }
};

struct guard_proj_f32 {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.request.runtime.proj_q8;
  }
};

struct guard_stage_ok {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.stage_ok;
  }
};

struct guard_stage_failed {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.ctx.stage_ok;
  }
};

struct guard_has_error_out {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_error_out {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out == nullptr;
  }
};

struct guard_has_done_callback {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_done_callback {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback{}(runtime_ev, ctx);
  }
};

struct guard_has_error_callback {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_error_callback {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::codec::mimi::encoder::guard
