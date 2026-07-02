#pragma once

#include <algorithm>

#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/encoder/context.hpp"
#include "emel/speech/codec/mimi/encoder/errors.hpp"
#include "emel/speech/codec/mimi/encoder/events.hpp"

namespace emel::speech::codec::mimi::encoder::action {

namespace detail {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

} // namespace detail

struct effect_begin_encode {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.ctx.stage_ok = true;
  }
};

struct effect_mark_runtime_unbound {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::runtime_unbound);
  }
};

struct effect_mark_request_shape_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::request_shape);
  }
};

struct effect_mark_buffer_capacity_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::buffer_capacity);
  }
};

// Stage the PCM frame into the io buffer and run the SEANet frontend
// (24 kHz -> 25 Hz). The operand class (f32 canonical vs reference f16) is
// selected by the transition rows via guard_conv_f16 / guard_conv_f32.
template <bool conv_f16> struct effect_run_frontend {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    auto &io = runtime_ev.ctx.io;
    std::copy(request.pcm.begin(), request.pcm.end(), request.frame.begin());
    io = mimi::detail::frame_buffer{request.frame.data(), 1,
                                    request.runtime.frame_samples};
    runtime_ev.ctx.stage_ok = mimi::detail::compute_seanet_stack<conv_f16>(
        request.runtime,
        std::span<const mimi::detail::seanet_layer_weights>{
            request.runtime.encoder_layers},
        request.streaming, io, request.workspace);
  }
};

struct effect_mark_frontend_failed {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::frontend_failed);
  }
};

struct effect_run_transformer {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    runtime_ev.ctx.stage_ok = mimi::detail::compute_transformer(
        request.runtime, request.runtime.encoder_transformer, request.streaming,
        request.streaming.encoder_positions, runtime_ev.ctx.io,
        request.workspace);
  }
};

struct effect_mark_transformer_failed {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::transformer_failed);
  }
};

// 25 Hz -> 12.5 Hz, then publish the latent column.
template <bool conv_f16> struct effect_run_downsample {
  void operator()(const event::encode_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    runtime_ev.ctx.stage_ok =
        mimi::detail::compute_streaming_conv<conv_f16>(
            request.runtime, request.runtime.downsample, request.streaming,
            runtime_ev.ctx.io, request.workspace) &&
        runtime_ev.ctx.io.length == 1 &&
        runtime_ev.ctx.io.channels == request.runtime.dim;
    const size_t dim = static_cast<size_t>(request.runtime.dim);
    const size_t copied = runtime_ev.ctx.stage_ok ? dim : 0u;
    std::copy_n(runtime_ev.ctx.io.data, copied, request.latent_out.begin());
    ctx.frames_encoded += runtime_ev.ctx.stage_ok ? 1u : 0u;
  }
};

struct effect_mark_downsample_failed {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::downsample_failed);
  }
};

struct effect_store_error_out {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_done {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::encode_done{
        .request = &runtime_ev.request,
    });
  }
};

struct effect_emit_error {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::encode_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

} // namespace emel::speech::codec::mimi::encoder::action
