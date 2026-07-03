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
// selected by the transition rows via guard_conv_f16 / guard_conv_f32; the
// compute is non-failing by contract (validation guards ran first).
template <bool conv_f16> struct effect_run_frontend {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    auto &io = runtime_ev.ctx.io;
    std::copy(request.pcm.begin(), request.pcm.end(), request.frame.begin());
    io = mimi::detail::frame_buffer{request.frame.data(), 1,
                                    request.runtime.frame_samples};
    mimi::detail::compute_seanet_encoder<conv_f16>(
        request.runtime, request.streaming, io, request.workspace);
  }
};

// The projection operand class (raw f32 vs pre-quantized q8_0) is selected
// by the transition rows via guard_proj_q8 / guard_proj_f32.
template <bool proj_q8> struct effect_run_transformer {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    mimi::detail::compute_transformer<proj_q8>(
        request.runtime, request.runtime.encoder_transformer, request.streaming,
        request.streaming.encoder_positions, runtime_ev.ctx.io,
        request.workspace);
  }
};

// 25 Hz -> 12.5 Hz, then publish the latent column (one column of dim
// floats, guaranteed by the bind-validated topology).
template <bool conv_f16> struct effect_run_downsample {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    mimi::detail::compute_encoder_downsample<conv_f16>(
        request.runtime, request.streaming, runtime_ev.ctx.io,
        request.workspace);
    std::copy_n(runtime_ev.ctx.io.data,
                static_cast<size_t>(request.runtime.dim),
                request.latent_out.begin());
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
