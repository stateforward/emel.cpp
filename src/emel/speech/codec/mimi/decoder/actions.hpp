#pragma once

#include <algorithm>

#include "emel/speech/codec/mimi/decoder/context.hpp"
#include "emel/speech/codec/mimi/decoder/errors.hpp"
#include "emel/speech/codec/mimi/decoder/events.hpp"
#include "emel/speech/codec/mimi/detail.hpp"

namespace emel::speech::codec::mimi::decoder::action {

namespace detail {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

} // namespace detail

struct effect_mark_runtime_unbound {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::runtime_unbound);
  }
};

struct effect_mark_request_shape_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::request_shape);
  }
};

struct effect_mark_buffer_capacity_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::buffer_capacity);
  }
};

// Stage the latent column into the io buffer and upsample 12.5 Hz -> 25 Hz
// (depthwise transposed conv; the depthwise variant is this stage's fixed
// algorithm, selected here by the transition, not by shape sniffing). The
// compute is non-failing by contract (validation guards ran first).
struct effect_run_upsample {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    auto &io = runtime_ev.ctx.io;
    std::copy(request.latent.begin(),
              request.latent.begin() + request.runtime.dim,
              request.frame.begin());
    io = mimi::detail::frame_buffer{request.frame.data(), request.runtime.dim,
                                    1};
    mimi::detail::compute_decoder_upsample(request.runtime, request.streaming,
                                           io, request.workspace);
  }
};

// The projection operand class (raw f32 vs pre-quantized q8_0) is selected
// by the transition rows via guard_proj_q8 / guard_proj_f32.
template <bool proj_q8> struct effect_run_transformer {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    mimi::detail::compute_transformer<proj_q8>(
        request.runtime, request.runtime.decoder_transformer, request.streaming,
        request.streaming.decoder_positions, runtime_ev.ctx.io,
        request.workspace);
  }
};

// SEANet decoder back to 24 kHz mono, then publish the PCM frame (one frame
// of frame_samples floats, guaranteed by the bind-validated topology). The
// operand class is selected by the transition rows via guard_conv_f16 /
// guard_conv_f32.
template <bool conv_f16> struct effect_run_backend {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    auto &io = runtime_ev.ctx.io;
    mimi::detail::compute_seanet_decoder<conv_f16>(
        request.runtime, request.streaming, io, request.workspace);
    std::copy_n(io.data, static_cast<size_t>(request.runtime.frame_samples),
                request.pcm_out.begin());
  }
};

struct effect_store_error_out {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_done {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::decode_done{
        .request = &runtime_ev.request,
    });
  }
};

struct effect_emit_error {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::decode_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

} // namespace emel::speech::codec::mimi::decoder::action
