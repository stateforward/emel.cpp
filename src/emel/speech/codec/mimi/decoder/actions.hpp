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

struct effect_begin_decode {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.ctx.stage_ok = true;
  }
};

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
// algorithm, selected here by the transition, not by shape sniffing).
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
    runtime_ev.ctx.stage_ok =
        mimi::detail::compute_streaming_conv_transpose_depthwise(
            request.runtime, request.runtime.upsample, request.streaming, io,
            request.workspace);
  }
};

struct effect_mark_upsample_failed {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::upsample_failed);
  }
};

struct effect_run_transformer {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    const auto &request = runtime_ev.request;
    runtime_ev.ctx.stage_ok = mimi::detail::compute_transformer(
        request.runtime, request.runtime.decoder_transformer, request.streaming,
        request.streaming.decoder_positions, runtime_ev.ctx.io,
        request.workspace);
  }
};

struct effect_mark_transformer_failed {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::transformer_failed);
  }
};

// SEANet decoder back to 24 kHz mono, then publish the PCM frame. The
// operand class is selected by the transition rows via guard_conv_f16 /
// guard_conv_f32.
template <bool conv_f16> struct effect_run_backend {
  void operator()(const event::decode_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    auto &io = runtime_ev.ctx.io;
    runtime_ev.ctx.stage_ok =
        mimi::detail::compute_seanet_stack<conv_f16>(
            request.runtime,
            std::span<const mimi::detail::seanet_layer_weights>{
                request.runtime.decoder_layers},
            request.streaming, io, request.workspace) &&
        io.channels == 1 && io.length == request.runtime.frame_samples;
    const size_t copied =
        runtime_ev.ctx.stage_ok
            ? static_cast<size_t>(request.runtime.frame_samples)
            : 0u;
    std::copy_n(io.data, copied, request.pcm_out.begin());
    ctx.frames_decoded += runtime_ev.ctx.stage_ok ? 1u : 0u;
  }
};

struct effect_mark_backend_failed {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::backend_failed);
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
