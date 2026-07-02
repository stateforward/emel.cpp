#pragma once

#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/quantizer/context.hpp"
#include "emel/speech/codec/mimi/quantizer/errors.hpp"
#include "emel/speech/codec/mimi/quantizer/events.hpp"

namespace emel::speech::codec::mimi::quantizer::action {

namespace detail {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

} // namespace detail

template <class runtime_event_type> struct effect_begin {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.ctx.stage_ok = true;
  }
};

template <class runtime_event_type> struct effect_mark_runtime_unbound {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::runtime_unbound);
  }
};

template <class runtime_event_type> struct effect_mark_request_shape_invalid {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::request_shape);
  }
};

// The projection operand class (f32 canonical vs reference f16 conv1x1) is
// selected by the transition rows via guard_conv_f16 / guard_conv_f32.
template <bool conv_f16, bool proj_q8> struct effect_run_quantize {
  void operator()(const event::encode_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    const mimi::detail::frame_buffer latent{request.latent.data(),
                                            request.runtime.dim, 1};
    runtime_ev.ctx.stage_ok =
        mimi::detail::compute_rvq_encode<conv_f16, proj_q8>(
            request.runtime, latent, request.codes_out, request.workspace);
    ctx.frames_quantized += runtime_ev.ctx.stage_ok ? 1u : 0u;
  }
};

struct effect_mark_quantize_failed {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::quantize_failed);
  }
};

template <bool conv_f16, bool proj_q8> struct effect_run_dequantize {
  void operator()(const event::decode_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    mimi::detail::frame_buffer latent{request.latent_out.data(),
                                      request.runtime.dim, 1};
    runtime_ev.ctx.stage_ok =
        mimi::detail::compute_rvq_decode<conv_f16, proj_q8>(
            request.runtime, request.codes, 1, latent, request.workspace);
    ctx.frames_dequantized += runtime_ev.ctx.stage_ok ? 1u : 0u;
  }
};

struct effect_mark_dequantize_failed {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::dequantize_failed);
  }
};

template <class runtime_event_type> struct effect_store_error_out {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_encode_done {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::encode_done{
        .request = &runtime_ev.request,
    });
  }
};

struct effect_emit_encode_error {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::encode_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_emit_decode_done {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::decode_done{
        .request = &runtime_ev.request,
    });
  }
};

struct effect_emit_decode_error {
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

} // namespace emel::speech::codec::mimi::quantizer::action
