#pragma once

#include <span>

#include "emel/speech/codec/mimi/context.hpp"
#include "emel/speech/codec/mimi/decoder/events.hpp"
#include "emel/speech/codec/mimi/encoder/events.hpp"
#include "emel/speech/codec/mimi/errors.hpp"
#include "emel/speech/codec/mimi/events.hpp"
#include "emel/speech/codec/mimi/quantizer/events.hpp"

namespace emel::speech::codec::mimi::action {

namespace detail_ns {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

} // namespace detail_ns

template <class runtime_event_type> struct effect_mark_not_initialized {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::not_initialized);
  }
};

// One-time bind of the caller arenas into the codec runtime. The latent
// staging column caps the supported model width; the bind is rejected when
// the model needs more.
struct effect_bind {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    ctx.workspace = request.workspace;
    ctx.frame = request.frame;
    runtime_ev.ctx.stage_ok =
        detail::bind_codec_runtime(request.model, request.prepared,
                                   request.state_arena, ctx.runtime,
                                   ctx.streaming) &&
        ctx.runtime.dim <= k_max_latent_floats &&
        request.workspace.size() >=
            detail::required_workspace_floats(request.model) &&
        request.frame.size() >= detail::required_frame_floats(request.model);
  }
};

struct effect_mark_bind_failed {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::bind_failed);
  }
};

// Frame encode phase 1: PCM -> latent through the frontend child actor.
struct effect_run_frontend_child {
  void operator()(const event::encode_frame_run &runtime_ev,
                  context &ctx) const noexcept {
    encoder::event::encode child_ev{
        ctx.runtime,
        ctx.streaming,
        runtime_ev.request.pcm,
        ctx.frame,
        ctx.workspace,
        std::span<float>{ctx.latent.data(),
                         static_cast<size_t>(ctx.runtime.dim)}};
    runtime_ev.ctx.stage_ok = ctx.frontend.process_event(child_ev);
  }
};

struct effect_mark_encode_failed {
  void operator()(const event::encode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::encode_failed);
  }
};

// Frame encode phase 2: latent -> codes through the quantizer child actor.
struct effect_run_quantize_child {
  void operator()(const event::encode_frame_run &runtime_ev,
                  context &ctx) const noexcept {
    quantizer::event::encode child_ev{
        ctx.runtime,
        std::span<float>{ctx.latent.data(),
                         static_cast<size_t>(ctx.runtime.dim)},
        runtime_ev.request.codes_out, ctx.workspace};
    runtime_ev.ctx.stage_ok = ctx.quantizer_machine.process_event(child_ev);
  }
};

struct effect_mark_quantize_failed {
  void operator()(const event::encode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::quantize_failed);
  }
};

// Frame decode phase 1: codes -> latent through the quantizer child actor.
struct effect_run_dequantize_child {
  void operator()(const event::decode_frame_run &runtime_ev,
                  context &ctx) const noexcept {
    quantizer::event::decode child_ev{
        ctx.runtime, runtime_ev.request.codes,
        std::span<float>{ctx.latent.data(),
                         static_cast<size_t>(ctx.runtime.dim)},
        ctx.workspace};
    runtime_ev.ctx.stage_ok = ctx.quantizer_machine.process_event(child_ev);
  }
};

struct effect_mark_dequantize_failed {
  void operator()(const event::decode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::dequantize_failed);
  }
};

// Frame decode phase 2: latent -> PCM through the backend child actor.
struct effect_run_backend_child {
  void operator()(const event::decode_frame_run &runtime_ev,
                  context &ctx) const noexcept {
    decoder::event::decode child_ev{
        ctx.runtime,
        ctx.streaming,
        std::span<const float>{ctx.latent.data(),
                               static_cast<size_t>(ctx.runtime.dim)},
        ctx.frame,
        ctx.workspace,
        runtime_ev.request.pcm_out};
    runtime_ev.ctx.stage_ok = ctx.backend.process_event(child_ev);
  }
};

struct effect_mark_decode_failed {
  void operator()(const event::decode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::decode_failed);
  }
};

struct effect_reset_stream {
  void operator()(const event::reset_stream &, context &ctx) const noexcept {
    detail::reset_streaming_state(ctx.runtime, ctx.streaming);
  }
};

template <class runtime_event_type> struct effect_store_error_out {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.request.on_done(events::initialize_done{
        .request = &runtime_ev.request,
        .frame_samples = ctx.runtime.frame_samples,
        .n_q = ctx.runtime.n_q,
    });
  }
};

struct effect_emit_initialize_error {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::initialize_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_emit_encode_done {
  void operator()(const event::encode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::encode_frame_done{
        .request = &runtime_ev.request,
    });
  }
};

struct effect_emit_encode_error {
  void operator()(const event::encode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::encode_frame_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_emit_decode_done {
  void operator()(const event::decode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::decode_frame_done{
        .request = &runtime_ev.request,
    });
  }
};

struct effect_emit_decode_error {
  void operator()(const event::decode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::decode_frame_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

} // namespace emel::speech::codec::mimi::action
