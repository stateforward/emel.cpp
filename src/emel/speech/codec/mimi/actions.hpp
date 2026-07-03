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

struct effect_mark_bind_failed {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::bind_failed);
  }
};

struct effect_mark_arena_capacity_invalid {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::arena_capacity);
  }
};

template <class runtime_event_type> struct effect_mark_request_shape_invalid {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::request_shape);
  }
};

struct effect_mark_code_range_invalid {
  void operator()(const event::decode_frame_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::code_range);
  }
};

// One-time bind of the caller arenas into the codec runtime. Non-failing by
// contract: the transition rows reach this only after guard_bind_contract_valid
// (pure dry-run of the same walk) and guard_arena_capacity_valid passed.
struct effect_bind {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    ctx.workspace = request.workspace;
    ctx.frame = request.frame;
    detail::bind_codec_runtime(request.model, request.prepared,
                               request.state_arena, ctx.runtime, ctx.streaming);
  }
};

// Frame encode phase 1: PCM -> latent through the frontend child actor. The
// child dispatch cannot fail: guard_encode_request_valid checked the request
// against the bound runtime, and the child's own validation guards accept
// exactly that contract, so its return value routes nothing.
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
    (void)ctx.frontend.process_event(child_ev);
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
    (void)ctx.quantizer_machine.process_event(child_ev);
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
    (void)ctx.quantizer_machine.process_event(child_ev);
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
    (void)ctx.backend.process_event(child_ev);
  }
};

struct effect_reset_stream {
  void operator()(const event::reset_stream_run &,
                  context &ctx) const noexcept {
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
  void operator()(const unexpected_event_type &ev, context &) const noexcept {
    // An external event in a state that does not model it is a caller
    // ordering error (for example initialize while session_ready or
    // reset_stream while uninitialized): record it so the dispatch reports
    // failure instead of silently succeeding.
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = detail_ns::to_error(error::unexpected_event);
      if constexpr (requires { ev.request.error_out; }) {
        if (ev.request.error_out != nullptr) {
          *ev.request.error_out = ev.ctx.err;
        }
      }
    }
  }
};

} // namespace emel::speech::codec::mimi::action
