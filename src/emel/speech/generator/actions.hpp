#pragma once

#include <algorithm>
#include <cstddef>

#include "emel/error/error.hpp"
#include "emel/memory/streaming/errors.hpp"
#include "emel/memory/streaming/events.hpp"
#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/errors.hpp"
#include "emel/speech/generator/events.hpp"

namespace emel::speech::generator::action {

inline emel::error::type error_code(const error value) noexcept {
  return emel::error::cast(value);
}

template <class runtime_event_type>
void effect_store_error(const runtime_event_type &runtime_ev,
                        const error value) noexcept {
  runtime_ev.ctx.err = error_code(value);
  runtime_ev.request.error_out = runtime_ev.ctx.err;
}

template <class dependencies_type> struct effect_initialize_temporal_positions {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    int32_t child_err = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.temporal_positions.process_event(
            emel::memory::streaming::event::initialize{child_err});
    runtime_ev.ctx.child_err = static_cast<emel::error::type>(child_err);
  }
};

template <class dependencies_type>
struct effect_initialize_secondary_positions {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    int32_t child_err = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.secondary_positions.process_event(
            emel::memory::streaming::event::initialize{child_err});
    runtime_ev.ctx.child_err = static_cast<emel::error::type>(child_err);
  }
};

template <class dependencies_type> struct effect_initialize_encoder {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    auto request = ctx.collaborators.encoder_initialize;
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.encoder.process_event(request);
  }
};

template <class dependencies_type> struct effect_initialize_decoder {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    auto request = ctx.collaborators.decoder_initialize;
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.decoder.process_event(request);
  }
};

template <class dependencies_type> struct effect_initialize_runtime {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    auto request = ctx.collaborators.runtime_initialize;
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.runtime.process_event(request);
  }
};

template <class dependencies_type> struct effect_initialize_predictor {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    auto request = ctx.collaborators.predictor_initialize;
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type> struct effect_initialize_conditioning {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    auto request = ctx.collaborators.conditioning_initialize;
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type> struct effect_publish_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type, error error_value>
struct effect_fail_initialize {
  void operator()(const event::initialize_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    effect_store_error(runtime_ev, error_value);
  }
};

template <class dependencies_type> struct effect_emit_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  const context<dependencies_type> &ctx) const noexcept {
    runtime_ev.request.on_done(events::initialize_done{
        runtime_ev.request, ctx.collaborators.frame_samples,
        ctx.collaborators.codebook_count});
  }
};

template <class dependencies_type> struct effect_emit_initialize_error {
  void operator()(const event::initialize_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_error(
        events::initialize_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

template <class dependencies_type> struct effect_condition_voice {
  void operator()(const event::condition_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.complete = false;
    runtime_ev.ctx.remaining = -1;
    typename dependencies_type::voice_condition_event request{};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    request.complete_out = &runtime_ev.ctx.complete;
    request.remaining_frames_out = &runtime_ev.ctx.remaining;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type> struct effect_begin_prompt_conditioning {
  void operator()(const event::condition_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    typename dependencies_type::prompt_begin_event request{};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type> struct effect_condition_prompt {
  void operator()(const event::condition_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.complete = false;
    runtime_ev.ctx.remaining = -1;
    typename dependencies_type::prompt_condition_event request{};
    request.text_token = runtime_ev.request.token;
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    request.complete_out = &runtime_ev.ctx.complete;
    request.remaining_frames_out = &runtime_ev.ctx.remaining;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type> struct effect_publish_condition_pending {
  void operator()(const event::condition_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.complete_out = false;
    runtime_ev.request.remaining_out = runtime_ev.ctx.remaining;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type> struct effect_publish_condition_complete {
  void operator()(const event::condition_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.complete_out = true;
    runtime_ev.request.remaining_out = 0;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type, error error_value>
struct effect_fail_condition {
  void operator()(const event::condition_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.complete_out = false;
    runtime_ev.request.remaining_out = -1;
    effect_store_error(runtime_ev, error_value);
  }
};

template <class dependencies_type> struct effect_emit_condition_done {
  void operator()(const event::condition_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_done(events::condition_done{
        runtime_ev.request, runtime_ev.request.complete_out,
        runtime_ev.request.remaining_out});
  }
};

template <class dependencies_type> struct effect_emit_condition_error {
  void operator()(const event::condition_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_error(
        events::condition_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

template <class dependencies_type> struct effect_prepare_generate {
  void operator()(const event::generate_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type, error error_value>
struct effect_fail_generate {
  void operator()(const event::generate_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    effect_store_error(runtime_ev, error_value);
  }
};

template <class dependencies_type> struct effect_emit_generation_error {
  void operator()(const event::generate_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_error(
        events::generation_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

template <class dependencies_type> struct effect_encode_stream_frame {
  void operator()(const event::stream_frame_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    typename dependencies_type::encode_event request{
        runtime_ev.request.pcm_in, ctx.collaborators.input_codes};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.encoder.process_event(request);
  }
};

template <class dependencies_type> struct effect_encode_flush_frame {
  void operator()(const event::flush_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    typename dependencies_type::encode_event request{
        std::span<const float>{ctx.collaborators.silence_pcm},
        ctx.collaborators.input_codes};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.encoder.process_event(request);
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_predict_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.produced = false;
    std::fill(ctx.collaborators.output_codes.begin(),
              ctx.collaborators.output_codes.end(), -1);
    typename dependencies_type::predict_event request{
        std::span<const int32_t>{ctx.collaborators.input_codes},
        ctx.collaborators.output_codes, runtime_ev.ctx.text_token};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    request.produced_out = &runtime_ev.ctx.produced;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_decode_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    typename dependencies_type::decode_event request{
        std::span<const int32_t>{ctx.collaborators.output_codes},
        runtime_ev.request.pcm_out};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.decoder.process_event(request);
  }
};

template <class runtime_event_type>
void effect_publish_frame_tokens(const runtime_event_type &runtime_ev,
                                 const auto &ctx) noexcept {
  std::copy(ctx.collaborators.input_codes.begin(),
            ctx.collaborators.input_codes.end(),
            runtime_ev.request.encoded_tokens_out.begin());
  std::copy(ctx.collaborators.output_codes.begin(),
            ctx.collaborators.output_codes.end(),
            runtime_ev.request.generated_tokens_out.begin());
  runtime_ev.request.text_token_out = runtime_ev.ctx.text_token;
}

template <class dependencies_type> struct effect_publish_stream_frame_produced {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context<dependencies_type> &ctx) const noexcept {
    effect_publish_frame_tokens(runtime_ev, ctx);
    runtime_ev.request.sample_count_out = ctx.collaborators.frame_samples;
    runtime_ev.request.produced_out = true;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type> struct effect_publish_stream_frame_pending {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context<dependencies_type> &ctx) const noexcept {
    effect_publish_frame_tokens(runtime_ev, ctx);
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type, error error_value>
struct effect_fail_stream_frame {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    effect_store_error(runtime_ev, error_value);
  }
};

template <class dependencies_type> struct effect_emit_stream_frame_done {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_done(events::stream_frame_done{
        runtime_ev.request, runtime_ev.request.sample_count_out,
        runtime_ev.request.produced_out});
  }
};

template <class dependencies_type> struct effect_emit_stream_frame_error {
  void operator()(const event::stream_frame_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_error(
        events::stream_frame_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

template <class dependencies_type> struct effect_publish_flush_produced {
  void operator()(const event::flush_run &runtime_ev,
                  const context<dependencies_type> &ctx) const noexcept {
    effect_publish_frame_tokens(runtime_ev, ctx);
    runtime_ev.request.sample_count_out = ctx.collaborators.frame_samples;
    runtime_ev.request.complete_out = false;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type> struct effect_publish_flush_pending {
  void operator()(const event::flush_run &runtime_ev,
                  const context<dependencies_type> &ctx) const noexcept {
    effect_publish_frame_tokens(runtime_ev, ctx);
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.complete_out = false;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type, error error_value> struct effect_fail_flush {
  void operator()(const event::flush_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.complete_out = false;
    effect_store_error(runtime_ev, error_value);
  }
};

template <class dependencies_type> struct effect_emit_flush_done {
  void operator()(const event::flush_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_done(events::flush_done{
        runtime_ev.request, runtime_ev.request.sample_count_out,
        runtime_ev.request.complete_out});
  }
};

template <class dependencies_type> struct effect_emit_flush_error {
  void operator()(const event::flush_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_error(
        events::flush_error{runtime_ev.request, runtime_ev.ctx.err});
  }
};

template <class dependencies_type, error error_value>
struct effect_reject_reset {
  void operator()(const event::reset &ev,
                  const context<dependencies_type> &) const noexcept {
    ev.error_out = error_code(error_value);
  }
};

template <class dependencies_type> struct effect_unexpected {
  template <class event_type>
  void operator()(const event_type &,
                  const context<dependencies_type> &) const noexcept {}
};

} // namespace emel::speech::generator::action
