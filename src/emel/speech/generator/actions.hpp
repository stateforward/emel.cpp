#pragma once

#include <algorithm>
#include <cstddef>

#include <stateforward/sml.hpp>

#include "emel/batch/planner/events.hpp"
#include "emel/error/error.hpp"
#include "emel/memory/streaming/errors.hpp"
#include "emel/memory/streaming/events.hpp"
#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/errors.hpp"
#include "emel/speech/generator/events.hpp"
#include "emel/speech/generator/frame/events.hpp"

namespace emel::speech::generator::action {

struct frame_plan_capture {
  event::frame_ctx &ctx;

  void on_done(const emel::batch::planner::events::plan_done &ev) noexcept {
    ctx.child_err = emel::error::cast(emel::batch::planner::error::none);
    ctx.plan_step_size = ev.step_sizes[0];
    ctx.plan_output_count = ev.total_outputs;
  }

  void on_error(const emel::batch::planner::events::plan_error &ev) noexcept {
    ctx.child_err = ev.err;
  }
};

inline emel::error::type error_code(const error value) noexcept {
  return emel::error::cast(value);
}

template <class runtime_event_type>
void effect_store_error(const runtime_event_type &runtime_ev,
                        const error value) noexcept {
  runtime_ev.ctx.err = error_code(value);
  runtime_ev.request.error_out = runtime_ev.ctx.err;
}

template <class runtime_event_type, class actor_type, class request_type>
void effect_dispatch_child(const runtime_event_type &runtime_ev,
                           actor_type &actor, request_type &request) noexcept {
  runtime_ev.ctx.child_err = 0;
  request.error_out = &runtime_ev.ctx.child_err;
  runtime_ev.ctx.child_accepted = actor.process_event(request);
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

template <class dependencies_type> struct effect_initialize_tokenizer {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.tokenizer_err = 0;
    typename dependencies_type::tokenizer_initialize_event request{
        runtime_ev.ctx.tokenizer_err};
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.tokenizer.process_event(request);
    runtime_ev.ctx.child_err =
        static_cast<emel::error::type>(runtime_ev.ctx.tokenizer_err);
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

template <class dependencies_type>
struct effect_emit_synthesis_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_done(
        events::initialize_done{runtime_ev.request, 0, 0});
  }
};

template <class dependencies_type>
struct effect_initialize_synthesis_conditioner {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    auto request = ctx.collaborators.conditioner_initialize;
    effect_dispatch_child(runtime_ev, ctx.collaborators.conditioner, request);
  }
};

template <class dependencies_type>
struct effect_initialize_synthesis_prefiller {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    auto request = ctx.collaborators.prefiller_initialize;
    effect_dispatch_child(runtime_ev, ctx.collaborators.prefiller, request);
  }
};

template <class dependencies_type>
struct effect_initialize_synthesis_predictor {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    auto request = ctx.collaborators.predictor_initialize;
    effect_dispatch_child(runtime_ev, ctx.collaborators.predictor, request);
  }
};

template <class dependencies_type> struct effect_initialize_synthesis_sampler {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    auto request = ctx.collaborators.sampler_initialize;
    effect_dispatch_child(runtime_ev, ctx.collaborators.sampler, request);
  }
};

template <class dependencies_type> struct effect_initialize_synthesis_decoder {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    auto request = ctx.collaborators.decoder_initialize;
    effect_dispatch_child(runtime_ev, ctx.collaborators.decoder, request);
  }
};

template <class dependencies_type>
struct effect_initialize_synthesis_postprocessor {
  void operator()(const event::initialize_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    auto request = ctx.collaborators.postprocessor_initialize;
    effect_dispatch_child(runtime_ev, ctx.collaborators.postprocessor, request);
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
    auto request = ctx.collaborators.prompt_begin;
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

template <class dependencies_type> struct effect_capture_tokenizer_state {
  void operator()(const event::condition_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.tokenizer_offset = 0;
    typename dependencies_type::capture_tokenizer_state_event request{
        ctx.collaborators.tokenizer_cache_snapshot,
        runtime_ev.ctx.tokenizer_offset, runtime_ev.ctx.child_err};
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type> struct effect_restore_tokenizer_state {
  void operator()(const event::condition_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.tokenizer_err = 0;
    typename dependencies_type::restore_tokenizer_state_event request{
        std::span<const int32_t>{ctx.collaborators.tokenizer_cache_snapshot},
        runtime_ev.ctx.tokenizer_offset, runtime_ev.ctx.tokenizer_err};
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.tokenizer.process_event(request);
    runtime_ev.ctx.child_err =
        static_cast<emel::error::type>(runtime_ev.ctx.tokenizer_err);
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

template <class dependencies_type> struct effect_condition_generate {
  void operator()(const event::generate_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    typename dependencies_type::condition_event request{runtime_ev.request};
    effect_dispatch_child(runtime_ev, ctx.collaborators.conditioner, request);
  }
};

template <class dependencies_type> struct effect_prefill_generate {
  void operator()(const event::generate_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    typename dependencies_type::prefill_event request{runtime_ev.request};
    effect_dispatch_child(runtime_ev, ctx.collaborators.prefiller, request);
  }
};

template <class dependencies_type> struct effect_predict_generate {
  void operator()(const event::generate_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    typename dependencies_type::predict_event request{runtime_ev.request};
    effect_dispatch_child(runtime_ev, ctx.collaborators.predictor, request);
  }
};

template <class dependencies_type> struct effect_sample_generate {
  void operator()(const event::generate_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    typename dependencies_type::sample_event request{runtime_ev.request};
    effect_dispatch_child(runtime_ev, ctx.collaborators.sampler, request);
  }
};

template <class dependencies_type> struct effect_decode_generate {
  void operator()(const event::generate_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    typename dependencies_type::decode_event request{
        runtime_ev.request, runtime_ev.ctx.sample_count};
    effect_dispatch_child(runtime_ev, ctx.collaborators.decoder, request);
  }
};

template <class dependencies_type> struct effect_postprocess_generate {
  void operator()(const event::generate_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    typename dependencies_type::postprocess_event request{
        runtime_ev.request, runtime_ev.ctx.sample_count};
    effect_dispatch_child(runtime_ev, ctx.collaborators.postprocessor, request);
  }
};

template <class dependencies_type> struct effect_publish_generation_done {
  void operator()(const event::generate_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.sample_count_out = runtime_ev.ctx.sample_count;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type> struct effect_emit_generation_done {
  void operator()(const event::generate_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.on_done(events::generation_done{
        runtime_ev.request, runtime_ev.request.sample_count_out});
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
struct effect_tokenize_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.tokenizer_err = 0;
    typename dependencies_type::tokenize_event request{
        ctx.collaborators.tokenize_input_codes, ctx.collaborators.model_codes,
        runtime_ev.ctx.tokenizer_err};
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.tokenizer.process_event(request);
    runtime_ev.ctx.child_err =
        static_cast<emel::error::type>(runtime_ev.ctx.tokenizer_err);
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_predict_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    typename dependencies_type::predict_event request{
        std::span<const int32_t>{ctx.collaborators.model_codes},
        ctx.collaborators.prediction_workspace, runtime_ev.ctx.plan_step_size,
        runtime_ev.ctx.plan_output_count};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.predictor.process_event(request);
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_plan_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err =
        emel::error::cast(emel::batch::planner::error::none);
    runtime_ev.ctx.plan_step_size = 0;
    runtime_ev.ctx.plan_output_count = 0;
    frame_plan_capture capture{runtime_ev.ctx};
    const auto on_done =
        emel::callback<void(const emel::batch::planner::events::plan_done &)>::
            template from<frame_plan_capture, &frame_plan_capture::on_done>(
                &capture);
    const auto on_error =
        emel::callback<void(const emel::batch::planner::events::plan_error &)>::
            template from<frame_plan_capture, &frame_plan_capture::on_error>(
                &capture);
    const emel::batch::planner::event::plan_request request{
        .token_ids = ctx.collaborators.model_codes.data(),
        .n_tokens = ctx.collaborators.frame_plan_token_count,
        .n_steps = ctx.collaborators.frame_plan_steps,
        .mode = ctx.collaborators.frame_plan_mode,
        .output_all = ctx.collaborators.frame_plan_output_all,
        .on_done = on_done,
        .on_error = on_error,
    };
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.planner.process_event(request);
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_execute_prediction_graph {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    typename dependencies_type::graph_event request{
        ctx.collaborators.prediction_workspace,
        std::span<const int32_t>{ctx.collaborators.model_codes}};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.graph.process_event(request);
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_sample_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.predicted_text_token = -1;
    std::fill(ctx.collaborators.predicted_codes.begin(),
              ctx.collaborators.predicted_codes.end(), -1);
    typename dependencies_type::sample_event request{
        ctx.collaborators.prediction_workspace,
        std::span<const int32_t>{ctx.collaborators.model_codes},
        ctx.collaborators.predicted_codes, runtime_ev.ctx.predicted_text_token};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.sampler.process_event(request);
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_detokenize_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx.tokenizer_err = 0;
    runtime_ev.ctx.produced = false;
    runtime_ev.ctx.text_token = -1;
    std::fill(ctx.collaborators.output_codes.begin(),
              ctx.collaborators.output_codes.end(), -1);
    typename dependencies_type::detokenize_event request{
        runtime_ev.ctx.predicted_text_token,
        std::span<const int32_t>{ctx.collaborators.predicted_codes},
        runtime_ev.ctx.text_token,
        ctx.collaborators.output_codes,
        runtime_ev.ctx.produced,
        runtime_ev.ctx.tokenizer_err};
    runtime_ev.ctx.child_accepted =
        ctx.collaborators.tokenizer.process_event(request);
    runtime_ev.ctx.child_err =
        static_cast<emel::error::type>(runtime_ev.ctx.tokenizer_err);
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
  static void
  effect_reject_origin(const event::initialize_run &runtime_ev,
                       const context<dependencies_type> &ctx) noexcept {
    effect_fail_initialize<dependencies_type, error::internal_error>{}(
        runtime_ev, ctx);
  }

  static void
  effect_reject_origin(const event::condition_run &runtime_ev,
                       const context<dependencies_type> &ctx) noexcept {
    effect_fail_condition<dependencies_type, error::internal_error>{}(
        runtime_ev, ctx);
  }

  static void
  effect_reject_origin(const event::generate_run &runtime_ev,
                       const context<dependencies_type> &ctx) noexcept {
    effect_fail_generate<dependencies_type, error::internal_error>{}(runtime_ev,
                                                                     ctx);
  }

  static void
  effect_reject_origin(const event::stream_frame_run &runtime_ev,
                       const context<dependencies_type> &ctx) noexcept {
    effect_fail_stream_frame<dependencies_type, error::internal_error>{}(
        runtime_ev, ctx);
  }

  static void
  effect_reject_origin(const event::flush_run &runtime_ev,
                       const context<dependencies_type> &ctx) noexcept {
    effect_fail_flush<dependencies_type, error::internal_error>{}(runtime_ev,
                                                                  ctx);
  }

  static void
  effect_reject_origin(const event::reset &ev,
                       const context<dependencies_type> &ctx) noexcept {
    effect_reject_reset<dependencies_type, error::internal_error>{}(ev, ctx);
  }

  template <class event_type>
  static void
  effect_reject_origin(const event_type &,
                       const context<dependencies_type> &) noexcept {}

  template <class unexpected_type>
  void operator()(const unexpected_type &unexpected,
                  const context<dependencies_type> &ctx) const noexcept {
    effect_reject_origin(stateforward::sml::back::get_origin_event(unexpected),
                         ctx);
  }
};

struct lane_zero {};
struct lane_one {};

template <class dependencies_type>
void effect_record_worker_entry(context<dependencies_type> &ctx) noexcept {
  if constexpr (requires { ctx.collaborators.stage_diagnostics; }) {
    ctx.collaborators.stage_diagnostics.worker_entries.fetch_add(
        1u, std::memory_order_relaxed);
  }
}

template <class dependencies_type>
void effect_record_worker_exit(context<dependencies_type> &ctx) noexcept {
  if constexpr (requires { ctx.collaborators.stage_diagnostics; }) {
    ctx.collaborators.stage_diagnostics.worker_exits.fetch_add(
        1u, std::memory_order_relaxed);
  }
}

template <class dependencies_type>
void effect_record_submission(context<dependencies_type> &ctx,
                              const bool submitted) noexcept {
  if constexpr (requires { ctx.collaborators.stage_diagnostics; }) {
    ctx.collaborators.stage_diagnostics.submissions.fetch_add(
        static_cast<uint64_t>(submitted), std::memory_order_relaxed);
  }
}

template <class dependencies_type>
void effect_record_joins(context<dependencies_type> &ctx,
                         const size_t joined) noexcept {
  if constexpr (requires { ctx.collaborators.stage_diagnostics; }) {
    ctx.collaborators.stage_diagnostics.joins.fetch_add(
        static_cast<uint64_t>(joined), std::memory_order_relaxed);
  }
}

template <class lane_type, class dependencies_type>
std::span<int32_t> encoded_lane(context<dependencies_type> &ctx) noexcept {
  if constexpr (std::same_as<lane_type, lane_zero>) {
    return ctx.encoded_lane0();
  } else {
    return ctx.encoded_lane1();
  }
}

template <class lane_type, class dependencies_type>
std::span<int32_t> generated_lane(context<dependencies_type> &ctx) noexcept {
  if constexpr (std::same_as<lane_type, lane_zero>) {
    return ctx.generated_lane0();
  } else {
    return ctx.generated_lane1();
  }
}

template <class lane_type, class dependencies_type>
event::wavefront_attribution &
encoded_attribution(context<dependencies_type> &ctx) noexcept {
  if constexpr (std::same_as<lane_type, lane_zero>) {
    return ctx.encoded_lane0_attribution;
  } else {
    return ctx.encoded_lane1_attribution;
  }
}

template <class lane_type, class dependencies_type>
event::wavefront_attribution &
generated_attribution(context<dependencies_type> &ctx) noexcept {
  if constexpr (std::same_as<lane_type, lane_zero>) {
    return ctx.generated_lane0_attribution;
  } else {
    return ctx.generated_lane1_attribution;
  }
}

template <class lane_type, class dependencies_type>
int32_t &generated_text_token(context<dependencies_type> &ctx) noexcept {
  if constexpr (std::same_as<lane_type, lane_zero>) {
    return ctx.generated_lane0_text_token;
  } else {
    return ctx.generated_lane1_text_token;
  }
}

template <class dependencies_type, class encode_lane_type,
          class middle_lane_type, class decode_lane_type, bool encode_active,
          bool middle_active, bool decode_active>
struct effect_execute_wavefront_phase_parallel {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx = {};
    runtime_ev.request.output_attribution = {};
    runtime_ev.request.text_token_out = -1;
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    if constexpr (requires { runtime_ev.request.complete_out; }) {
      runtime_ev.request.complete_out = false;
    }

    wavefront_stage_pool::join_group group{};
    emel::policy::fork_join_start_gate gate{};
    size_t submitted_tasks = 0u;
    bool all_submitted = true;

    if constexpr (encode_active) {
      const bool submitted = ctx.collaborators.stage_pool->try_submit(
          group, [&runtime_ev, &ctx, &gate]() noexcept {
            effect_record_worker_entry(ctx);
            gate.arrive_and_wait();
            typename dependencies_type::wavefront_encode_event request{
                runtime_ev.request.pcm_in, encoded_lane<encode_lane_type>(ctx)};
            request.error_out = &runtime_ev.ctx.encode_err;
            runtime_ev.ctx.encode_accepted =
                ctx.collaborators.wavefront_encoder.process_event(request);
            encoded_attribution<encode_lane_type>(ctx) =
                runtime_ev.request.input_attribution;
            effect_record_worker_exit(ctx);
          });
      submitted_tasks += static_cast<size_t>(submitted);
      all_submitted = all_submitted && submitted;
      effect_record_submission(ctx, submitted);
    }

    if constexpr (decode_active) {
      const bool submitted = ctx.collaborators.stage_pool->try_submit(
          group, [&runtime_ev, &ctx, &gate]() noexcept {
            effect_record_worker_entry(ctx);
            gate.arrive_and_wait();
            typename dependencies_type::wavefront_decode_event request{
                std::span<const int32_t>{generated_lane<decode_lane_type>(ctx)},
                ctx.decoded_pcm()};
            request.error_out = &runtime_ev.ctx.decode_err;
            runtime_ev.ctx.decode_accepted =
                ctx.collaborators.wavefront_decoder.process_event(request);
            runtime_ev.ctx.decoded_attribution =
                generated_attribution<decode_lane_type>(ctx);
            runtime_ev.ctx.decoded_text_token =
                generated_text_token<decode_lane_type>(ctx);
            effect_record_worker_exit(ctx);
          });
      submitted_tasks += static_cast<size_t>(submitted);
      all_submitted = all_submitted && submitted;
      effect_record_submission(ctx, submitted);
    }

    gate.open_after_arrivals(submitted_tasks);

    if constexpr (middle_active) {
      frame::event::run request{
          std::span<const int32_t>{encoded_lane<middle_lane_type>(ctx)},
          generated_lane<middle_lane_type>(ctx),
          generated_text_token<middle_lane_type>(ctx),
          runtime_ev.ctx.middle_err};
      runtime_ev.ctx.middle_accepted =
          ctx.collaborators.wavefront_middle.process_event(request);
      generated_attribution<middle_lane_type>(ctx) =
          encoded_attribution<middle_lane_type>(ctx);
    }

    (void)group.wait();
    effect_record_joins(ctx, submitted_tasks);
    runtime_ev.ctx.all_submitted = all_submitted;
    runtime_ev.ctx.joined = true;
  }
};

template <class dependencies_type, class encode_lane_type,
          class middle_lane_type, class decode_lane_type, bool encode_active,
          bool middle_active, bool decode_active>
struct effect_execute_wavefront_phase_serial {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx = {};
    runtime_ev.request.output_attribution = {};
    runtime_ev.request.text_token_out = -1;
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    if constexpr (requires { runtime_ev.request.complete_out; }) {
      runtime_ev.request.complete_out = false;
    }

    if constexpr (encode_active) {
      typename dependencies_type::wavefront_encode_event request{
          runtime_ev.request.pcm_in, encoded_lane<encode_lane_type>(ctx)};
      request.error_out = &runtime_ev.ctx.encode_err;
      runtime_ev.ctx.encode_accepted =
          ctx.collaborators.wavefront_encoder.process_event(request);
      encoded_attribution<encode_lane_type>(ctx) =
          runtime_ev.request.input_attribution;
    }

    if constexpr (middle_active) {
      frame::event::run request{
          std::span<const int32_t>{encoded_lane<middle_lane_type>(ctx)},
          generated_lane<middle_lane_type>(ctx),
          generated_text_token<middle_lane_type>(ctx),
          runtime_ev.ctx.middle_err};
      runtime_ev.ctx.middle_accepted =
          ctx.collaborators.wavefront_middle.process_event(request);
      generated_attribution<middle_lane_type>(ctx) =
          encoded_attribution<middle_lane_type>(ctx);
    }

    if constexpr (decode_active) {
      typename dependencies_type::wavefront_decode_event request{
          std::span<const int32_t>{generated_lane<decode_lane_type>(ctx)},
          ctx.decoded_pcm()};
      request.error_out = &runtime_ev.ctx.decode_err;
      runtime_ev.ctx.decode_accepted =
          ctx.collaborators.wavefront_decoder.process_event(request);
      runtime_ev.ctx.decoded_attribution =
          generated_attribution<decode_lane_type>(ctx);
      runtime_ev.ctx.decoded_text_token =
          generated_text_token<decode_lane_type>(ctx);
    }

    runtime_ev.ctx.all_submitted = true;
    runtime_ev.ctx.joined = true;
  }
};

template <class dependencies_type> struct effect_prepare_wavefront_empty_flush {
  void operator()(const detail::wavefront_flush_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.ctx = {};
    runtime_ev.request.output_attribution = {};
    runtime_ev.request.text_token_out = -1;
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    runtime_ev.request.complete_out = false;
  }
};

template <class dependencies_type>
void effect_commit_wavefront_input(
    const detail::wavefront_frame_run &runtime_ev,
    context<dependencies_type> &ctx) noexcept {
  ctx.expected_input.sequence =
      runtime_ev.request.input_attribution.sequence + 1u;
  ctx.expected_input.source = runtime_ev.request.input_attribution.source;
}

template <class dependencies_type, bool emit_callback>
struct effect_publish_wavefront_frame_pending {
  void operator()(const detail::wavefront_frame_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    effect_commit_wavefront_input(runtime_ev, ctx);
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
    if constexpr (emit_callback) {
      runtime_ev.request.on_done(
          events::wavefront_frame_done{runtime_ev.request, {}, 0, false});
    }
  }
};

template <class dependencies_type, class output_lane_type, bool emit_callback>
struct effect_publish_wavefront_frame_produced {
  void operator()(const detail::wavefront_frame_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    std::copy(ctx.decoded_pcm().begin(), ctx.decoded_pcm().end(),
              runtime_ev.request.pcm_out.begin());
    std::copy(generated_lane<output_lane_type>(ctx).begin(),
              generated_lane<output_lane_type>(ctx).end(),
              runtime_ev.request.generated_tokens_out.begin());
    runtime_ev.request.output_attribution = runtime_ev.ctx.decoded_attribution;
    runtime_ev.request.text_token_out = runtime_ev.ctx.decoded_text_token;
    runtime_ev.request.sample_count_out = ctx.collaborators.frame_samples;
    runtime_ev.request.produced_out = true;
    effect_commit_wavefront_input(runtime_ev, ctx);
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
    if constexpr (emit_callback) {
      runtime_ev.request.on_done(events::wavefront_frame_done{
          runtime_ev.request, runtime_ev.ctx.decoded_attribution,
          ctx.collaborators.frame_samples, true});
    }
  }
};

template <class dependencies_type, class output_lane_type, bool produced,
          bool complete, bool emit_callback>
struct effect_publish_wavefront_flush {
  void operator()(const detail::wavefront_flush_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    if constexpr (produced) {
      std::copy(ctx.decoded_pcm().begin(), ctx.decoded_pcm().end(),
                runtime_ev.request.pcm_out.begin());
      std::copy(generated_lane<output_lane_type>(ctx).begin(),
                generated_lane<output_lane_type>(ctx).end(),
                runtime_ev.request.generated_tokens_out.begin());
      runtime_ev.request.output_attribution =
          runtime_ev.ctx.decoded_attribution;
      runtime_ev.request.text_token_out = runtime_ev.ctx.decoded_text_token;
      runtime_ev.request.sample_count_out = ctx.collaborators.frame_samples;
      runtime_ev.request.produced_out = true;
    }
    runtime_ev.request.complete_out = complete;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
    if constexpr (emit_callback) {
      runtime_ev.request.on_done(events::wavefront_flush_done{
          runtime_ev.request, runtime_ev.request.output_attribution,
          runtime_ev.request.sample_count_out, runtime_ev.request.produced_out,
          complete});
    }
  }
};

template <class dependencies_type, class runtime_event_type, error error_value,
          bool emit_callback, error qualifier = error::none,
          error second_qualifier = error::none>
struct effect_fail_wavefront {
  void operator()(const runtime_event_type &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.request.output_attribution = {};
    runtime_ev.request.text_token_out = -1;
    runtime_ev.request.sample_count_out = 0;
    runtime_ev.request.produced_out = false;
    if constexpr (requires { runtime_ev.request.complete_out; }) {
      runtime_ev.request.complete_out = false;
    }
    runtime_ev.ctx.err = error_code(error_value) | error_code(qualifier) |
                         error_code(second_qualifier);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
    if constexpr (emit_callback && std::same_as<runtime_event_type,
                                                detail::wavefront_frame_run>) {
      runtime_ev.request.on_error(events::wavefront_frame_error{
          runtime_ev.request, runtime_ev.ctx.err});
    }
    if constexpr (emit_callback && std::same_as<runtime_event_type,
                                                detail::wavefront_flush_run>) {
      runtime_ev.request.on_error(events::wavefront_flush_error{
          runtime_ev.request, runtime_ev.ctx.err});
    }
  }
};

template <class dependencies_type, class runtime_event_type>
struct effect_emit_wavefront_error {
  void operator()(const runtime_event_type &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    if constexpr (std::same_as<runtime_event_type,
                               detail::wavefront_frame_run>) {
      runtime_ev.request.on_error(events::wavefront_frame_error{
          runtime_ev.request, runtime_ev.ctx.err});
    } else {
      runtime_ev.request.on_error(events::wavefront_flush_error{
          runtime_ev.request, runtime_ev.ctx.err});
    }
  }
};

template <class dependencies_type>
struct effect_reset_wavefront_children_parallel {
  void operator()(const detail::event_wavefront_reset_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx = {};
    wavefront_stage_pool::join_group group{};
    size_t submitted_tasks = 0u;
    bool all_submitted = true;

    const bool encode_submitted = ctx.collaborators.stage_pool->try_submit(
        group, [&runtime_ev, &ctx]() noexcept {
          effect_record_worker_entry(ctx);
          typename dependencies_type::wavefront_encode_reset_event request{
              runtime_ev.ctx.encode_err};
          runtime_ev.ctx.encode_accepted =
              ctx.collaborators.wavefront_encoder.process_event(request);
          effect_record_worker_exit(ctx);
        });
    submitted_tasks += static_cast<size_t>(encode_submitted);
    all_submitted = all_submitted && encode_submitted;
    effect_record_submission(ctx, encode_submitted);

    const bool decode_submitted = ctx.collaborators.stage_pool->try_submit(
        group, [&runtime_ev, &ctx]() noexcept {
          effect_record_worker_entry(ctx);
          typename dependencies_type::wavefront_decode_reset_event request{
              runtime_ev.ctx.decode_err};
          runtime_ev.ctx.decode_accepted =
              ctx.collaborators.wavefront_decoder.process_event(request);
          effect_record_worker_exit(ctx);
        });
    submitted_tasks += static_cast<size_t>(decode_submitted);
    all_submitted = all_submitted && decode_submitted;
    effect_record_submission(ctx, decode_submitted);

    typename dependencies_type::wavefront_middle_reset_event middle_request{
        runtime_ev.ctx.middle_err};
    runtime_ev.ctx.middle_accepted =
        ctx.collaborators.wavefront_middle.process_event(middle_request);

    (void)group.wait();
    effect_record_joins(ctx, submitted_tasks);
    runtime_ev.ctx.all_submitted = all_submitted;
    runtime_ev.ctx.joined = true;
  }
};

template <class dependencies_type>
struct effect_reset_wavefront_children_serial {
  void operator()(const detail::event_wavefront_reset_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    runtime_ev.ctx = {};

    typename dependencies_type::wavefront_encode_reset_event encode_request{
        runtime_ev.ctx.encode_err};
    runtime_ev.ctx.encode_accepted =
        ctx.collaborators.wavefront_encoder.process_event(encode_request);

    typename dependencies_type::wavefront_middle_reset_event middle_request{
        runtime_ev.ctx.middle_err};
    runtime_ev.ctx.middle_accepted =
        ctx.collaborators.wavefront_middle.process_event(middle_request);

    typename dependencies_type::wavefront_decode_reset_event decode_request{
        runtime_ev.ctx.decode_err};
    runtime_ev.ctx.decode_accepted =
        ctx.collaborators.wavefront_decoder.process_event(decode_request);

    runtime_ev.ctx.all_submitted = true;
    runtime_ev.ctx.joined = true;
  }
};

template <class dependencies_type> struct effect_reset_wavefront_parent {
  void operator()(const detail::event_wavefront_reset_run &runtime_ev,
                  context<dependencies_type> &ctx) const noexcept {
    std::fill(ctx.encoded_lane0_storage.begin(),
              ctx.encoded_lane0_storage.end(), -1);
    std::fill(ctx.encoded_lane1_storage.begin(),
              ctx.encoded_lane1_storage.end(), -1);
    std::fill(ctx.generated_lane0_storage.begin(),
              ctx.generated_lane0_storage.end(), -1);
    std::fill(ctx.generated_lane1_storage.begin(),
              ctx.generated_lane1_storage.end(), -1);
    std::fill(ctx.decoded_pcm_storage.begin(), ctx.decoded_pcm_storage.end(),
              0.0f);
    ctx.encoded_lane0_attribution = {};
    ctx.encoded_lane1_attribution = {};
    ctx.generated_lane0_attribution = {};
    ctx.generated_lane1_attribution = {};
    ctx.expected_input = {.sequence = 0u, .source = 0u};
    ctx.generated_lane0_text_token = -1;
    ctx.generated_lane1_text_token = -1;
    runtime_ev.ctx.err = error_code(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type, error error_value>
struct effect_fail_wavefront_reset {
  void operator()(const detail::event_wavefront_reset_run &runtime_ev,
                  const context<dependencies_type> &) const noexcept {
    runtime_ev.ctx.err = error_code(error_value);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <class dependencies_type> struct effect_unexpected_wavefront {
  template <class unexpected_type>
  void operator()(const unexpected_type &unexpected,
                  const context<dependencies_type> &) const noexcept {
    const auto &ev = stateforward::sml::back::get_origin_event(unexpected);
    if constexpr (requires { ev.error_out; }) {
      ev.error_out = error_code(error::internal_error);
    }
    if constexpr (requires { ev.produced_out; }) {
      ev.produced_out = false;
    }
    if constexpr (requires { ev.complete_out; }) {
      ev.complete_out = false;
    }
  }
};

} // namespace emel::speech::generator::action
