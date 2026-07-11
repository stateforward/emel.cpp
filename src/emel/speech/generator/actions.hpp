#pragma once

#include <algorithm>
#include <cstddef>

#include "emel/batch/planner/events.hpp"
#include "emel/error/error.hpp"
#include "emel/memory/streaming/errors.hpp"
#include "emel/memory/streaming/events.hpp"
#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/errors.hpp"
#include "emel/speech/generator/events.hpp"

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
        ctx.collaborators.predicted_codes, runtime_ev.ctx.predicted_text_token};
    request.error_out = &runtime_ev.ctx.child_err;
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
  template <class event_type>
  void operator()(const event_type &,
                  const context<dependencies_type> &) const noexcept {}
};

} // namespace emel::speech::generator::action
