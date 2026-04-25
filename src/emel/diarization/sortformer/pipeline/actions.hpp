#pragma once

#include "emel/diarization/request/events.hpp"
#include "emel/diarization/sortformer/encoder/detail.hpp"
#include "emel/diarization/sortformer/executor/events.hpp"
#include "emel/diarization/sortformer/output/detail.hpp"
#include "emel/diarization/sortformer/pipeline/context.hpp"
#include "emel/diarization/sortformer/pipeline/detail.hpp"
#include "emel/diarization/sortformer/pipeline/events.hpp"

namespace emel::diarization::sortformer::pipeline::action {

struct effect_begin_run {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.request.error_out = detail::to_error(error::none);
    runtime_ev.request.frame_count_out = 0;
    runtime_ev.request.probability_count_out = 0;
    runtime_ev.request.segment_count_out = 0;
  }
};

struct effect_mark_model_invalid {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::model_invalid);
  }
};

struct effect_mark_sample_rate_invalid {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::sample_rate);
  }
};

struct effect_mark_channel_count_invalid {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::channel_count);
  }
};

struct effect_mark_pcm_shape_invalid {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::pcm_shape);
  }
};

struct effect_mark_probability_capacity_invalid {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::probability_capacity);
  }
};

struct effect_mark_segment_capacity_invalid {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::segment_capacity);
  }
};

struct effect_mark_tensor_contract_invalid {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::tensor_contract);
  }
};

struct effect_prepare_features {
  void operator()(const event::run_flow & runtime_ev, context & ctx) const noexcept {
    int32_t frame_count = 0;
    int32_t feature_bin_count = 0;
    emel::diarization::request::event::prepare request{
      runtime_ev.request.contract,
      runtime_ev.request.pcm,
      runtime_ev.request.sample_rate,
      runtime_ev.request.channel_count,
      ctx.features,
      frame_count,
      feature_bin_count,
    };
    request.error_out = &runtime_ev.ctx.err;
    ctx.request.process_event(request);
  }
};

struct effect_bind_encoder {
  void operator()(const event::run_flow & runtime_ev, context & ctx) const noexcept {
    emel::diarization::sortformer::encoder::detail::bind_contract(
        *runtime_ev.request.contract.model, ctx.encoder);
  }
};

struct effect_compute_encoder_frames {
  void operator()(const event::run_flow &, context & ctx) const noexcept {
    emel::diarization::sortformer::encoder::detail::compute_encoder_frames_from_features(
        ctx.features,
        ctx.encoder,
        ctx.encoder_workspace,
        ctx.encoder_frames);
  }
};

struct effect_execute_hidden {
  void operator()(const event::run_flow & runtime_ev, context & ctx) const noexcept {
    int32_t frame_count = 0;
    int32_t hidden_dim = 0;
    emel::diarization::sortformer::executor::event::execute request{
      runtime_ev.request.contract,
      ctx.encoder_frames,
      ctx.hidden,
      frame_count,
      hidden_dim,
    };
    request.error_out = &runtime_ev.ctx.err;
    ctx.executor.process_event(request);
  }
};

struct effect_bind_modules {
  void operator()(const event::run_flow & runtime_ev, context & ctx) const noexcept {
    emel::diarization::sortformer::modules::detail::bind_contract(
        *runtime_ev.request.contract.model, ctx.modules);
  }
};

struct effect_compute_probabilities {
  void operator()(const event::run_flow & runtime_ev, context & ctx) const noexcept {
    emel::diarization::sortformer::output::detail::compute_speaker_probabilities(
        ctx.hidden,
        ctx.modules,
        runtime_ev.request.probabilities.first(
            static_cast<size_t>(detail::k_required_probability_value_count)));
    runtime_ev.request.probability_count_out = detail::k_required_probability_value_count;
  }
};

struct effect_decode_segments {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    emel::diarization::sortformer::output::detail::decode_segments(
        runtime_ev.request.probabilities.first(
            static_cast<size_t>(detail::k_required_probability_value_count)),
        emel::diarization::sortformer::output::detail::k_default_activity_threshold,
        runtime_ev.request.segments,
        runtime_ev.request.segment_count_out);
    runtime_ev.request.frame_count_out = detail::k_frame_count;
  }
};

struct effect_publish_success {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_publish_error {
  void operator()(const event::run_flow & runtime_ev, context &) const noexcept {
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

inline constexpr effect_begin_run effect_begin_run{};
inline constexpr effect_mark_model_invalid effect_mark_model_invalid{};
inline constexpr effect_mark_sample_rate_invalid effect_mark_sample_rate_invalid{};
inline constexpr effect_mark_channel_count_invalid effect_mark_channel_count_invalid{};
inline constexpr effect_mark_pcm_shape_invalid effect_mark_pcm_shape_invalid{};
inline constexpr effect_mark_probability_capacity_invalid effect_mark_probability_capacity_invalid{};
inline constexpr effect_mark_segment_capacity_invalid effect_mark_segment_capacity_invalid{};
inline constexpr effect_mark_tensor_contract_invalid effect_mark_tensor_contract_invalid{};
inline constexpr effect_prepare_features effect_prepare_features{};
inline constexpr effect_bind_encoder effect_bind_encoder{};
inline constexpr effect_compute_encoder_frames effect_compute_encoder_frames{};
inline constexpr effect_execute_hidden effect_execute_hidden{};
inline constexpr effect_bind_modules effect_bind_modules{};
inline constexpr effect_compute_probabilities effect_compute_probabilities{};
inline constexpr effect_decode_segments effect_decode_segments{};
inline constexpr effect_publish_success effect_publish_success{};
inline constexpr effect_publish_error effect_publish_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

}  // namespace emel::diarization::sortformer::pipeline::action
