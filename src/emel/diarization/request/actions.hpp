#pragma once

#include <algorithm>
#include <cstddef>

#include "emel/diarization/request/context.hpp"
#include "emel/diarization/request/detail.hpp"
#include "emel/diarization/request/events.hpp"
#include "emel/diarization/sortformer/encoder/feature_extractor/detail.hpp"

namespace emel::diarization::request::action {

struct effect_begin_prepare {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.request.frame_count_out = 0;
    runtime_ev.request.feature_bin_count_out = 0;
  }
};

struct effect_mark_model_invalid {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::model_invalid);
  }
};

struct effect_mark_sample_rate_invalid {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::sample_rate);
  }
};

struct effect_mark_channel_count_invalid {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::channel_count);
  }
};

struct effect_mark_pcm_shape_invalid {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::pcm_shape);
  }
};

struct effect_mark_capacity_invalid {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::capacity);
  }
};

struct effect_extract_features {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    const auto feature_contract =
        emel::diarization::sortformer::encoder::feature_extractor::detail::make_contract(
            *runtime_ev.request.contract.model);
    auto features = runtime_ev.request.features.first(
        static_cast<size_t>(detail::k_required_feature_count));
    std::fill(features.begin(), features.end(), 0.0f);
    emel::diarization::sortformer::encoder::feature_extractor::detail::compute(
        runtime_ev.request.pcm, feature_contract, features);
    runtime_ev.request.frame_count_out =
        emel::diarization::sortformer::encoder::feature_extractor::detail::k_feature_frame_count;
    runtime_ev.request.feature_bin_count_out = detail::k_feature_bin_count;
    runtime_ev.ctx.err = detail::to_error(error::none);
  }
};

struct effect_store_success_error {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_done {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.request.on_done(events::prepare_done{
      .request = &runtime_ev.request,
      .frame_count = runtime_ev.request.frame_count_out,
      .feature_bin_count = runtime_ev.request.feature_bin_count_out,
    });
  }
};

struct effect_publish_success_and_emit_done {
  void operator()(const event::prepare_run & runtime_ev, context & ctx) const noexcept {
    effect_store_success_error{}(runtime_ev, ctx);
    effect_emit_done{}(runtime_ev, ctx);
  }
};

struct effect_store_error_error {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_error {
  void operator()(const event::prepare_run & runtime_ev, context &) const noexcept {
    runtime_ev.request.on_error(events::prepare_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_publish_error_and_emit_error {
  void operator()(const event::prepare_run & runtime_ev, context & ctx) const noexcept {
    effect_store_error_error{}(runtime_ev, ctx);
    effect_emit_error{}(runtime_ev, ctx);
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

inline constexpr effect_begin_prepare effect_begin_prepare{};
inline constexpr effect_mark_model_invalid effect_mark_model_invalid{};
inline constexpr effect_mark_sample_rate_invalid effect_mark_sample_rate_invalid{};
inline constexpr effect_mark_channel_count_invalid effect_mark_channel_count_invalid{};
inline constexpr effect_mark_pcm_shape_invalid effect_mark_pcm_shape_invalid{};
inline constexpr effect_mark_capacity_invalid effect_mark_capacity_invalid{};
inline constexpr effect_extract_features effect_extract_features{};
inline constexpr effect_store_success_error effect_store_success_error{};
inline constexpr effect_emit_done effect_emit_done{};
inline constexpr effect_publish_success_and_emit_done effect_publish_success_and_emit_done{};
inline constexpr effect_store_error_error effect_store_error_error{};
inline constexpr effect_emit_error effect_emit_error{};
inline constexpr effect_publish_error_and_emit_error effect_publish_error_and_emit_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

}  // namespace emel::diarization::request::action
