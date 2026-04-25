#pragma once

#include "emel/diarization/sortformer/encoder/detail.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/pipeline/context.hpp"
#include "emel/diarization/sortformer/pipeline/detail.hpp"
#include "emel/diarization/sortformer/pipeline/events.hpp"
#include "emel/diarization/sortformer/transformer/detail.hpp"

namespace emel::diarization::sortformer::pipeline::guard {

struct guard_model_contract_valid {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    const auto & contract = runtime_ev.request.contract;
    return contract.model != nullptr &&
        contract.sample_rate == detail::k_sample_rate &&
        contract.speaker_count == detail::k_speaker_count &&
        contract.chunk_len == detail::k_frame_count &&
        contract.feature_extractor.tensor_count != 0u &&
        contract.encoder.tensor_count != 0u &&
        contract.modules.tensor_count != 0u &&
        contract.transformer_encoder.tensor_count != 0u;
  }
};

struct guard_model_contract_invalid {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_model_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_sample_rate_valid {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    return runtime_ev.request.sample_rate == detail::k_sample_rate;
  }
};

struct guard_sample_rate_invalid {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_sample_rate_valid{}(runtime_ev, ctx);
  }
};

struct guard_channel_count_valid {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    return runtime_ev.request.channel_count == detail::k_channel_count;
  }
};

struct guard_channel_count_invalid {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_channel_count_valid{}(runtime_ev, ctx);
  }
};

struct guard_pcm_shape_valid {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    return runtime_ev.request.pcm.data() != nullptr &&
        runtime_ev.request.pcm.size() == static_cast<size_t>(detail::k_required_sample_count);
  }
};

struct guard_pcm_shape_invalid {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_pcm_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_probability_capacity_valid {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    return runtime_ev.request.probabilities.data() != nullptr &&
        runtime_ev.request.probabilities.size() >=
        static_cast<size_t>(detail::k_required_probability_value_count);
  }
};

struct guard_probability_capacity_invalid {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_probability_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_segment_capacity_valid {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    return runtime_ev.request.segments.data() != nullptr &&
        runtime_ev.request.segments.size() >= static_cast<size_t>(detail::k_max_segment_count);
  }
};

struct guard_segment_capacity_invalid {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_segment_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_tensor_contract_valid {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    emel::diarization::sortformer::encoder::detail::contract encoder = {};
    emel::diarization::sortformer::modules::detail::contract modules = {};
    emel::diarization::sortformer::transformer::detail::contract transformer = {};
    return emel::diarization::sortformer::encoder::detail::bind_contract(
        *runtime_ev.request.contract.model, encoder) &&
        emel::diarization::sortformer::modules::detail::bind_contract(
            *runtime_ev.request.contract.model, modules) &&
        emel::diarization::sortformer::transformer::detail::bind_contract(
            *runtime_ev.request.contract.model, transformer);
  }
};

struct guard_tensor_contract_invalid {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_tensor_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_no_error {
  bool operator()(const event::run_flow & runtime_ev, const action::context &) const noexcept {
    return runtime_ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_has_error {
  bool operator()(const event::run_flow & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_no_error{}(runtime_ev, ctx);
  }
};

}  // namespace emel::diarization::sortformer::pipeline::guard
