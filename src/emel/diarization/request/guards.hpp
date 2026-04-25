#pragma once

#include "emel/diarization/request/context.hpp"
#include "emel/diarization/request/detail.hpp"
#include "emel/diarization/request/events.hpp"
#include "emel/diarization/sortformer/encoder/feature_extractor/detail.hpp"

namespace emel::diarization::request::guard {

struct guard_model_contract_valid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context &) const noexcept {
    const auto & contract = runtime_ev.request.contract;
    if (contract.model == nullptr) {
      return false;
    }

    const auto feature_contract =
        emel::diarization::sortformer::encoder::feature_extractor::detail::make_contract(
            *contract.model);
    return contract.model != nullptr &&
        contract.sample_rate == detail::k_sample_rate &&
        contract.speaker_count == detail::k_speaker_count &&
        contract.frame_shift_ms == detail::k_frame_shift_ms &&
        contract.chunk_len == detail::k_chunk_len &&
        contract.chunk_right_context == detail::k_chunk_right_context &&
        emel::diarization::sortformer::encoder::feature_extractor::detail::contract_valid(
            feature_contract) &&
        contract.feature_extractor.tensor_count != 0u &&
        contract.encoder.tensor_count != 0u &&
        contract.modules.tensor_count != 0u &&
        contract.transformer_encoder.tensor_count != 0u;
  }
};

struct guard_model_contract_invalid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_model_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_sample_rate_valid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sample_rate == detail::k_sample_rate;
  }
};

struct guard_sample_rate_invalid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_sample_rate_valid{}(runtime_ev, ctx);
  }
};

struct guard_channel_count_valid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.channel_count == detail::k_channel_count;
  }
};

struct guard_channel_count_invalid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_channel_count_valid{}(runtime_ev, ctx);
  }
};

struct guard_pcm_shape_valid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.pcm.data() != nullptr &&
        runtime_ev.request.pcm.size() ==
        static_cast<size_t>(detail::k_required_sample_count);
  }
};

struct guard_pcm_shape_invalid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_pcm_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_output_capacity_valid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.features.data() != nullptr &&
        runtime_ev.request.features.size() >=
        static_cast<size_t>(detail::k_required_feature_count);
  }
};

struct guard_output_capacity_invalid {
  bool operator()(const event::prepare_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_output_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_has_done_callback {
  bool operator()(const event::prepare_run & runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_done_callback {
  bool operator()(const event::prepare_run & runtime_ev) const noexcept {
    return !guard_has_done_callback{}(runtime_ev);
  }
};

struct guard_has_error_out {
  bool operator()(const event::prepare_run & runtime_ev) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_error_out {
  bool operator()(const event::prepare_run & runtime_ev) const noexcept {
    return !guard_has_error_out{}(runtime_ev);
  }
};

struct guard_has_error_callback {
  bool operator()(const event::prepare_run & runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_error_callback {
  bool operator()(const event::prepare_run & runtime_ev) const noexcept {
    return !guard_has_error_callback{}(runtime_ev);
  }
};

}  // namespace emel::diarization::request::guard
