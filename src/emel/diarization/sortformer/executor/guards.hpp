#pragma once

#include "emel/diarization/sortformer/executor/context.hpp"
#include "emel/diarization/sortformer/executor/detail.hpp"
#include "emel/diarization/sortformer/executor/events.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/transformer/detail.hpp"

namespace emel::diarization::sortformer::executor::guard {

struct guard_model_contract_valid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context &) const noexcept {
    const auto & contract = runtime_ev.request.contract;
    return contract.model != nullptr &&
        contract.chunk_len == detail::k_frame_count &&
        contract.speaker_count == detail::k_speaker_count &&
        contract.encoder.tensor_count != 0u &&
        contract.modules.tensor_count != 0u &&
        contract.transformer_encoder.tensor_count != 0u;
  }
};

struct guard_model_contract_invalid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_model_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_tensor_contract_valid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context &) const noexcept {
    emel::diarization::sortformer::modules::detail::contract modules = {};
    emel::diarization::sortformer::transformer::detail::contract transformer = {};
    return emel::diarization::sortformer::modules::detail::bind_contract(
        *runtime_ev.request.contract.model, modules) &&
        emel::diarization::sortformer::transformer::detail::bind_contract(
            *runtime_ev.request.contract.model, transformer);
  }
};

struct guard_tensor_contract_invalid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_tensor_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_input_shape_valid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.encoder_frames.data() != nullptr &&
        runtime_ev.request.encoder_frames.size() ==
        static_cast<size_t>(detail::k_required_encoder_value_count);
  }
};

struct guard_input_shape_invalid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_input_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_output_capacity_valid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.hidden_out.data() != nullptr &&
        runtime_ev.request.hidden_out.size() >=
        static_cast<size_t>(detail::k_required_hidden_value_count);
  }
};

struct guard_output_capacity_invalid {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_output_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_execution_ok {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_execution_failed {
  bool operator()(const event::execute_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_execution_ok{}(runtime_ev, ctx);
  }
};

struct guard_has_error_out {
  bool operator()(const event::execute_run & runtime_ev) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_error_out {
  bool operator()(const event::execute_run & runtime_ev) const noexcept {
    return !guard_has_error_out{}(runtime_ev);
  }
};

struct guard_has_done_callback {
  bool operator()(const event::execute_run & runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_done_callback {
  bool operator()(const event::execute_run & runtime_ev) const noexcept {
    return !guard_has_done_callback{}(runtime_ev);
  }
};

struct guard_has_error_callback {
  bool operator()(const event::execute_run & runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_error_callback {
  bool operator()(const event::execute_run & runtime_ev) const noexcept {
    return !guard_has_error_callback{}(runtime_ev);
  }
};

}  // namespace emel::diarization::sortformer::executor::guard
