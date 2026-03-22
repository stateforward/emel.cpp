#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/graph/processor/events.hpp"
#include "emel/tensor/errors.hpp"
#include "emel/tensor/events.hpp"
#include "emel/tensor/sm.hpp"

namespace emel::graph::processor::detail {

inline const event::lifecycle_phase & lifecycle_phase(const event::execute & request) noexcept {
  return *request.lifecycle->phase;
}

inline bool capture_tensor_state(tensor::sm & tensor_machine, const int32_t tensor_id,
                                 tensor::event::tensor_state & state_out) noexcept {
  int32_t tensor_err = static_cast<int32_t>(emel::error::cast(tensor::error::none));
  return tensor_machine.process_event(tensor::event::capture_tensor_state{
      .tensor_id = tensor_id,
      .state_out = &state_out,
      .error_out = &tensor_err,
  });
}

inline bool lifecycle_state_allowed(
    const tensor::event::lifecycle lifecycle_state,
    const std::array<uint8_t, 5> & allowed_states) noexcept {
  return allowed_states[static_cast<size_t>(lifecycle_state)] != 0u;
}

inline bool required_inputs_ready(const event::execute & request) noexcept {
  static constexpr std::array<uint8_t, 5> filled_states{0u, 0u, 1u, 1u, 0u};
  const auto & phase = lifecycle_phase(request);
  bool all_ready = true;
  for (int32_t idx = 0; idx < phase.required_filled_count; ++idx) {
    tensor::event::tensor_state tensor_state{};
    const bool captured =
        capture_tensor_state(*request.tensor_machine, phase.required_filled_ids[idx], tensor_state);
    const bool tensor_ready =
        captured && lifecycle_state_allowed(tensor_state.lifecycle_state, filled_states);
    all_ready = tensor_ready && all_ready;
  }
  return all_ready;
}

inline bool publish_targets_reusable(const event::execute & request) noexcept {
  static constexpr std::array<uint8_t, 5> empty_states{0u, 1u, 0u, 0u, 0u};
  const auto & phase = lifecycle_phase(request);
  bool all_reusable = true;
  for (int32_t idx = 0; idx < phase.publish_count; ++idx) {
    tensor::event::tensor_state tensor_state{};
    const bool captured =
        capture_tensor_state(*request.tensor_machine, phase.publish_ids[idx], tensor_state);
    const bool tensor_reusable =
        captured && lifecycle_state_allowed(tensor_state.lifecycle_state, empty_states);
    all_reusable = tensor_reusable && all_reusable;
  }
  return all_reusable;
}

inline bool publish_phase_tensors(const event::execute & request) noexcept {
  const auto & phase = lifecycle_phase(request);
  bool all_ok = true;
  for (int32_t idx = 0; idx < phase.publish_count; ++idx) {
    int32_t tensor_err = static_cast<int32_t>(emel::error::cast(tensor::error::none));
    const bool published = request.tensor_machine->process_event(tensor::event::publish_filled_tensor{
        .tensor_id = phase.publish_ids[idx],
        .error_out = &tensor_err,
    });
    all_ok = published && all_ok;
  }
  return all_ok;
}

inline bool release_phase_tensors(const event::execute & request) noexcept {
  const auto & phase = lifecycle_phase(request);
  bool all_ok = true;
  for (int32_t idx = 0; idx < phase.release_count; ++idx) {
    int32_t tensor_err = static_cast<int32_t>(emel::error::cast(tensor::error::none));
    const bool released = request.tensor_machine->process_event(tensor::event::release_tensor_ref{
        .tensor_id = phase.release_ids[idx],
        .error_out = &tensor_err,
    });
    all_ok = released && all_ok;
  }
  return all_ok;
}

}  // namespace emel::graph::processor::detail
