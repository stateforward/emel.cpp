#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/graph/processor/context.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/tensor/errors.hpp"
#include "emel/tensor/events.hpp"
#include "emel/tensor/sm.hpp"

namespace emel::graph::processor::action {

namespace detail {

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

}  // namespace detail

inline void reset_output(event::execution_output & output) noexcept {
  output.outputs_produced = 0;
  output.graph_reused = 0;
  output.lifecycle = nullptr;
}

struct reject_invalid_execute_with_dispatch {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_output(*ev.request.output_out);
    ev.request.dispatch_error(events::execution_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct reject_invalid_execute_with_output_only {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
    reset_output(*ev.request.output_out);
  }
};

struct reject_invalid_execute_without_output {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct begin_execute {
  void operator()(const event::execute_step & ev, context & ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    ev.ctx.validate_outcome = validate_step::events::phase_outcome::unknown;
    ev.ctx.prepare_outcome = prepare_step::events::phase_outcome::unknown;
    ev.ctx.alloc_outcome = alloc_step::events::phase_outcome::unknown;
    ev.ctx.bind_outcome = bind_step::events::phase_outcome::unknown;
    ev.ctx.kernel_outcome = kernel_step::events::phase_outcome::unknown;
    ev.ctx.gate_outcome = event::lifecycle_outcome::unknown;
    ev.ctx.publish_outcome = event::lifecycle_outcome::unknown;
    ev.ctx.extract_outcome = extract_step::events::phase_outcome::unknown;
    ev.ctx.release_outcome = event::lifecycle_outcome::unknown;
    ev.ctx.graph_reused = 0;
    ev.ctx.outputs_produced = 0;
    ev.ctx.phase_callback_ok = false;
    ev.ctx.phase_callback_err = 0;
    ++ctx.dispatch_generation;
    reset_output(*ev.request.output_out);
  }
};

struct request_lifecycle_gate {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    const bool inputs_ready = detail::required_inputs_ready(ev.request);
    const bool outputs_reusable = detail::publish_targets_reusable(ev.request);
    const std::array<event::lifecycle_outcome, 2> outcomes{
      event::lifecycle_outcome::failed,
      event::lifecycle_outcome::done,
    };
    const std::array<emel::error::type, 2> errors{
      emel::error::cast(error::internal_error),
      emel::error::cast(error::none),
    };
    const size_t result = static_cast<size_t>(inputs_ready && outputs_reusable);
    ev.ctx.gate_outcome = outcomes[result];
    ev.ctx.err = errors[result];
  }
};

struct request_lifecycle_publish {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    const bool publish_ok = detail::publish_phase_tensors(ev.request);
    const std::array<event::lifecycle_outcome, 2> outcomes{
      event::lifecycle_outcome::failed,
      event::lifecycle_outcome::done,
    };
    const std::array<emel::error::type, 2> errors{
      emel::error::cast(error::internal_error),
      emel::error::cast(error::none),
    };
    const size_t result = static_cast<size_t>(publish_ok);
    ev.ctx.publish_outcome = outcomes[result];
    ev.ctx.err = errors[result];
  }
};

struct request_lifecycle_release {
  void operator()(const event::execute_step & ev, context &) const noexcept {
    const bool release_ok = detail::release_phase_tensors(ev.request);
    const std::array<event::lifecycle_outcome, 2> outcomes{
      event::lifecycle_outcome::failed,
      event::lifecycle_outcome::done,
    };
    const std::array<emel::error::type, 2> errors{
      emel::error::cast(error::internal_error),
      emel::error::cast(error::none),
    };
    const size_t result = static_cast<size_t>(release_ok);
    ev.ctx.release_outcome = outcomes[result];
    ev.ctx.err = errors[result];
  }
};

struct commit_output {
  void operator()(const event::execute_step & ev, const context &) const noexcept {
    ev.request.output_out->outputs_produced = ev.ctx.outputs_produced;
    ev.request.output_out->graph_reused = ev.ctx.graph_reused;
    ev.request.output_out->lifecycle = ev.request.lifecycle;
  }
};

struct dispatch_done {
  void operator()(const event::execute_step & ev, const context &) const noexcept {
    ev.request.dispatch_done(events::execution_done{*ev.request.output_out});
  }
};

struct dispatch_error {
  void operator()(const event::execute_step & ev, const context &) const noexcept {
    ev.request.dispatch_error(events::execution_error{
      *ev.request.output_out,
      static_cast<int32_t>(ev.ctx.err),
    });
  }
};

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = emel::error::cast(error::internal_error);
    }
  }
};

inline constexpr reject_invalid_execute_with_dispatch reject_invalid_execute_with_dispatch{};
inline constexpr reject_invalid_execute_with_output_only reject_invalid_execute_with_output_only{};
inline constexpr reject_invalid_execute_without_output reject_invalid_execute_without_output{};
inline constexpr begin_execute begin_execute{};
inline constexpr request_lifecycle_gate request_lifecycle_gate{};
inline constexpr request_lifecycle_publish request_lifecycle_publish{};
inline constexpr request_lifecycle_release request_lifecycle_release{};
inline constexpr commit_output commit_output{};
inline constexpr dispatch_done dispatch_done{};
inline constexpr dispatch_error dispatch_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::graph::processor::action
