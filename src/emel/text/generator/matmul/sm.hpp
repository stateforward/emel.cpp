#pragma once
// benchmark: designed

#include <span>
#include <utility>

#include "emel/sm.hpp"
#include "emel/text/generator/matmul/actions.hpp"
#include "emel/text/generator/matmul/guards.hpp"

namespace emel::text::generator::matmul {

struct state_ready {};
struct state_serial_result_decision {};
struct state_parallel_result_decision {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Kernel binding.
        sml::state<state_ready> <= *sml::state<state_ready>
                 + sml::event<event::configure_kernel_kind>
                 / action::effect_configure_kernel_kind

      //------------------------------------------------------------------------------//
      // Explicit serial versus parallel matmul execution.
      , sml::state<state_serial_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_serial>
                 / action::effect_execute_serial

      , sml::state<state_ready> <= sml::state<state_serial_result_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_accepted{} ]
                 / action::effect_accept_serial_execution

      , sml::state<state_ready> <= sml::state<state_serial_result_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_rejected{} ]
                 / action::effect_reject_serial_execution

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready{} ]
                 / action::effect_execute_parallel

      , sml::state<state_ready> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_unavailable{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_submission_failed{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_join_failed{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_all_lanes_accepted{} ]
                 / action::effect_accept_parallel_execution

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_serial_result_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  explicit sm(const execution_policy & policy) : base_type() {
    this->context_.parallel_matmul_lanes = policy.parallel_matmul_lanes;
    this->context_.kernel_kind = policy.kernel_kind;
    this->context_.active_lanes = policy.active_lanes;
    (void) action::reserve_lane_storage(
        this->context_, policy.parallel_matmul_lanes.lane_capacity);
    process_event(event::configure_kernel_kind{policy.kernel_kind});
  }

  bool process_event(const event::configure_kernel_kind & ev) {
    return base_type::process_event(ev);
  }

  bool process_event(const event::execute_serial & ev) {
    return base_type::process_event(ev);
  }

  bool process_event(const event::execute_parallel & ev) {
    return base_type::process_event(ev);
  }

  template <class counter_fn>
  uint64_t kernel_counter_total(counter_fn && counter) const noexcept {
    return action::compute_kernel_counter_total(
        this->context_, std::forward<counter_fn>(counter));
  }

  const emel::kernel::sm & serial_kernel() const noexcept {
    return this->context_.kernel;
  }

  bool parallel_lanes_available() const noexcept {
    return this->context_.parallel_matmul_lanes.valid() &&
           action::lane_storage_ready(this->context_);
  }

  size_t active_lane_count() const noexcept {
    return this->context_.active_lanes;
  }

  std::span<const emel::kernel::sm> parallel_lane_kernels() const noexcept {
    if (this->context_.lane_kernels == nullptr) {
      return {};
    }
    return {this->context_.lane_kernels.get(), this->context_.lane_capacity};
  }
};

}  // namespace emel::text::generator::matmul
