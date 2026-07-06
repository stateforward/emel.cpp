#pragma once

#include <utility>

#include "emel/sm.hpp"
#include "emel/text/generator/matmul/actions.hpp"
#include "emel/text/generator/matmul/guards.hpp"

namespace emel::text::generator::matmul {

using lane_pool = action::lane_pool;

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
                 [ guard::guard_parallel_lane_rejected<0>{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected<1>{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected<2>{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected<3>{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected<4>{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected<5>{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected<6>{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_ready> <= sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected<7>{} ]
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

  sm() : base_type() {
    process_event(event::configure_kernel_kind{emel::kernel::detect_host_kind()});
  }

  explicit sm(lane_pool & parallel_matmul_lanes_ref) : base_type() {
    this->context_.parallel_matmul_lanes = &parallel_matmul_lanes_ref;
    process_event(event::configure_kernel_kind{emel::kernel::detect_host_kind()});
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
    return this->context_.parallel_matmul_lanes != nullptr;
  }

  const std::array<emel::kernel::sm, k_matmul_lanes> & parallel_lane_kernels() const noexcept {
    return this->context_.lane_kernels;
  }
};

}  // namespace emel::text::generator::matmul
