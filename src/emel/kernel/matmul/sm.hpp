#pragma once
// benchmark: designed

#include <utility>

#include "emel/kernel/matmul/actions.hpp"
#include "emel/kernel/matmul/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::matmul {

struct state_ready {};
// These choice states turn kernel and join outcomes into explicit accepted or
// rejected event results without storing per-dispatch status in actor context.
struct state_serial_result_decision {};
struct state_parallel_result_decision {};
struct state_done_callback_decision {};
struct state_error_callback_decision {};
struct state_done {};
struct state_errored {};

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

      , sml::state<state_done_callback_decision> <=
          sml::state<state_serial_result_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_accepted{} ]
                 / action::effect_accept_serial_execution

      , sml::state<state_error_callback_decision> <=
          sml::state<state_serial_result_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_rejected{} ]
                 / action::effect_reject_serial_execution

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<8u, emel::kernel::detail::quant::Q4_K_X8_ROWS>{} ]
                 / action::effect_execute_parallel<8u, emel::kernel::detail::quant::Q4_K_X8_ROWS>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<8u, emel::kernel::detail::quant::Q8_0_X4_ROWS>{} ]
                 / action::effect_execute_parallel<8u, emel::kernel::detail::quant::Q8_0_X4_ROWS>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<8u, 1u>{} ]
                 / action::effect_execute_parallel<8u, 1u>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<4u, emel::kernel::detail::quant::Q4_K_X8_ROWS>{} ]
                 / action::effect_execute_parallel<4u, emel::kernel::detail::quant::Q4_K_X8_ROWS>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<4u, emel::kernel::detail::quant::Q8_0_X4_ROWS>{} ]
                 / action::effect_execute_parallel<4u, emel::kernel::detail::quant::Q8_0_X4_ROWS>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<4u, 1u>{} ]
                 / action::effect_execute_parallel<4u, 1u>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<2u, emel::kernel::detail::quant::Q4_K_X8_ROWS>{} ]
                 / action::effect_execute_parallel<2u, emel::kernel::detail::quant::Q4_K_X8_ROWS>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<2u, emel::kernel::detail::quant::Q8_0_X4_ROWS>{} ]
                 / action::effect_execute_parallel<2u, emel::kernel::detail::quant::Q8_0_X4_ROWS>{}

      , sml::state<state_parallel_result_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_ready<2u, 1u>{} ]
                 / action::effect_execute_parallel<2u, 1u>{}

      , sml::state<state_error_callback_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_request_invalid{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_error_callback_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_unavailable{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_error_callback_decision> <= sml::state<state_ready>
                 + sml::event<event::execute_parallel>
                 [ guard::guard_parallel_no_lane_count{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_error_callback_decision> <=
          sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_submission_failed{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_error_callback_decision> <=
          sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_lane_rejected{} ]
                 / action::effect_reject_parallel_execution

      , sml::state<state_done_callback_decision> <=
          sml::state<state_parallel_result_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_all_lanes_accepted{} ]
                 / action::effect_accept_parallel_execution

      //------------------------------------------------------------------------------//
      // Publish explicit same-RTC outcomes.
      , sml::state<state_done> <= sml::state<state_done_callback_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_has_done_callback{} ]
                 / action::effect_emit_serial_done{}

      , sml::state<state_done> <= sml::state<state_done_callback_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_no_done_callback{} ]

      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_has_error_callback{} ]
                 / action::effect_emit_serial_error{}

      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
                 + sml::completion<event::execute_serial>
                 [ guard::guard_serial_no_error_callback{} ]

      , sml::state<state_done> <= sml::state<state_done_callback_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_has_done_callback{} ]
                 / action::effect_emit_parallel_done{}

      , sml::state<state_done> <= sml::state<state_done_callback_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_no_done_callback{} ]

      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_has_error_callback{} ]
                 / action::effect_emit_parallel_error{}

      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
                 + sml::completion<event::execute_parallel>
                 [ guard::guard_parallel_no_error_callback{} ]

      , sml::state<state_ready> <= sml::state<state_done>
                 + sml::completion<event::execute_serial>

      , sml::state<state_ready> <= sml::state<state_done>
                 + sml::completion<event::execute_parallel>

      , sml::state<state_ready> <= sml::state<state_errored>
                 + sml::completion<event::execute_serial>

      , sml::state<state_ready> <= sml::state<state_errored>
                 + sml::completion<event::execute_parallel>

      //------------------------------------------------------------------------------//
      // Bounded observability through an explicit query event.
      , sml::state<state_ready> <= sml::state<state_ready>
                 + sml::event<event::capture_diagnostics>
                 / action::effect_capture_diagnostics

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
      , sml::state<state_ready> <= sml::state<state_done_callback_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_error_callback_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_done>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_errored>
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

  explicit sm(const execution_policy &policy)
      : base_type(std::in_place, policy) {}

  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;
  sm(sm &&) = delete;
  sm &operator=(sm &&) = delete;

  bool process_event(const event::configure_kernel_kind &ev) {
    return base_type::process_event(ev);
  }

  bool process_event(const event::execute_serial &ev) {
    return base_type::process_event(ev);
  }

  bool process_event(const event::execute_parallel &ev) {
    return base_type::process_event(ev);
  }

  bool process_event(const event::capture_diagnostics &ev) {
    return base_type::process_event(ev);
  }
};

} // namespace emel::kernel::matmul
