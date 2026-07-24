#pragma once
// benchmark: designed

#include "emel/kernel/f32_matvec/actions.hpp"
#include "emel/kernel/f32_matvec/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::f32_matvec {

struct state_ready {};
struct state_prepare_f32_result_decision {};
struct state_prepare_f16_result_decision {};
struct state_reference_result_decision {};
struct state_exact_x4_result_decision {};
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
      // Initialization-only four-row packing.
        sml::state<state_prepare_f32_result_decision> <= *sml::state<state_ready>
          + sml::event<event::prepare_f32>
          [ guard::guard_prepare_f32_supported{} ]
          / action::effect_prepare_f32{}
      , sml::state<state_error_callback_decision> <= sml::state<state_ready>
          + sml::event<event::prepare_f32>
          [ guard::guard_prepare_f32_unsupported{} ]
          / action::effect_reject_prepare_f32{}
      , sml::state<state_done_callback_decision> <=
          sml::state<state_prepare_f32_result_decision>
          + sml::completion<event::prepare_f32>
          [ guard::guard_prepare_f32_succeeded{} ]
          / action::effect_accept_prepare_f32{}
      , sml::state<state_error_callback_decision> <=
          sml::state<state_prepare_f32_result_decision>
          + sml::completion<event::prepare_f32>
          [ guard::guard_prepare_f32_failed{} ]
          / action::effect_reject_prepare_f32{}

      , sml::state<state_prepare_f16_result_decision> <= sml::state<state_ready>
          + sml::event<event::prepare_f16>
          [ guard::guard_prepare_f16_supported{} ]
          / action::effect_prepare_f16{}
      , sml::state<state_error_callback_decision> <= sml::state<state_ready>
          + sml::event<event::prepare_f16>
          [ guard::guard_prepare_f16_unsupported{} ]
          / action::effect_reject_prepare_f16{}
      , sml::state<state_done_callback_decision> <=
          sml::state<state_prepare_f16_result_decision>
          + sml::completion<event::prepare_f16>
          [ guard::guard_prepare_f16_succeeded{} ]
          / action::effect_accept_prepare_f16{}
      , sml::state<state_error_callback_decision> <=
          sml::state<state_prepare_f16_result_decision>
          + sml::completion<event::prepare_f16>
          [ guard::guard_prepare_f16_failed{} ]
          / action::effect_reject_prepare_f16{}

      //------------------------------------------------------------------------------//
      // Same-binary oracle and AArch64 exact x4 inference.
      , sml::state<state_reference_result_decision> <= sml::state<state_ready>
          + sml::event<event::execute_reference>
          [ guard::guard_execute_reference_supported{} ]
          / action::effect_execute_reference{}
      , sml::state<state_error_callback_decision> <= sml::state<state_ready>
          + sml::event<event::execute_reference>
          [ guard::guard_execute_reference_unsupported{} ]
          / action::effect_reject_execute_reference{}
      , sml::state<state_done_callback_decision> <=
          sml::state<state_reference_result_decision>
          + sml::completion<event::execute_reference>
          [ guard::guard_execute_reference_succeeded{} ]
          / action::effect_accept_execute_reference{}
      , sml::state<state_error_callback_decision> <=
          sml::state<state_reference_result_decision>
          + sml::completion<event::execute_reference>
          [ guard::guard_execute_reference_failed{} ]
          / action::effect_reject_execute_reference{}

      , sml::state<state_exact_x4_result_decision> <= sml::state<state_ready>
          + sml::event<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_supported{} ]
          / action::effect_execute_exact_x4{}
      , sml::state<state_error_callback_decision> <= sml::state<state_ready>
          + sml::event<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_unsupported{} ]
          / action::effect_reject_execute_exact_x4{}
      , sml::state<state_done_callback_decision> <=
          sml::state<state_exact_x4_result_decision>
          + sml::completion<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_succeeded{} ]
          / action::effect_accept_execute_exact_x4{}
      , sml::state<state_error_callback_decision> <=
          sml::state<state_exact_x4_result_decision>
          + sml::completion<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_failed{} ]
          / action::effect_reject_execute_exact_x4{}

      //------------------------------------------------------------------------------//
      // Publish explicit same-RTC outcomes.
      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::prepare_f32>
          [ guard::guard_prepare_f32_has_done_callback{} ]
          / action::effect_emit_prepare_f32_done{}
      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::prepare_f32>
          [ guard::guard_prepare_f32_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::prepare_f32>
          [ guard::guard_prepare_f32_has_error_callback{} ]
          / action::effect_emit_prepare_f32_error{}
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::prepare_f32>
          [ guard::guard_prepare_f32_no_error_callback{} ]

      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::prepare_f16>
          [ guard::guard_prepare_f16_has_done_callback{} ]
          / action::effect_emit_prepare_f16_done{}
      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::prepare_f16>
          [ guard::guard_prepare_f16_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::prepare_f16>
          [ guard::guard_prepare_f16_has_error_callback{} ]
          / action::effect_emit_prepare_f16_error{}
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::prepare_f16>
          [ guard::guard_prepare_f16_no_error_callback{} ]

      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::execute_reference>
          [ guard::guard_execute_reference_has_done_callback{} ]
          / action::effect_emit_execute_reference_done{}
      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::execute_reference>
          [ guard::guard_execute_reference_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::execute_reference>
          [ guard::guard_execute_reference_has_error_callback{} ]
          / action::effect_emit_execute_reference_error{}
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::execute_reference>
          [ guard::guard_execute_reference_no_error_callback{} ]

      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_has_done_callback{} ]
          / action::effect_emit_execute_exact_x4_done{}
      , sml::state<state_done> <= sml::state<state_done_callback_decision>
          + sml::completion<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_has_error_callback{} ]
          / action::effect_emit_execute_exact_x4_error{}
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::execute_exact_x4>
          [ guard::guard_execute_exact_x4_no_error_callback{} ]

      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::prepare_f32>
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::prepare_f16>
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::execute_reference>
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::execute_exact_x4>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::prepare_f32>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::prepare_f16>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::execute_reference>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::execute_exact_x4>

      //------------------------------------------------------------------------------//
      // Diagnostics and unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::capture_diagnostics>
          / action::effect_capture_diagnostics{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_done_callback_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_error_callback_decision>
          + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;
  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;
  sm(sm &&) = delete;
  sm &operator=(sm &&) = delete;

  template <class event_type> bool process_event(const event_type &ev) {
    if constexpr (requires { ev.result.accepted; }) {
      ev.result.accepted = false;
    }
    const bool handled = base_type::process_event(ev);
    if constexpr (requires { ev.result.accepted; }) {
      return handled && ev.result.accepted;
    } else {
      return false;
    }
  }
};

} // namespace emel::kernel::f32_matvec
