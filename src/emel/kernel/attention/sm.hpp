#pragma once
// benchmark: designed

#include "emel/kernel/attention/actions.hpp"
#include "emel/kernel/attention/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::attention {

struct state_ready {};
struct state_execute_result_decision {};
struct state_execute_done_callback_decision {};
struct state_execute_error_callback_decision {};
struct state_execute_done {};
struct state_execute_errored {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Validate and execute one disjoint contiguous range of attention heads.
        sml::state<state_execute_result_decision> <= *sml::state<state_ready>
                 + sml::event<event::execute>
                 [ guard::guard_execute_supported{} ]
                 / action::effect_execute_head_range{}

      , sml::state<state_execute_error_callback_decision> <= sml::state<state_ready>
                 + sml::event<event::execute>
                 [ guard::guard_execute_unsupported{} ]
                 / action::effect_reject_execute{}

      , sml::state<state_execute_done_callback_decision> <=
          sml::state<state_execute_result_decision>
                 + sml::completion<event::execute>
                 [ guard::guard_execute_succeeded{} ]
                 / action::effect_accept_execute{}

      , sml::state<state_execute_error_callback_decision> <=
          sml::state<state_execute_result_decision>
                 + sml::completion<event::execute>
                 [ guard::guard_execute_failed{} ]
                 / action::effect_reject_execute{}

      //------------------------------------------------------------------------------//
      // Publish explicit same-RTC outcomes, then return to ready.
      , sml::state<state_execute_done> <=
          sml::state<state_execute_done_callback_decision>
                 + sml::completion<event::execute>
                 [ guard::guard_has_done_callback{} ]
                 / action::effect_emit_execute_done{}

      , sml::state<state_execute_done> <=
          sml::state<state_execute_done_callback_decision>
                 + sml::completion<event::execute>
                 [ guard::guard_no_done_callback{} ]

      , sml::state<state_execute_errored> <=
          sml::state<state_execute_error_callback_decision>
                 + sml::completion<event::execute>
                 [ guard::guard_has_error_callback{} ]
                 / action::effect_emit_execute_error{}

      , sml::state<state_execute_errored> <=
          sml::state<state_execute_error_callback_decision>
                 + sml::completion<event::execute>
                 [ guard::guard_no_error_callback{} ]

      , sml::state<state_ready> <= sml::state<state_execute_done>
                 + sml::completion<event::execute>

      , sml::state<state_ready> <= sml::state<state_execute_errored>
                 + sml::completion<event::execute>

      //------------------------------------------------------------------------------//
      // Unexpected events are rejected without retaining their payload.
      , sml::state<state_ready> <= sml::state<state_ready>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected{}

      , sml::state<state_ready> <= sml::state<state_execute_result_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected{}

      , sml::state<state_ready> <= sml::state<state_execute_done_callback_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected{}

      , sml::state<state_ready> <= sml::state<state_execute_error_callback_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected{}

      , sml::state<state_ready> <= sml::state<state_execute_done>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected{}

      , sml::state<state_ready> <= sml::state<state_execute_errored>
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

  bool process_event(const event::execute &ev) {
    const bool handled = base_type::process_event(ev);
    return handled && ev.result.accepted;
  }
};

} // namespace emel::kernel::attention
