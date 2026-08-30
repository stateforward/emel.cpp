#pragma once
// benchmark: kernel

#include "emel/kernel/zcrms/actions.hpp"
#include "emel/kernel/zcrms/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::zcrms {

struct state_ready {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_ready> <= *sml::state<state_ready>
          + sml::event<event::execute_norm_rows>
          [ guard::guard_execute_norm_rows{} ] / action::effect_execute_norm_rows{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_unit_rows>
          [ guard::guard_execute_unit_rows{} ] / action::effect_execute_unit_rows{}
      , sml::state<state_ready> <= sml::state<state_ready>
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

  template <class event_type> bool process_event(const event_type &ev) {
    if constexpr (requires { ev.result.accepted; })
      ev.result.accepted = false;
    const bool handled = base_type::process_event(ev);
    if constexpr (requires { ev.result.accepted; })
      return handled && ev.result.accepted;
    return handled;
  }
};

} // namespace emel::kernel::zcrms
