#pragma once
// benchmark: kernel

#include "emel/kernel/swa/actions.hpp"
#include "emel/kernel/swa/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::swa {

struct state_ready {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_ready> <= *sml::state<state_ready>
          + sml::event<event::execute_attend>
          [ guard::guard_execute_attend{} ] / action::effect_execute_attend{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_cache_write>
          [ guard::guard_execute_cache_write{} ] / action::effect_execute_cache_write{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_gate_mul>
          [ guard::guard_execute_gate_mul{} ] / action::effect_execute_gate_mul{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_residual_gate>
          [ guard::guard_execute_residual_gate{} ] / action::effect_execute_residual_gate{}
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

} // namespace emel::kernel::swa
