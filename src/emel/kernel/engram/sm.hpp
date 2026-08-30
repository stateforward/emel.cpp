#pragma once
// benchmark: kernel

#include "emel/kernel/engram/actions.hpp"
#include "emel/kernel/engram/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::engram {

struct state_ready {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_ready> <= *sml::state<state_ready>
          + sml::event<event::execute_hash_rows>
          [ guard::guard_execute_hash_rows{} ] / action::effect_execute_hash_rows{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_conv_taps>
          [ guard::guard_execute_conv_taps{} ] / action::effect_execute_conv_taps{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_alpha_gate>
          [ guard::guard_execute_alpha_gate{} ] / action::effect_execute_alpha_gate{}
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

} // namespace emel::kernel::engram
