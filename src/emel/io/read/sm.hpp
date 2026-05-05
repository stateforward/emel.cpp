#pragma once

#include "emel/io/read/actions.hpp"
#include "emel/io/read/context.hpp"
#include "emel/io/read/detail.hpp"
#include "emel/io/read/events.hpp"
#include "emel/io/read/guards.hpp"
#include "emel/sm.hpp"

namespace emel::io::read {

// Boundary state set for Phase 212. Phase 213 introduces real validation and
// platform decision states (file/offset/length/layout/target-buffer/platform);
// Phase 214 introduces the file-open / read-attempt / done-callback chain.
// Phase 212 only needs ready, the request decision, the fail-closed boundary
// error decision, and the synchronous error callback state.
struct state_ready {};
struct state_request_decision {};
struct state_unsupported_platform_error_decision {};
struct state_error_callback {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Read_tensor acceptance: every well-formed read_tensor enters the
      // boundary decision chain. Phase 212 routes accepted requests to the
      // unsupported_platform fail-closed leg; Phase 213 will replace this
      // single fork with the full validation chain.
        sml::state<state_request_decision> <= *sml::state<state_ready>
          + sml::event<detail::read_tensor_runtime>
          / action::effect_begin_read_tensor

      //------------------------------------------------------------------------------//
      // Boundary request decision.
      , sml::state<state_unsupported_platform_error_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::guard_request_accepted{} ]
          / action::effect_mark_unsupported_platform

      //------------------------------------------------------------------------------//
      // Read_tensor error publication for the unsupported_platform leg.
      , sml::state<state_error_callback> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_read_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_read_tensor_error

      //------------------------------------------------------------------------------//
      // Synchronous error callback recovery.
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::completion<detail::read_tensor_runtime>
          / action::effect_record_read_tensor_error

      //------------------------------------------------------------------------------//
      // Unexpected event handling for every reachable state.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_request_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::read_tensor &ev) {
    detail::read_attempt_status status{};
    detail::read_tensor_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }
};

} // namespace emel::io::read
