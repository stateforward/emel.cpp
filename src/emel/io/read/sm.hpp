#pragma once

#include "emel/io/read/actions.hpp"
#include "emel/io/read/context.hpp"
#include "emel/io/read/detail.hpp"
#include "emel/io/read/events.hpp"
#include "emel/io/read/guards.hpp"
#include "emel/sm.hpp"

namespace emel::io::read {

struct state_ready {};
struct state_request_decision {};
struct state_file_path_decision {};
struct state_file_decision {};
struct state_length_decision {};
struct state_layout_decision {};
struct state_target_buffer_decision {};
struct state_platform_decision {};
struct state_read_attempt_decision {};
struct state_invalid_request_error_decision {};
struct state_unsupported_resource_error_decision {};
struct state_unsupported_platform_error_decision {};
struct state_error_callback {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Read_tensor acceptance: every read_tensor enters the validation chain.
      // The chain explicitly walks request, file_path, file, length, layout,
      // target-buffer, and platform preconditions before the read-attempt
      // placeholder is structurally reachable.
        sml::state<state_request_decision> <= *sml::state<state_ready>
          + sml::event<detail::read_tensor_runtime>
          / action::effect_begin_read_tensor

      //------------------------------------------------------------------------------//
      // Request validation.
      , sml::state<state_file_path_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::request_span_valid{} ]
      , sml::state<state_invalid_request_error_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::request_span_invalid{} ]
          / action::effect_mark_invalid_request

      //------------------------------------------------------------------------------//
      // File path validation.
      , sml::state<state_file_decision> <= sml::state<state_file_path_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::file_path_valid{} ]
      , sml::state<state_invalid_request_error_decision> <=
          sml::state<state_file_path_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::file_path_invalid{} ]
          / action::effect_mark_invalid_request

      //------------------------------------------------------------------------------//
      // File index validation.
      , sml::state<state_length_decision> <= sml::state<state_file_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::file_index_valid{} ]
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_file_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::file_index_invalid{} ]
          / action::effect_mark_unsupported_resource

      //------------------------------------------------------------------------------//
      // Length validation.
      , sml::state<state_layout_decision> <= sml::state<state_length_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::length_within_bounds{} ]
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_length_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::length_overflow{} ]
          / action::effect_mark_unsupported_resource

      //------------------------------------------------------------------------------//
      // Layout validation.
      , sml::state<state_target_buffer_decision> <=
          sml::state<state_layout_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::layout_supported{} ]
      , sml::state<state_unsupported_resource_error_decision> <=
          sml::state<state_layout_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::layout_unsupported{} ]
          / action::effect_mark_unsupported_resource

      //------------------------------------------------------------------------------//
      // Target-buffer validation.
      , sml::state<state_platform_decision> <=
          sml::state<state_target_buffer_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::target_buffer_valid{} ]
      , sml::state<state_invalid_request_error_decision> <=
          sml::state<state_target_buffer_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::target_buffer_invalid{} ]
          / action::effect_mark_invalid_request

      //------------------------------------------------------------------------------//
      // Platform validation. Phase 214 replaces the read-attempt placeholder
      // with concrete open/seek/read execution and lifetime management.
      , sml::state<state_read_attempt_decision> <=
          sml::state<state_platform_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::platform_read_supported{} ]
          / action::effect_mark_unsupported_resource
      , sml::state<state_unsupported_platform_error_decision> <=
          sml::state<state_platform_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::platform_read_unsupported{} ]
          / action::effect_mark_unsupported_platform

      //------------------------------------------------------------------------------//
      // Read-attempt placeholder publication. Reaching this state proves all
      // Phase 213 preconditions passed; Phase 214 supplies the actual attempt.
      , sml::state<state_error_callback> <=
          sml::state<state_read_attempt_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_read_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_read_attempt_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_read_tensor_error

      //------------------------------------------------------------------------------//
      // Read_tensor error publication for validation/platform legs.
      , sml::state<state_error_callback> <=
          sml::state<state_invalid_request_error_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_read_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_invalid_request_error_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_read_tensor_error
      , sml::state<state_error_callback> <=
          sml::state<state_unsupported_resource_error_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_read_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_unsupported_resource_error_decision>
          + sml::completion<detail::read_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_read_tensor_error
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
      , sml::state<state_ready> <= sml::state<state_file_path_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_file_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_length_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_layout_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_target_buffer_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_platform_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_read_attempt_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_invalid_request_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_unsupported_resource_error_decision>
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
