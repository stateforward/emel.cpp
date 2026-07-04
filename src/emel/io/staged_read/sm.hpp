#pragma once

// benchmark: designed

#include "emel/io/staged_read/actions.hpp"
#include "emel/io/staged_read/context.hpp"
#include "emel/io/staged_read/detail.hpp"
#include "emel/io/staged_read/events.hpp"
#include "emel/io/staged_read/guards.hpp"
#include "emel/sm.hpp"

namespace emel::io::staged_read {

struct state_ready {};
struct state_guard_staged_callbacks_decision {};
struct state_guard_source_contract_decision {};
struct state_guard_target_window_decision {};
struct state_guard_platform_decision {};
struct state_guard_copy_source_decision {};
struct state_guard_source_span_decision {};
struct state_staged_pre_ready {};
struct state_batch_guard_callbacks_decision {};
struct state_batch_guard_requests_decision {};
struct state_batch_guard_platform_decision {};
struct state_batch_done_callback {};
struct state_invalid_callbacks_error_decision {};
struct state_invalid_staging_contract_error_decision {};
struct state_invalid_target_window_error_decision {};
struct state_unsupported_platform_error_decision {};
struct state_null_source_span_error_decision {};
struct state_source_span_size_mismatch_error_decision {};
struct state_insufficient_source_span_error_decision {};
struct state_staged_window_error_callback {};
struct state_batch_invalid_callbacks_error_decision {};
struct state_batch_invalid_staging_contract_error_decision {};
struct state_batch_unsupported_platform_error_decision {};
struct state_staged_window_batch_error_callback {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Preconditions (228) plus copy-source readiness (229). Copy executes in-action
      // after these guards settle (single-terminal success per dispatch).
        sml::state<state_guard_staged_callbacks_decision> <= *sml::state<state_ready>
          + sml::event<detail::staged_window_runtime>
          / action::effect_begin_staged_window

      , sml::state<state_guard_source_contract_decision> <=
          sml::state<state_guard_staged_callbacks_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_staged_window_callbacks_present{} ]
      , sml::state<state_invalid_callbacks_error_decision> <=
          sml::state<state_guard_staged_callbacks_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_staged_window_callbacks_missing{} ]
          / action::effect_mark_invalid_callbacks

      , sml::state<state_guard_target_window_decision> <=
          sml::state<state_guard_source_contract_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_source_contract_valid{} ]
      , sml::state<state_invalid_staging_contract_error_decision> <=
          sml::state<state_guard_source_contract_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_source_contract_invalid{} ]
          / action::effect_mark_invalid_staging_contract

      , sml::state<state_guard_platform_decision> <=
          sml::state<state_guard_target_window_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_target_window_valid{} ]
      , sml::state<state_invalid_target_window_error_decision> <=
          sml::state<state_guard_target_window_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_target_window_invalid{} ]
          / action::effect_mark_invalid_target_window

      , sml::state<state_guard_copy_source_decision> <=
          sml::state<state_guard_platform_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_platform_staged_read_supported{} ]
      , sml::state<state_unsupported_platform_error_decision> <=
          sml::state<state_guard_platform_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_platform_staged_read_unsupported{} ]
          / action::effect_mark_unsupported_platform

      , sml::state<state_staged_pre_ready> <=
          sml::state<state_guard_source_span_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_copy_span_valid{} ]
          / action::effect_mark_staged_validation_accepted
      , sml::state<state_guard_source_span_decision> <=
          sml::state<state_guard_copy_source_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_source_span_present{} ]
      , sml::state<state_null_source_span_error_decision> <=
          sml::state<state_guard_copy_source_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_source_span_missing{} ]
          / action::effect_mark_null_source_span

      , sml::state<state_insufficient_source_span_error_decision> <=
          sml::state<state_guard_source_span_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_source_span_insufficient{} ]
          / action::effect_mark_insufficient_source_span
      , sml::state<state_source_span_size_mismatch_error_decision> <=
          sml::state<state_guard_source_span_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_source_span_size_mismatch{} ]
          / action::effect_mark_source_span_size_mismatch

      , sml::state<state_ready> <= sml::state<state_staged_pre_ready>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_logical_chunk_aligned{} ]
          / action::effect_publish_staged_window_done_aligned
      , sml::state<state_ready> <= sml::state<state_staged_pre_ready>
          + sml::completion<detail::staged_window_runtime>
          [ guard::guard_stg_logical_chunk_remainder{} ]
          / action::effect_publish_staged_window_done_remainder

      //------------------------------------------------------------------------------//
      // Error publication (symmetric with io/read validation legs).
      , sml::state<state_staged_window_error_callback> <=
          sml::state<state_invalid_callbacks_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_staged_window_error
      , sml::state<state_ready> <=
          sml::state<state_invalid_callbacks_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_staged_window_error
      , sml::state<state_staged_window_error_callback> <=
          sml::state<state_invalid_staging_contract_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_staged_window_error
      , sml::state<state_ready> <=
          sml::state<state_invalid_staging_contract_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_staged_window_error
      , sml::state<state_staged_window_error_callback> <=
          sml::state<state_invalid_target_window_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_staged_window_error
      , sml::state<state_ready> <=
          sml::state<state_invalid_target_window_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_staged_window_error
      , sml::state<state_staged_window_error_callback> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_staged_window_error
      , sml::state<state_ready> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_staged_window_error
      , sml::state<state_staged_window_error_callback> <=
          sml::state<state_null_source_span_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_staged_window_error
      , sml::state<state_ready> <=
          sml::state<state_null_source_span_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_staged_window_error
      , sml::state<state_staged_window_error_callback> <=
          sml::state<state_source_span_size_mismatch_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_staged_window_error
      , sml::state<state_ready> <=
          sml::state<state_source_span_size_mismatch_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_staged_window_error
      , sml::state<state_staged_window_error_callback> <=
          sml::state<state_insufficient_source_span_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_staged_window_error
      , sml::state<state_ready> <=
          sml::state<state_insufficient_source_span_error_decision>
          + sml::completion<detail::staged_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_staged_window_error

      , sml::state<state_ready> <= sml::state<state_staged_window_error_callback>
          + sml::completion<detail::staged_window_runtime>
          / action::effect_record_staged_window_error

      //------------------------------------------------------------------------------//
      // Public staged batch path with explicit batch contract validation.
      , sml::state<state_batch_guard_callbacks_decision> <= *sml::state<state_ready>
          + sml::event<detail::staged_window_batch_runtime>
          / action::effect_begin_staged_window_batch
      , sml::state<state_batch_guard_requests_decision> <=
          sml::state<state_batch_guard_callbacks_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::guard_staged_window_batch_callbacks_present{} ]
      , sml::state<state_batch_invalid_callbacks_error_decision> <=
          sml::state<state_batch_guard_callbacks_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::guard_staged_window_batch_callbacks_missing{} ]
          / action::effect_mark_batch_invalid_callbacks
      , sml::state<state_batch_guard_platform_decision> <=
          sml::state<state_batch_guard_requests_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::guard_stg_batch_requests_valid{} ]
      , sml::state<state_batch_invalid_staging_contract_error_decision> <=
          sml::state<state_batch_guard_requests_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::guard_stg_batch_requests_invalid{} ]
          / action::effect_mark_batch_invalid_staging_contract
      , sml::state<state_batch_done_callback> <=
          sml::state<state_batch_guard_platform_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::guard_platform_staged_read_batch_supported{} ]
          / action::effect_publish_staged_window_batch_done
      , sml::state<state_batch_unsupported_platform_error_decision> <=
          sml::state<state_batch_guard_platform_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::guard_platform_staged_read_batch_unsupported{} ]
          / action::effect_mark_batch_unsupported_platform
      , sml::state<state_ready> <= sml::state<state_batch_done_callback>
          + sml::completion<detail::staged_window_batch_runtime>
          / action::effect_record_staged_window_batch_done
      , sml::state<state_staged_window_batch_error_callback> <=
          sml::state<state_batch_invalid_callbacks_error_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::batch_error_callback_present{} ]
          / action::effect_publish_staged_window_batch_error
      , sml::state<state_ready> <=
          sml::state<state_batch_invalid_callbacks_error_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::batch_error_callback_absent{} ]
          / action::effect_record_staged_window_batch_error
      , sml::state<state_staged_window_batch_error_callback> <=
          sml::state<state_batch_invalid_staging_contract_error_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::batch_error_callback_present{} ]
          / action::effect_publish_staged_window_batch_error
      , sml::state<state_ready> <=
          sml::state<state_batch_invalid_staging_contract_error_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::batch_error_callback_absent{} ]
          / action::effect_record_staged_window_batch_error
      , sml::state<state_staged_window_batch_error_callback> <=
          sml::state<state_batch_unsupported_platform_error_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::batch_error_callback_present{} ]
          / action::effect_publish_staged_window_batch_error
      , sml::state<state_ready> <=
          sml::state<state_batch_unsupported_platform_error_decision>
          + sml::completion<detail::staged_window_batch_runtime>
          [ guard::batch_error_callback_absent{} ]
          / action::effect_record_staged_window_batch_error
      , sml::state<state_ready> <= sml::state<state_staged_window_batch_error_callback>
          + sml::completion<detail::staged_window_batch_runtime>
          / action::effect_record_staged_window_batch_error

      //------------------------------------------------------------------------------//
      // Unexpected handling.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_guard_staged_callbacks_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_guard_source_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_guard_target_window_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_guard_platform_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_guard_copy_source_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_guard_source_span_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_staged_pre_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_invalid_callbacks_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_invalid_staging_contract_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_invalid_target_window_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_unsupported_platform_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_null_source_span_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_source_span_size_mismatch_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_insufficient_source_span_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_staged_window_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_guard_callbacks_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_guard_requests_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_guard_platform_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_done_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_invalid_callbacks_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_invalid_staging_contract_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_unsupported_platform_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_staged_window_batch_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::staged_window &ev) {
    detail::staged_window_attempt_status status{};
    detail::staged_window_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }

  bool process_event(const event::staged_window_batch &ev) {
    detail::staged_window_batch_attempt_status status{};
    detail::staged_window_batch_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }
};

} // namespace emel::io::staged_read
