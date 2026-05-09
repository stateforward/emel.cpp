#pragma once

// benchmark: designed

#include "emel/io/async/actions.hpp"
#include "emel/io/async/context.hpp"
#include "emel/io/async/detail.hpp"
#include "emel/io/async/events.hpp"
#include "emel/io/async/guards.hpp"
#include "emel/sm.hpp"

namespace emel::io::async {

struct state_ready {};
struct state_guard_callbacks_decision {};
struct state_guard_source_contract_decision {};
struct state_guard_target_window_decision {};
struct state_guard_progress_contract_decision {};
struct state_guard_cancel_decision {};
struct state_guard_scheduler_contract_decision {};
struct state_guard_progress_kind_decision {};
struct state_invalid_callbacks_error_decision {};
struct state_invalid_source_contract_error_decision {};
struct state_invalid_target_window_error_decision {};
struct state_invalid_progress_contract_error_decision {};
struct state_cancelled_error_decision {};
struct state_invalid_scheduler_contract_error_decision {};
struct state_error_callback {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Phase 243 cooperative progress: each public dispatch validates stable
      // caller-owned storage and then advances at most one configured chunk.
        sml::state<state_guard_callbacks_decision> <= *sml::state<state_ready>
          + sml::event<detail::load_window_runtime>
          / action::effect_begin_load_window{}

      , sml::state<state_guard_source_contract_decision> <=
          sml::state<state_guard_callbacks_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_load_window_callbacks_present{} ]
      , sml::state<state_invalid_callbacks_error_decision> <=
          sml::state<state_guard_callbacks_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_load_window_callbacks_missing{} ]
          / action::effect_mark_invalid_callbacks{}

      , sml::state<state_guard_target_window_decision> <=
          sml::state<state_guard_source_contract_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_source_contract_valid{} ]
      , sml::state<state_invalid_source_contract_error_decision> <=
          sml::state<state_guard_source_contract_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_source_contract_invalid{} ]
          / action::effect_mark_invalid_source_contract{}

      , sml::state<state_guard_progress_contract_decision> <=
          sml::state<state_guard_target_window_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_target_window_valid{} ]
      , sml::state<state_invalid_target_window_error_decision> <=
          sml::state<state_guard_target_window_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_target_window_invalid{} ]
          / action::effect_mark_invalid_target_window{}

      , sml::state<state_guard_cancel_decision> <=
          sml::state<state_guard_progress_contract_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_progress_contract_valid{} ]
      , sml::state<state_invalid_progress_contract_error_decision> <=
          sml::state<state_guard_progress_contract_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_progress_contract_invalid{} ]
          / action::effect_mark_invalid_progress_contract{}

      , sml::state<state_cancelled_error_decision> <=
          sml::state<state_guard_cancel_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_cancel_requested{} ]
          / action::effect_mark_cancelled{}
      , sml::state<state_guard_scheduler_contract_decision> <=
          sml::state<state_guard_cancel_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_cancel_absent{} ]

      , sml::state<state_guard_progress_kind_decision> <=
          sml::state<state_guard_scheduler_contract_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_scheduler_contract_valid{} ]
      , sml::state<state_invalid_scheduler_contract_error_decision> <=
          sml::state<state_guard_scheduler_contract_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_scheduler_contract_invalid{} ]
          / action::effect_mark_invalid_scheduler_contract{}

      , sml::state<state_ready> <=
          sml::state<state_guard_progress_kind_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_partial_progress_ready{} ]
          / action::effect_publish_load_window_progress_done{}
      , sml::state<state_ready> <=
          sml::state<state_guard_progress_kind_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::guard_terminal_progress_ready{} ]
          / action::effect_publish_load_window_done{}

      , sml::state<state_error_callback> <=
          sml::state<state_invalid_callbacks_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_window_error{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_callbacks_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_window_error{}
      , sml::state<state_error_callback> <=
          sml::state<state_invalid_source_contract_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_window_error{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_source_contract_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_window_error{}
      , sml::state<state_error_callback> <=
          sml::state<state_invalid_target_window_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_window_error{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_target_window_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_window_error{}
      , sml::state<state_error_callback> <=
          sml::state<state_invalid_progress_contract_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_window_error{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_progress_contract_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_window_error{}
      , sml::state<state_error_callback> <=
          sml::state<state_cancelled_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_window_error{}
      , sml::state<state_ready> <=
          sml::state<state_cancelled_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_window_error{}
      , sml::state<state_error_callback> <=
          sml::state<state_invalid_scheduler_contract_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_window_error{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_scheduler_contract_error_decision>
          + sml::completion<detail::load_window_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_window_error{}
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::completion<detail::load_window_runtime>
          / action::effect_record_load_window_error{}

      //------------------------------------------------------------------------------//
      // Unexpected handling.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_guard_callbacks_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_guard_source_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_guard_target_window_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_guard_progress_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_guard_cancel_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_guard_scheduler_contract_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_guard_progress_kind_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_callbacks_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_source_contract_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_target_window_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_progress_contract_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_cancelled_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <=
          sml::state<state_invalid_scheduler_contract_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::co_sm<model, action::context> {
  using base_type = emel::co_sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::process_event_async;
  using base_type::visit_current_states;

  static_assert(emel::policy::strict_ordering_scheduler_contract<
                typename base_type::scheduler_type>);

  bool process_event(const event::load_window &ev) {
    detail::load_window_status status{};
    detail::load_window_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }
};

} // namespace emel::io::async
