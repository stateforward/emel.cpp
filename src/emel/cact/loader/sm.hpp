#pragma once

// benchmark: scaffold

#include "emel/cact/loader/actions.hpp"
#include "emel/cact/loader/context.hpp"
#include "emel/cact/loader/errors.hpp"
#include "emel/cact/loader/events.hpp"
#include "emel/cact/loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::cact::loader {

struct state_uninitialized {};
struct state_probed {};
struct state_bound {};
struct state_parsed {};
struct state_errored {};

struct state_probe_request_decision {};
struct state_probe_outcome_dispatch {};
struct state_probe_geometry_dispatch {};
struct state_probe_done_callback_decision {};
struct state_probe_error_callback_decision {};
struct state_bind_request_decision {};
struct state_bind_request_shape_decision {};
struct state_bind_capacity_decision {};
struct state_bind_outcome_dispatch {};
struct state_bind_done_callback_decision {};
struct state_bind_error_callback_decision {};
struct state_parse_request_decision {};
struct state_parse_file_image_decision {};
struct state_parse_file_identity_decision {};
struct state_parse_bound_storage_decision {};
struct state_parse_capacity_decision {};
struct state_parse_outcome_dispatch {};
struct state_parse_done_callback_decision {};
struct state_parse_error_callback_decision {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Probe op.
        sml::state<state_probe_request_decision> <= *sml::state<state_uninitialized>
          + sml::event<event::probe_runtime> / action::effect_begin_probe
      , sml::state<state_probe_request_decision> <= sml::state<state_probed>
          + sml::event<event::probe_runtime> / action::effect_begin_probe
      , sml::state<state_probe_request_decision> <= sml::state<state_bound>
          + sml::event<event::probe_runtime> / action::effect_begin_probe
      , sml::state<state_probe_request_decision> <= sml::state<state_parsed>
          + sml::event<event::probe_runtime> / action::effect_begin_probe
      , sml::state<state_probe_request_decision> <= sml::state<state_errored>
          + sml::event<event::probe_runtime> / action::effect_begin_probe

      , sml::state<state_probe_outcome_dispatch> <= sml::state<state_probe_request_decision>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_valid_request{} ]
          / action::effect_exec_probe
      , sml::state<state_probe_outcome_dispatch> <= sml::state<state_probe_request_decision>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_invalid_request{} ]
          / action::effect_mark_probe_invalid_request

      , sml::state<state_probe_geometry_dispatch> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_none{} ]
          / action::effect_commit_probe_geometry
      , sml::state<state_probe_done_callback_decision> <= sml::state<state_probe_geometry_dispatch>
          + sml::completion<event::probe_runtime>
      , sml::state<state_probed> <= sml::state<state_probe_done_callback_decision>
          + sml::completion<event::probe_runtime>
          [ guard::guard_done_callback_present<event::probe_runtime>{} ]
          / action::effect_publish_probe_done
      , sml::state<state_probed> <= sml::state<state_probe_done_callback_decision>
          + sml::completion<event::probe_runtime>
          [ guard::guard_done_callback_absent<event::probe_runtime>{} ]
      , sml::state<state_probe_error_callback_decision> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_invalid_request{} ]
      , sml::state<state_probe_error_callback_decision> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_model_invalid{} ]
      , sml::state<state_probe_error_callback_decision> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_capacity{} ]
      , sml::state<state_probe_error_callback_decision> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_parse_failed{} ]
      , sml::state<state_probe_error_callback_decision> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_internal_error{} ]
      , sml::state<state_probe_error_callback_decision> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_untracked{} ]
      , sml::state<state_probe_error_callback_decision> <= sml::state<state_probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::guard_probe_error_unknown{} ]
      , sml::state<state_errored> <= sml::state<state_probe_error_callback_decision>
          + sml::completion<event::probe_runtime>
          [ guard::guard_error_callback_present<event::probe_runtime>{} ]
          / action::effect_publish_probe_error
      , sml::state<state_errored> <= sml::state<state_probe_error_callback_decision>
          + sml::completion<event::probe_runtime>
          [ guard::guard_error_callback_absent<event::probe_runtime>{} ]

      //------------------------------------------------------------------------------//
      // Bind op.
      , sml::state<state_bind_request_decision> <= sml::state<state_probed>
          + sml::event<event::bind_runtime> / action::effect_begin_bind
      , sml::state<state_bind_request_decision> <= sml::state<state_bound>
          + sml::event<event::bind_runtime> / action::effect_begin_bind
      , sml::state<state_bind_request_decision> <= sml::state<state_parsed>
          + sml::event<event::bind_runtime> / action::effect_begin_bind
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_uninitialized>
          + sml::event<event::bind_runtime> / action::effect_mark_bind_invalid_request
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_errored>
          + sml::event<event::bind_runtime> / action::effect_mark_bind_invalid_request

      , sml::state<state_bind_request_shape_decision> <= sml::state<state_bind_request_decision>
          + sml::completion<event::bind_runtime>
      , sml::state<state_bind_capacity_decision> <= sml::state<state_bind_request_shape_decision>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_valid_request{} ]
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_bind_request_shape_decision>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_invalid_request{} ]
          / action::effect_mark_bind_invalid_request
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_bind_request_shape_decision>
          + sml::completion<event::bind_runtime>
          / action::effect_mark_bind_invalid_request
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_bind_capacity_decision>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_capacity_sufficient{} ]
          / action::effect_exec_bind
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_bind_capacity_decision>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_capacity_insufficient{} ]
          / action::effect_mark_bind_capacity
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_bind_capacity_decision>
          + sml::completion<event::bind_runtime>
          / action::effect_mark_bind_capacity

      , sml::state<state_bind_done_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_none{} ]
      , sml::state<state_bound> <= sml::state<state_bind_done_callback_decision>
          + sml::completion<event::bind_runtime>
          [ guard::guard_done_callback_present<event::bind_runtime>{} ]
          / action::effect_publish_bind_done
      , sml::state<state_bound> <= sml::state<state_bind_done_callback_decision>
          + sml::completion<event::bind_runtime>
          [ guard::guard_done_callback_absent<event::bind_runtime>{} ]
      , sml::state<state_bind_error_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_invalid_request{} ]
      , sml::state<state_bind_error_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_model_invalid{} ]
      , sml::state<state_bind_error_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_capacity{} ]
      , sml::state<state_bind_error_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_parse_failed{} ]
      , sml::state<state_bind_error_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_internal_error{} ]
      , sml::state<state_bind_error_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_untracked{} ]
      , sml::state<state_bind_error_callback_decision> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_unknown{} ]
      , sml::state<state_errored> <= sml::state<state_bind_error_callback_decision>
          + sml::completion<event::bind_runtime>
          [ guard::guard_error_callback_present<event::bind_runtime>{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_error_callback_decision>
          + sml::completion<event::bind_runtime>
          [ guard::guard_error_callback_absent<event::bind_runtime>{} ]

      //------------------------------------------------------------------------------//
      // Parse op.
      , sml::state<state_parse_request_decision> <= sml::state<state_bound>
          + sml::event<event::parse_runtime> / action::effect_begin_parse
      , sml::state<state_parse_request_decision> <= sml::state<state_parsed>
          + sml::event<event::parse_runtime> / action::effect_begin_parse
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_uninitialized>
          + sml::event<event::parse_runtime> / action::effect_mark_parse_invalid_request
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_probed>
          + sml::event<event::parse_runtime> / action::effect_mark_parse_invalid_request
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_errored>
          + sml::event<event::parse_runtime> / action::effect_mark_parse_invalid_request

      , sml::state<state_parse_file_image_decision> <= sml::state<state_parse_request_decision>
          + sml::completion<event::parse_runtime>
      , sml::state<state_parse_file_identity_decision> <= sml::state<state_parse_file_image_decision>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_has_file_image{} ]
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_file_image_decision>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_missing_file_image{} ]
          / action::effect_mark_parse_invalid_request
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_file_image_decision>
          + sml::completion<event::parse_runtime>
          / action::effect_mark_parse_invalid_request

      , sml::state<state_parse_bound_storage_decision> <= sml::state<state_parse_file_identity_decision>
          + sml::completion<event::parse_runtime>
          [ guard::guard_parse_matches_probed_file_image{} ]
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_file_identity_decision>
          + sml::completion<event::parse_runtime>
          [ guard::guard_parse_mismatches_probed_file_image{} ]
          / action::effect_mark_parse_invalid_request
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_file_identity_decision>
          + sml::completion<event::parse_runtime>
          / action::effect_mark_parse_invalid_request

      , sml::state<state_parse_capacity_decision> <= sml::state<state_parse_bound_storage_decision>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_has_bound_storage{} ]
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_bound_storage_decision>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_missing_bound_storage{} ]
          / action::effect_mark_parse_invalid_request
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_bound_storage_decision>
          + sml::completion<event::parse_runtime>
          / action::effect_mark_parse_invalid_request

      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_capacity_decision>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_bound_capacity_sufficient{} ]
          / action::effect_exec_parse
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_capacity_decision>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_bound_capacity_insufficient{} ]
          / action::effect_mark_parse_capacity
      , sml::state<state_parse_outcome_dispatch> <= sml::state<state_parse_capacity_decision>
          + sml::completion<event::parse_runtime>
          / action::effect_mark_parse_capacity

      , sml::state<state_parse_done_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_none{} ]
      , sml::state<state_parsed> <= sml::state<state_parse_done_callback_decision>
          + sml::completion<event::parse_runtime>
          [ guard::guard_done_callback_present<event::parse_runtime>{} ]
          / action::effect_publish_parse_done
      , sml::state<state_parsed> <= sml::state<state_parse_done_callback_decision>
          + sml::completion<event::parse_runtime>
          [ guard::guard_done_callback_absent<event::parse_runtime>{} ]
      , sml::state<state_parse_error_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_invalid_request{} ]
      , sml::state<state_parse_error_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_model_invalid{} ]
      , sml::state<state_parse_error_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_capacity{} ]
      , sml::state<state_parse_error_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_parse_failed{} ]
      , sml::state<state_parse_error_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_internal_error{} ]
      , sml::state<state_parse_error_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_untracked{} ]
      , sml::state<state_parse_error_callback_decision> <= sml::state<state_parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::guard_parse_error_unknown{} ]
      , sml::state<state_errored> <= sml::state<state_parse_error_callback_decision>
          + sml::completion<event::parse_runtime>
          [ guard::guard_error_callback_present<event::parse_runtime>{} ]
          / action::effect_publish_parse_error
      , sml::state<state_errored> <= sml::state<state_parse_error_callback_decision>
          + sml::completion<event::parse_runtime>
          [ guard::guard_error_callback_absent<event::parse_runtime>{} ]

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_errored> <= sml::state<state_uninitialized> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_probed> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bound> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parsed> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_errored> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_probe_request_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_probe_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_probe_geometry_dispatch> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_probe_done_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_probe_error_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_request_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_request_shape_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_capacity_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_done_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_error_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_request_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_file_image_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_file_identity_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_bound_storage_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_capacity_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_done_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_parse_error_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  sm() : base_type() {}

  bool process_event(const event::probe &ev) {
    event::probe_ctx ctx{};
    event::probe_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::bind_storage &ev) {
    event::bind_ctx ctx{};
    event::bind_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::parse &ev) {
    event::parse_ctx ctx{};
    event::parse_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Loader = sm;

} // namespace emel::cact::loader
