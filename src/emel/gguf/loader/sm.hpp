#pragma once

#include "emel/gguf/loader/actions.hpp"
#include "emel/gguf/loader/context.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gguf::loader {

struct uninitialized {};
struct probed {};
struct bound {};
struct parsed {};
struct errored {};

struct probe_request_decision {};
struct probe_outcome_dispatch {};
struct probe_requirements_dispatch {};
struct bind_request_decision {};
struct bind_request_shape_decision {};
struct bind_capacity_decision {};
struct bind_outcome_dispatch {};
struct parse_request_decision {};
struct parse_file_image_decision {};
struct parse_bound_storage_decision {};
struct parse_capacity_decision {};
struct parse_outcome_dispatch {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Probe op.
        sml::state<probe_request_decision> <= *sml::state<uninitialized>
          + sml::event<event::probe_runtime> / action::begin_probe
      , sml::state<probe_request_decision> <= sml::state<probed>
          + sml::event<event::probe_runtime> / action::begin_probe
      , sml::state<probe_request_decision> <= sml::state<bound>
          + sml::event<event::probe_runtime> / action::begin_probe
      , sml::state<probe_request_decision> <= sml::state<parsed>
          + sml::event<event::probe_runtime> / action::begin_probe
      , sml::state<probe_request_decision> <= sml::state<errored>
          + sml::event<event::probe_runtime> / action::begin_probe

      , sml::state<probe_outcome_dispatch> <= sml::state<probe_request_decision>
          + sml::completion<event::probe_runtime> [ guard::probe_valid_request{} ]
          / action::exec_probe
      , sml::state<probe_outcome_dispatch> <= sml::state<probe_request_decision>
          + sml::completion<event::probe_runtime> [ guard::probe_invalid_request{} ]
          / action::mark_probe_invalid_request

      , sml::state<probe_requirements_dispatch> <= sml::state<probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::probe_phase_ok{} ]
          / action::commit_probe_requirements
      , sml::state<probed> <= sml::state<probe_requirements_dispatch>
          + sml::completion<event::probe_runtime>
          / action::publish_probe_done
      , sml::state<errored> <= sml::state<probe_outcome_dispatch>
          + sml::completion<event::probe_runtime> [ guard::probe_phase_failed{} ]
          / action::publish_probe_error

      //------------------------------------------------------------------------------//
      // Bind op.
      , sml::state<bind_request_decision> <= sml::state<probed>
          + sml::event<event::bind_runtime> / action::begin_bind
      , sml::state<bind_request_decision> <= sml::state<bound>
          + sml::event<event::bind_runtime> / action::begin_bind
      , sml::state<bind_request_decision> <= sml::state<parsed>
          + sml::event<event::bind_runtime> / action::begin_bind
      , sml::state<bind_outcome_dispatch> <= sml::state<uninitialized>
          + sml::event<event::bind_runtime> / action::mark_bind_invalid_request
      , sml::state<bind_outcome_dispatch> <= sml::state<errored>
          + sml::event<event::bind_runtime> / action::mark_bind_invalid_request

      , sml::state<bind_request_shape_decision> <= sml::state<bind_request_decision>
          + sml::completion<event::bind_runtime>
      , sml::state<bind_capacity_decision> <= sml::state<bind_request_shape_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_valid_request{} ]
      , sml::state<bind_outcome_dispatch> <= sml::state<bind_request_shape_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_invalid_request{} ]
          / action::mark_bind_invalid_request
      , sml::state<bind_outcome_dispatch> <= sml::state<bind_request_shape_decision>
          + sml::completion<event::bind_runtime>
          / action::mark_bind_invalid_request
      , sml::state<bind_outcome_dispatch> <= sml::state<bind_capacity_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_capacity_sufficient{} ]
          / action::exec_bind
      , sml::state<bind_outcome_dispatch> <= sml::state<bind_capacity_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_capacity_insufficient{} ]
          / action::mark_bind_capacity
      , sml::state<bind_outcome_dispatch> <= sml::state<bind_capacity_decision>
          + sml::completion<event::bind_runtime>
          / action::mark_bind_capacity

      , sml::state<bound> <= sml::state<bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::bind_phase_ok{} ]
          / action::publish_bind_done
      , sml::state<errored> <= sml::state<bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::bind_phase_failed{} ]
          / action::publish_bind_error

      //------------------------------------------------------------------------------//
      // Parse op.
      , sml::state<parse_request_decision> <= sml::state<bound>
          + sml::event<event::parse_runtime> / action::begin_parse
      , sml::state<parse_request_decision> <= sml::state<parsed>
          + sml::event<event::parse_runtime> / action::begin_parse
      , sml::state<parse_outcome_dispatch> <= sml::state<uninitialized>
          + sml::event<event::parse_runtime> / action::mark_parse_invalid_request
      , sml::state<parse_outcome_dispatch> <= sml::state<probed>
          + sml::event<event::parse_runtime> / action::mark_parse_invalid_request
      , sml::state<parse_outcome_dispatch> <= sml::state<errored>
          + sml::event<event::parse_runtime> / action::mark_parse_invalid_request

      , sml::state<parse_file_image_decision> <= sml::state<parse_request_decision>
          + sml::completion<event::parse_runtime>
      , sml::state<parse_bound_storage_decision> <= sml::state<parse_file_image_decision>
          + sml::completion<event::parse_runtime> [ guard::parse_has_file_image{} ]
      , sml::state<parse_outcome_dispatch> <= sml::state<parse_file_image_decision>
          + sml::completion<event::parse_runtime> [ guard::parse_missing_file_image{} ]
          / action::mark_parse_invalid_request
      , sml::state<parse_outcome_dispatch> <= sml::state<parse_file_image_decision>
          + sml::completion<event::parse_runtime>
          / action::mark_parse_invalid_request

      , sml::state<parse_capacity_decision> <= sml::state<parse_bound_storage_decision>
          + sml::completion<event::parse_runtime> [ guard::parse_has_bound_storage{} ]
      , sml::state<parse_outcome_dispatch> <= sml::state<parse_bound_storage_decision>
          + sml::completion<event::parse_runtime> [ guard::parse_missing_bound_storage{} ]
          / action::mark_parse_invalid_request
      , sml::state<parse_outcome_dispatch> <= sml::state<parse_bound_storage_decision>
          + sml::completion<event::parse_runtime>
          / action::mark_parse_invalid_request

      , sml::state<parse_outcome_dispatch> <= sml::state<parse_capacity_decision>
          + sml::completion<event::parse_runtime> [ guard::parse_bound_capacity_sufficient{} ]
          / action::exec_parse
      , sml::state<parse_outcome_dispatch> <= sml::state<parse_capacity_decision>
          + sml::completion<event::parse_runtime> [ guard::parse_bound_capacity_insufficient{} ]
          / action::mark_parse_invalid_request
      , sml::state<parse_outcome_dispatch> <= sml::state<parse_capacity_decision>
          + sml::completion<event::parse_runtime>
          / action::mark_parse_invalid_request

      , sml::state<parsed> <= sml::state<parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::parse_phase_ok{} ]
          / action::publish_parse_done
      , sml::state<errored> <= sml::state<parse_outcome_dispatch>
          + sml::completion<event::parse_runtime> [ guard::parse_phase_failed{} ]
          / action::publish_parse_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<errored> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<probed> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bound> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<parsed> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<probe_request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<probe_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<probe_requirements_dispatch> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_request_shape_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_capacity_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<parse_request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<parse_file_image_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<parse_bound_storage_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<parse_capacity_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<parse_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::on_unexpected
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

  bool process_event(const event::probe & ev) {
    event::probe_ctx ctx{};
    event::probe_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::bind_storage & ev) {
    event::bind_ctx ctx{};
    event::bind_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::parse & ev) {
    event::parse_ctx ctx{};
    event::parse_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Loader = sm;

}  // namespace emel::gguf::loader
