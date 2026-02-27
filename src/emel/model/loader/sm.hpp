#pragma once

#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/context.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::loader {

struct ready {};
struct request_decision {};
struct parsing {};
struct parse_decision {};
struct loading_weights {};
struct load_decision {};
struct mapping_layers {};
struct map_layers_decision {};
struct structure_decision {};
struct validating_structure {};
struct structure_validation_decision {};
struct architecture_decision {};
struct validating_architecture {};
struct architecture_validation_decision {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<request_decision> <= *sml::state<ready> + sml::event<event::load_runtime>
          / action::begin_load

      , sml::state<parsing> <= sml::state<request_decision>
          + sml::completion<event::load_runtime> [ guard::valid_request{} ]
      , sml::state<errored> <= sml::state<request_decision>
          + sml::completion<event::load_runtime> [ guard::invalid_request{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<parse_decision> <= sml::state<parsing>
          + sml::completion<event::load_runtime> / action::run_parse

      , sml::state<loading_weights> <= sml::state<parse_decision>
          + sml::completion<event::load_runtime>
          [ guard::phase_ok_and_should_load_weights_and_can_load_weights{} ]
      , sml::state<errored> <= sml::state<parse_decision>
          + sml::completion<event::load_runtime> [ guard::phase_failed{} ]
      , sml::state<errored> <= sml::state<parse_decision>
          + sml::completion<event::load_runtime>
          [ guard::phase_ok_and_should_load_weights_and_cannot_load_weights{} ]
          / action::mark_invalid_request
      , sml::state<structure_decision> <= sml::state<parse_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_skip_load_weights{} ]

      //------------------------------------------------------------------------------//
      , sml::state<load_decision> <= sml::state<loading_weights>
          + sml::completion<event::load_runtime> / action::run_load_weights

      , sml::state<mapping_layers> <= sml::state<load_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_can_map_layers{} ]
      , sml::state<errored> <= sml::state<load_decision>
          + sml::completion<event::load_runtime> [ guard::phase_failed{} ]
      , sml::state<errored> <= sml::state<load_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_cannot_map_layers{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<map_layers_decision> <= sml::state<mapping_layers>
          + sml::completion<event::load_runtime> / action::run_map_layers

      , sml::state<structure_decision> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok{} ]
      , sml::state<errored> <= sml::state<map_layers_decision>
          + sml::completion<event::load_runtime> [ guard::phase_failed{} ]

      //------------------------------------------------------------------------------//
      , sml::state<architecture_decision> <= sml::state<structure_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_skip_validate_structure{} ]
      , sml::state<validating_structure> <= sml::state<structure_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_can_validate_structure{} ]
      , sml::state<errored> <= sml::state<structure_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_cannot_validate_structure{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<structure_validation_decision> <= sml::state<validating_structure>
          + sml::completion<event::load_runtime> / action::run_validate_structure

      , sml::state<architecture_decision> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok{} ]
      , sml::state<errored> <= sml::state<structure_validation_decision>
          + sml::completion<event::load_runtime> [ guard::phase_failed{} ]

      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<architecture_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_skip_validate_architecture{} ]
      , sml::state<validating_architecture> <= sml::state<architecture_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok_and_can_validate_architecture{} ]
      , sml::state<errored> <= sml::state<architecture_decision>
          + sml::completion<event::load_runtime>
          [ guard::phase_ok_and_cannot_validate_architecture{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<architecture_validation_decision> <= sml::state<validating_architecture>
          + sml::completion<event::load_runtime> / action::run_validate_architecture

      , sml::state<done> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::phase_ok{} ]
      , sml::state<errored> <= sml::state<architecture_validation_decision>
          + sml::completion<event::load_runtime> [ guard::phase_failed{} ]

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::load_runtime>
          [ guard::done_callback_present{} ]
          / action::publish_done
      , sml::state<ready> <= sml::state<done> + sml::completion<event::load_runtime>
          [ guard::done_callback_absent{} ]
          / action::publish_done_noop
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::load_runtime>
          [ guard::error_callback_present{} ]
          / action::publish_error
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::load_runtime>
          [ guard::error_callback_absent{} ]
          / action::publish_error_noop

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<parsing> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<parse_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<loading_weights> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<load_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<mapping_layers> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<map_layers_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<structure_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<validating_structure> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<structure_validation_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<architecture_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<validating_architecture> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<architecture_validation_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<errored> + sml::unexpected_event<sml::_>
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

  bool process_event(const event::load & ev) {
    event::load_ctx ctx{};
    event::load_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

}  // namespace emel::model::loader
