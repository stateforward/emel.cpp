#pragma once

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/context.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/guards.hpp"
#include "emel/batch/planner/modes/equal/sm.hpp"
#include "emel/batch/planner/modes/sequential/sm.hpp"
#include "emel/batch/planner/modes/simple/sm.hpp"
#include "emel/sm.hpp"

namespace emel::batch::planner {

struct initialized {};
struct validate_decision {};
struct normalizing_batch {};
struct mode_decision {};
struct publishing {};
struct done {};
struct invalid_request {};
struct plan_failed {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<validate_decision> <= *sml::state<initialized> + sml::event<event::request_runtime>
          / action::begin_plan
      , sml::state<normalizing_batch> <= sml::state<validate_decision>
          + sml::completion<event::request_runtime> [ guard::inputs_are_valid ]
      , sml::state<invalid_request> <= sml::state<validate_decision>
          + sml::completion<event::request_runtime> [ guard::inputs_are_invalid ]
          / action::mark_invalid_request
      //------------------------------------------------------------------------------//
      , sml::state<mode_decision> <= sml::state<normalizing_batch>
          + sml::completion<event::request_runtime> / action::normalize_batch
      //------------------------------------------------------------------------------//
      , sml::state<modes::simple::model> <= sml::state<mode_decision>
          + sml::completion<event::request_runtime> [ guard::mode_is_simple ]
      , sml::state<modes::equal::model> <= sml::state<mode_decision>
          + sml::completion<event::request_runtime> [ guard::mode_is_equal ]
      , sml::state<modes::sequential::model> <= sml::state<mode_decision>
          + sml::completion<event::request_runtime> [ guard::mode_is_seq ]
      , sml::state<invalid_request> <= sml::state<mode_decision>
          + sml::completion<event::request_runtime> [ guard::mode_is_invalid ]
          / action::mark_invalid_mode
      //------------------------------------------------------------------------------//
      , sml::state<publishing> <= sml::state<modes::simple::model>
          + sml::completion<event::request_runtime> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::simple::model>
          + sml::completion<event::request_runtime> [ guard::planning_failed ]
      , sml::state<publishing> <= sml::state<modes::equal::model>
          + sml::completion<event::request_runtime> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::equal::model>
          + sml::completion<event::request_runtime> [ guard::planning_failed ]
      , sml::state<publishing> <= sml::state<modes::sequential::model>
          + sml::completion<event::request_runtime> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::sequential::model>
          + sml::completion<event::request_runtime> [ guard::planning_failed ]
      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<plan_failed>
          + sml::completion<event::request_runtime> [ guard::plan_error_present ]
          / action::dispatch_plan_failed_with_ctx_error
      , sml::state<done> <= sml::state<plan_failed>
          + sml::completion<event::request_runtime> [ guard::plan_error_absent ]
          / action::dispatch_plan_failed_internal
      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<publishing>
          + sml::completion<event::request_runtime> / action::dispatch_done
      //------------------------------------------------------------------------------//
      , sml::state<validate_decision> <= sml::state<done> + sml::event<event::request_runtime>
          / action::begin_plan
      , sml::state<validate_decision> <= sml::state<invalid_request>
          + sml::event<event::request_runtime> / action::begin_plan
      , sml::state<validate_decision> <= sml::state<plan_failed>
          + sml::event<event::request_runtime> / action::begin_plan
      //------------------------------------------------------------------------------//
      , sml::state<initialized> <= sml::state<initialized> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<initialized> <= sml::state<validate_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<initialized> <= sml::state<normalizing_batch> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<initialized> <= sml::state<mode_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<initialized> <= sml::state<publishing> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<initialized> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<initialized> <= sml::state<invalid_request> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<initialized> <= sml::state<plan_failed> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  bool process_event(const event::request & ev) {
    event::request_ctx ctx{};
    event::request_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Planner = sm;

}  // namespace emel::batch::planner
