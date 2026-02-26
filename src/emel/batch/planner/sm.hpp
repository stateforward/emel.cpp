#pragma once
// benchmark: scaffold

#include <cstdint>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/guards.hpp"
#include "emel/batch/planner/modes/equal/sm.hpp"
#include "emel/batch/planner/modes/sequential/sm.hpp"
#include "emel/batch/planner/modes/simple/sm.hpp"
#include "emel/sm.hpp"

namespace emel::batch::planner {


// ready state. invariant: no active plan in progress.
struct initialized {};
// validates input payload on current request.
struct validate_decision {};
// normalizes batch sizing parameters.
struct normalizing_batch {};
// delegates planning algorithm by request mode to mode submachines.
struct mode_decision {};
// checks output capacity and finalizes results.
struct publishing {};
// terminal success state.
struct done {};
// terminal error: invalid arguments.
struct invalid_request {};
// terminal error: split computation failed.
struct plan_failed {};
// terminal error: unexpected event.
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<validate_decision> <= *sml::state<initialized> + sml::event<event::request>
          / action::begin_plan
      //------------------------------------------------------------------------------//
      , sml::state<normalizing_batch> <= sml::state<validate_decision>
          + sml::completion<event::request> [ guard::inputs_are_valid ]
      , sml::state<invalid_request> <= sml::state<validate_decision>
          + sml::completion<event::request> / action::dispatch_invalid_request_default
      //------------------------------------------------------------------------------//
      , sml::state<mode_decision> <= sml::state<normalizing_batch>
          + sml::completion<event::request> / action::normalize_batch
      //------------------------------------------------------------------------------//
      , sml::state<modes::simple::model> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_simple ]
      , sml::state<modes::equal::model> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_equal ]
      , sml::state<modes::sequential::model> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_seq ]
      , sml::state<invalid_request> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_invalid ] / action::dispatch_invalid_mode
      //------------------------------------------------------------------------------//
      , sml::state<publishing> <= sml::state<modes::simple::model>
          + sml::completion<event::request> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::simple::model>
          + sml::completion<event::request> [ guard::planning_failed ]
      , sml::state<publishing> <= sml::state<modes::equal::model>
          + sml::completion<event::request> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::equal::model>
          + sml::completion<event::request> [ guard::planning_failed ]
      , sml::state<publishing> <= sml::state<modes::sequential::model>
          + sml::completion<event::request> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::sequential::model>
          + sml::completion<event::request> [ guard::planning_failed ]
      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<publishing>
          + sml::completion<event::request> / action::dispatch_done
      //------------------------------------------------------------------------------//
      , sml::state<validate_decision> <= sml::state<done> + sml::event<event::request> / action::begin_plan
      , sml::state<validate_decision> <= sml::state<invalid_request>
          + sml::event<event::request> / action::begin_plan
      , sml::state<done> <= sml::state<plan_failed>
          + sml::completion<event::request> / action::dispatch_plan_failed_default
      , sml::state<validate_decision> <= sml::state<plan_failed>
          + sml::event<event::request> / action::begin_plan
      , sml::state<validate_decision> <= sml::state<unexpected_event>
          + sml::event<event::request> / action::begin_plan
      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<initialized> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<validate_decision> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<normalizing_batch> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<mode_decision> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<publishing> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<done> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<invalid_request> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<plan_failed> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<unexpected_event> + sml::unexpected_event<sml::_>
    );
    // clang-format on
  }
};

using sm = emel::sm_with_context<model, action::context>;
}  // namespace emel::batch::planner
