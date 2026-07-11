#pragma once

// benchmark: designed

#include <utility>

#include "emel/memory/streaming/actions.hpp"
#include "emel/memory/streaming/context.hpp"
#include "emel/memory/streaming/events.hpp"
#include "emel/memory/streaming/guards.hpp"
#include "emel/sm.hpp"

namespace emel::memory::streaming {

struct state_uninitialized {};
struct state_empty {};
struct state_filling {};
struct state_full {};
struct state_errored {};

/*
streaming memory architecture notes (single source of truth)

state purpose
- state_uninitialized rejects use until the injected capacity is validated.
- state_empty represents an initialized ring with no visible history.
- state_filling represents 0 < visible positions < capacity.
- state_full represents a capacity-sized rolling window.
- state_errored makes an unexpected external event observable until a valid
  initialize event recovers the actor.

control invariants
- one actor instance owns one temporal ring. Independent temporal and depformer
  caches compose as independent actors, so stream ids, active bits, ownership
  maps, free stacks, and per-slot reference actors are unnecessary here.
- actor state, rather than context flags, owns empty/filling/full behavior.
- context stores only the absolute next logical position and a cached physical
  cursor; valid length and window bounds are derived into event outputs.
- every fill, full, wrap, overflow, reset, and error choice is an explicit
  guard/transition. Actions execute only the already-selected cursor update.
- reset is O(1): stale payload bytes are excluded by the empty state and are
  overwritten on later advances, so the actor never scans or clears storage.

guard semantics and action side effects
- capacity guards validate the constructor-injected ring geometry.
- filling guards select partial versus first-full publication.
- full guards select cursor increment, cursor wrap, overflow, or invariant
  failure without action-local runtime branching.
- successful advance actions publish the write slot and visible window, then
  update both cursors. No action allocates or performs I/O.
*/
struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    using effect_advance_filling =
        action::effect_advance<action::window_mode::filling,
                               action::cursor_mode::advance>;
    using effect_advance_full =
        action::effect_advance<action::window_mode::full,
                               action::cursor_mode::advance>;
    using effect_advance_full_wrap =
        action::effect_advance<action::window_mode::full,
                               action::cursor_mode::wrap>;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialization validates injected capacity. Duplicate initialization
      // is explicit and preserves the current ring.
        sml::state<state_empty> <= *sml::state<state_uninitialized>
          + sml::event<event::initialize> [ guard::guard_configuration_valid{} ]
          / action::effect_initialize{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::initialize> [ guard::guard_configuration_invalid{} ]
          / action::effect_reject<error::invalid_configuration>{}
      , sml::state<state_empty> <= sml::state<state_empty>
          + sml::event<event::initialize>
          / action::effect_reject<error::already_initialized>{}
      , sml::state<state_filling> <= sml::state<state_filling>
          + sml::event<event::initialize>
          / action::effect_reject<error::already_initialized>{}
      , sml::state<state_full> <= sml::state<state_full>
          + sml::event<event::initialize>
          / action::effect_reject<error::already_initialized>{}
      , sml::state<state_empty> <= sml::state<state_errored>
          + sml::event<event::initialize> [ guard::guard_configuration_valid{} ]
          / action::effect_initialize{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::initialize> [ guard::guard_configuration_invalid{} ]
          / action::effect_reject<error::invalid_configuration>{}

      //------------------------------------------------------------------------------//
      // Advance from empty or partially filled rings.
      , sml::state<state_full> <= sml::state<state_empty>
          + sml::event<event::advance> [ guard::guard_capacity_one{} ]
          / effect_advance_full_wrap{}
      , sml::state<state_filling> <= sml::state<state_empty>
          + sml::event<event::advance> [ guard::guard_capacity_many{} ]
          / effect_advance_filling{}
      , sml::state<state_filling> <= sml::state<state_filling>
          + sml::event<event::advance> [ guard::guard_filling_remains_partial{} ]
          / effect_advance_filling{}
      , sml::state<state_full> <= sml::state<state_filling>
          + sml::event<event::advance> [ guard::guard_filling_becomes_full{} ]
          / effect_advance_full_wrap{}

      //------------------------------------------------------------------------------//
      // Full-ring advance explicitly selects increment, wrap, or failure.
      , sml::state<state_full> <= sml::state<state_full>
          + sml::event<event::advance>
          [ guard::guard_full_position_available_before_wrap{} ]
          / effect_advance_full{}
      , sml::state<state_full> <= sml::state<state_full>
          + sml::event<event::advance>
          [ guard::guard_full_position_available_at_wrap{} ]
          / effect_advance_full_wrap{}
      , sml::state<state_full> <= sml::state<state_full>
          + sml::event<event::advance> [ guard::guard_full_position_overflow{} ]
          / action::effect_reject<error::position_overflow>{}
      , sml::state<state_full> <= sml::state<state_full>
          + sml::event<event::advance> [ guard::guard_full_cursor_invalid{} ]
          / action::effect_reject<error::internal_error>{}

      //------------------------------------------------------------------------------//
      // O(1) reset and derived view publication.
      , sml::state<state_empty> <= sml::state<state_empty>
          + sml::event<event::reset> / action::effect_reset{}
      , sml::state<state_empty> <= sml::state<state_filling>
          + sml::event<event::reset> / action::effect_reset{}
      , sml::state<state_empty> <= sml::state<state_full>
          + sml::event<event::reset> / action::effect_reset{}
      , sml::state<state_empty> <= sml::state<state_empty>
          + sml::event<event::capture_view>
          / action::effect_capture_view<action::window_mode::empty>{}
      , sml::state<state_filling> <= sml::state<state_filling>
          + sml::event<event::capture_view>
          / action::effect_capture_view<action::window_mode::filling>{}
      , sml::state<state_full> <= sml::state<state_full>
          + sml::event<event::capture_view>
          / action::effect_capture_view<action::window_mode::full>{}

      //------------------------------------------------------------------------------//
      // Known operations before initialization are explicit failures.
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::advance>
          / action::effect_reject<error::uninitialized>{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::reset>
          / action::effect_reject<error::uninitialized>{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::capture_view>
          / action::effect_reject<error::uninitialized>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::advance>
          / action::effect_reject<error::internal_error>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::reset>
          / action::effect_reject<error::internal_error>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::capture_view>
          / action::effect_reject<error::internal_error>{}

      //------------------------------------------------------------------------------//
      // Every external event outside the public contract is explicit.
      , sml::state<state_errored> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_empty>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_filling>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_full>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  explicit sm(const dependencies &deps) : base_type(std::in_place, deps) {}
};

} // namespace emel::memory::streaming
