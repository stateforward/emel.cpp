#pragma once

#include "emel/graph/actions.hpp"
#include "emel/graph/errors.hpp"
#include "emel/graph/events.hpp"
#include "emel/graph/guards.hpp"
#include "emel/sm.hpp"

namespace emel::graph {

struct uninitialized {};
struct reserved {};

struct reserving {};
struct reserve_decision {};

struct assembling {};
struct assemble_decision {};

struct executing {};
struct execute_decision {};
struct compute_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Reserve request validation.
        sml::state<reserving> <= *sml::state<uninitialized> + sml::event<event::reserve_graph>
                 [ guard::valid_reserve{} ]
                 / action::begin_reserve

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::reserve_graph>
                 [ guard::invalid_reserve_with_dispatchable_output{} ]
                 / action::reject_invalid_reserve_with_dispatch

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::reserve_graph>
                 [ guard::invalid_reserve_with_output_only{} ]
                 / action::reject_invalid_reserve_with_output_only

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::reserve_graph>
                 [ guard::invalid_reserve_without_output{} ]
                 / action::reject_invalid_reserve_without_output

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::reserve_graph>
                 [ guard::valid_reserve{} ]
                 / action::reject_invalid_reserve_with_dispatch

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::reserve_graph>
                 [ guard::invalid_reserve_with_dispatchable_output{} ]
                 / action::reject_invalid_reserve_with_dispatch

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::reserve_graph>
                 [ guard::invalid_reserve_with_output_only{} ]
                 / action::reject_invalid_reserve_with_output_only

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::reserve_graph>
                 [ guard::invalid_reserve_without_output{} ]
                 / action::reject_invalid_reserve_without_output

      //------------------------------------------------------------------------------//
      // Reserve pipeline.
      , sml::state<reserve_decision> <= sml::state<reserving> + sml::completion<event::reserve_graph>
                 / action::request_reserve

      , sml::state<reserved> <= sml::state<reserve_decision> + sml::completion<event::reserve_graph>
                 [ guard::reserve_done{} ]
                 / action::dispatch_reserve_done

      , sml::state<uninitialized> <= sml::state<reserve_decision> + sml::completion<event::reserve_graph>
                 [ guard::reserve_failed{} ]
                 / action::dispatch_reserve_error

      //------------------------------------------------------------------------------//
      // Compute request validation.
      , sml::state<assembling> <= sml::state<reserved> + sml::event<event::compute_graph>
                 [ guard::valid_compute{} ]
                 / action::begin_compute

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::compute_graph>
                 [ guard::invalid_compute_with_dispatchable_output{} ]
                 / action::reject_invalid_compute_with_dispatch

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::compute_graph>
                 [ guard::invalid_compute_with_output_only{} ]
                 / action::reject_invalid_compute_with_output_only

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::compute_graph>
                 [ guard::invalid_compute_without_output{} ]
                 / action::reject_invalid_compute_without_output

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::compute_graph>
                 [ guard::valid_compute{} ]
                 / action::reject_invalid_compute_with_dispatch

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::compute_graph>
                 [ guard::invalid_compute_with_dispatchable_output{} ]
                 / action::reject_invalid_compute_with_dispatch

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::compute_graph>
                 [ guard::invalid_compute_with_output_only{} ]
                 / action::reject_invalid_compute_with_output_only

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::compute_graph>
                 [ guard::invalid_compute_without_output{} ]
                 / action::reject_invalid_compute_without_output

      //------------------------------------------------------------------------------//
      // Assemble phase.
      , sml::state<assemble_decision> <= sml::state<assembling> + sml::completion<event::compute_graph>
                 / action::request_assemble

      , sml::state<executing> <= sml::state<assemble_decision> + sml::completion<event::compute_graph>
                 [ guard::assemble_done{} ]

      , sml::state<compute_decision> <= sml::state<assemble_decision> + sml::completion<event::compute_graph>
                 [ guard::assemble_failed{} ]

      //------------------------------------------------------------------------------//
      // Execute phase.
      , sml::state<execute_decision> <= sml::state<executing> + sml::completion<event::compute_graph>
                 / action::request_execute

      , sml::state<compute_decision> <= sml::state<execute_decision> + sml::completion<event::compute_graph>
                 [ guard::execute_done{} ]

      , sml::state<compute_decision> <= sml::state<execute_decision> + sml::completion<event::compute_graph>
                 [ guard::execute_failed{} ]

      //------------------------------------------------------------------------------//
      // Compute finalization.
      , sml::state<reserved> <= sml::state<compute_decision> + sml::completion<event::compute_graph>
                 [ guard::compute_phase_ok{} ]
                 / action::dispatch_compute_done

      , sml::state<reserved> <= sml::state<compute_decision> + sml::completion<event::compute_graph>
                 [ guard::compute_phase_failed{} ]
                 / action::dispatch_compute_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<reserved> <= sml::state<reserved> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<uninitialized> <= sml::state<reserving> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<uninitialized> <= sml::state<reserve_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<reserved> <= sml::state<assembling> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<reserved> <= sml::state<assemble_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<reserved> <= sml::state<executing> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<reserved> <= sml::state<execute_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<reserved> <= sml::state<compute_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm_with_context<model, action::context> {
  using base_type = emel::sm_with_context<model, action::context>;
  using base_type::base_type;
  using base_type::process_event;

  bool process_event(const event::reserve & ev) {
    event::reserve_ctx ctx{};
    event::reserve_graph evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::compute & ev) {
    event::compute_ctx ctx{};
    event::compute_graph evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};


}  // namespace emel::graph
