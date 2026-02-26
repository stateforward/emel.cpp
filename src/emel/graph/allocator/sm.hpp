#pragma once

#include "emel/graph/allocator/liveness_pass/sm.hpp"
#include "emel/graph/allocator/ordering_pass/sm.hpp"
#include "emel/graph/allocator/placement_pass/sm.hpp"
#include "emel/graph/allocator/actions.hpp"
#include "emel/graph/allocator/errors.hpp"
#include "emel/graph/allocator/events.hpp"
#include "emel/graph/allocator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::graph::allocator {

struct ready {};
struct liveness_decision {};
struct ordering_decision {};
struct placement_decision {};
struct allocation_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<liveness_pass::model> <= *sml::state<ready> + sml::event<event::allocate_graph_plan>
                 [ guard::valid_allocate{} ]
                 / action::begin_allocate

      , sml::state<ready> <= sml::state<ready> + sml::event<event::allocate_graph_plan>
                 [ guard::invalid_allocate_with_dispatchable_output{} ]
                 / action::reject_invalid_allocate_with_dispatch

      , sml::state<ready> <= sml::state<ready> + sml::event<event::allocate_graph_plan>
                 [ guard::invalid_allocate_with_output_only{} ]
                 / action::reject_invalid_allocate_with_output_only

      , sml::state<ready> <= sml::state<ready> + sml::event<event::allocate_graph_plan>
                 [ guard::invalid_allocate_without_output{} ]
                 / action::reject_invalid_allocate_without_output

      //------------------------------------------------------------------------------//
      // Liveness phase.
      , sml::state<liveness_decision> <= sml::state<liveness_pass::model> +
               sml::completion<event::allocate_graph_plan>

      , sml::state<ordering_pass::model> <= sml::state<liveness_decision> +
               sml::completion<event::allocate_graph_plan>
                 [ guard::liveness_done{} ]

      , sml::state<allocation_decision> <= sml::state<liveness_decision> +
               sml::completion<event::allocate_graph_plan>
                 [ guard::liveness_failed{} ]

      //------------------------------------------------------------------------------//
      // Ordering phase.
      , sml::state<ordering_decision> <= sml::state<ordering_pass::model> +
               sml::completion<event::allocate_graph_plan>

      , sml::state<placement_pass::model> <= sml::state<ordering_decision> +
               sml::completion<event::allocate_graph_plan>
                 [ guard::ordering_done{} ]

      , sml::state<allocation_decision> <= sml::state<ordering_decision> +
               sml::completion<event::allocate_graph_plan>
                 [ guard::ordering_failed{} ]

      //------------------------------------------------------------------------------//
      // Placement phase.
      , sml::state<placement_decision> <= sml::state<placement_pass::model> +
               sml::completion<event::allocate_graph_plan>

      , sml::state<allocation_decision> <= sml::state<placement_decision> +
               sml::completion<event::allocate_graph_plan>
                 [ guard::placement_done{} ]
                 / action::commit_plan

      , sml::state<allocation_decision> <= sml::state<placement_decision> +
               sml::completion<event::allocate_graph_plan>
                 [ guard::placement_failed{} ]

      //------------------------------------------------------------------------------//
      // Finalization and callback dispatch.
      , sml::state<ready> <= sml::state<allocation_decision> + sml::completion<event::allocate_graph_plan>
                 [ guard::phase_ok{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<allocation_decision> + sml::completion<event::allocate_graph_plan>
                 [ guard::phase_failed{} ]
                 / action::dispatch_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<allocation_decision> <= sml::state<liveness_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<allocation_decision> <= sml::state<ordering_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<allocation_decision> <= sml::state<placement_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<allocation_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm_with_context<model, action::context> {
  using base_type = emel::sm_with_context<model, action::context>;
  using base_type::base_type;

  bool process_event(const event::allocate_graph & ev) {
    event::allocate_graph_ctx ctx{};
    event::allocate_graph_plan evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Allocator = sm;

}  // namespace emel::graph::allocator
