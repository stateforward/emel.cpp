#pragma once

#include "emel/graph/processor/alloc_step/sm.hpp"
#include "emel/graph/processor/actions.hpp"
#include "emel/graph/processor/kernel_step/sm.hpp"
#include "emel/graph/processor/bind_step/sm.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/graph/processor/extract_step/sm.hpp"
#include "emel/graph/processor/guards.hpp"
#include "emel/graph/processor/prepare_step/sm.hpp"
#include "emel/graph/processor/validate_step/sm.hpp"
#include "emel/sm.hpp"

namespace emel::graph::processor {

struct ready {};
struct validate_decision {};
struct prepare_decision {};
struct alloc_decision {};
struct bind_decision {};
struct kernel_decision {};
struct extract_decision {};
struct execution_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<validate_step::model> <= *sml::state<ready> + sml::event<event::execute_step>
                 [ guard::valid_execute{} ]
                 / action::begin_execute

      , sml::state<ready> <= sml::state<ready> + sml::event<event::execute_step>
                 [ guard::invalid_execute_with_dispatchable_output{} ]
                 / action::reject_invalid_execute_with_dispatch

      , sml::state<ready> <= sml::state<ready> + sml::event<event::execute_step>
                 [ guard::invalid_execute_with_output_only{} ]
                 / action::reject_invalid_execute_with_output_only

      , sml::state<ready> <= sml::state<ready> + sml::event<event::execute_step>
                 [ guard::invalid_execute_without_output{} ]
                 / action::reject_invalid_execute_without_output

      //------------------------------------------------------------------------------//
      // Validate phase.
      , sml::state<validate_decision> <= sml::state<validate_step::model> +
               sml::completion<event::execute_step>

      , sml::state<prepare_step::model> <= sml::state<validate_decision> +
               sml::completion<event::execute_step>
                 [ guard::validate_done{} ]

      , sml::state<execution_decision> <= sml::state<validate_decision> +
               sml::completion<event::execute_step>
                 [ guard::validate_failed{} ]

      //------------------------------------------------------------------------------//
      // Prepare phase.
      , sml::state<prepare_decision> <= sml::state<prepare_step::model> +
               sml::completion<event::execute_step>

      , sml::state<bind_step::model> <= sml::state<prepare_decision> +
               sml::completion<event::execute_step>
                 [ guard::prepare_done_reused{} ]

      , sml::state<alloc_step::model> <= sml::state<prepare_decision> +
               sml::completion<event::execute_step>
                 [ guard::prepare_done_needs_allocation{} ]

      , sml::state<execution_decision> <= sml::state<prepare_decision> +
               sml::completion<event::execute_step>
                 [ guard::prepare_failed{} ]

      //------------------------------------------------------------------------------//
      // Alloc phase.
      , sml::state<alloc_decision> <= sml::state<alloc_step::model> +
               sml::completion<event::execute_step>

      , sml::state<bind_step::model> <= sml::state<alloc_decision> +
               sml::completion<event::execute_step>
                 [ guard::alloc_done{} ]

      , sml::state<execution_decision> <= sml::state<alloc_decision> +
               sml::completion<event::execute_step>
                 [ guard::alloc_failed{} ]

      //------------------------------------------------------------------------------//
      // Bind phase.
      , sml::state<bind_decision> <= sml::state<bind_step::model> +
               sml::completion<event::execute_step>

      , sml::state<kernel_step::model> <= sml::state<bind_decision> +
               sml::completion<event::execute_step>
                 [ guard::bind_done{} ]

      , sml::state<execution_decision> <= sml::state<bind_decision> +
               sml::completion<event::execute_step>
                 [ guard::bind_failed{} ]

      //------------------------------------------------------------------------------//
      // Kernel phase.
      , sml::state<kernel_decision> <= sml::state<kernel_step::model> +
               sml::completion<event::execute_step>

      , sml::state<extract_step::model> <= sml::state<kernel_decision> +
               sml::completion<event::execute_step>
                 [ guard::kernel_done{} ]

      , sml::state<execution_decision> <= sml::state<kernel_decision> +
               sml::completion<event::execute_step>
                 [ guard::kernel_failed{} ]

      //------------------------------------------------------------------------------//
      // Extract phase.
      , sml::state<extract_decision> <= sml::state<extract_step::model> +
               sml::completion<event::execute_step>

      , sml::state<execution_decision> <= sml::state<extract_decision> +
               sml::completion<event::execute_step>
                 [ guard::extract_done{} ]
                 / action::commit_output

      , sml::state<execution_decision> <= sml::state<extract_decision> +
               sml::completion<event::execute_step>
                 [ guard::extract_failed{} ]

      //------------------------------------------------------------------------------//
      // Finalization and callback dispatch.
      , sml::state<ready> <= sml::state<execution_decision> + sml::completion<event::execute_step>
                 [ guard::phase_ok{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<execution_decision> + sml::completion<event::execute_step>
                 [ guard::phase_failed{} ]
                 / action::dispatch_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<execution_decision> <= sml::state<validate_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<execution_decision> <= sml::state<prepare_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<execution_decision> <= sml::state<alloc_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<execution_decision> <= sml::state<bind_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<execution_decision> <= sml::state<kernel_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<execution_decision> <= sml::state<extract_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<execution_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;

  bool process_event(const event::execute & ev) {
    event::execute_ctx ctx{};
    event::execute_step evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Processor = sm;

}  // namespace emel::graph::processor
