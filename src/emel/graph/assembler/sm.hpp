#pragma once

#include "emel/graph/assembler/assemble_alloc_pass/sm.hpp"
#include "emel/graph/assembler/assemble_build_pass/sm.hpp"
#include "emel/graph/assembler/assemble_validate_pass/sm.hpp"
#include "emel/graph/assembler/actions.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/guards.hpp"
#include "emel/graph/assembler/reserve_alloc_pass/sm.hpp"
#include "emel/graph/assembler/reserve_build_pass/sm.hpp"
#include "emel/graph/assembler/reserve_validate_pass/sm.hpp"
#include "emel/graph/assembler/reuse_decision_pass/sm.hpp"
#include "emel/sm.hpp"

namespace emel::graph::assembler {

struct uninitialized {};
struct reserved {};

struct reserve_validate_decision {};
struct reserve_build_decision {};
struct reserve_alloc_decision {};
struct reserve_dispatch_decision {};

struct assemble_validate_decision {};
struct reuse_decision {};
struct assemble_build_decision {};
struct assemble_alloc_decision {};
struct assemble_dispatch_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Reserve request validation.
        sml::state<reserve_validate_pass::model> <= *sml::state<uninitialized> + sml::event<event::reserve_graph>
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
      // Reserve phase pipeline.
      , sml::state<reserve_validate_decision> <= sml::state<reserve_validate_pass::model> +
               sml::completion<event::reserve_graph>

      , sml::state<reserve_build_pass::model> <= sml::state<reserve_validate_decision> +
               sml::completion<event::reserve_graph>
                 [ guard::reserve_validate_done{} ]

      , sml::state<reserve_dispatch_decision> <= sml::state<reserve_validate_decision> +
               sml::completion<event::reserve_graph>
                 [ guard::reserve_validate_failed{} ]

      , sml::state<reserve_build_decision> <= sml::state<reserve_build_pass::model> +
               sml::completion<event::reserve_graph>

      , sml::state<reserve_alloc_pass::model> <= sml::state<reserve_build_decision> +
               sml::completion<event::reserve_graph>
                 [ guard::reserve_build_done{} ]

      , sml::state<reserve_dispatch_decision> <= sml::state<reserve_build_decision> +
               sml::completion<event::reserve_graph>
                 [ guard::reserve_build_failed{} ]

      , sml::state<reserve_alloc_decision> <= sml::state<reserve_alloc_pass::model> +
               sml::completion<event::reserve_graph>

      , sml::state<reserve_dispatch_decision> <= sml::state<reserve_alloc_decision> +
               sml::completion<event::reserve_graph>
                 [ guard::reserve_alloc_done{} ]
                 / action::commit_reserve_result

      , sml::state<reserve_dispatch_decision> <= sml::state<reserve_alloc_decision> +
               sml::completion<event::reserve_graph>
                 [ guard::reserve_alloc_failed{} ]

      , sml::state<reserved> <= sml::state<reserve_dispatch_decision> + sml::completion<event::reserve_graph>
                 [ guard::reserve_phase_ok{} ]
                 / action::dispatch_reserve_done

      , sml::state<uninitialized> <= sml::state<reserve_dispatch_decision> +
               sml::completion<event::reserve_graph>
                 [ guard::reserve_phase_failed{} ]
                 / action::dispatch_reserve_error

      //------------------------------------------------------------------------------//
      // Assemble request validation.
      , sml::state<assemble_validate_pass::model> <= sml::state<reserved> + sml::event<event::assemble_graph>
                 [ guard::valid_assemble{} ]
                 / action::begin_assemble

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::assemble_graph>
                 [ guard::invalid_assemble_with_dispatchable_output{} ]
                 / action::reject_invalid_assemble_with_dispatch

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::assemble_graph>
                 [ guard::invalid_assemble_with_output_only{} ]
                 / action::reject_invalid_assemble_with_output_only

      , sml::state<reserved> <= sml::state<reserved> + sml::event<event::assemble_graph>
                 [ guard::invalid_assemble_without_output{} ]
                 / action::reject_invalid_assemble_without_output

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::assemble_graph>
                 [ guard::valid_assemble{} ]
                 / action::reject_invalid_assemble_with_dispatch

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::assemble_graph>
                 [ guard::invalid_assemble_with_dispatchable_output{} ]
                 / action::reject_invalid_assemble_with_dispatch

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::assemble_graph>
                 [ guard::invalid_assemble_with_output_only{} ]
                 / action::reject_invalid_assemble_with_output_only

      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::event<event::assemble_graph>
                 [ guard::invalid_assemble_without_output{} ]
                 / action::reject_invalid_assemble_without_output

      //------------------------------------------------------------------------------//
      // Assemble phase pipeline.
      , sml::state<assemble_validate_decision> <= sml::state<assemble_validate_pass::model> +
               sml::completion<event::assemble_graph>

      , sml::state<reuse_decision_pass::model> <= sml::state<assemble_validate_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_validate_done{} ]

      , sml::state<assemble_dispatch_decision> <= sml::state<assemble_validate_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_validate_failed{} ]

      , sml::state<reuse_decision> <= sml::state<reuse_decision_pass::model> +
               sml::completion<event::assemble_graph>

      , sml::state<assemble_dispatch_decision> <= sml::state<reuse_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::reuse_decision_reused{} ]
                 / action::commit_assemble_reuse_result

      , sml::state<assemble_build_pass::model> <= sml::state<reuse_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::reuse_decision_rebuild{} ]

      , sml::state<assemble_dispatch_decision> <= sml::state<reuse_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::reuse_decision_failed{} ]

      , sml::state<assemble_build_decision> <= sml::state<assemble_build_pass::model> +
               sml::completion<event::assemble_graph>

      , sml::state<assemble_alloc_pass::model> <= sml::state<assemble_build_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_build_done{} ]

      , sml::state<assemble_dispatch_decision> <= sml::state<assemble_build_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_build_failed{} ]

      , sml::state<assemble_alloc_decision> <= sml::state<assemble_alloc_pass::model> +
               sml::completion<event::assemble_graph>

      , sml::state<assemble_dispatch_decision> <= sml::state<assemble_alloc_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_alloc_done{} ]
                 / action::commit_assemble_rebuild_result

      , sml::state<assemble_dispatch_decision> <= sml::state<assemble_alloc_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_alloc_failed{} ]

      , sml::state<reserved> <= sml::state<assemble_dispatch_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_phase_ok{} ]
                 / action::dispatch_assemble_done

      , sml::state<reserved> <= sml::state<assemble_dispatch_decision> +
               sml::completion<event::assemble_graph>
                 [ guard::assemble_phase_failed{} ]
                 / action::dispatch_assemble_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<uninitialized> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<reserved> <= sml::state<reserved> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<reserve_dispatch_decision> <= sml::state<reserve_validate_decision> +
               sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<reserve_dispatch_decision> <= sml::state<reserve_build_decision> +
               sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<reserve_dispatch_decision> <= sml::state<reserve_alloc_decision> +
               sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<uninitialized> <= sml::state<reserve_dispatch_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<assemble_dispatch_decision> <= sml::state<assemble_validate_decision> +
               sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<assemble_dispatch_decision> <= sml::state<reuse_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<assemble_dispatch_decision> <= sml::state<assemble_build_decision> +
               sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<assemble_dispatch_decision> <= sml::state<assemble_alloc_decision> +
               sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<reserved> <= sml::state<assemble_dispatch_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;

  bool process_event(const event::reserve & ev) {
    event::reserve_ctx ctx{};
    event::reserve_graph evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::assemble & ev) {
    event::assemble_ctx ctx{};
    event::assemble_graph evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using Assembler = sm;

}  // namespace emel::graph::assembler
