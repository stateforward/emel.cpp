#pragma once

#include "emel/graph/assembler/assemble_alloc_pass/actions.hpp"
#include "emel/graph/assembler/assemble_alloc_pass/guards.hpp"
#include "emel/sm.hpp"

namespace emel::graph::assembler::assemble_alloc_pass {

struct deciding {};
struct assembled {};
struct assemble_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<assembled> <= *sml::state<deciding> + sml::completion<assembler::event::assemble_graph>
                 [ guard::phase_request_allocator{} ]
                 / action::request_allocator_plan

      , sml::state<assemble_failed> <= sml::state<deciding> + sml::completion<assembler::event::assemble_graph>
                 [ guard::phase_prereq_failed{} ]
                 / action::mark_failed_prereq

      , sml::state<assemble_failed> <= sml::state<deciding> + sml::completion<assembler::event::assemble_graph>
                 [ guard::phase_invalid_request{} ]
                 / action::mark_failed_invalid_request

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<assembled>
      , sml::X <= sml::state<assemble_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<assembled> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<assemble_failed> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<unexpected_event> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : emel::sm<model> {
  using model_type = model;
};

}  // namespace emel::graph::assembler::assemble_alloc_pass
