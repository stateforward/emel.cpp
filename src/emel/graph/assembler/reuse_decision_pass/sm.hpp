#pragma once
// benchmark: scaffold

#include "emel/graph/assembler/reuse_decision_pass/actions.hpp"
#include "emel/graph/assembler/reuse_decision_pass/guards.hpp"
#include "emel/sm.hpp"

namespace emel::graph::assembler::reuse_decision_pass {

struct deciding {};
struct reuse_selected {};
struct rebuild_selected {};
struct assemble_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<reuse_selected> <= *sml::state<deciding> + sml::completion<assembler::event::assemble_graph>
                 [ guard::phase_reuse{} ]
                 / action::mark_reuse

      , sml::state<rebuild_selected> <= sml::state<deciding> + sml::completion<assembler::event::assemble_graph>
                 [ guard::phase_rebuild{} ]
                 / action::mark_rebuild

      , sml::state<assemble_failed> <= sml::state<deciding> + sml::completion<assembler::event::assemble_graph>
                 [ guard::phase_prereq_failed{} ]
                 / action::mark_failed_prereq

      , sml::state<assemble_failed> <= sml::state<deciding> + sml::completion<assembler::event::assemble_graph>
                 [ guard::phase_invalid_request{} ]
                 / action::mark_failed_invalid_request

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<reuse_selected>
      , sml::X <= sml::state<rebuild_selected>
      , sml::X <= sml::state<assemble_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<reuse_selected> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<rebuild_selected> + sml::unexpected_event<sml::_>
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

}  // namespace emel::graph::assembler::reuse_decision_pass
