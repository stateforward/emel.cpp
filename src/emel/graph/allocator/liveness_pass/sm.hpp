#pragma once
// benchmark: designed

#include "emel/graph/allocator/liveness_pass/actions.hpp"
#include "emel/graph/allocator/liveness_pass/guards.hpp"
#include "emel/sm.hpp"

namespace emel::graph::allocator::liveness_pass {

struct deciding {};
struct allocated {};
struct allocate_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<allocated> <= *sml::state<deciding> + sml::completion<allocator::event::allocate_graph_plan>
                 [ guard::phase_done{} ]
                 / action::mark_done

      , sml::state<allocate_failed> <= sml::state<deciding> + sml::completion<allocator::event::allocate_graph_plan>
                 [ guard::phase_invalid_request{} ]
                 / action::mark_failed_invalid_request

      , sml::state<allocate_failed> <= sml::state<deciding> + sml::completion<allocator::event::allocate_graph_plan>
                 [ guard::phase_capacity_exceeded{} ]
                 / action::mark_failed_capacity

      , sml::state<allocate_failed> <= sml::state<deciding> + sml::completion<allocator::event::allocate_graph_plan>
                 [ guard::phase_unclassified_failure{} ]
                 / action::mark_failed_internal

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<allocated>
      , sml::X <= sml::state<allocate_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<allocated> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<allocate_failed> + sml::unexpected_event<sml::_>
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

}  // namespace emel::graph::allocator::liveness_pass
