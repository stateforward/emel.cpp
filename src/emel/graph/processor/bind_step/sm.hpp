#pragma once
// benchmark: designed

#include "emel/graph/processor/bind_step/actions.hpp"
#include "emel/graph/processor/bind_step/guards.hpp"
#include "emel/sm.hpp"

namespace emel::graph::processor::bind_step {

struct deciding {};
struct callback_decision {};
struct executed {};
struct execute_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<execute_failed> <= *sml::state<deciding> + sml::completion<processor::event::execute_step>
                 [ guard::phase_prefailed{} ]
                 / action::mark_failed_existing_error

      , sml::state<callback_decision> <= sml::state<deciding> + sml::completion<processor::event::execute_step>
                 [ guard::phase_request_callback{} ]
                 / action::run_callback

      , sml::state<execute_failed> <= sml::state<deciding> + sml::completion<processor::event::execute_step>
                 [ guard::phase_missing_callback{} ]
                 / action::mark_failed_invalid_request

      , sml::state<executed> <= sml::state<callback_decision> + sml::completion<processor::event::execute_step>
                 [ guard::callback_ok{} ]
                 / action::mark_done

      , sml::state<execute_failed> <= sml::state<callback_decision> +
               sml::completion<processor::event::execute_step>
                 [ guard::callback_error{} ]
                 / action::mark_failed_callback_error

      , sml::state<execute_failed> <= sml::state<callback_decision> +
               sml::completion<processor::event::execute_step>
                 [ guard::callback_failed_without_error{} ]
                 / action::mark_failed_callback_without_error

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<executed>
      , sml::X <= sml::state<execute_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<callback_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<executed> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<execute_failed> + sml::unexpected_event<sml::_>
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

}  // namespace emel::graph::processor::bind_step
