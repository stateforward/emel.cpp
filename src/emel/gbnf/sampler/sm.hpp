#pragma once

#include "emel/gbnf/sampler/accept_parser/sm.hpp"
#include "emel/gbnf/sampler/actions.hpp"
#include "emel/gbnf/sampler/candidate_parser/sm.hpp"
#include "emel/gbnf/sampler/events.hpp"
#include "emel/gbnf/sampler/guards.hpp"
#include "emel/gbnf/sampler/matcher_parser/sm.hpp"
#include "emel/gbnf/sampler/token_parser/sm.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::sampler {

struct ready {};

struct apply_begin {};
struct apply_loop_decision {};
struct apply_result_decision {};

struct accept_begin {};
struct accept_result_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Apply request validation.
        sml::state<apply_begin> <= *sml::state<ready> + sml::event<event::apply_runtime>
                 [ guard::valid_apply{} ]
                 / action::begin_apply

      , sml::state<ready> <= sml::state<ready> + sml::event<event::apply_runtime>
                 [ guard::invalid_apply{} ]
                 / action::reject_invalid_apply

      //------------------------------------------------------------------------------//
      // Apply flow.
      , sml::state<apply_loop_decision> <= sml::state<apply_begin> + sml::completion<event::apply_runtime>
                 / action::prepare_candidate_parse

      , sml::state<apply_result_decision> <= sml::state<apply_loop_decision> + sml::completion<event::apply_runtime>
                 [ guard::phase_failed_apply{} ]

      , sml::state<candidate_parser::model> <= sml::state<apply_loop_decision> + sml::completion<event::apply_runtime>
                 [ guard::phase_ok_apply{} ]

      , sml::state<apply_result_decision> <= sml::state<candidate_parser::model> + sml::completion<event::apply_runtime>
                 [ guard::candidate_failed{} ]

      , sml::state<token_parser::model> <= sml::state<candidate_parser::model> + sml::completion<event::apply_runtime>
                 [ guard::candidate_done{} ]
                 / action::prepare_token_parse

      , sml::state<apply_result_decision> <= sml::state<token_parser::model> + sml::completion<event::apply_runtime>
                 [ guard::token_failed{} ]

      , sml::state<matcher_parser::model> <= sml::state<token_parser::model> + sml::completion<event::apply_runtime>
                 [ guard::token_done{} ]
                 / action::prepare_match_parse

      , sml::state<apply_result_decision> <= sml::state<matcher_parser::model> + sml::completion<event::apply_runtime>
                 [ guard::matcher_done{} ]

      , sml::state<apply_result_decision> <= sml::state<matcher_parser::model> + sml::completion<event::apply_runtime>
                 [ guard::matcher_failed{} ]

      , sml::state<ready> <= sml::state<apply_result_decision> + sml::completion<event::apply_runtime>
                 [ guard::phase_ok_apply{} ]
                 / action::dispatch_apply_done

      , sml::state<ready> <= sml::state<apply_result_decision> + sml::completion<event::apply_runtime>
                 [ guard::phase_failed_apply{} ]
                 / action::dispatch_apply_error

      //------------------------------------------------------------------------------//
      // Accept request validation.
      , sml::state<accept_begin> <= sml::state<ready> + sml::event<event::accept_runtime>
                 [ guard::valid_accept{} ]
                 / action::begin_accept

      , sml::state<ready> <= sml::state<ready> + sml::event<event::accept_runtime>
                 [ guard::invalid_accept{} ]
                 / action::reject_invalid_accept

      //------------------------------------------------------------------------------//
      // Accept flow.
      , sml::state<accept_result_decision> <= sml::state<accept_begin> + sml::completion<event::accept_runtime>
                 / action::prepare_accept_parse

      , sml::state<ready> <= sml::state<accept_result_decision> + sml::completion<event::accept_runtime>
                 [ guard::phase_failed_accept{} ]
                 / action::dispatch_accept_error

      , sml::state<accept_parser::model> <= sml::state<accept_result_decision> + sml::completion<event::accept_runtime>
                 [ guard::phase_ok_accept{} ]

      , sml::state<ready> <= sml::state<accept_parser::model> + sml::completion<event::accept_runtime>
                 [ guard::accept_done{} ]
                 / action::dispatch_accept_done

      , sml::state<ready> <= sml::state<accept_parser::model> + sml::completion<event::accept_runtime>
                 [ guard::accept_failed{} ]
                 / action::dispatch_accept_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<apply_result_decision> <= sml::state<apply_begin> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<apply_result_decision> <= sml::state<apply_loop_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<apply_result_decision> <= sml::state<apply_result_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<accept_result_decision> <= sml::state<accept_begin> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<accept_result_decision> <= sml::state<accept_result_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm_with_context<model, action::context> {
  using base_type = emel::sm_with_context<model, action::context>;
  using base_type::base_type;
  using base_type::process_event;

  bool process_event(const event::apply & ev) {
    event::apply_flow flow{};
    event::apply_runtime runtime{ev, flow};
    const bool accepted = base_type::process_event(runtime);
    return accepted && flow.err == emel::error::cast(error::none);
  }

  bool process_event(const event::accept & ev) {
    event::accept_flow flow{};
    event::accept_runtime runtime{ev, flow};
    const bool accepted = base_type::process_event(runtime);
    return accepted && flow.err == emel::error::cast(error::none);
  }
};

}  // namespace emel::gbnf::sampler
