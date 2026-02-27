#pragma once

#include "emel/logits/validator/actions.hpp"
#include "emel/logits/validator/context.hpp"
#include "emel/logits/validator/events.hpp"
#include "emel/logits/validator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::logits::validator {

struct ready {};
struct request_decision {};
struct preparing_candidates_begin {};
struct preparing_candidates_decision {};
struct preparing_candidates_step {};
struct preparing_candidates_advance {};
struct scanning_max_begin {};
struct scanning_max_decision {};
struct scanning_max_step {};
struct scanning_max_advance {};
struct normalizing_scores_begin {};
struct normalizing_scores_decision {};
struct normalizing_scores_step {};
struct normalizing_scores_advance {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<request_decision> <= *sml::state<ready> + sml::event<event::build_runtime>
          / action::begin_build
      , sml::state<preparing_candidates_begin> <= sml::state<request_decision>
          + sml::completion<event::build_runtime> [ guard::valid_request{} ]
      , sml::state<errored> <= sml::state<request_decision>
          + sml::completion<event::build_runtime> [ guard::invalid_request{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<preparing_candidates_decision> <= sml::state<preparing_candidates_begin>
          + sml::completion<event::build_runtime> / action::prepare_candidates_begin
      , sml::state<preparing_candidates_step> <= sml::state<preparing_candidates_decision>
          + sml::completion<event::build_runtime> [ guard::prepare_has_more{} ]
      , sml::state<scanning_max_begin> <= sml::state<preparing_candidates_decision>
          + sml::completion<event::build_runtime> [ guard::prepare_done{} ]
          / action::set_candidate_count_from_vocab

      , sml::state<preparing_candidates_advance> <= sml::state<preparing_candidates_step>
          + sml::completion<event::build_runtime> / action::prepare_candidate_step
      , sml::state<preparing_candidates_decision> <= sml::state<preparing_candidates_advance>
          + sml::completion<event::build_runtime> / action::advance_prepare_cursor

      //------------------------------------------------------------------------------//
      , sml::state<scanning_max_decision> <= sml::state<scanning_max_begin>
          + sml::completion<event::build_runtime> / action::begin_max_scan
      , sml::state<scanning_max_step> <= sml::state<scanning_max_decision>
          + sml::completion<event::build_runtime> [ guard::max_scan_has_more{} ]
      , sml::state<normalizing_scores_begin> <= sml::state<scanning_max_decision>
          + sml::completion<event::build_runtime> [ guard::max_scan_done{} ]

      , sml::state<scanning_max_advance> <= sml::state<scanning_max_step>
          + sml::completion<event::build_runtime> [ guard::current_score_exceeds_max{} ]
          / action::update_max_score
      , sml::state<scanning_max_advance> <= sml::state<scanning_max_step>
          + sml::completion<event::build_runtime> [ guard::current_score_not_exceeds_max{} ]
      , sml::state<scanning_max_decision> <= sml::state<scanning_max_advance>
          + sml::completion<event::build_runtime> / action::advance_max_cursor

      //------------------------------------------------------------------------------//
      , sml::state<normalizing_scores_decision> <= sml::state<normalizing_scores_begin>
          + sml::completion<event::build_runtime> / action::begin_normalize
      , sml::state<normalizing_scores_step> <= sml::state<normalizing_scores_decision>
          + sml::completion<event::build_runtime> [ guard::normalize_has_more{} ]
      , sml::state<done> <= sml::state<normalizing_scores_decision>
          + sml::completion<event::build_runtime> [ guard::normalize_done{} ]

      , sml::state<normalizing_scores_advance> <= sml::state<normalizing_scores_step>
          + sml::completion<event::build_runtime> / action::normalize_score_step
      , sml::state<normalizing_scores_decision> <= sml::state<normalizing_scores_advance>
          + sml::completion<event::build_runtime> / action::advance_normalize_cursor

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::build_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::build_runtime>
          / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<preparing_candidates_begin> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<preparing_candidates_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<preparing_candidates_step> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<preparing_candidates_advance> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<scanning_max_begin> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<scanning_max_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<scanning_max_step> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<scanning_max_advance> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<normalizing_scores_begin> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<normalizing_scores_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<normalizing_scores_step> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<normalizing_scores_advance> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  sm() : base_type() {}

  bool process_event(const event::build & ev) {
    event::build_ctx ctx{};
    event::build_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

 private:
};

using Validator = sm;

}  // namespace emel::logits::validator
