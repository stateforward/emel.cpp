#pragma once

#include <functional>

#include "emel/gbnf/sampler/accept_parser/sm.hpp"
#include "emel/gbnf/sampler/actions.hpp"
#include "emel/gbnf/sampler/events.hpp"
#include "emel/gbnf/sampler/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::sampler {

struct ready {};
struct sample_begin {};
struct sample_loop_decision {};
struct sample_candidate_prepare {};
struct sample_candidate_decision {};
struct sample_candidate_compact {};
struct sample_candidate_advance {};
struct sample_finalize_decision {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Sample request validation.
        sml::state<sample_begin> <= *sml::state<ready> + sml::event<event::sample_runtime>
                 / action::begin_sample

      , sml::state<sample_loop_decision> <= sml::state<sample_begin> + sml::completion<event::sample_runtime>
                 [ guard::valid_sample_request{} ]

      , sml::state<errored> <= sml::state<sample_begin> + sml::completion<event::sample_runtime>
                 [ guard::invalid_sample_request{} ]
                 / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      // Candidate filter loop.
      , sml::state<sample_candidate_prepare> <= sml::state<sample_loop_decision>
                 + sml::completion<event::sample_runtime>
                 [ guard::has_more_candidates{} ]

      , sml::state<sample_finalize_decision> <= sml::state<sample_loop_decision>
                 + sml::completion<event::sample_runtime>
                 [ guard::no_more_candidates{} ]

      , sml::state<accept_parser::model> <= sml::state<sample_candidate_prepare>
                 + sml::completion<event::sample_runtime>
                 / action::load_candidate_token

      , sml::state<sample_candidate_decision> <= sml::state<accept_parser::model>
                 + sml::completion<event::sample_runtime>
                 [ guard::accept_done{} ]

      , sml::state<errored> <= sml::state<accept_parser::model>
                 + sml::completion<event::sample_runtime>
                 [ guard::accept_failed{} ]

      , sml::state<sample_candidate_compact> <= sml::state<sample_candidate_decision>
                 + sml::completion<event::sample_runtime>
                 [ guard::candidate_accepted{} ]

      , sml::state<sample_candidate_advance> <= sml::state<sample_candidate_decision>
                 + sml::completion<event::sample_runtime>
                 [ guard::candidate_rejected{} ]

      , sml::state<sample_candidate_advance> <= sml::state<sample_candidate_compact>
                 + sml::completion<event::sample_runtime>
                 / action::compact_candidate

      , sml::state<sample_loop_decision> <= sml::state<sample_candidate_advance>
                 + sml::completion<event::sample_runtime>
                 / action::advance_candidate_cursor

      //------------------------------------------------------------------------------//
      // Sample finalization.
      , sml::state<done> <= sml::state<sample_finalize_decision> + sml::completion<event::sample_runtime>
                 [ guard::filtered_candidates_available{} ]

      , sml::state<errored> <= sml::state<sample_finalize_decision> + sml::completion<event::sample_runtime>
                 [ guard::no_filtered_candidates{} ]
                 / action::mark_parse_failed

      //------------------------------------------------------------------------------//
      // Dispatch completion.
      , sml::state<ready> <= sml::state<done> + sml::completion<event::sample_runtime>
                 / action::publish_done

      , sml::state<ready> <= sml::state<errored> + sml::completion<event::sample_runtime>
                 / action::publish_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<sample_begin> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_loop_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_candidate_prepare> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_candidate_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_candidate_compact> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_candidate_advance> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_finalize_decision> + sml::unexpected_event<sml::_>
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
  using base_type::visit_current_states;

  sm() = default;
  explicit sm(const action::context & ctx) : base_type(ctx) {}
  explicit sm(const emel::gbnf::grammar & grammar, const uint32_t start_rule_id = 0)
      : base_type(make_context(grammar, start_rule_id)) {}

  bool process_event(const event::sample & ev) {
    event::sample_ctx ctx{};
    event::sample_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  emel::error::type sample(int32_t & candidate_ids,
                           float & candidate_scores,
                           int32_t & candidate_count,
                           int32_t & selected_token_out) {
    emel::error::type err = emel::error::cast(error::none);
    const event::sample request{
      candidate_ids,
      candidate_scores,
      candidate_count,
      selected_token_out,
      err,
    };
    (void)process_event(request);
    return err;
  }

 private:
  static action::context make_context(const emel::gbnf::grammar & grammar,
                                      const uint32_t start_rule_id) {
    action::context ctx{};
    ctx.grammar = std::cref(grammar);
    ctx.start_rule_id = start_rule_id;
    return ctx;
  }
};

using Sampler = sm;

}  // namespace emel::gbnf::sampler
