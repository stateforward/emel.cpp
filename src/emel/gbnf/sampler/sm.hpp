#pragma once
// benchmark: scaffold

#include <functional>

#include "emel/gbnf/sampler/actions.hpp"
#include "emel/gbnf/sampler/events.hpp"
#include "emel/gbnf/sampler/guards.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::sampler {

struct ready {};
struct request_decision {};
struct filter_candidates {};
struct finalize_decision {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<request_decision> <= *sml::state<ready> + sml::event<event::sample_runtime>
                 / action::begin_sample

      , sml::state<filter_candidates> <= sml::state<request_decision> + sml::completion<event::sample_runtime>
                 [ guard::valid_sample_request{} ]

      , sml::state<errored> <= sml::state<request_decision> + sml::completion<event::sample_runtime>
                 [ guard::invalid_sample_request{} ]
                 / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      // Candidate filtering.
      , sml::state<finalize_decision> <= sml::state<filter_candidates>
                 + sml::completion<event::sample_runtime> / action::filter_candidates

      //------------------------------------------------------------------------------//
      // Sample finalization.
      , sml::state<done> <= sml::state<finalize_decision> + sml::completion<event::sample_runtime>
                 [ guard::filtered_candidates_available{} ]

      , sml::state<errored> <= sml::state<finalize_decision> + sml::completion<event::sample_runtime>
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

      , sml::state<ready> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<filter_candidates> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<ready> <= sml::state<finalize_decision> + sml::unexpected_event<sml::_>
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

  emel::logits::sampler::fn as_logits_sampler_fn() noexcept {
    return emel::logits::sampler::fn::from<sm, &sm::sample>(this);
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

inline emel::logits::sampler::fn make_logits_sampler_fn(sm & machine) noexcept {
  return machine.as_logits_sampler_fn();
}

using Sampler = sm;

}  // namespace emel::gbnf::sampler
