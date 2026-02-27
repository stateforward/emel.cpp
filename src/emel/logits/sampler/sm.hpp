#pragma once

#include "emel/logits/sampler/actions.hpp"
#include "emel/logits/sampler/context.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/logits/sampler/guards.hpp"
#include "emel/sm.hpp"

namespace emel::logits::sampler {

struct ready {};
struct request_logits_decision {};
struct request_preselected_decision {};
struct preparing_candidates {};
struct apply_samplers {};
struct sample_decision {};
struct sample_call {};
struct sample_call_decision {};
struct sample_complete_decision {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<request_logits_decision> <= *sml::state<ready> + sml::event<event::sample_logits_runtime>
      , sml::state<request_preselected_decision> <= *sml::state<ready>
          + sml::event<event::sample_preselected_runtime>

      , sml::state<done> <= sml::state<request_preselected_decision>
          + sml::completion<event::sample_preselected_runtime> [ guard::preselected_token_valid{} ]
      , sml::state<errored> <= sml::state<request_preselected_decision>
          + sml::completion<event::sample_preselected_runtime> [ guard::preselected_token_invalid{} ]
          / action::mark_invalid_request

      , sml::state<preparing_candidates> <= sml::state<request_logits_decision>
          + sml::completion<event::sample_logits_runtime> [ guard::valid_request{} ]
          / action::begin_sample
      , sml::state<errored> <= sml::state<request_logits_decision>
          + sml::completion<event::sample_logits_runtime> [ guard::invalid_request{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<apply_samplers> <= sml::state<preparing_candidates>
          + sml::completion<event::sample_logits_runtime> / action::prepare_candidates

      //------------------------------------------------------------------------------//
      , sml::state<sample_decision> <= sml::state<apply_samplers>
          + sml::completion<event::sample_logits_runtime> [ guard::has_more_samplers{} ]
      , sml::state<sample_complete_decision> <= sml::state<apply_samplers>
          + sml::completion<event::sample_logits_runtime> [ guard::no_more_samplers{} ]

      , sml::state<sample_call> <= sml::state<sample_decision>
          + sml::completion<event::sample_logits_runtime> [ guard::sampler_fn_available{} ]
          / action::apply_sampler
      , sml::state<errored> <= sml::state<sample_decision>
          + sml::completion<event::sample_logits_runtime> [ guard::sampler_fn_missing{} ]
          / action::mark_invalid_request

      , sml::state<sample_call_decision> <= sml::state<sample_call>
          + sml::completion<event::sample_logits_runtime>
      , sml::state<apply_samplers> <= sml::state<sample_call_decision>
          + sml::completion<event::sample_logits_runtime>
          [ guard::sampler_call_succeeded_with_valid_candidate_count{} ]
          / action::advance_sampler_index
      , sml::state<errored> <= sml::state<sample_call_decision>
          + sml::completion<event::sample_logits_runtime>
          [ guard::sampler_call_succeeded_with_invalid_candidate_count{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<sample_call_decision>
          + sml::completion<event::sample_logits_runtime> [ guard::sampler_call_failed{} ]
          / action::mark_sampler_error

      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<sample_complete_decision>
          + sml::completion<event::sample_logits_runtime> [ guard::selected_token_valid{} ]
      , sml::state<errored> <= sml::state<sample_complete_decision>
          + sml::completion<event::sample_logits_runtime> [ guard::selected_token_missing_or_invalid{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::sample_logits_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::sample_logits_runtime>
          / action::publish_error
      , sml::state<ready> <= sml::state<done> + sml::completion<event::sample_preselected_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::sample_preselected_runtime>
          / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_logits_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_preselected_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<preparing_candidates> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<apply_samplers> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_call> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_call_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<sample_complete_decision> + sml::unexpected_event<sml::_>
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

  explicit sm(const action::context & sampler_context) : base_type() {
    this->context_ = sampler_context;
  }

  sm(fn * sampler_fns, int32_t sampler_count)
      : base_type() {
    this->context_.sampler_fns = sampler_fns;
    this->context_.sampler_count = sampler_count;
  }

  bool process_event(const event::sample_logits & ev) {
    event::sample_logits_ctx ctx{};
    event::sample_logits_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::sample_preselected & ev) {
    event::sample_preselected_ctx ctx{};
    event::sample_preselected_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

 private:
};

using Sampler = sm;

}  // namespace emel::logits::sampler
