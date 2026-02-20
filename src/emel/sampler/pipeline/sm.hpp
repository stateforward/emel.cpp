#pragma once

#include "emel/sampler/pipeline/actions.hpp"
#include "emel/sampler/pipeline/events.hpp"
#include "emel/sampler/pipeline/guards.hpp"
#include "emel/sm.hpp"

namespace emel::sampler::pipeline {

struct initialized {};
struct preparing_candidates {};
struct prepare_decision {};
struct sampling {};
struct sampling_decision {};
struct selecting_token {};
struct select_decision {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::sample> / action::begin_sample =
        sml::state<preparing_candidates>,

      sml::state<preparing_candidates> / action::run_prepare_candidates =
        sml::state<prepare_decision>,
      sml::state<prepare_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<prepare_decision> [guard::phase_ok_and_has_more_samplers{}] =
        sml::state<sampling>,
      sml::state<prepare_decision> [guard::phase_ok_and_no_more_samplers{}] =
        sml::state<selecting_token>,

      sml::state<sampling> / action::run_apply_sampling = sml::state<sampling_decision>,
      sml::state<sampling_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<sampling_decision> [guard::phase_ok_and_has_more_samplers{}] =
        sml::state<sampling>,
      sml::state<sampling_decision> [guard::phase_ok_and_no_more_samplers{}] =
        sml::state<selecting_token>,

      sml::state<selecting_token> / action::run_select_token = sml::state<select_decision>,
      sml::state<select_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<select_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::publish_done = sml::state<initialized>,
      sml::state<errored> / action::publish_error = sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<preparing_candidates> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<prepare_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<sampling> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<sampling_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<selecting_token> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<select_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t selected_token() const noexcept { return context_.selected_token; }

 private:
  action::context context_{};
};

}  // namespace emel::sampler::pipeline
