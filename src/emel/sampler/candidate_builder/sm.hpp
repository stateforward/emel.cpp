#pragma once

#include "emel/sampler/candidate_builder/actions.hpp"
#include "emel/sampler/candidate_builder/events.hpp"
#include "emel/sampler/candidate_builder/guards.hpp"
#include "emel/sm.hpp"
#include "emel/sm.hpp"

namespace emel::sampler::candidate_builder {

struct initialized {};
struct validating {};
struct validate_decision {};
struct building_candidates {};
struct build_decision {};
struct normalizing_scores {};
struct normalize_decision {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::build> / action::begin_build =
        sml::state<validating>,

      sml::state<validating> / action::run_validate = sml::state<validate_decision>,
      sml::state<validate_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<validate_decision> [guard::phase_ok{}] = sml::state<building_candidates>,

      sml::state<building_candidates> / action::run_build_candidates =
        sml::state<build_decision>,
      sml::state<build_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<build_decision> [guard::phase_ok{}] = sml::state<normalizing_scores>,

      sml::state<normalizing_scores> / action::run_normalize_scores =
        sml::state<normalize_decision>,
      sml::state<normalize_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<normalize_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::publish_done = sml::state<initialized>,
      sml::state<errored> / action::publish_error = sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<validating> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<validate_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<building_candidates> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<build_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<normalizing_scores> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<normalize_decision> + sml::unexpected_event<sml::_> /
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

  int32_t candidate_count() const noexcept { return context_.candidate_count; }

 private:
  action::context context_{};
};

}  // namespace emel::sampler::candidate_builder
