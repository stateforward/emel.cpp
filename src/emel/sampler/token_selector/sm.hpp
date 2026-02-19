#pragma once

#include "emel/sampler/token_selector/actions.hpp"
#include "emel/sampler/token_selector/events.hpp"
#include "emel/sampler/token_selector/guards.hpp"
#include "emel/sm.hpp"

namespace emel::sampler::token_selector {

struct initialized {};
struct validating {};
struct validate_decision {};
struct selecting_token {};
struct select_decision {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    const auto not_anonymous = [](const auto & ev) {
      using event_type = std::decay_t<decltype(ev)>;
      return !std::is_same_v<event_type, boost::sml::anonymous>;
    };
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::select_token> / action::begin_select_token =
        sml::state<validating>,

      sml::state<validating> / action::run_validate = sml::state<validate_decision>,
      sml::state<validate_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<validate_decision> [guard::phase_ok{}] = sml::state<selecting_token>,

      sml::state<selecting_token> / action::run_select = sml::state<select_decision>,
      sml::state<select_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<select_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::publish_done = sml::state<initialized>,
      sml::state<errored> / action::publish_error = sml::state<initialized>,

      sml::state<initialized> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<validating> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<validate_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<selecting_token> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<select_decision> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::event<sml::_> [not_anonymous] /
        action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::event<sml::_> [not_anonymous] /
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

}  // namespace emel::sampler::token_selector
