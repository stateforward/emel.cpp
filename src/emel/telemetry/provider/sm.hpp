#pragma once

#include "emel/sm.hpp"
#include "emel/telemetry/provider/actions.hpp"
#include "emel/telemetry/provider/events.hpp"
#include "emel/telemetry/provider/guards.hpp"

namespace emel::telemetry::provider {

struct initialized {};
struct configure_decision {};
struct start_decision {};
struct running {};
struct publish_decision {};
struct stop_decision {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::configure> / action::run_configure =
        sml::state<configure_decision>,
      sml::state<configure_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<configure_decision> [guard::phase_ok{}] = sml::state<initialized>,

      sml::state<initialized> + sml::event<event::start> / action::run_start =
        sml::state<start_decision>,
      sml::state<start_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<start_decision> [guard::phase_ok{}] = sml::state<running>,

      sml::state<running> + sml::event<event::publish> / action::run_publish =
        sml::state<publish_decision>,
      sml::state<publish_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<publish_decision> [guard::phase_ok{}] = sml::state<running>,

      sml::state<running> + sml::event<event::stop> / action::run_stop =
        sml::state<stop_decision>,
      sml::state<initialized> + sml::event<event::stop> / action::run_stop =
        sml::state<stop_decision>,
      sml::state<errored> + sml::event<event::stop> / action::run_stop =
        sml::state<stop_decision>,
      sml::state<stop_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<stop_decision> [guard::phase_ok{}] = sml::state<initialized>,

      sml::state<initialized> + sml::event<event::reset> / action::run_reset =
        sml::state<initialized>,
      sml::state<errored> + sml::event<event::reset> / action::run_reset =
        sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<configure_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<start_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<running> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<publish_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<stop_decision> + sml::unexpected_event<sml::_> /
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

  uint64_t sessions_started() const noexcept { return context_.sessions_started; }
  uint64_t records_emitted() const noexcept { return context_.records_emitted; }
  uint64_t records_dropped() const noexcept { return context_.records_dropped; }

 private:
  action::context context_{};
};

}  // namespace emel::telemetry::provider
