#pragma once

#include "emel/sm.hpp"
#include "emel/telemetry/exporter/actions.hpp"
#include "emel/telemetry/exporter/events.hpp"
#include "emel/telemetry/exporter/guards.hpp"

namespace emel::telemetry::exporter {

struct initialized {};
struct configuring {};
struct configure_decision {};
struct starting {};
struct start_decision {};
struct running {};
struct collecting {};
struct collect_decision {};
struct flushing {};
struct flush_decision {};
struct backing_off {};
struct backoff_decision {};
struct stopping {};
struct stop_decision {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::configure> / action::begin_configure =
        sml::state<configuring>,
      sml::state<configuring> / action::run_validate_config = sml::state<configure_decision>,
      sml::state<configure_decision> [guard::phase_failed{}] / action::publish_error =
        sml::state<errored>,
      sml::state<configure_decision> [guard::phase_ok{}] / action::publish_done =
        sml::state<initialized>,

      sml::state<initialized> + sml::event<event::start> / action::begin_start =
        sml::state<starting>,
      sml::state<starting> / action::run_start = sml::state<start_decision>,
      sml::state<start_decision> [guard::phase_failed{}] / action::publish_error =
        sml::state<errored>,
      sml::state<start_decision> [guard::phase_ok{}] / action::publish_done =
        sml::state<running>,

      sml::state<running> + sml::event<event::tick> / action::begin_tick =
        sml::state<collecting>,
      sml::state<collecting> / action::run_collect_batch = sml::state<collect_decision>,
      sml::state<collect_decision> [guard::phase_failed{}] / action::publish_error =
        sml::state<errored>,
      sml::state<collect_decision> [guard::phase_ok_and_has_batch{}] =
        sml::state<flushing>,
      sml::state<collect_decision> [guard::phase_ok_and_no_batch{}] / action::publish_done =
        sml::state<running>,

      sml::state<flushing> / action::run_flush_batch = sml::state<flush_decision>,
      sml::state<flush_decision> [guard::phase_failed{}] = sml::state<backing_off>,
      sml::state<flush_decision> [guard::phase_ok{}] / action::publish_done =
        sml::state<running>,

      sml::state<backing_off> / action::run_backoff = sml::state<backoff_decision>,
      sml::state<backoff_decision> [guard::phase_failed{}] / action::publish_error =
        sml::state<errored>,
      sml::state<backoff_decision> [guard::phase_ok{}] / action::publish_error =
        sml::state<running>,

      sml::state<running> + sml::event<event::stop> / action::begin_stop =
        sml::state<stopping>,
      sml::state<initialized> + sml::event<event::stop> / action::begin_stop =
        sml::state<stopping>,
      sml::state<errored> + sml::event<event::stop> / action::begin_stop =
        sml::state<stopping>,
      sml::state<stopping> / action::run_stop = sml::state<stop_decision>,
      sml::state<stop_decision> [guard::phase_failed{}] / action::publish_error =
        sml::state<errored>,
      sml::state<stop_decision> [guard::phase_ok{}] / action::publish_done =
        sml::state<initialized>,

      sml::state<initialized> + sml::event<event::reset> / action::run_reset =
        sml::state<initialized>,
      sml::state<errored> + sml::event<event::reset> / action::run_reset =
        sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<configuring> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<configure_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<starting> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<start_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<running> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<collecting> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<collect_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<flushing> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<flush_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<backing_off> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<backoff_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<stopping> + sml::unexpected_event<sml::_> /
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

  uint64_t flushed_records() const noexcept { return context_.flushed_records; }
  uint64_t dropped_records() const noexcept { return context_.dropped_records; }
  uint32_t backoff_count() const noexcept { return context_.backoff_count; }

 private:
  action::context context_{};
};

}  // namespace emel::telemetry::exporter
