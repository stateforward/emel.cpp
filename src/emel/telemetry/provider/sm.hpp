#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/telemetry/provider/actions.hpp"
#include "emel/telemetry/provider/events.hpp"
#include "emel/telemetry/provider/guards.hpp"

namespace emel::telemetry::provider {

using Process = boost::sml::back::process<
  event::validate_config,
  events::configure_done,
  events::configure_error,
  event::run_start,
  events::start_done,
  events::start_error,
  event::publish_record,
  events::publish_record_done,
  events::publish_record_error,
  event::run_stop,
  events::stop_done,
  events::stop_error>;

struct initialized {};
struct configuring {};
struct starting {};
struct running {};
struct publishing {};
struct stopping {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::configure> / action::begin_configure =
          sml::state<configuring>,
      sml::state<configuring> + sml::on_entry<event::configure> /
          [](const event::configure & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate_config validate{
              .error_out = &phase_error,
            };
            process(validate);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::configure_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::configure_done{
              .request = &ev,
            });
          },
      sml::state<configuring> + sml::event<event::validate_config> / action::run_validate_config =
          sml::state<configuring>,
      sml::state<configuring> + sml::event<events::configure_done> = sml::state<initialized>,
      sml::state<configuring> + sml::event<events::configure_error> = sml::state<errored>,

      sml::state<initialized> + sml::event<event::start> / action::begin_start =
          sml::state<starting>,
      sml::state<starting> + sml::on_entry<event::start> /
          [](const event::start & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::run_start run{
              .error_out = &phase_error,
            };
            process(run);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::start_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::start_done{
              .request = &ev,
            });
          },
      sml::state<starting> + sml::event<event::run_start> / action::run_start =
          sml::state<starting>,
      sml::state<starting> + sml::event<events::start_done> = sml::state<running>,
      sml::state<starting> + sml::event<events::start_error> = sml::state<errored>,

      sml::state<running> + sml::event<event::publish> / action::begin_publish =
          sml::state<publishing>,
      sml::state<publishing> + sml::on_entry<event::publish> /
          [](const event::publish & ev, action::context & ctx, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::publish_record publish{
              .error_out = &phase_error,
            };
            process(publish);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (ev.dropped_out != nullptr) {
              *ev.dropped_out = phase_error != EMEL_OK ? true : ctx.pending_dropped;
            }
            if (phase_error != EMEL_OK) {
              process(events::publish_record_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::publish_record_done{
              .request = &ev,
            });
          },
      sml::state<publishing> + sml::event<event::publish_record> / action::run_publish_record =
          sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_record_done> = sml::state<running>,
      sml::state<publishing> + sml::event<events::publish_record_error> = sml::state<errored>,

      sml::state<running> + sml::event<event::stop> / action::begin_stop = sml::state<stopping>,
      sml::state<initialized> + sml::event<event::stop> / action::begin_stop = sml::state<stopping>,
      sml::state<errored> + sml::event<event::stop> / action::begin_stop = sml::state<stopping>,
      sml::state<stopping> + sml::on_entry<event::stop> /
          [](const event::stop & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::run_stop run{
              .error_out = &phase_error,
            };
            process(run);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::stop_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::stop_done{
              .request = &ev,
            });
          },
      sml::state<stopping> + sml::event<event::run_stop> / action::run_stop =
          sml::state<stopping>,
      sml::state<stopping> + sml::event<events::stop_done> = sml::state<initialized>,
      sml::state<stopping> + sml::event<events::stop_error> = sml::state<errored>,

      sml::state<initialized> + sml::event<event::reset> / action::run_reset =
          sml::state<initialized>,
      sml::state<errored> + sml::event<event::reset> / action::run_reset =
          sml::state<initialized>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  uint64_t sessions_started() const noexcept { return context_.sessions_started; }
  uint64_t records_emitted() const noexcept { return context_.records_emitted; }
  uint64_t records_dropped() const noexcept { return context_.records_dropped; }

 private:
  action::context context_{};
};

}  // namespace emel::telemetry::provider
