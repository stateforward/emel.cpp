#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/telemetry/exporter/actions.hpp"
#include "emel/telemetry/exporter/events.hpp"
#include "emel/telemetry/exporter/guards.hpp"

namespace emel::telemetry::exporter {

using Process = boost::sml::back::process<
  event::validate_config,
  events::configure_done,
  events::configure_error,
  event::run_start,
  events::start_done,
  events::start_error,
  event::collect_batch,
  events::collect_batch_done,
  events::collect_batch_error,
  event::flush_batch,
  events::flush_batch_done,
  events::flush_batch_error,
  event::run_backoff,
  events::backoff_done,
  events::backoff_error,
  event::run_stop,
  events::stop_done,
  events::stop_error>;

struct initialized {};
struct configuring {};
struct starting {};
struct running {};
struct collecting {};
struct flushing {};
struct backing_off {};
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

      sml::state<running> + sml::event<event::tick> / action::begin_tick = sml::state<collecting>,
      sml::state<collecting> + sml::on_entry<event::tick> /
          [](const event::tick & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::collect_batch collect{
              .error_out = &phase_error,
            };
            process(collect);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::collect_batch_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::collect_batch_done{
              .request = &ev,
            });
          },
      sml::state<collecting> + sml::event<event::collect_batch> / action::run_collect_batch =
          sml::state<collecting>,
      sml::state<collecting> +
          sml::event<events::collect_batch_done>[guard::has_batch{}] =
          sml::state<flushing>,
      sml::state<collecting> +
          sml::event<events::collect_batch_done>[guard::no_batch{}] =
          sml::state<running>,
      sml::state<collecting> + sml::event<events::collect_batch_error> = sml::state<errored>,

      sml::state<flushing> + sml::on_entry<events::collect_batch_done> /
          [](const events::collect_batch_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::flush_batch flush{
              .error_out = &phase_error,
            };
            process(flush);
            const event::tick * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::flush_batch_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::flush_batch_done{
              .request = request,
            });
          },
      sml::state<flushing> + sml::event<event::flush_batch> / action::run_flush_batch =
          sml::state<flushing>,
      sml::state<flushing> + sml::event<events::flush_batch_done> = sml::state<running>,
      sml::state<flushing> + sml::event<events::flush_batch_error> = sml::state<backing_off>,

      sml::state<backing_off> + sml::on_entry<events::flush_batch_error> /
          [](const events::flush_batch_error & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::run_backoff backoff{
              .error_out = &phase_error,
            };
            process(backoff);
            const event::tick * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = ev.err;
            }
            if (phase_error != EMEL_OK) {
              process(events::backoff_error{
                .err = ev.err,
                .request = request,
              });
              return;
            }
            process(events::backoff_done{
              .request = request,
            });
          },
      sml::state<backing_off> + sml::event<event::run_backoff> / action::run_backoff =
          sml::state<backing_off>,
      sml::state<backing_off> + sml::event<events::backoff_done> = sml::state<running>,
      sml::state<backing_off> + sml::event<events::backoff_error> = sml::state<errored>,

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

struct sm : private emel::detail::process_support<sm, Process>,
            public emel::co_sm<
                model,
                emel::policy::coroutine_scheduler<emel::policy::fifo_scheduler<>>,
                emel::policy::coroutine_allocator<emel::policy::pooled_coroutine_allocator<>>,
                Process> {
  using base_type = emel::co_sm<
      model,
      emel::policy::coroutine_scheduler<emel::policy::fifo_scheduler<>>,
      emel::policy::coroutine_allocator<emel::policy::pooled_coroutine_allocator<>>,
      Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  uint64_t flushed_records() const noexcept { return context_.flushed_records; }
  uint64_t dropped_records() const noexcept { return context_.dropped_records; }
  uint32_t backoff_count() const noexcept { return context_.backoff_count; }

 private:
  action::context context_{};
};

}  // namespace emel::telemetry::exporter
