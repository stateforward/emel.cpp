#pragma once

#include <cstdint>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/batch/splitter/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::batch::splitter {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::normalize_batch,
  events::normalize_done,
  events::normalize_error,
  event::create_ubatches,
  events::split_done,
  events::split_error,
  event::publish,
  events::publish_done,
  events::publish_error,
  events::splitting_done,
  events::splitting_error>;

struct initialized {};
struct validating {};
struct normalizing_batch {};
struct splitting {};
struct publishing {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::split> / action::begin_split = sml::state<validating>,
      sml::state<validating> + sml::on_entry<event::split> /
          [](const event::split & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            if (ev.ubatch_sizes_capacity < 0 ||
                (ev.ubatch_sizes_out == nullptr && ev.ubatch_sizes_capacity != 0)) {
              phase_error = EMEL_ERR_INVALID_ARGUMENT;
            }
            if (phase_error != EMEL_OK) {
              process(events::validate_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            event::validate validate{
              .error_out = &phase_error,
            };
            process(validate);
            if (phase_error != EMEL_OK) {
              process(events::validate_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::validate_done{
              .request = &ev,
            });
          },

      sml::state<validating> + sml::event<event::validate> / action::run_validate =
          sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> = sml::state<normalizing_batch>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<normalizing_batch> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::normalize_batch normalize{
              .error_out = &phase_error,
            };
            process(normalize);
            if (phase_error != EMEL_OK) {
              process(events::normalize_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::normalize_done{
              .request = ev.request,
            });
          },
      sml::state<normalizing_batch> + sml::event<event::normalize_batch> / action::run_normalize_batch =
          sml::state<normalizing_batch>,
      sml::state<normalizing_batch> + sml::event<events::normalize_done> = sml::state<splitting>,
      sml::state<normalizing_batch> + sml::event<events::normalize_error> = sml::state<errored>,

      sml::state<splitting> + sml::on_entry<events::normalize_done> /
          [](const events::normalize_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::create_ubatches create{
              .error_out = &phase_error,
            };
            process(create);
            if (phase_error != EMEL_OK) {
              process(events::split_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::split_done{
              .request = ev.request,
            });
          },
      sml::state<splitting> + sml::event<event::create_ubatches> / action::run_create_ubatches =
          sml::state<splitting>,
      sml::state<splitting> + sml::event<events::split_done> = sml::state<publishing>,
      sml::state<splitting> + sml::event<events::split_error> = sml::state<errored>,

      sml::state<publishing> + sml::on_entry<events::split_done> /
          [](const events::split_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::publish publish{
              .error_out = &phase_error,
            };
            process(publish);
            if (phase_error != EMEL_OK) {
              process(events::publish_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::publish_done{
              .request = ev.request,
            });
          },
      sml::state<publishing> + sml::event<event::publish> / action::run_publish = sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<done> + sml::on_entry<events::publish_done> /
          [](const events::publish_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::split * request = ev.request;
            int32_t err = EMEL_OK;
            if (request != nullptr && request->ubatch_sizes_out != nullptr &&
                request->ubatch_sizes_capacity < ctx.ubatch_count) {
              err = EMEL_ERR_INVALID_ARGUMENT;
            }
            if (err != EMEL_OK) {
              process(events::splitting_error{
                .err = err,
                .request = request,
              });
              return;
            }
            process(events::splitting_done{
              .request = request,
              .ubatch_count = ctx.ubatch_count,
              .total_outputs = ctx.total_outputs,
            });
          },
      sml::state<done> + sml::event<events::splitting_done> / action::on_splitting_done =
          sml::state<initialized>,
      sml::state<done> + sml::event<events::splitting_error> / action::on_splitting_error =
          sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_INVALID_ARGUMENT;
            const event::split * request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            process(events::splitting_error{
              .err = err,
              .request = request,
            });
          },
      sml::state<errored> + sml::event<events::splitting_error> / action::on_splitting_error =
          sml::state<initialized>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

 private:
  action::context context_{};
};

}  // namespace emel::batch::splitter
