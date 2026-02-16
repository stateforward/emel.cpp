#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/coordinator/actions.hpp"
#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::memory::coordinator {

using Process = boost::sml::back::process<
  event::validate_update,
  event::validate_batch,
  event::validate_full,
  events::validate_done,
  events::validate_error,
  event::prepare_update_step,
  event::prepare_batch_step,
  event::prepare_full_step,
  events::prepare_done,
  events::prepare_error,
  event::apply_update_step,
  events::apply_done,
  events::apply_error,
  event::publish_update,
  event::publish_batch,
  event::publish_full,
  events::publish_done,
  events::publish_error,
  events::memory_done,
  events::memory_error>;

struct initialized {};
struct validating_update {};
struct validating_batch {};
struct validating_full {};
struct preparing_update {};
struct preparing_batch {};
struct preparing_full {};
struct applying_update {};
struct publishing_update {};
struct publishing_batch {};
struct publishing_full {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;
    auto is_prepare_success = [](const events::prepare_done & ev) noexcept {
      return ev.prepared_status == event::memory_status::success;
    };
    auto is_prepare_no_update = [](const events::prepare_done & ev) noexcept {
      return ev.prepared_status == event::memory_status::no_update;
    };

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::prepare_update> / action::begin_prepare_update =
          sml::state<validating_update>,
      sml::state<initialized> + sml::event<event::prepare_batch> / action::begin_prepare_batch =
          sml::state<validating_batch>,
      sml::state<initialized> + sml::event<event::prepare_full> / action::begin_prepare_full =
          sml::state<validating_full>,

      sml::state<validating_update> + sml::on_entry<event::prepare_update> /
          [](const event::prepare_update & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate_update validate{
              .request = &ev,
              .error_out = &phase_error,
            };
            process(validate);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::validate_error{
                .err = phase_error,
                .update_request = &ev,
              });
              return;
            }
            process(events::validate_done{
              .update_request = &ev,
            });
          },
      sml::state<validating_update> + sml::event<event::validate_update> / action::run_validate_update =
          sml::state<validating_update>,
      sml::state<validating_update> + sml::event<events::validate_done> = sml::state<preparing_update>,
      sml::state<validating_update> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<validating_batch> + sml::on_entry<event::prepare_batch> /
          [](const event::prepare_batch & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate_batch validate{
              .request = &ev,
              .error_out = &phase_error,
            };
            process(validate);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::validate_error{
                .err = phase_error,
                .batch_request = &ev,
              });
              return;
            }
            process(events::validate_done{
              .batch_request = &ev,
            });
          },
      sml::state<validating_batch> + sml::event<event::validate_batch> / action::run_validate_batch =
          sml::state<validating_batch>,
      sml::state<validating_batch> + sml::event<events::validate_done> = sml::state<preparing_batch>,
      sml::state<validating_batch> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<validating_full> + sml::on_entry<event::prepare_full> /
          [](const event::prepare_full & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate_full validate{
              .request = &ev,
              .error_out = &phase_error,
            };
            process(validate);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::validate_error{
                .err = phase_error,
                .full_request = &ev,
              });
              return;
            }
            process(events::validate_done{
              .full_request = &ev,
            });
          },
      sml::state<validating_full> + sml::event<event::validate_full> / action::run_validate_full =
          sml::state<validating_full>,
      sml::state<validating_full> + sml::event<events::validate_done> = sml::state<preparing_full>,
      sml::state<validating_full> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<preparing_update> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_update * request = ev.update_request;
            int32_t phase_error = EMEL_OK;
            event::memory_status prepared_status = event::memory_status::success;
            event::prepare_update_step step{
              .request = request,
              .prepared_status_out = &prepared_status,
              .error_out = &phase_error,
            };
            process(step);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_error{
                .err = phase_error,
                .prepared_status = event::memory_status::failed_prepare,
                .update_request = request,
              });
              return;
            }
            if (prepared_status != event::memory_status::success &&
                prepared_status != event::memory_status::no_update) {
              process(events::prepare_error{
                .err = EMEL_ERR_BACKEND,
                .prepared_status = prepared_status,
                .update_request = request,
              });
              return;
            }
            process(events::prepare_done{
              .prepared_status = prepared_status,
              .update_request = request,
            });
          },
      sml::state<preparing_update> + sml::event<event::prepare_update_step> / action::run_prepare_update_step =
          sml::state<preparing_update>,
      sml::state<preparing_update> + sml::event<events::prepare_done>[is_prepare_success] =
          sml::state<applying_update>,
      sml::state<preparing_update> + sml::event<events::prepare_done>[is_prepare_no_update] =
          sml::state<publishing_update>,
      sml::state<preparing_update> + sml::event<events::prepare_error> = sml::state<errored>,

      sml::state<preparing_batch> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_batch * request = ev.batch_request;
            int32_t phase_error = EMEL_OK;
            event::memory_status prepared_status = event::memory_status::success;
            event::prepare_batch_step step{
              .request = request,
              .prepared_status_out = &prepared_status,
              .error_out = &phase_error,
            };
            process(step);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_error{
                .err = phase_error,
                .prepared_status = event::memory_status::failed_prepare,
                .batch_request = request,
              });
              return;
            }
            process(events::prepare_done{
              .prepared_status = prepared_status,
              .batch_request = request,
            });
          },
      sml::state<preparing_batch> + sml::event<event::prepare_batch_step> / action::run_prepare_batch_step =
          sml::state<preparing_batch>,
      sml::state<preparing_batch> + sml::event<events::prepare_done> = sml::state<publishing_batch>,
      sml::state<preparing_batch> + sml::event<events::prepare_error> = sml::state<errored>,

      sml::state<preparing_full> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_full * request = ev.full_request;
            int32_t phase_error = EMEL_OK;
            event::memory_status prepared_status = event::memory_status::success;
            event::prepare_full_step step{
              .request = request,
              .prepared_status_out = &prepared_status,
              .error_out = &phase_error,
            };
            process(step);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_error{
                .err = phase_error,
                .prepared_status = event::memory_status::failed_prepare,
                .full_request = request,
              });
              return;
            }
            process(events::prepare_done{
              .prepared_status = prepared_status,
              .full_request = request,
            });
          },
      sml::state<preparing_full> + sml::event<event::prepare_full_step> / action::run_prepare_full_step =
          sml::state<preparing_full>,
      sml::state<preparing_full> + sml::event<events::prepare_done> = sml::state<publishing_full>,
      sml::state<preparing_full> + sml::event<events::prepare_error> = sml::state<errored>,

      sml::state<applying_update> + sml::on_entry<events::prepare_done> /
          [](const events::prepare_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_update * request = ev.update_request;
            int32_t phase_error = EMEL_OK;
            event::apply_update_step step{
              .request = request,
              .prepared_status = ev.prepared_status,
              .error_out = &phase_error,
            };
            process(step);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::apply_error{
                .err = phase_error,
                .prepared_status = ev.prepared_status,
                .update_request = request,
              });
              return;
            }
            process(events::apply_done{
              .prepared_status = ev.prepared_status,
              .update_request = request,
            });
          },
      sml::state<applying_update> + sml::event<event::apply_update_step> / action::run_apply_update_step =
          sml::state<applying_update>,
      sml::state<applying_update> + sml::event<events::apply_done> = sml::state<publishing_update>,
      sml::state<applying_update> + sml::event<events::apply_error> = sml::state<errored>,

      sml::state<publishing_update> + sml::on_entry<events::apply_done> /
          [](const events::apply_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_update * request = ev.update_request;
            int32_t phase_error = EMEL_OK;
            event::publish_update publish{
              .request = request,
              .prepared_status = ev.prepared_status,
              .error_out = &phase_error,
            };
            process(publish);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::publish_error{
                .err = phase_error,
                .prepared_status = ev.prepared_status,
                .update_request = request,
              });
              return;
            }
            process(events::publish_done{
              .prepared_status = ev.prepared_status,
              .update_request = request,
            });
          },
      sml::state<publishing_update> + sml::on_entry<events::prepare_done> /
          [](const events::prepare_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_update * request = ev.update_request;
            int32_t phase_error = EMEL_OK;
            event::publish_update publish{
              .request = request,
              .prepared_status = ev.prepared_status,
              .error_out = &phase_error,
            };
            process(publish);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::publish_error{
                .err = phase_error,
                .prepared_status = ev.prepared_status,
                .update_request = request,
              });
              return;
            }
            process(events::publish_done{
              .prepared_status = ev.prepared_status,
              .update_request = request,
            });
          },
      sml::state<publishing_update> + sml::event<event::publish_update> / action::run_publish_update =
          sml::state<publishing_update>,
      sml::state<publishing_update> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing_update> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<publishing_batch> + sml::on_entry<events::prepare_done> /
          [](const events::prepare_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_batch * request = ev.batch_request;
            int32_t phase_error = EMEL_OK;
            event::publish_batch publish{
              .request = request,
              .prepared_status = ev.prepared_status,
              .error_out = &phase_error,
            };
            process(publish);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::publish_error{
                .err = phase_error,
                .prepared_status = ev.prepared_status,
                .batch_request = request,
              });
              return;
            }
            process(events::publish_done{
              .prepared_status = ev.prepared_status,
              .batch_request = request,
            });
          },
      sml::state<publishing_batch> + sml::event<event::publish_batch> / action::run_publish_batch =
          sml::state<publishing_batch>,
      sml::state<publishing_batch> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing_batch> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<publishing_full> + sml::on_entry<events::prepare_done> /
          [](const events::prepare_done & ev, action::context &, process_t & process) noexcept {
            const event::prepare_full * request = ev.full_request;
            int32_t phase_error = EMEL_OK;
            event::publish_full publish{
              .request = request,
              .prepared_status = ev.prepared_status,
              .error_out = &phase_error,
            };
            process(publish);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::publish_error{
                .err = phase_error,
                .prepared_status = ev.prepared_status,
                .full_request = request,
              });
              return;
            }
            process(events::publish_done{
              .prepared_status = ev.prepared_status,
              .full_request = request,
            });
          },
      sml::state<publishing_full> + sml::event<event::publish_full> / action::run_publish_full =
          sml::state<publishing_full>,
      sml::state<publishing_full> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing_full> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<done> + sml::on_entry<events::publish_done> /
          [](const events::publish_done & ev, action::context &, process_t & process) noexcept {
            process(events::memory_done{
              .status = ev.prepared_status,
              .update_request = ev.update_request,
              .batch_request = ev.batch_request,
              .full_request = ev.full_request,
            });
          },
      sml::state<done> + sml::event<events::memory_done> / action::on_memory_done =
          sml::state<initialized>,
      sml::state<done> + sml::event<events::memory_error> / action::on_memory_error =
          sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            event::memory_status status = event::memory_status::failed_prepare;
            const event::prepare_update * update_request = nullptr;
            const event::prepare_batch * batch_request = nullptr;
            const event::prepare_full * full_request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.prepared_status; }) {
              status = ev.prepared_status;
            } else if constexpr (requires { ev.status; }) {
              status = ev.status;
            }
            if constexpr (requires { ev.update_request; }) {
              update_request = ev.update_request;
            }
            if constexpr (requires { ev.batch_request; }) {
              batch_request = ev.batch_request;
            }
            if constexpr (requires { ev.full_request; }) {
              full_request = ev.full_request;
            }
            if (update_request != nullptr || batch_request != nullptr || full_request != nullptr) {
              process(events::memory_error{
                .err = err,
                .status = status,
                .update_request = update_request,
                .batch_request = batch_request,
                .full_request = full_request,
              });
            }
          },
      sml::state<errored> + sml::event<events::memory_error> / action::on_memory_error =
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

}  // namespace emel::memory::coordinator
