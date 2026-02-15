#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/memory/coordinator/actions.hpp"
#include "emel/memory/coordinator/events.hpp"
#include "emel/memory/coordinator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::memory::coordinator {

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

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::prepare_update> / action::begin_prepare_update =
          sml::state<validating_update>,
      sml::state<initialized> + sml::event<event::prepare_batch> / action::begin_prepare_batch =
          sml::state<validating_batch>,
      sml::state<initialized> + sml::event<event::prepare_full> / action::begin_prepare_full =
          sml::state<validating_full>,

      sml::state<validating_update> + sml::event<event::validate_update> / action::run_validate_update =
          sml::state<validating_update>,
      sml::state<validating_update> + sml::event<events::validate_done> = sml::state<preparing_update>,
      sml::state<validating_update> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<validating_batch> + sml::event<event::validate_batch> / action::run_validate_batch =
          sml::state<validating_batch>,
      sml::state<validating_batch> + sml::event<events::validate_done> = sml::state<preparing_batch>,
      sml::state<validating_batch> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<validating_full> + sml::event<event::validate_full> / action::run_validate_full =
          sml::state<validating_full>,
      sml::state<validating_full> + sml::event<events::validate_done> = sml::state<preparing_full>,
      sml::state<validating_full> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<preparing_update> + sml::event<event::prepare_update_step> / action::run_prepare_update_step =
          sml::state<preparing_update>,
      sml::state<preparing_update> + sml::event<events::prepare_done> = sml::state<applying_update>,
      sml::state<preparing_update> + sml::event<events::prepare_error> = sml::state<errored>,

      sml::state<preparing_batch> + sml::event<event::prepare_batch_step> / action::run_prepare_batch_step =
          sml::state<preparing_batch>,
      sml::state<preparing_batch> + sml::event<events::prepare_done> = sml::state<publishing_batch>,
      sml::state<preparing_batch> + sml::event<events::prepare_error> = sml::state<errored>,

      sml::state<preparing_full> + sml::event<event::prepare_full_step> / action::run_prepare_full_step =
          sml::state<preparing_full>,
      sml::state<preparing_full> + sml::event<events::prepare_done> = sml::state<publishing_full>,
      sml::state<preparing_full> + sml::event<events::prepare_error> = sml::state<errored>,

      sml::state<applying_update> + sml::event<event::apply_update_step> / action::run_apply_update_step =
          sml::state<applying_update>,
      sml::state<applying_update> + sml::event<events::apply_done> = sml::state<publishing_update>,
      sml::state<applying_update> + sml::event<events::apply_error> = sml::state<errored>,

      sml::state<publishing_update> + sml::event<event::publish_update> / action::run_publish_update =
          sml::state<publishing_update>,
      sml::state<publishing_update> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing_update> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<publishing_batch> + sml::event<event::publish_batch> / action::run_publish_batch =
          sml::state<publishing_batch>,
      sml::state<publishing_batch> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing_batch> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<publishing_full> + sml::event<event::publish_full> / action::run_publish_full =
          sml::state<publishing_full>,
      sml::state<publishing_full> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing_full> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<done> + sml::event<events::memory_done> / action::on_memory_done =
          sml::state<initialized>,
      sml::state<errored> + sml::event<events::memory_error> / action::on_memory_error =
          sml::state<initialized>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::prepare_update & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    event::memory_status prepared_status = event::memory_status::success;
    if (!run_phase(
            event::validate_update{
              .request = &ev,
            },
            events::validate_done{},
            events::validate_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }
    if (!run_phase(
            event::prepare_update_step{
              .request = &ev,
              .prepared_status_out = &prepared_status,
            },
            events::prepare_done{},
            events::prepare_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }
    if (prepared_status == event::memory_status::success) {
      if (!run_phase(
              event::apply_update_step{
                .request = &ev,
                .prepared_status = prepared_status,
              },
              events::apply_done{},
              events::apply_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
        return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
      }
    } else if (prepared_status == event::memory_status::no_update) {
      if (!base_type::process_event(events::apply_done{})) {
        return finalize_error(EMEL_ERR_BACKEND, prepared_status);  // GCOVR_EXCL_LINE
      }
    } else {  // GCOVR_EXCL_BR_LINE
      return finalize_error(EMEL_ERR_BACKEND, prepared_status);  // GCOVR_EXCL_LINE
    }
    if (!run_phase(
            event::publish_update{
              .request = &ev,
              .prepared_status = prepared_status,
            },
            events::publish_done{},
            events::publish_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }

    return base_type::process_event(events::memory_done{
      .status = prepared_status,
    });
  }

  bool process_event(const event::prepare_batch & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    event::memory_status prepared_status = event::memory_status::success;
    if (!run_phase(
            event::validate_batch{
              .request = &ev,
            },
            events::validate_done{},
            events::validate_error{},
            phase_error)) {
      return finalize_error(phase_error, prepared_status);
    }
    if (!run_phase(
            event::prepare_batch_step{
              .request = &ev,
              .prepared_status_out = &prepared_status,
            },
            events::prepare_done{},
            events::prepare_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }
    if (!run_phase(
            event::publish_batch{
              .request = &ev,
              .prepared_status = prepared_status,
            },
            events::publish_done{},
            events::publish_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }

    return base_type::process_event(events::memory_done{
      .status = prepared_status,
    });
  }

  bool process_event(const event::prepare_full & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    event::memory_status prepared_status = event::memory_status::success;
    if (!run_phase(
            event::validate_full{
              .request = &ev,
            },
            events::validate_done{},
            events::validate_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }
    if (!run_phase(
            event::prepare_full_step{
              .request = &ev,
              .prepared_status_out = &prepared_status,
            },
            events::prepare_done{},
            events::prepare_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }
    if (!run_phase(
            event::publish_full{
              .request = &ev,
              .prepared_status = prepared_status,
            },
            events::publish_done{},
            events::publish_error{},
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error, prepared_status);  // GCOVR_EXCL_LINE
    }

    return base_type::process_event(events::memory_done{
      .status = prepared_status,
    });
  }

 private:
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(
      TriggerEvent trigger, const DoneEvent & done_ev, const ErrorEvent &, int32_t & error_out) {
    error_out = EMEL_OK;
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {  // GCOVR_EXCL_BR_LINE
      error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return false;  // GCOVR_EXCL_LINE
    }
    if (error_out == EMEL_OK) {  // GCOVR_EXCL_BR_LINE
      return base_type::process_event(done_ev);
    }
    (void)base_type::process_event(events::memory_error{
      .err = error_out,
      .status = event::memory_status::failed_prepare,
    });
    return false;
  }

  bool finalize_error(const int32_t error_code, const event::memory_status status) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    return base_type::process_event(events::memory_error{
      .err = err,
      .status = status,
    });
  }

  action::context context_{};
};

}  // namespace emel::memory::coordinator
