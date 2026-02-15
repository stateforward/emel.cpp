#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"
#include "emel/kv/cache/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kv::cache {

struct initialized {};
struct preparing {};
struct prepared {};
struct applying_ubatch {};
struct rolling_back {};
struct publishing {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::prepare> / action::begin_prepare =
          sml::state<preparing>,
      sml::state<prepared> + sml::event<event::prepare> / action::begin_prepare =
          sml::state<preparing>,

      sml::state<preparing> + sml::event<event::validate> / action::run_validate =
          sml::state<preparing>,
      sml::state<preparing> + sml::event<events::validate_done> = sml::state<preparing>,
      sml::state<preparing> + sml::event<events::validate_error> = sml::state<errored>,
      sml::state<preparing> + sml::event<event::prepare_slots> / action::run_prepare_slots =
          sml::state<preparing>,
      sml::state<preparing> + sml::event<events::prepare_slots_done> = sml::state<publishing>,
      sml::state<preparing> + sml::event<events::prepare_slots_error> = sml::state<errored>,

      sml::state<prepared> + sml::event<event::apply_ubatch> / action::begin_apply =
          sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::event<event::validate> / action::run_validate =
          sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::event<events::validate_done> = sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::event<events::validate_error> = sml::state<errored>,
      sml::state<applying_ubatch> + sml::event<event::apply_step> / action::run_apply_step =
          sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::event<events::apply_done> = sml::state<publishing>,
      sml::state<applying_ubatch> + sml::event<events::apply_error> = sml::state<errored>,

      sml::state<prepared> + sml::event<event::rollback> / action::begin_rollback =
          sml::state<rolling_back>,
      sml::state<errored> + sml::event<event::rollback> / action::begin_rollback =
          sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<event::validate> / action::run_validate =
          sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::validate_done> = sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::validate_error> = sml::state<errored>,
      sml::state<rolling_back> + sml::event<event::rollback_step> / action::run_rollback_step =
          sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::rollback_done> = sml::state<publishing>,
      sml::state<rolling_back> + sml::event<events::rollback_error> = sml::state<errored>,

      sml::state<publishing> + sml::event<event::publish> / action::run_publish = sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<done> + sml::event<events::kv_done> / action::on_kv_done = sml::state<prepared>,
      sml::state<errored> + sml::event<events::kv_error> / action::on_kv_error = sml::state<prepared>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::prepare & ev) {
    if (!base_type::process_event(ev)) return false;
    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<
            event::prepare_slots,
            events::prepare_slots_done,
            events::prepare_slots_error>(phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<event::publish, events::publish_done, events::publish_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error);
    }
    return base_type::process_event(events::kv_done{});
  }

  bool process_event(const event::apply_ubatch & ev) {
    if (!base_type::process_event(ev)) return false;
    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<event::apply_step, events::apply_done, events::apply_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error);
    }
    if (!run_phase<event::publish, events::publish_done, events::publish_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error);
    }
    return base_type::process_event(events::kv_done{});
  }

  bool process_event(const event::rollback & ev) {
    if (!base_type::process_event(ev)) return false;
    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<
            event::rollback_step,
            events::rollback_done,
            events::rollback_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error);
    }
    if (!run_phase<event::publish, events::publish_done, events::publish_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_error(phase_error);
    }
    return base_type::process_event(events::kv_done{});
  }

 private:
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(int32_t & error_out) {
    error_out = EMEL_OK;
    TriggerEvent trigger{};
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {  // GCOVR_EXCL_BR_LINE
      error_out = EMEL_ERR_BACKEND;
      return false;
    }
    if (error_out == EMEL_OK) {  // GCOVR_EXCL_BR_LINE
      return base_type::process_event(DoneEvent{});
    }
    (void)base_type::process_event(ErrorEvent{
      .err = error_out,
    });
    return false;
  }

  bool finalize_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::kv_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::kv::cache
