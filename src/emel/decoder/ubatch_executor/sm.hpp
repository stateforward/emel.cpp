#pragma once

#include <cstdint>

#include "emel/decoder/ubatch_executor/actions.hpp"
#include "emel/decoder/ubatch_executor/events.hpp"
#include "emel/decoder/ubatch_executor/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::decoder::ubatch_executor {

struct initialized {};
struct validating {};
struct preparing_memory {};
struct preparing_kv {};
struct running_compute {};
struct extracting_outputs {};
struct rolling_back {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::execute> / action::begin_execute =
          sml::state<validating>,

      sml::state<validating> + sml::event<event::validate> / action::run_validate =
          sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> = sml::state<preparing_memory>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<preparing_memory> + sml::event<event::prepare_memory> / action::run_prepare_memory =
          sml::state<preparing_memory>,
      sml::state<preparing_memory> + sml::event<events::prepare_memory_done> =
          sml::state<preparing_kv>,
      sml::state<preparing_memory> + sml::event<events::prepare_memory_error> =
          sml::state<errored>,

      sml::state<preparing_kv> + sml::event<event::prepare_kv> / action::run_prepare_kv =
          sml::state<preparing_kv>,
      sml::state<preparing_kv> + sml::event<events::prepare_kv_done> =
          sml::state<running_compute>,
      sml::state<preparing_kv> + sml::event<events::prepare_kv_error> = sml::state<errored>,

      sml::state<running_compute> + sml::event<event::run_compute> / action::run_compute =
          sml::state<running_compute>,
      sml::state<running_compute> + sml::event<events::run_compute_done> =
          sml::state<extracting_outputs>,
      sml::state<running_compute> + sml::event<events::run_compute_error> =
          sml::state<rolling_back>,

      sml::state<extracting_outputs> + sml::event<event::extract_outputs> /
          action::run_extract_outputs = sml::state<extracting_outputs>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_done> =
          sml::state<done>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_error> =
          sml::state<rolling_back>,

      sml::state<rolling_back> + sml::event<event::rollback> / action::run_rollback =
          sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::rollback_done> = sml::state<errored>,
      sml::state<rolling_back> + sml::event<events::rollback_error> = sml::state<errored>,

      sml::state<done> + sml::event<events::ubatch_execution_done> / action::on_ubatch_execution_done =
          sml::state<initialized>,
      sml::state<errored> + sml::event<events::ubatch_execution_error> /
          action::on_ubatch_execution_error = sml::state<initialized>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::execute & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<
            event::prepare_memory,
            events::prepare_memory_done,
            events::prepare_memory_error>(phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<event::prepare_kv, events::prepare_kv_done, events::prepare_kv_error>(
            phase_error)) {
      return finalize_error(phase_error);  // GCOVR_EXCL_LINE
    }

    if (!run_phase<event::run_compute, events::run_compute_done, events::run_compute_error>(
            phase_error)) {
      const int32_t compute_err = phase_error;
      (void)run_phase<event::rollback, events::rollback_done, events::rollback_error>(phase_error);
      return finalize_error(compute_err);
    }

    if (!run_phase<
            event::extract_outputs,
            events::extract_outputs_done,
            events::extract_outputs_error>(phase_error)) {
      const int32_t extract_err = phase_error;  // GCOVR_EXCL_LINE
      (void)run_phase<event::rollback, events::rollback_done, events::rollback_error>(phase_error);  // GCOVR_EXCL_LINE
      return finalize_error(extract_err);  // GCOVR_EXCL_LINE
    }

    return base_type::process_event(events::ubatch_execution_done{
      .outputs_produced = context_.outputs_produced,
      .kv_tokens = context_.kv_tokens,
    });
  }

  int32_t status_code() const noexcept { return context_.status_code; }
  int32_t outputs_produced() const noexcept { return context_.outputs_produced; }
  int32_t kv_tokens() const noexcept { return context_.kv_tokens; }
  bool rollback_attempted() const noexcept { return context_.rollback_attempted; }

 private:
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(int32_t & error_out) {
    error_out = EMEL_OK;
    TriggerEvent trigger{};
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {  // GCOVR_EXCL_BR_LINE
      error_out = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
      return false;  // GCOVR_EXCL_LINE
    }
    if (error_out == EMEL_OK) {
      return base_type::process_event(DoneEvent{});
    }
    (void)base_type::process_event(ErrorEvent{
      .err = error_out,
    });
    return false;
  }

  bool finalize_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::ubatch_execution_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::decoder::ubatch_executor
