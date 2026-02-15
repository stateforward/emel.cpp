#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/actions.hpp"
#include "emel/decoder/compute_executor/events.hpp"
#include "emel/decoder/compute_executor/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::decoder::compute_executor {

struct initialized {};
struct validating {};
struct binding_inputs {};
struct running_backend {};
struct extracting_outputs {};
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
      sml::state<validating> + sml::event<events::validate_done> = sml::state<binding_inputs>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<binding_inputs> + sml::event<event::bind_inputs> / action::run_bind_inputs =
          sml::state<binding_inputs>,
      sml::state<binding_inputs> + sml::event<events::bind_inputs_done> = sml::state<running_backend>,
      sml::state<binding_inputs> + sml::event<events::bind_inputs_error> = sml::state<errored>,

      sml::state<running_backend> + sml::event<event::run_backend> / action::run_backend =
          sml::state<running_backend>,
      sml::state<running_backend> + sml::event<events::run_backend_done> =
          sml::state<extracting_outputs>,
      sml::state<running_backend> + sml::event<events::run_backend_error> = sml::state<errored>,

      sml::state<extracting_outputs> + sml::event<event::extract_outputs> /
          action::run_extract_outputs = sml::state<extracting_outputs>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_done> = sml::state<done>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_error> =
          sml::state<errored>,

      sml::state<done> + sml::event<events::compute_done> / action::on_compute_done =
          sml::state<initialized>,
      sml::state<errored> + sml::event<events::compute_error> / action::on_compute_error =
          sml::state<initialized>
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
    if (!run_phase<event::bind_inputs, events::bind_inputs_done, events::bind_inputs_error>(
            phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<event::run_backend, events::run_backend_done, events::run_backend_error>(
            phase_error)) {
      return finalize_error(phase_error);
    }
    if (!run_phase<
            event::extract_outputs,
            events::extract_outputs_done,
            events::extract_outputs_error>(
            phase_error)) {
      return finalize_error(phase_error);
    }

    return base_type::process_event(events::compute_done{
      .outputs_produced = context_.outputs_produced,
    });
  }

  int32_t status_code() const noexcept { return context_.status_code; }
  int32_t outputs_produced() const noexcept { return context_.outputs_produced; }

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
    (void)base_type::process_event(events::compute_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::decoder::compute_executor
