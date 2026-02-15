#pragma once

#include <cstdint>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/batch/splitter/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::batch::splitter {

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

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::split> / action::begin_split = sml::state<validating>,

      sml::state<validating> + sml::event<event::validate> / action::run_validate =
          sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> = sml::state<normalizing_batch>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<normalizing_batch> + sml::event<event::normalize_batch> / action::run_normalize_batch =
          sml::state<normalizing_batch>,
      sml::state<normalizing_batch> + sml::event<events::normalize_done> = sml::state<splitting>,
      sml::state<normalizing_batch> + sml::event<events::normalize_error> = sml::state<errored>,

      sml::state<splitting> + sml::event<event::create_ubatches> / action::run_create_ubatches =
          sml::state<splitting>,
      sml::state<splitting> + sml::event<events::split_done> = sml::state<publishing>,
      sml::state<splitting> + sml::event<events::split_error> = sml::state<errored>,

      sml::state<publishing> + sml::event<event::publish> / action::run_publish = sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<done> + sml::event<events::splitting_done> / action::on_splitting_done =
          sml::state<initialized>,
      sml::state<errored> + sml::event<events::splitting_error> / action::on_splitting_error =
          sml::state<initialized>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::split & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_split_error(phase_error);
    }
    if (!run_phase<
            event::normalize_batch,
            events::normalize_done,
            events::normalize_error>(phase_error)) {
      return finalize_split_error(phase_error);
    }
    if (!run_phase<event::create_ubatches, events::split_done, events::split_error>(phase_error)) {
      return finalize_split_error(phase_error);
    }
    if (!run_phase<event::publish, events::publish_done, events::publish_error>(phase_error)) {
      return finalize_split_error(phase_error);
    }

    return base_type::process_event(events::splitting_done{
      .ubatch_count = context_.ubatch_count,
      .total_outputs = context_.total_outputs,
    });
  }

 private:
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(int32_t & error_out) {
    error_out = EMEL_OK;
    TriggerEvent trigger{};
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {
      error_out = EMEL_ERR_BACKEND;
      return false;
    }
    if (error_out == EMEL_OK) {
      return base_type::process_event(DoneEvent{});
    }
    (void)base_type::process_event(ErrorEvent{
      .err = error_out,
    });
    return false;
  }

  bool finalize_split_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::splitting_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::batch::splitter
