#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/decoder/actions.hpp"
#include "emel/decoder/events.hpp"
#include "emel/decoder/guards.hpp"

namespace emel::decoder {

struct initialized {};
struct validating_request {};
struct initializing_batch {};
struct updating_memory_pre {};
struct preparing_memory_batch_initial {};
struct optimizing_memory {};
struct preparing_memory_batch_retry {};
struct reserving_output {};
struct processing_ubatch {};
struct handling_ubatch_failure {};
struct finalizing_outputs {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::decode> / action::begin_decode =
          sml::state<validating_request>,

      sml::state<validating_request> + sml::event<event::validate> / action::run_validate =
          sml::state<validating_request>,
      sml::state<validating_request> + sml::event<events::validate_done> =
          sml::state<initializing_batch>,
      sml::state<validating_request> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<initializing_batch> + sml::event<event::initialize_batch> /
          action::run_initialize_batch = sml::state<initializing_batch>,
      sml::state<initializing_batch> + sml::event<events::initialize_batch_done> =
          sml::state<updating_memory_pre>,
      sml::state<initializing_batch> + sml::event<events::initialize_batch_error> =
          sml::state<errored>,

      sml::state<updating_memory_pre> + sml::event<event::update_memory> / action::run_update_memory =
          sml::state<updating_memory_pre>,
      sml::state<updating_memory_pre> + sml::event<events::update_memory_done> =
          sml::state<preparing_memory_batch_initial>,
      sml::state<updating_memory_pre> + sml::event<events::update_memory_error> =
          sml::state<errored>,

      sml::state<preparing_memory_batch_initial> + sml::event<event::prepare_memory_batch> /
          action::run_prepare_memory_batch = sml::state<preparing_memory_batch_initial>,
      sml::state<preparing_memory_batch_initial> + sml::event<events::prepare_memory_batch_done> =
          sml::state<reserving_output>,
      sml::state<preparing_memory_batch_initial> +
          sml::event<events::prepare_memory_batch_retryable_error> = sml::state<optimizing_memory>,
      sml::state<preparing_memory_batch_initial> +
          sml::event<events::prepare_memory_batch_permanent_error> = sml::state<errored>,

      sml::state<optimizing_memory> + sml::event<event::optimize_memory> /
          action::run_optimize_memory = sml::state<optimizing_memory>,
      sml::state<optimizing_memory> + sml::event<events::optimize_memory_done> =
          sml::state<preparing_memory_batch_retry>,
      sml::state<optimizing_memory> + sml::event<events::optimize_memory_error> =
          sml::state<errored>,

      sml::state<preparing_memory_batch_retry> + sml::event<event::prepare_memory_batch> /
          action::run_prepare_memory_batch = sml::state<preparing_memory_batch_retry>,
      sml::state<preparing_memory_batch_retry> + sml::event<events::prepare_memory_batch_done> =
          sml::state<reserving_output>,
      sml::state<preparing_memory_batch_retry> +
          sml::event<events::prepare_memory_batch_retryable_error> = sml::state<errored>,
      sml::state<preparing_memory_batch_retry> +
          sml::event<events::prepare_memory_batch_permanent_error> = sml::state<errored>,

      sml::state<reserving_output> + sml::event<event::reserve_output> / action::run_reserve_output =
          sml::state<reserving_output>,
      sml::state<reserving_output> + sml::event<events::reserve_output_done> =
          sml::state<processing_ubatch>,
      sml::state<reserving_output> + sml::event<events::reserve_output_error> =
          sml::state<errored>,

      sml::state<processing_ubatch> + sml::event<event::process_ubatch> / action::run_process_ubatch =
          sml::state<processing_ubatch>,
      sml::state<processing_ubatch> + sml::event<events::ubatch_done>[guard::has_more_ubatches] =
          sml::state<processing_ubatch>,
      sml::state<processing_ubatch> + sml::event<events::ubatch_done>[guard::no_more_ubatches] =
          sml::state<finalizing_outputs>,
      sml::state<processing_ubatch> + sml::event<events::ubatch_error> =
          sml::state<handling_ubatch_failure>,

      sml::state<handling_ubatch_failure> + sml::event<event::rollback_ubatch> /
          action::run_rollback_ubatch = sml::state<handling_ubatch_failure>,
      sml::state<handling_ubatch_failure> + sml::event<events::rollback_done> =
          sml::state<errored>,
      sml::state<handling_ubatch_failure> + sml::event<events::rollback_error> =
          sml::state<errored>,

      sml::state<finalizing_outputs> + sml::event<event::finalize_outputs> /
          action::run_finalize_outputs = sml::state<finalizing_outputs>,
      sml::state<finalizing_outputs> + sml::event<events::finalize_outputs_done> =
          sml::state<done>,
      sml::state<finalizing_outputs> + sml::event<events::finalize_outputs_error> =
          sml::state<errored>,

      sml::state<done> + sml::event<events::decoding_done> / action::dispatch_decoding_done_to_owner =
          sml::state<initialized>,
      sml::state<errored> + sml::event<events::decoding_error> /
          action::dispatch_decoding_error_to_owner = sml::state<initialized>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::decode & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;

    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_decode_error(phase_error);
    }
    if (!run_phase<
            event::initialize_batch,
            events::initialize_batch_done,
            events::initialize_batch_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_decode_error(phase_error);
    }
    if (!run_phase<event::update_memory, events::update_memory_done, events::update_memory_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_decode_error(phase_error);  // GCOVR_EXCL_LINE
    }
    bool retryable_prepare_failure = false;
    if (!run_prepare_memory_batch_phase(phase_error, retryable_prepare_failure)) {
      if (!retryable_prepare_failure) {
        return finalize_decode_error(phase_error);
      }

      if (!run_phase<event::optimize_memory, events::optimize_memory_done, events::optimize_memory_error>(
              phase_error)) {  // GCOVR_EXCL_BR_LINE
        return finalize_decode_error(phase_error);  // GCOVR_EXCL_LINE
      }

      bool retryable_after_retry = false;
      if (!run_prepare_memory_batch_phase(phase_error, retryable_after_retry)) {
        return finalize_decode_error(phase_error);  // GCOVR_EXCL_LINE
      }
    }
    if (!run_phase<event::reserve_output, events::reserve_output_done, events::reserve_output_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_decode_error(phase_error);  // GCOVR_EXCL_LINE
    }

    while (context_.ubatches_processed < context_.ubatches_total) {
      if (run_phase<event::process_ubatch, events::ubatch_done, events::ubatch_error>(phase_error)) {
        continue;
      }

      (void)run_phase<event::rollback_ubatch, events::rollback_done, events::rollback_error>(  // GCOVR_EXCL_BR_LINE GCOVR_EXCL_LINE
          phase_error);
      return finalize_decode_error(phase_error);  // GCOVR_EXCL_LINE
    }

    if (!run_phase<
            event::finalize_outputs,
            events::finalize_outputs_done,
            events::finalize_outputs_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_decode_error(phase_error);  // GCOVR_EXCL_LINE
    }

    return base_type::process_event(events::decoding_done{
      .outputs = context_.outputs_total,
    });
  }

  int32_t status_code() const noexcept { return context_.status_code; }
  int32_t outputs_processed() const noexcept { return context_.outputs_processed; }

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

    if (error_out == EMEL_OK) {  // GCOVR_EXCL_BR_LINE
      return base_type::process_event(DoneEvent{});
    }

    (void)base_type::process_event(ErrorEvent{
      .err = error_out,
    });
    return false;
  }

  bool run_prepare_memory_batch_phase(int32_t & error_out, bool & retryable_error) {
    error_out = EMEL_OK;
    retryable_error = false;

    event::prepare_memory_batch trigger{};
    trigger.error_out = &error_out;
    trigger.retryable_out = &retryable_error;
    if (!base_type::process_event(trigger)) {
      error_out = EMEL_ERR_BACKEND;
      retryable_error = false;
      return false;
    }

    if (error_out == EMEL_OK) {
      return base_type::process_event(events::prepare_memory_batch_done{});
    }

    if (retryable_error) {
      (void)base_type::process_event(events::prepare_memory_batch_retryable_error{
        .err = error_out,
      });
    } else {
      (void)base_type::process_event(events::prepare_memory_batch_permanent_error{
        .err = error_out,
      });
    }
    return false;
  }

  bool finalize_decode_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::decoding_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::decoder
