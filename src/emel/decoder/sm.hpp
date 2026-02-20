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
struct validate_decision {};
struct sanitizing_batch {};
struct sanitize_decision {};
struct initializing_batch {};
struct initialize_batch_decision {};
struct updating_memory_pre {};
struct update_memory_decision {};
struct preparing_memory_batch_initial {};
struct prepare_memory_batch_initial_decision {};
struct optimizing_memory {};
struct optimize_memory_decision {};
struct preparing_memory_batch_retry {};
struct prepare_memory_batch_retry_decision {};
struct reserving_output {};
struct reserve_decision {};
struct processing_ubatch {};
struct ubatch_decision {};
struct handling_ubatch_failure {};
struct rollback_decision {};
struct finalizing_outputs {};
struct finalize_decision {};
struct done {};
struct errored {};

/**
 * Decoder orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting decode intent.
 * - `validating_request`/`validate_decision`: validate token inputs before orchestration.
 * - `sanitizing_batch`/`sanitize_decision`: sanitize and auto-generate batch metadata.
 * - `initializing_batch`/`initialize_batch_decision`: compute ubatch sizes and outputs count.
 * - `updating_memory_pre`/`update_memory_decision`: update memory coordinator before batch prep.
 * - `preparing_memory_batch_initial`/`prepare_memory_batch_initial_decision`: prepare memory & kv cache.
 * - `optimizing_memory`/`optimize_memory_decision`: one-shot optimization retry for prepare failures.
 * - `preparing_memory_batch_retry`/`prepare_memory_batch_retry_decision`: retry prepare after optimize.
 * - `reserving_output`/`reserve_decision`: validate output totals before execution.
 * - `processing_ubatch`/`ubatch_decision`: run ubatch execution loop, bounded by ubatch count.
 * - `handling_ubatch_failure`/`rollback_decision`: attempt rollback after ubatch error.
 * - `finalizing_outputs`/`finalize_decision`: verify outputs and close request.
 * - `done`/`errored`: terminal outcomes, immediately return to initialized.
 *
 * Guard semantics:
 * - `valid_*`/`invalid_*` guards are pure predicates of context.
 * - `phase_*` guards observe phase error flags set by actions.
 *
 * Action side effects:
 * - Actions run bounded, allocation-free orchestration steps and update context fields.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::decode> / action::begin_decode =
          sml::state<validating_request>,

      sml::state<validating_request> [guard::invalid_token_inputs] /
          action::reject_invalid_validate = sml::state<errored>,
      sml::state<validating_request> [guard::valid_token_inputs] / action::run_validate =
          sml::state<validate_decision>,

      sml::state<validate_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<validate_decision> [guard::phase_ok] = sml::state<sanitizing_batch>,

      sml::state<sanitizing_batch> / action::run_sanitize_batch =
          sml::state<sanitize_decision>,
      sml::state<sanitize_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<sanitize_decision> [guard::phase_ok] = sml::state<initializing_batch>,

      sml::state<initializing_batch> / action::run_initialize_batch =
          sml::state<initialize_batch_decision>,
      sml::state<initialize_batch_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<initialize_batch_decision> [guard::phase_ok] =
          sml::state<updating_memory_pre>,

      sml::state<updating_memory_pre> / action::run_update_memory =
          sml::state<update_memory_decision>,
      sml::state<update_memory_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<update_memory_decision> [guard::phase_ok] =
          sml::state<preparing_memory_batch_initial>,

      sml::state<preparing_memory_batch_initial> / action::run_prepare_memory_batch =
          sml::state<prepare_memory_batch_initial_decision>,
      sml::state<prepare_memory_batch_initial_decision> [guard::phase_failed_retryable] =
          sml::state<optimizing_memory>,
      sml::state<prepare_memory_batch_initial_decision> [guard::phase_failed_permanent] =
          sml::state<errored>,
      sml::state<prepare_memory_batch_initial_decision> [guard::phase_ok] =
          sml::state<reserving_output>,

      sml::state<optimizing_memory> / action::run_optimize_memory =
          sml::state<optimize_memory_decision>,
      sml::state<optimize_memory_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<optimize_memory_decision> [guard::phase_ok] =
          sml::state<preparing_memory_batch_retry>,

      sml::state<preparing_memory_batch_retry> / action::run_prepare_memory_batch =
          sml::state<prepare_memory_batch_retry_decision>,
      sml::state<prepare_memory_batch_retry_decision> [guard::phase_failed] =
          sml::state<errored>,
      sml::state<prepare_memory_batch_retry_decision> [guard::phase_ok] =
          sml::state<reserving_output>,

      sml::state<reserving_output> [guard::invalid_outputs_total] /
          action::reject_invalid_reserve_output = sml::state<errored>,
      sml::state<reserving_output> [guard::valid_outputs_total] / action::run_reserve_output =
          sml::state<reserve_decision>,

      sml::state<reserve_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<reserve_decision> [guard::phase_ok] = sml::state<processing_ubatch>,

      sml::state<processing_ubatch> [guard::no_more_ubatches] =
          sml::state<finalizing_outputs>,
      sml::state<processing_ubatch> [guard::cannot_process_ubatch] /
          action::on_invalid_ubatch_size = sml::state<ubatch_decision>,
      sml::state<processing_ubatch> [guard::can_process_ubatch] / action::run_process_ubatch =
          sml::state<ubatch_decision>,

      sml::state<ubatch_decision> [guard::phase_failed] = sml::state<handling_ubatch_failure>,
      sml::state<ubatch_decision> [guard::phase_ok] = sml::state<processing_ubatch>,

      sml::state<handling_ubatch_failure> / action::run_rollback_ubatch =
          sml::state<rollback_decision>,
      sml::state<rollback_decision> [guard::phase_failed] / action::capture_rollback_error =
          sml::state<errored>,
      sml::state<rollback_decision> [guard::phase_ok] / action::capture_ubatch_error =
          sml::state<errored>,

      sml::state<finalizing_outputs> / action::run_finalize_outputs =
          sml::state<finalize_decision>,
      sml::state<finalize_decision> [guard::phase_failed] = sml::state<errored>,
      sml::state<finalize_decision> [guard::phase_ok] = sml::state<done>,

      sml::state<done> / action::mark_done = sml::state<initialized>,
      sml::state<errored> / action::ensure_last_error = sml::state<initialized>,

      sml::state<validating_request> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<validate_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<sanitizing_batch> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<sanitize_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<initializing_batch> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<initialize_batch_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<updating_memory_pre> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<update_memory_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<preparing_memory_batch_initial> + sml::event<event::decode> /
          action::on_unexpected = sml::state<errored>,
      sml::state<prepare_memory_batch_initial_decision> + sml::event<event::decode> /
          action::on_unexpected = sml::state<errored>,
      sml::state<optimizing_memory> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<optimize_memory_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<preparing_memory_batch_retry> + sml::event<event::decode> /
          action::on_unexpected = sml::state<errored>,
      sml::state<prepare_memory_batch_retry_decision> + sml::event<event::decode> /
          action::on_unexpected = sml::state<errored>,
      sml::state<reserving_output> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<reserve_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<processing_ubatch> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<ubatch_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<handling_ubatch_failure> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<rollback_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<finalizing_outputs> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<finalize_decision> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<done> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>,
      sml::state<errored> + sml::event<event::decode> / action::on_unexpected =
          sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::decode & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (ev.error_out != nullptr) {
      *ev.error_out = context_.last_error;
    }
    if (ev.dispatch_event != nullptr) {
      if (context_.last_error == EMEL_OK) {
        (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                               .type = events::owner_event::kind::done,
                                               .done = events::decoding_done{
                                                 .outputs = context_.outputs_total,
                                                 .error_out = ev.error_out,
                                                 .owner_sm = ev.owner_sm,
                                                 .dispatch_event = ev.dispatch_event,
                                                 .request = &ev,
                                               },
                                             });
      } else {
        (void)ev.dispatch_event(ev.owner_sm, events::owner_event{
                                               .type = events::owner_event::kind::error,
                                               .error = events::decoding_error{
                                                 .err = context_.last_error,
                                                 .error_out = ev.error_out,
                                                 .owner_sm = ev.owner_sm,
                                                 .dispatch_event = ev.dispatch_event,
                                                 .request = &ev,
                                               },
                                             });
      }
    }
    return accepted && context_.last_error == EMEL_OK;
  }

  using base_type::process_event;

  int32_t outputs_processed() const noexcept { return context_.outputs_processed; }
  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  action::context context_{};
};

}  // namespace emel::decoder
