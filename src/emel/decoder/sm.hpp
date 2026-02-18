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

/**
 * Decoder orchestration model.
 *
 * State purposes:
 * - `initialized`: idle state awaiting decode intent.
 * - `validating_request`: validate token inputs before orchestration.
 * - `initializing_batch`: compute ubatch sizes and outputs count.
 * - `updating_memory_pre`: update memory coordinator before batch prep.
 * - `preparing_memory_batch_initial`/`preparing_memory_batch_retry`: prepare memory & kv cache.
 * - `optimizing_memory`: one-shot optimization retry for prepare failures.
 * - `reserving_output`: validate output totals before execution.
 * - `processing_ubatch`: run ubatch execution loop, bounded by ubatch count.
 * - `handling_ubatch_failure`: attempt rollback after ubatch error.
 * - `finalizing_outputs`: verify outputs and close request.
 * - `done`/`errored`: terminal outcomes, immediately return to initialized.
 *
 * Guard semantics:
 * - `valid_*`/`invalid_*` guards are pure predicates of context.
 * - `phase_*` guards observe phase error flags set by entry actions.
 *
 * Action side effects:
 * - Entry actions run bounded, allocation-free orchestration steps and update context fields.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::decode> / action::begin_decode =
          sml::state<validating_request>,

      sml::state<validating_request> [guard::invalid_token_inputs] /
          action::reject_invalid_validate_phase = sml::state<errored>,
      sml::state<validating_request> [guard::valid_token_inputs] /
          action::run_validate_phase = sml::state<initializing_batch>,

      sml::state<initializing_batch> + sml::on_entry<sml::_> / action::run_initialize_batch_phase,
      sml::state<initializing_batch> [guard::phase_failed] = sml::state<errored>,
      sml::state<initializing_batch> [guard::phase_ok] = sml::state<updating_memory_pre>,

      sml::state<updating_memory_pre> + sml::on_entry<sml::_> / action::run_update_memory_phase,
      sml::state<updating_memory_pre> [guard::phase_failed] = sml::state<errored>,
      sml::state<updating_memory_pre> [guard::phase_ok] =
          sml::state<preparing_memory_batch_initial>,

      sml::state<preparing_memory_batch_initial> + sml::on_entry<sml::_> /
          action::run_prepare_memory_batch_phase,
      sml::state<preparing_memory_batch_initial> [guard::phase_failed_retryable] =
          sml::state<optimizing_memory>,
      sml::state<preparing_memory_batch_initial> [guard::phase_failed_permanent] =
          sml::state<errored>,
      sml::state<preparing_memory_batch_initial> [guard::phase_ok] =
          sml::state<reserving_output>,

      sml::state<optimizing_memory> + sml::on_entry<sml::_> / action::run_optimize_memory_phase,
      sml::state<optimizing_memory> [guard::phase_failed] = sml::state<errored>,
      sml::state<optimizing_memory> [guard::phase_ok] =
          sml::state<preparing_memory_batch_retry>,

      sml::state<preparing_memory_batch_retry> + sml::on_entry<sml::_> /
          action::run_prepare_memory_batch_phase,
      sml::state<preparing_memory_batch_retry> [guard::phase_failed] = sml::state<errored>,
      sml::state<preparing_memory_batch_retry> [guard::phase_ok] = sml::state<reserving_output>,

      sml::state<reserving_output> [guard::invalid_outputs_total] /
          action::reject_invalid_reserve_output_phase = sml::state<errored>,
      sml::state<reserving_output> [guard::valid_outputs_total] /
          action::run_reserve_output_phase = sml::state<processing_ubatch>,

      sml::state<processing_ubatch> [guard::phase_failed] = sml::state<handling_ubatch_failure>,
      sml::state<processing_ubatch> [guard::no_more_ubatches] = sml::state<finalizing_outputs>,
      sml::state<processing_ubatch> [guard::cannot_process_ubatch] /
          action::on_invalid_ubatch_size_phase = sml::state<processing_ubatch>,
      sml::state<processing_ubatch> [guard::can_process_ubatch] /
          action::run_process_ubatch_phase = sml::state<processing_ubatch>,

      sml::state<handling_ubatch_failure> + sml::on_entry<sml::_> /
          action::run_rollback_ubatch_phase,
      sml::state<handling_ubatch_failure> [guard::phase_failed] /
          action::capture_rollback_error = sml::state<errored>,
      sml::state<handling_ubatch_failure> [guard::phase_ok] /
          action::capture_ubatch_error = sml::state<errored>,

      sml::state<finalizing_outputs> + sml::on_entry<sml::_> / action::run_finalize_outputs_phase,
      sml::state<finalizing_outputs> [guard::phase_failed] = sml::state<errored>,
      sml::state<finalizing_outputs> [guard::phase_ok] = sml::state<done>,

      sml::state<done> + sml::on_entry<sml::_> / action::mark_done,
      sml::state<done> [guard::always] = sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> / action::ensure_last_error,
      sml::state<errored> [guard::always] = sml::state<initialized>,

      sml::state<initialized> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<validating_request> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<initializing_batch> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<updating_memory_pre> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_memory_batch_initial> + sml::event<event::decode> /
          action::on_unexpected{} =
          sml::state<errored>,
      sml::state<optimizing_memory> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_memory_batch_retry> + sml::event<event::decode> /
          action::on_unexpected{} =
          sml::state<errored>,
      sml::state<reserving_output> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<processing_ubatch> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<handling_ubatch_failure> + sml::event<event::decode> /
          action::on_unexpected{} =
          sml::state<errored>,
      sml::state<finalizing_outputs> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<done> + sml::event<event::decode> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<errored> + sml::event<event::decode> / action::on_unexpected{} =
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
