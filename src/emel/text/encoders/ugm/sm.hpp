#pragma once

#include <cstdint>

#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/ugm/actions.hpp"
#include "emel/text/encoders/ugm/errors.hpp"
#include "emel/text/encoders/ugm/guards.hpp"
#include "emel/sm.hpp"

namespace emel::text::encoders::ugm {

struct initialized {};
struct encode_validity_decision {};
struct encode_vocab_sync_decision {};
struct encode_precheck_decision {};
struct table_policy_decision {};
struct table_sync_exec {};
struct table_sync_result_decision {};
struct unk_resolution_decision {};
struct unk_lookup_exec {};
struct normalize_exec {};
struct normalize_result_decision {};
struct input_prepare_exec {};
struct input_prepare_result_decision {};
struct dp_forward_exec {};
struct dp_forward_result_decision {};
struct dp_backtrace_exec {};
struct dp_backtrace_result_decision {};
struct emit_exec {};
struct encode_result_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * UGM encoder orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting encode intent.
 * - `encode_validity_decision`/`encode_vocab_sync_decision`: explicit intake routing.
 * - `encode_precheck_decision`: request prechecks before phase execution.
 * - `table_policy_decision`: explicit UGM table readiness routing for non-empty text.
 * - `table_sync_exec`/`table_sync_result_decision`: explicit UGM table preparation.
 * - `unk_resolution_decision`/`unk_lookup_exec`: explicit unknown-token ID resolution.
 * - `normalize_exec`/`normalize_result_decision`: explicit normalization execution and status branch.
 * - `input_prepare_exec`/`input_prepare_result_decision`: explicit input-size and DP setup branch.
 * - `dp_forward_exec`/`dp_forward_result_decision`: explicit DP forward-pass execution status branch.
 * - `dp_backtrace_exec`/`dp_backtrace_result_decision`: explicit DP backtrace status branch.
 * - `emit_exec`/`encode_result_decision`: explicit output emission status branch.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_encode`/`invalid_encode` validate request payload shape.
 * - `vocab_changed`/`vocab_unchanged` route explicit vocabulary-sync behavior.
 * - `text_empty`/`text_non_empty` route precheck work.
 * - `tables_ready`/`tables_missing` route explicit table-policy work.
 * - `vocab_unk_present`/`vocab_unk_missing` route explicit unknown-ID resolution.
 * - `normalized_empty`/`normalized_non_empty` route explicit no-input vs DP execution.
 * - `phase_ok`/`phase_failed` route per-phase error status.
 *
 * action side effects:
 * - `begin_encode`/`begin_encode_sync_vocab` reset runtime outputs and vocabulary bindings.
 * - `sync_tables` prepares UGM tables in an explicit phase.
 * - `resolve_vocab_unk`/`lookup_unk_id` set the runtime unknown-token ID.
 * - `normalize_input`, `prepare_dp_input`, `run_dp_forward`, `run_dp_backtrace`, and `emit_tokens`
 *   execute kernels per phase.
 * - `mark_done`/`ensure_last_error` finalize runtime status.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Encode Intake
      //------------------------------------------------------------------------------//
        sml::state<encode_validity_decision> <= *sml::state<initialized>
          + sml::event<runtime::encode_runtime>
      , sml::state<encode_validity_decision> <= sml::state<done>
          + sml::event<runtime::encode_runtime>
      , sml::state<encode_validity_decision> <= sml::state<errored>
          + sml::event<runtime::encode_runtime>
      , sml::state<encode_validity_decision> <= sml::state<unexpected>
          + sml::event<runtime::encode_runtime>

      //------------------------------------------------------------------------------//
      // Encode Intake Validation
      //------------------------------------------------------------------------------//
      , sml::state<encode_vocab_sync_decision> <= sml::state<encode_validity_decision>
          + sml::completion<runtime::encode_runtime>[guard::valid_encode{}]
      , sml::state<errored> <= sml::state<encode_validity_decision>
          + sml::completion<runtime::encode_runtime>[guard::invalid_encode{}]
          / action::reject_invalid_encode
      , sml::state<errored> <= sml::state<encode_validity_decision>
          + sml::completion<runtime::encode_runtime> / action::reject_invalid_encode

      //------------------------------------------------------------------------------//
      // Encode Intake Vocab Sync
      //------------------------------------------------------------------------------//
      , sml::state<encode_precheck_decision> <= sml::state<encode_vocab_sync_decision>
          + sml::completion<runtime::encode_runtime>[guard::vocab_changed{}]
          / action::begin_encode_sync_vocab
      , sml::state<encode_precheck_decision> <= sml::state<encode_vocab_sync_decision>
          + sml::completion<runtime::encode_runtime>[guard::vocab_unchanged{}]
          / action::begin_encode
      , sml::state<errored> <= sml::state<encode_vocab_sync_decision>
          + sml::completion<runtime::encode_runtime> / action::reject_invalid_encode

      //------------------------------------------------------------------------------//
      // Encode Precheck
      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<encode_precheck_decision>
          + sml::completion<runtime::encode_runtime>[guard::text_empty{}] / action::mark_done
      , sml::state<table_policy_decision> <= sml::state<encode_precheck_decision>
          + sml::completion<runtime::encode_runtime>[guard::text_non_empty{}]
      , sml::state<errored> <= sml::state<encode_precheck_decision>
          + sml::completion<runtime::encode_runtime> / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // UGM Table Policy
      //------------------------------------------------------------------------------//
      , sml::state<table_sync_exec> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime>[guard::tables_missing{}]
      , sml::state<unk_resolution_decision> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime>[guard::tables_ready{}]
      , sml::state<errored> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime> / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // UGM Table Sync
      //------------------------------------------------------------------------------//
      , sml::state<table_sync_result_decision> <= sml::state<table_sync_exec>
          + sml::completion<runtime::encode_runtime> / action::sync_tables
      , sml::state<unk_resolution_decision> <= sml::state<table_sync_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_ok{}]
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_failed{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Unknown-Token Resolution
      //------------------------------------------------------------------------------//
      , sml::state<normalize_exec> <= sml::state<unk_resolution_decision>
          + sml::completion<runtime::encode_runtime>[guard::vocab_unk_present{}]
          / action::resolve_vocab_unk
      , sml::state<unk_lookup_exec> <= sml::state<unk_resolution_decision>
          + sml::completion<runtime::encode_runtime>[guard::vocab_unk_missing{}]
      , sml::state<normalize_exec> <= sml::state<unk_lookup_exec>
          + sml::completion<runtime::encode_runtime> / action::lookup_unk_id

      //------------------------------------------------------------------------------//
      // Normalization
      //------------------------------------------------------------------------------//
      , sml::state<normalize_result_decision> <= sml::state<normalize_exec>
          + sml::completion<runtime::encode_runtime> / action::normalize_input
      , sml::state<input_prepare_exec> <= sml::state<normalize_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_ok{}]
      , sml::state<errored> <= sml::state<normalize_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_failed{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Input Preparation
      //------------------------------------------------------------------------------//
      , sml::state<input_prepare_result_decision> <= sml::state<input_prepare_exec>
          + sml::completion<runtime::encode_runtime> / action::prepare_dp_input
      , sml::state<dp_forward_exec> <= sml::state<input_prepare_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::normalized_non_empty{}]
      , sml::state<done> <= sml::state<input_prepare_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::normalized_empty{}]
          / action::mark_done
      , sml::state<errored> <= sml::state<input_prepare_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_failed{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Dynamic Programming + Emit
      //------------------------------------------------------------------------------//
      , sml::state<dp_forward_result_decision> <= sml::state<dp_forward_exec>
          + sml::completion<runtime::encode_runtime> / action::run_dp_forward
      , sml::state<dp_backtrace_exec> <= sml::state<dp_forward_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_ok{}]
      , sml::state<errored> <= sml::state<dp_forward_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_failed{}]
          / action::ensure_last_error
      , sml::state<dp_backtrace_result_decision> <= sml::state<dp_backtrace_exec>
          + sml::completion<runtime::encode_runtime> / action::run_dp_backtrace
      , sml::state<emit_exec> <= sml::state<dp_backtrace_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_ok{}]
      , sml::state<errored> <= sml::state<dp_backtrace_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_failed{}]
          / action::ensure_last_error
      , sml::state<encode_result_decision> <= sml::state<emit_exec>
          + sml::completion<runtime::encode_runtime> / action::emit_tokens
      , sml::state<done> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_ok{}] / action::mark_done
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::phase_failed{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Explicit Unexpected-Event Handling
      //------------------------------------------------------------------------------//
      , sml::state<unexpected> <= sml::state<encode_validity_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_vocab_sync_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_precheck_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_policy_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_resolution_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_lookup_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected

      , sml::state<unexpected> <= sml::state<initialized>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<initialized>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_validity_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_validity_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_vocab_sync_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_vocab_sync_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_precheck_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_precheck_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_policy_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_policy_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_resolution_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_resolution_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_lookup_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_lookup_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unexpected>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unexpected>
          + sml::event<events::encoding_error> / action::on_unexpected

      , sml::state<unexpected> <= sml::state<initialized>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_validity_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_vocab_sync_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_precheck_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_policy_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_resolution_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_lookup_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<normalize_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<input_prepare_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_forward_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_backtrace_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unexpected>
          + sml::unexpected_event<sml::_> / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() : base_type() {}

  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::encode & ev) {
    event::encode_ctx runtime_ctx{};
    event::encode_runtime base_runtime_ev{ev, runtime_ctx};
    runtime::encode_runtime runtime_ev{base_runtime_ev};
    const bool accepted = base_type::process_event(runtime_ev);

    runtime_ctx.err = emel::text::encoders::detail::select_final_error(accepted, runtime_ctx.err);

    int32_t token_count_sink = 0;
    int32_t error_sink = EMEL_OK;
    emel::text::encoders::detail::write_optional(
      ev.token_count_out, token_count_sink, runtime_ctx.token_count);
    emel::text::encoders::detail::write_optional(ev.error_out, error_sink, runtime_ctx.err);

    emel::text::encoders::detail::publish_result(ev, runtime_ctx);
    last_error_ = runtime_ctx.err;
    return runtime_ctx.err == EMEL_OK;
  }

  int32_t last_error() const noexcept { return last_error_; }

 private:
  int32_t last_error_ = EMEL_OK;
};

using Ugm = sm;

}  // namespace emel::text::encoders::ugm
