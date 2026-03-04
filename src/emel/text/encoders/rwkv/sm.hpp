#pragma once

#include <cstdint>

#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/rwkv/actions.hpp"
#include "emel/text/encoders/rwkv/errors.hpp"
#include "emel/text/encoders/rwkv/guards.hpp"
#include "emel/sm.hpp"

namespace emel::text::encoders::rwkv {

struct initialized {};
struct encode_validity_decision {};
struct encode_vocab_sync_decision {};
struct encode_precheck_decision {};
struct table_policy_decision {};
struct table_sync_exec {};
struct table_sync_result_decision {};
struct unk_resolution_decision {};
struct unk_lookup_exec {};
struct unk_lookup_result_decision {};
struct encode_exec {};
struct encode_result_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * RWKV encoder orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting encode intent.
 * - `encode_validity_decision`/`encode_vocab_sync_decision`: explicit intake routing.
 * - `encode_precheck_decision`: explicit request prechecks before phase execution.
 * - `table_policy_decision`: explicit RWKV table readiness routing for non-empty text.
 * - `table_sync_exec`/`table_sync_result_decision`: explicit RWKV table preparation.
 * - `unk_resolution_decision`/`unk_lookup_exec`/`unk_lookup_result_decision`:
 *   explicit unknown-token ID resolution.
 * - `encode_exec`/`encode_result_decision`: explicit encode execution and status branch.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_encode`/`invalid_encode` validate request payload shape.
 * - `vocab_changed`/`vocab_unchanged` route explicit vocabulary-sync behavior.
 * - `text_empty`/`text_non_empty` route explicit precheck decisions.
 * - `tables_ready`/`tables_missing` route explicit table-policy behavior.
 * - `vocab_unk_present`/`vocab_unk_missing` route unknown-token resolution.
 * - `phase_ok`/`phase_failed` route phase error status.
 *
 * action side effects:
 * - `begin_encode`/`begin_encode_sync_vocab` reset runtime outputs and vocabulary bindings.
 * - `sync_tables` rebuilds RWKV lookup tables in an explicit phase.
 * - `resolve_vocab_unk`/`lookup_unk_candidate`/`set_unk_*` set runtime unknown-token ID.
 * - `run_encode` performs bounded encode scanning and output emission.
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
          + sml::completion<runtime::encode_runtime>
          / action::reject_invalid_encode

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
          + sml::completion<runtime::encode_runtime>
          / action::reject_invalid_encode

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
      // RWKV Table Policy
      //------------------------------------------------------------------------------//
      , sml::state<table_sync_exec> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime>[guard::tables_missing{}]
      , sml::state<unk_resolution_decision> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime>[guard::tables_ready{}]
      , sml::state<errored> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime> / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // RWKV Table Sync
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
      , sml::state<encode_exec> <= sml::state<unk_resolution_decision>
          + sml::completion<runtime::encode_runtime>[guard::vocab_unk_present{}]
          / action::resolve_vocab_unk
      , sml::state<unk_lookup_exec> <= sml::state<unk_resolution_decision>
          + sml::completion<runtime::encode_runtime>[guard::vocab_unk_missing{}]
      , sml::state<unk_lookup_result_decision> <= sml::state<unk_lookup_exec>
          + sml::completion<runtime::encode_runtime> / action::lookup_unk_candidate
      , sml::state<encode_exec> <= sml::state<unk_lookup_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::unk_lookup_found{}]
          / action::set_unk_from_lookup
      , sml::state<encode_exec> <= sml::state<unk_lookup_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::unk_lookup_missing{}]
          / action::set_unk_missing

      //------------------------------------------------------------------------------//
      // Encode Execution
      //------------------------------------------------------------------------------//
      , sml::state<encode_result_decision> <= sml::state<encode_exec>
          + sml::completion<runtime::encode_runtime> / action::run_encode
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
      , sml::state<unexpected> <= sml::state<unk_lookup_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
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
      , sml::state<unexpected> <= sml::state<unk_lookup_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unk_lookup_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
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
      , sml::state<unexpected> <= sml::state<unk_lookup_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
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

using Rwkv = sm;

}  // namespace emel::text::encoders::rwkv
