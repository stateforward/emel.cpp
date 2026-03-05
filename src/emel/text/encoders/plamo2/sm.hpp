#pragma once

#include <cstdint>

#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/plamo2/actions.hpp"
#include "emel/text/encoders/plamo2/errors.hpp"
#include "emel/text/encoders/plamo2/guards.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/sm.hpp"

namespace emel::text::encoders::plamo2 {

struct initialized {};
struct encode_validity_decision {};
struct encode_vocab_sync_decision {};
struct encode_precheck_decision {};
struct table_policy_decision {};
struct table_sync_exec {};
struct table_sync_result_decision {};
struct decode_exec {};
struct decode_result_decision {};
struct dp_prepare_exec {};
struct dp_exec {};
struct emit_exec {};
struct emit_result_decision {};
struct encode_result_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * PLaMo2 encoder orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting encode intent.
 * - `encode_validity_decision`/`encode_vocab_sync_decision`: explicit intake routing.
 * - `encode_precheck_decision`: explicit request prechecks before phase execution.
 * - `table_policy_decision`: explicit PLaMo2 table readiness routing for non-empty text.
 * - `table_sync_exec`/`table_sync_result_decision`: explicit table preparation and status branch.
 * - `decode_exec`/`decode_result_decision`: explicit UTF-8 decode/BOM-strip phase and branch.
 * - `dp_prepare_exec`/`dp_exec`: explicit dynamic-programming setup and forward phase.
 * - `emit_exec`/`emit_result_decision`: explicit output emission and emit-status branch.
 * - `encode_result_decision`: explicit final encode result branch.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_encode`/`invalid_encode` validate request payload shape.
 * - `vocab_changed`/`vocab_unchanged` route explicit vocabulary-sync behavior.
 * - `text_empty`/`text_non_empty` route explicit precheck decisions.
 * - `tables_ready`/`tables_missing` route explicit table-policy behavior.
 * - `decode_result_*` route explicit decode outcome and error-class status.
 * - `emit_result_ok`/`emit_result_failed` route explicit emission outcomes.
 * - `table_sync_*` and `encode_result_*` route explicit per-phase error status.
 *
 * action side effects:
 * - `begin_encode`/`begin_encode_sync_vocab` reset runtime outputs and vocabulary bindings.
 * - `sync_tables` rebuilds PLaMo2 lookup tables in an explicit phase.
 * - `decode_input` decodes UTF-8 input into runtime codepoints and strips BOM.
 * - `prepare_dp`/`run_dp` perform bounded DP setup and scoring.
 * - `emit_tokens` performs bounded output emission from explicit DP path data.
 * - `apply_emit_result_ok`/`apply_emit_result_failed` commit explicit emission outcomes.
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

      , sml::state<encode_vocab_sync_decision> <= sml::state<encode_validity_decision>
          + sml::completion<runtime::encode_runtime>[guard::valid_encode{}]
      , sml::state<errored> <= sml::state<encode_validity_decision>
          + sml::completion<runtime::encode_runtime>[guard::invalid_encode{}]
          / action::reject_invalid_encode
      , sml::state<errored> <= sml::state<encode_validity_decision>
          + sml::completion<runtime::encode_runtime>
          / action::reject_invalid_encode

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
          + sml::completion<runtime::encode_runtime>
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Table Policy + Sync
      //------------------------------------------------------------------------------//
      , sml::state<table_sync_exec> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime>[guard::tables_missing{}]
      , sml::state<decode_exec> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime>[guard::tables_ready{}]
      , sml::state<errored> <= sml::state<table_policy_decision>
          + sml::completion<runtime::encode_runtime>
          / action::ensure_last_error

      , sml::state<table_sync_result_decision> <= sml::state<table_sync_exec>
          + sml::completion<runtime::encode_runtime> / action::sync_tables
      , sml::state<decode_exec> <= sml::state<table_sync_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::table_sync_ok{}]
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::table_sync_invalid_argument_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::table_sync_backend_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::table_sync_model_invalid_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::table_sync_unknown_error{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Decode + Dynamic Programming + Emit
      //------------------------------------------------------------------------------//
      , sml::state<decode_result_decision> <= sml::state<decode_exec>
          + sml::completion<runtime::encode_runtime> / action::decode_input
      , sml::state<done> <= sml::state<decode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::decode_result_empty_ok{}]
          / action::mark_done
      , sml::state<dp_prepare_exec> <= sml::state<decode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::decode_result_non_empty_ok{}]
      , sml::state<errored> <= sml::state<decode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::decode_result_invalid_argument_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<decode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::decode_result_backend_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<decode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::decode_result_model_invalid_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<decode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::decode_result_unknown_error{}]
          / action::ensure_last_error

      , sml::state<dp_exec> <= sml::state<dp_prepare_exec>
          + sml::completion<runtime::encode_runtime> / action::prepare_dp
      , sml::state<emit_exec> <= sml::state<dp_exec>
          + sml::completion<runtime::encode_runtime> / action::run_dp
      , sml::state<emit_result_decision> <= sml::state<emit_exec>
          + sml::completion<runtime::encode_runtime> / action::emit_tokens
      , sml::state<encode_result_decision> <= sml::state<emit_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::emit_result_ok{}]
          / action::apply_emit_result_ok
      , sml::state<encode_result_decision> <= sml::state<emit_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::emit_result_failed{}]
          / action::apply_emit_result_failed
      , sml::state<errored> <= sml::state<emit_result_decision>
          + sml::completion<runtime::encode_runtime>
          / action::ensure_last_error
      , sml::state<done> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::encode_result_ok{}]
          / action::mark_done
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::encode_result_invalid_argument_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::encode_result_backend_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::encode_result_model_invalid_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>[guard::encode_result_unknown_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<runtime::encode_runtime>
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
      , sml::state<unexpected> <= sml::state<decode_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_result_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_prepare_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_result_decision>
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
      , sml::state<unexpected> <= sml::state<decode_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_result_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_prepare_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_prepare_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_result_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_result_decision>
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
      , sml::state<unexpected> <= sml::state<decode_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<decode_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_prepare_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<dp_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<emit_result_decision>
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
    int32_t error_sink = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
    emel::text::encoders::detail::write_optional(
      ev.token_count_out, token_count_sink, runtime_ctx.token_count);
    emel::text::encoders::detail::write_optional(ev.error_out, error_sink, runtime_ctx.err);

    emel::text::encoders::detail::publish_result(ev, runtime_ctx);
    last_error_ = runtime_ctx.err;
    return runtime_ctx.err == emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  }

  int32_t last_error() const noexcept { return last_error_; }

 private:
  int32_t last_error_ = emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
};

using Plamo2 = sm;

}  // namespace emel::text::encoders::plamo2
