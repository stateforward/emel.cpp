#pragma once

#include <cstdint>

#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/fallback/actions.hpp"
#include "emel/text/encoders/fallback/errors.hpp"
#include "emel/text/encoders/fallback/guards.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/sm.hpp"

namespace emel::text::encoders::fallback {

struct initialized {};
struct encode_validity_decision {};
struct encode_vocab_sync_decision {};
struct encode_precheck_decision {};
struct encode_table_prepare {};
struct encode_exec {};
struct emit_result_decision {};
struct encode_result_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * Fallback encoder orchestration model.
 *
 * state purposes:
 * - 'initialized': idle state awaiting encode intent.
 * - 'encode_validity_decision': explicit request validity routing before runtime setup.
 * - 'encode_vocab_sync_decision': explicit vocabulary-sync policy routing.
 * - 'encode_precheck_decision': explicit request prechecks before kernel execution.
 * - 'encode_table_prepare': ensure per-vocab tables before encode execution.
 * - 'encode_exec'/'emit_result_decision': explicit kernel execution and emit outcome routing.
 * - 'encode_result_decision': explicit final runtime-error routing.
 * - 'done'/'errored': terminal outcomes.
 * - 'unexpected': sequencing contract violation.
 *
 * guard semantics:
 * - 'valid_encode'/'invalid_encode' validate request pointers and context.
 * - 'vocab_changed'/'vocab_unchanged' route vocabulary sync work.
 * - 'text_empty'/'text_non_empty' route explicit precheck decisions.
 * - 'emit_result_ok'/'emit_result_failed' route explicit emit outcomes.
 * - 'table_prepare_*' and 'encode_result_*' guards route explicit error-class outcomes.
 *
 * action side effects:
 * - 'begin_encode' resets runtime per-request outputs.
 * - 'begin_encode_sync_vocab' refreshes per-vocab cached tables.
 * - 'prepare_tables' builds lookup tables before execution.
 * - 'run_encode_exec' computes explicit emit outcome data.
 * - 'apply_emit_result_ok'/'apply_emit_result_failed' commit explicit emit outcomes.
 * - 'mark_done'/'ensure_last_error' finalize runtime status.
 * - 'on_unexpected' reports sequencing violations.
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
      , sml::state<encode_table_prepare> <= sml::state<encode_precheck_decision>
          + sml::completion<runtime::encode_runtime>[guard::text_non_empty{}]
          / action::prepare_tables

      //------------------------------------------------------------------------------//
      // Table Preparation
      //------------------------------------------------------------------------------//
      , sml::state<encode_exec> <= sml::state<encode_table_prepare>
          + sml::completion<runtime::encode_runtime>[guard::table_prepare_ok{}]
      , sml::state<errored> <= sml::state<encode_table_prepare>
          + sml::completion<runtime::encode_runtime>[guard::table_prepare_invalid_argument_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_table_prepare>
          + sml::completion<runtime::encode_runtime>[guard::table_prepare_backend_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_table_prepare>
          + sml::completion<runtime::encode_runtime>[guard::table_prepare_model_invalid_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_table_prepare>
          + sml::completion<runtime::encode_runtime>[guard::table_prepare_unknown_error{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Encode Execution
      //------------------------------------------------------------------------------//
      , sml::state<emit_result_decision> <= sml::state<encode_exec>
          + sml::completion<runtime::encode_runtime> / action::run_encode_exec
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

      //------------------------------------------------------------------------------//
      // Explicit Unexpected-Event Handling
      //------------------------------------------------------------------------------//
      , sml::state<unexpected> <= sml::state<encode_validity_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_vocab_sync_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_precheck_decision>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::event<runtime::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
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
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
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
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
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

using Fallback = sm;

}  // namespace emel::text::encoders::fallback
