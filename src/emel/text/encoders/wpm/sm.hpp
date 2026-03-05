#pragma once

#include <cstdint>

#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/wpm/actions.hpp"
#include "emel/text/encoders/wpm/errors.hpp"
#include "emel/text/encoders/wpm/guards.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/sm.hpp"

namespace emel::text::encoders::wpm {

struct initialized {};
struct encode_validity_decision {};
struct encode_vocab_sync_decision {};
struct encode_precheck_decision {};
struct table_policy_decision {};
struct table_sync_exec {};
struct table_sync_result_decision {};
struct encode_input_capacity_decision {};
struct encode_exec {};
struct encode_result_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * WPM encoder orchestration model.
 *
 * state purposes:
 * - 'initialized': idle state awaiting encode intent.
 * - 'encode_validity_decision': explicit request validity routing before runtime setup.
 * - 'encode_vocab_sync_decision': explicit vocabulary-sync policy routing.
 * - 'encode_precheck_decision': explicit request prechecks before kernel execution.
 * - 'table_policy_decision': explicit non-empty-input table-policy routing.
 * - 'table_sync_exec'/'table_sync_result_decision': explicit WPM table-prep phase.
 * - 'encode_input_capacity_decision': explicit input-prefix-capacity routing.
 * - 'encode_exec'/'encode_result_decision': run kernel and branch on phase error.
 * - 'done'/'errored': terminal outcomes.
 * - 'unexpected': sequencing contract violation.
 *
 * guard semantics:
 * - 'valid_encode'/'invalid_encode' validate request pointers and context.
 * - 'vocab_changed'/'vocab_unchanged' route vocabulary sync work.
 * - 'text_empty'/'text_non_empty' route explicit precheck decisions.
 * - 'tables_ready'/'tables_missing' route table-sync execution.
 * - 'prefix_buffer_capacity_within_limit'/'prefix_buffer_capacity_exceeded'
 *   route encode-input capacity policy.
 * - 'phase_*' guards observe runtime phase errors.
 *
 * action side effects:
 * - 'begin_encode' resets runtime per-request outputs.
 * - 'begin_encode_sync_vocab' refreshes per-vocab cached tables.
 * - 'sync_tables' builds WPM lookup tables in an explicit phase.
 * - 'run_encode' performs bounded encoding work.
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
          + sml::event<event::encode_runtime>
      , sml::state<encode_validity_decision> <= sml::state<done>
          + sml::event<event::encode_runtime>
      , sml::state<encode_validity_decision> <= sml::state<errored>
          + sml::event<event::encode_runtime>
      , sml::state<encode_validity_decision> <= sml::state<unexpected>
          + sml::event<event::encode_runtime>

      , sml::state<encode_vocab_sync_decision> <= sml::state<encode_validity_decision>
          + sml::completion<event::encode_runtime>[guard::valid_encode{}]
      , sml::state<errored> <= sml::state<encode_validity_decision>
          + sml::completion<event::encode_runtime>[guard::invalid_encode{}]
          / action::reject_invalid_encode
      , sml::state<errored> <= sml::state<encode_validity_decision>
          + sml::completion<event::encode_runtime>
          / action::reject_invalid_encode

      , sml::state<encode_precheck_decision> <= sml::state<encode_vocab_sync_decision>
          + sml::completion<event::encode_runtime>[guard::vocab_changed{}]
          / action::begin_encode_sync_vocab
      , sml::state<encode_precheck_decision> <= sml::state<encode_vocab_sync_decision>
          + sml::completion<event::encode_runtime>[guard::vocab_unchanged{}]
          / action::begin_encode
      , sml::state<errored> <= sml::state<encode_vocab_sync_decision>
          + sml::completion<event::encode_runtime>
          / action::reject_invalid_encode

      //------------------------------------------------------------------------------//
      // Encode Precheck
      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<encode_precheck_decision>
          + sml::completion<event::encode_runtime>[guard::text_empty{}] / action::mark_done
      , sml::state<table_policy_decision> <= sml::state<encode_precheck_decision>
          + sml::completion<event::encode_runtime>[guard::text_non_empty{}]
      , sml::state<errored> <= sml::state<encode_precheck_decision>
          + sml::completion<event::encode_runtime>
          / action::ensure_last_error

      , sml::state<table_sync_exec> <= sml::state<table_policy_decision>
          + sml::completion<event::encode_runtime>[guard::tables_missing{}]
      , sml::state<encode_input_capacity_decision> <= sml::state<table_policy_decision>
          + sml::completion<event::encode_runtime>[guard::tables_ready{}]
      , sml::state<errored> <= sml::state<table_policy_decision>
          + sml::completion<event::encode_runtime>
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // WPM Table Sync
      //------------------------------------------------------------------------------//
      , sml::state<table_sync_result_decision> <= sml::state<table_sync_exec>
          + sml::completion<event::encode_runtime> / action::sync_tables
      , sml::state<encode_input_capacity_decision> <= sml::state<table_sync_result_decision>
          + sml::completion<event::encode_runtime>[guard::table_sync_ok{}]
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<event::encode_runtime>[guard::table_sync_invalid_argument_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<event::encode_runtime>[guard::table_sync_backend_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<event::encode_runtime>[guard::table_sync_model_invalid_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<table_sync_result_decision>
          + sml::completion<event::encode_runtime>[guard::table_sync_unclassified_error_code{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Input Capacity Decision
      //------------------------------------------------------------------------------//
      , sml::state<encode_exec> <= sml::state<encode_input_capacity_decision>
          + sml::completion<event::encode_runtime>[guard::prefix_buffer_capacity_within_limit{}]
      , sml::state<errored> <= sml::state<encode_input_capacity_decision>
          + sml::completion<event::encode_runtime>[guard::prefix_buffer_capacity_exceeded{}]
          / action::reject_invalid_encode
      , sml::state<errored> <= sml::state<encode_input_capacity_decision>
          + sml::completion<event::encode_runtime>
          / action::reject_invalid_encode

      //------------------------------------------------------------------------------//
      // Encode Execution
      //------------------------------------------------------------------------------//
      , sml::state<encode_result_decision> <= sml::state<encode_exec>
          + sml::completion<event::encode_runtime> / action::run_encode
      , sml::state<done> <= sml::state<encode_result_decision>
          + sml::completion<event::encode_runtime>[guard::encode_result_ok{}]
          / action::mark_done
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<event::encode_runtime>[guard::encode_result_invalid_argument_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<event::encode_runtime>[guard::encode_result_backend_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<event::encode_runtime>[guard::encode_result_model_invalid_error{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<event::encode_runtime>[guard::encode_result_unclassified_error_code{}]
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Explicit Unexpected-Event Handling
      //------------------------------------------------------------------------------//
      , sml::state<unexpected> <= sml::state<encode_validity_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_vocab_sync_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_precheck_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_policy_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_exec>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<table_sync_result_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_input_capacity_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_exec>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_result_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected

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
      , sml::state<unexpected> <= sml::state<encode_input_capacity_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_input_capacity_decision>
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
      , sml::state<unexpected> <= sml::state<encode_input_capacity_decision>
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
    event::encode_runtime runtime_ev{ev, runtime_ctx};
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

using Wpm = sm;

}  // namespace emel::text::encoders::wpm
