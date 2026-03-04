#pragma once

#include <cstdint>

#include "emel/text/encoders/detail.hpp"
#include "emel/text/encoders/bpe/actions.hpp"
#include "emel/text/encoders/bpe/errors.hpp"
#include "emel/text/encoders/bpe/guards.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/sm.hpp"

namespace emel::text::encoders::bpe {

struct initialized {};
struct encode_validity_decision {};
struct encode_vocab_sync_decision {};
struct encode_precheck_decision {};
struct encode_input_policy_decision {};
struct encode_table_prepare {};
struct encode_path_decision {};
struct encode_direct_word_policy_decision {};
struct encode_merge_input_capacity_decision {};
struct encode_exec {};
struct encode_result_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * BPE encoder orchestration model.
 *
 * state purposes:
 * - 'initialized': idle state awaiting encode intent.
 * - 'encode_validity_decision': explicit request validity routing before runtime setup.
 * - 'encode_vocab_sync_decision': explicit vocabulary-sync policy routing.
 * - 'encode_precheck_decision': explicit request prechecks before kernel execution.
 * - 'encode_input_policy_decision': explicit preprocessed-input policy routing.
 * - 'encode_table_prepare': ensure per-vocab tables for deterministic path guards.
 * - 'encode_path_decision': explicit `ignore_merges` policy routing.
 * - 'encode_direct_word_policy_decision': explicit direct-word availability routing.
 * - 'encode_merge_input_capacity_decision': explicit merge-path symbol-capacity routing.
 * - 'encode_exec'/'encode_result_decision': run selected kernel and branch on phase error.
 * - 'done'/'errored': terminal outcomes.
 * - 'unexpected': sequencing contract violation.
 *
 * guard semantics:
 * - 'valid_encode'/'invalid_encode' validate request pointers and context.
 * - 'vocab_changed'/'vocab_unchanged' route vocabulary sync work.
 * - 'text_empty'/'text_non_empty' and 'preprocessed'/'not_preprocessed' route precheck decisions.
 * - 'ignore_merges_enabled' and 'direct_word_token_available' route algorithm path selection.
 * - 'merge_symbol_capacity_within_limit'/'merge_symbol_capacity_exceeded' route merge-path intake.
 * - 'phase_*' guards observe runtime phase errors.
 *
 * action side effects:
 * - 'begin_encode' resets runtime per-request outputs.
 * - 'begin_encode_sync_vocab' refreshes per-vocab cached tables.
 * - 'prepare_tables' builds lookup tables before path routing.
 * - 'run_encode_ignore_merges' and 'run_encode_merge_path' execute bounded kernels.
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
      , sml::state<encode_input_policy_decision> <= sml::state<encode_precheck_decision>
          + sml::completion<event::encode_runtime>[guard::text_non_empty{}]
      , sml::state<errored> <= sml::state<encode_precheck_decision>
          + sml::completion<event::encode_runtime>
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Input Policy Decision
      //------------------------------------------------------------------------------//
      , sml::state<encode_table_prepare> <= sml::state<encode_input_policy_decision>
          + sml::completion<event::encode_runtime>[guard::preprocessed{}]
          / action::prepare_tables
      , sml::state<errored> <= sml::state<encode_input_policy_decision>
          + sml::completion<event::encode_runtime>[guard::not_preprocessed{}]
          / action::reject_invalid_encode
      , sml::state<errored> <= sml::state<encode_input_policy_decision>
          + sml::completion<event::encode_runtime>
          / action::reject_invalid_encode

      //------------------------------------------------------------------------------//
      // Table Preparation
      //------------------------------------------------------------------------------//
      , sml::state<encode_path_decision> <= sml::state<encode_table_prepare>
          + sml::completion<event::encode_runtime>[guard::phase_ok{}]
      , sml::state<errored> <= sml::state<encode_table_prepare>
          + sml::completion<event::encode_runtime>[guard::phase_failed{}]
          / action::ensure_last_error
      , sml::state<errored> <= sml::state<encode_table_prepare>
          + sml::completion<event::encode_runtime>
          / action::ensure_last_error

      //------------------------------------------------------------------------------//
      // Encode Path Decision
      //------------------------------------------------------------------------------//
      , sml::state<encode_direct_word_policy_decision> <= sml::state<encode_path_decision>
          + sml::completion<event::encode_runtime>[guard::ignore_merges_enabled{}]
      , sml::state<encode_exec> <= sml::state<encode_path_decision>
          + sml::completion<event::encode_runtime>

      , sml::state<encode_result_decision> <= sml::state<encode_direct_word_policy_decision>
          + sml::completion<event::encode_runtime>[guard::direct_word_token_available{}]
          / action::run_encode_ignore_merges
      , sml::state<encode_merge_input_capacity_decision> <= sml::state<encode_direct_word_policy_decision>
          + sml::completion<event::encode_runtime>

      //------------------------------------------------------------------------------//
      // Merge Input Capacity Decision
      //------------------------------------------------------------------------------//
      , sml::state<encode_exec> <= sml::state<encode_merge_input_capacity_decision>
          + sml::completion<event::encode_runtime>[guard::merge_symbol_capacity_within_limit{}]
      , sml::state<errored> <= sml::state<encode_merge_input_capacity_decision>
          + sml::completion<event::encode_runtime>[guard::merge_symbol_capacity_exceeded{}]
          / action::reject_invalid_encode
      , sml::state<errored> <= sml::state<encode_merge_input_capacity_decision>
          + sml::completion<event::encode_runtime>
          / action::reject_invalid_encode

      //------------------------------------------------------------------------------//
      // Encode Execution
      //------------------------------------------------------------------------------//
      , sml::state<encode_result_decision> <= sml::state<encode_exec>
          + sml::completion<event::encode_runtime> / action::run_encode_merge_path
      , sml::state<done> <= sml::state<encode_result_decision>
          + sml::completion<event::encode_runtime>[guard::phase_ok{}] / action::mark_done
      , sml::state<errored> <= sml::state<encode_result_decision>
          + sml::completion<event::encode_runtime>[guard::phase_failed{}]
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
      , sml::state<unexpected> <= sml::state<encode_input_policy_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_path_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_direct_word_policy_decision>
          + sml::event<event::encode_runtime> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_merge_input_capacity_decision>
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
      , sml::state<unexpected> <= sml::state<encode_input_policy_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_input_policy_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_path_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_path_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_direct_word_policy_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_direct_word_policy_decision>
          + sml::event<events::encoding_error> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_merge_input_capacity_decision>
          + sml::event<events::encoding_done> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_merge_input_capacity_decision>
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
      , sml::state<unexpected> <= sml::state<encode_input_policy_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_table_prepare>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_path_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_direct_word_policy_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<unexpected> <= sml::state<encode_merge_input_capacity_decision>
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

using Bpe = sm;

}  // namespace emel::text::encoders::bpe
