#pragma once

#include "emel/generator/actions.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::generator {

struct initialized {};
struct tokenizing_prompt {};
struct tokenize_decision {};
struct prefilling {};
struct prefill_decision {};
struct decoding {};
struct decode_decision {};
struct done {};
struct errored {};
struct unexpected_event {};

/**
 * generator orchestration scaffold.
 *
 * state purposes:
 * - `initialized`: idle state awaiting generation intent.
 * - `tokenizing_prompt`/`tokenize_decision`: run prompt tokenization phase and branch by phase result.
 * - `prefilling`/`prefill_decision`: run prefill phase and branch by phase result.
 * - `decoding`/`decode_decision`: bounded decode loop, one decode step per internal completion.
 * - `done`/`errored`: terminal outcomes for the current request.
 * - `unexpected_event`: catchall for unhandled external events.
 *
 * guard semantics:
 * - `valid_generate`/`invalid_generate`: validate `event::generate` shape and decode bound.
 * - `phase_ok`/`phase_failed`: observe action-reported phase status.
 * - `should_continue_decode`/`stop_condition_met`: enforce bounded decode progress.
 *
 * action side effects:
 * - `begin_generate`: reset per-request counters and clear error-out.
 * - `tokenize_prompt`/`run_prefill`/`run_decode_step`: execute phase work and update context.
 * - `dispatch_generation_done_to_owner`/`dispatch_generation_error_to_owner`: publish outcomes
 *   synchronously before dispatch returns.
 * - `on_unexpected`: record backend sequencing violation.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::generate>[guard::valid_generate] /
          action::begin_generate = sml::state<tokenizing_prompt>,
      sml::state<initialized> + sml::event<event::generate>[guard::invalid_generate] /
          action::reject_invalid_generate = sml::state<errored>,

      sml::state<tokenizing_prompt> + sml::completion<event::generate> / action::tokenize_prompt =
          sml::state<tokenize_decision>,
      sml::state<tokenize_decision> + sml::completion<event::generate>[guard::phase_ok] =
          sml::state<prefilling>,
      sml::state<tokenize_decision> + sml::completion<event::generate>[guard::phase_failed] =
          sml::state<errored>,

      sml::state<prefilling> + sml::completion<event::generate> / action::run_prefill =
          sml::state<prefill_decision>,
      sml::state<prefill_decision> + sml::completion<event::generate>[guard::phase_ok] =
          sml::state<decoding>,
      sml::state<prefill_decision> + sml::completion<event::generate>[guard::phase_failed] =
          sml::state<errored>,

      sml::state<decoding> + sml::completion<event::generate> / action::run_decode_step =
          sml::state<decode_decision>,
      sml::state<decode_decision> + sml::completion<event::generate>[guard::phase_failed] =
          sml::state<errored>,
      sml::state<decode_decision> +
          sml::completion<event::generate>[guard::should_continue_decode] =
          sml::state<decoding>,
      sml::state<decode_decision> +
          sml::completion<event::generate>[guard::stop_condition_met] =
          sml::state<done>,

      sml::state<done> + sml::completion<event::generate> /
          action::dispatch_generation_done_to_owner = sml::state<initialized>,
      sml::state<errored> + sml::completion<event::generate> /
          action::dispatch_generation_error_to_owner = sml::state<initialized>,

      sml::state<unexpected_event> + sml::event<event::generate>[guard::valid_generate] /
          action::begin_generate = sml::state<tokenizing_prompt>,
      sml::state<unexpected_event> + sml::event<event::generate>[guard::invalid_generate] /
          action::reject_invalid_generate = sml::state<errored>,

      sml::state<initialized> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<tokenizing_prompt> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<tokenize_decision> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<prefilling> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<prefill_decision> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<decoding> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<decode_decision> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<done> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>,
      sml::state<unexpected_event> + sml::unexpected_event<sml::_> / action::on_unexpected =
          sml::state<unexpected_event>);
  }
};

using sm = emel::sm_with_context<model, action::context>;

}  // namespace emel::generator
