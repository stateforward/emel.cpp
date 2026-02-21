#pragma once

#include "emel/sm.hpp"
#include "emel/tokenizer/preprocessor/actions.hpp"
#include "emel/tokenizer/preprocessor/events.hpp"
#include "emel/tokenizer/preprocessor/guards.hpp"

namespace emel::tokenizer::preprocessor::detail {

struct idle {};
struct preparing {};
struct partitioning {};
struct partition_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * tokenizer preprocessor orchestration model (variant-agnostic scaffold).
 *
 * state purposes:
 * - `idle`: wait for preprocess intent.
 * - `preparing`: build special-token cache for the vocab.
 * - `partitioning`: split raw text into fragments with special tokens isolated.
 * - `partition_decision`: branch on partition success/failure.
 * - `done`/`errored`: terminal outcomes for a request.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_request`/`invalid_request`: validate request pointers and capacity.
 * - `phase_ok`/`phase_failed`: observe error set by actions.
 *
 * action side effects:
 * - `begin_preprocess`: capture inputs and reset outputs.
 * - `build_specials`: build cached special-token inventory.
 * - `partition_specials`: populate output fragments.
 * - `mark_done`: clear error state.
 * - `ensure_last_error`: provide a terminal error code when missing.
 * - `on_unexpected`: report sequencing violations.
 */
template <class tag>
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<idle> + sml::event<event::preprocess>[guard::valid_request{}] /
                action::begin_preprocess = sml::state<preparing>,
        sml::state<idle> + sml::event<event::preprocess>[guard::invalid_request{}] /
                action::reject_invalid = sml::state<errored>,

        sml::state<preparing> / action::build_specials = sml::state<partitioning>,
        sml::state<partitioning> / action::partition_specials =
                sml::state<partition_decision>,

        sml::state<partition_decision>[guard::phase_failed{}] /
                action::ensure_last_error = sml::state<errored>,
        sml::state<partition_decision>[guard::phase_ok{}] /
                action::mark_done = sml::state<done>,

        sml::state<done> + sml::event<event::preprocess>[guard::valid_request{}] /
                action::begin_preprocess = sml::state<preparing>,
        sml::state<done> + sml::event<event::preprocess>[guard::invalid_request{}] /
                action::reject_invalid = sml::state<errored>,

        sml::state<errored> + sml::event<event::preprocess>[guard::valid_request{}] /
                action::begin_preprocess = sml::state<preparing>,
        sml::state<errored> + sml::event<event::preprocess>[guard::invalid_request{}] /
                action::reject_invalid = sml::state<errored>,

        sml::state<unexpected> + sml::event<event::preprocess>[guard::valid_request{}] /
                action::begin_preprocess = sml::state<preparing>,
        sml::state<unexpected> + sml::event<event::preprocess>[guard::invalid_request{}] /
                action::reject_invalid = sml::state<errored>,

        sml::state<idle> + sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<preparing> + sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<partitioning> + sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<partition_decision> + sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>,
        sml::state<unexpected> + sml::unexpected_event<sml::_> /
                action::on_unexpected = sml::state<unexpected>);
  }
};

}  // namespace emel::tokenizer::preprocessor::detail
