#pragma once

#include "emel/sm.hpp"
#include "emel/tokenizer/preprocessor/rwkv/actions.hpp"
#include "emel/tokenizer/preprocessor/rwkv/guards.hpp"

namespace emel::tokenizer::preprocessor::rwkv {

struct idle {};
struct preparing {};
struct partitioning_non_bpe {};
struct partition_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * tokenizer preprocessor orchestration model.
 *
 * state purposes:
 * - `idle`: wait for preprocess intent.
 * - `preparing`: build special-token cache for the vocab.
 * - `partitioning_non_bpe`: split raw text into fragments with specials isolated.
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
 * - `partition_non_bpe`: populate output fragments for non-BPE vocabularies.
 * - `mark_done`: clear error state.
 * - `ensure_last_error`: provide a terminal error code when missing.
 * - `on_unexpected`: report sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<idle> + sml::event<event::preprocess>[guard::valid_request{}] /
                action::begin_preprocess = sml::state<preparing>,
        sml::state<idle> + sml::event<event::preprocess>[guard::invalid_request{}] /
                action::reject_invalid = sml::state<errored>,

        sml::state<preparing> / action::build_specials =
                sml::state<partitioning_non_bpe>,

        sml::state<partitioning_non_bpe> / action::partition_non_bpe =
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
        sml::state<partitioning_non_bpe> + sml::unexpected_event<sml::_> /
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

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::preprocess & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err =
        ok ? EMEL_OK
           : (context_.last_error != EMEL_OK ? context_.last_error
                                             : EMEL_ERR_BACKEND);

    if (ev.fragment_count_out != nullptr) {
      *ev.fragment_count_out = context_.fragment_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm,
                         events::preprocess_done{&ev, context_.fragment_count});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::preprocess_error{&ev, err});
      }
    }

    action::clear_request(context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }
  size_t fragment_count() const noexcept { return context_.fragment_count; }

 private:
  action::context context_{};
};

}  // namespace emel::tokenizer::preprocessor::rwkv
