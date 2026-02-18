#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/tokenizer/actions.hpp"
#include "emel/tokenizer/events.hpp"
#include "emel/tokenizer/guards.hpp"

namespace emel::tokenizer {

struct initialized {};
struct building_special_tokens {};
struct special_tokens_decision {};
struct partitioning_raw {};
struct partitioning_with_specials {};
struct partitioning_decision {};
struct selecting_backend {};
struct selecting_backend_decision {};
struct prefix_decision {};
struct encoding_ready {};
struct encoding_token_fragment {};
struct encoding_raw_fragment {};
struct encoding_decision {};
struct suffix_decision {};
struct finalizing {};
struct done {};
struct errored {};
struct unexpected {};

/*
Tokenizer architecture notes (single source of truth)

Scope
- Component boundary: tokenizer
- Goal: tokenize text into token ids with special-token partitioning and
model-aware encoding.

State purpose
- initialized: idle, accepts tokenize requests.
- building_special_tokens: builds token inventory for special-token parsing.
- special_tokens_decision: routes to raw or special partitioning.
- partitioning_raw/partitioning_with_specials: build fragment list.
- selecting_backend: binds encoder context and selects encoder machine.
- prefix_decision: applies optional BOS prefix or errors.
- encoding_ready/encoding_*: encodes fragments in a bounded loop.
- suffix_decision: applies optional SEP/EOS suffix or errors.
- finalizing: marks success.
- done: last request completed successfully.
- errored: last request failed with an error code.
- unexpected: sequencing contract violation.

Key invariants
- Per-request outputs are written only through the triggering event payload.
- Context owns only runtime data (fragments, encoder context, counters).
- Internal progress uses anonymous transitions (no self-dispatch).

Guard semantics
- can_tokenize: validates request pointers and capacity.
- phase_ok/phase_failed: observe errors set by actions.
- has_special_tokens: indicates whether special-token inventory is available.
- has_capacity: checks remaining output capacity before encoding.
- should_add_bos/sep/eos: determines prefix/suffix requirements.
- has_more_fragments: indicates more fragments to encode.

Action side effects
- begin_tokenize: resets request outputs and context runtime state.
- build_special_tokens: builds special-token inventory.
- partition_raw/partition_with_specials: builds fragment list, honoring
parse_special.
- append_bos/sep/eos: appends prefix/suffix tokens as configured by vocab.
- append_fragment_token/encode_raw_fragment: encode a fragment or append a
literal token.
- set_capacity_error/set_invalid_id_error: records validation failures.
- finalize: marks success.
- on_unexpected: reports sequencing violations.
*/
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<initialized> +
            sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<building_special_tokens>,
        sml::state<initialized> +
            sml::event<event::tokenize> / action::reject_invalid =
            sml::state<errored>,

        sml::state<done> + sml::event<event::tokenize>[guard::can_tokenize{}] /
                               action::begin_tokenize =
            sml::state<building_special_tokens>,
        sml::state<done> + sml::event<event::tokenize> /
                               action::reject_invalid = sml::state<errored>,

        sml::state<errored> +
            sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<building_special_tokens>,
        sml::state<errored> + sml::event<event::tokenize> /
                                  action::reject_invalid = sml::state<errored>,

        sml::state<unexpected> +
            sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<building_special_tokens>,
        sml::state<unexpected> +
            sml::event<event::tokenize> / action::reject_invalid =
            sml::state<unexpected>,

        sml::state<building_special_tokens> / action::build_special_tokens =
            sml::state<special_tokens_decision>,
        sml::state<special_tokens_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<special_tokens_decision>[guard::has_special_tokens{}] =
            sml::state<partitioning_with_specials>,
        sml::state<special_tokens_decision>[guard::no_special_tokens{}] =
            sml::state<partitioning_raw>,

        sml::state<partitioning_raw> / action::partition_raw =
            sml::state<partitioning_decision>,
        sml::state<partitioning_with_specials> /
            action::partition_with_specials = sml::state<partitioning_decision>,
        sml::state<partitioning_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<partitioning_decision>[guard::phase_ok{}] =
            sml::state<selecting_backend>,

        sml::state<selecting_backend> / action::select_backend =
            sml::state<selecting_backend_decision>,
        sml::state<selecting_backend_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<selecting_backend_decision>[guard::phase_ok{}] =
            sml::state<prefix_decision>,

        sml::state<prefix_decision>[guard::should_add_bos{} &&
                                    guard::bos_id_valid{} &&
                                    guard::has_capacity{}] /
            action::append_bos = sml::state<encoding_ready>,
        sml::state<prefix_decision>[guard::should_add_bos{} &&
                                    guard::bos_id_valid{} &&
                                    guard::no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<prefix_decision>[guard::should_add_bos{} &&
                                    guard::bos_id_invalid{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<prefix_decision>[guard::no_prefix{}] =
            sml::state<encoding_ready>,

        sml::state<encoding_ready>[guard::no_more_fragments{}] =
            sml::state<suffix_decision>,
        sml::state<encoding_ready>[guard::has_more_fragments{} &&
                                   guard::no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<encoding_ready>[guard::has_more_fragments{} &&
                                   guard::fragment_is_token{}] =
            sml::state<encoding_token_fragment>,
        sml::state<encoding_ready>[guard::has_more_fragments{} &&
                                   guard::fragment_is_raw{}] =
            sml::state<encoding_raw_fragment>,

        sml::state<encoding_token_fragment> / action::append_fragment_token =
            sml::state<encoding_decision>,
        sml::state<encoding_raw_fragment> / action::encode_raw_fragment =
            sml::state<encoding_decision>,
        sml::state<encoding_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<encoding_decision>[guard::phase_ok{}] =
            sml::state<encoding_ready>,

        sml::state<suffix_decision>[guard::should_add_sep{} &&
                                    guard::sep_id_valid{} &&
                                    guard::has_capacity{}] /
            action::append_sep = sml::state<finalizing>,
        sml::state<suffix_decision>[guard::should_add_sep{} &&
                                    guard::sep_id_valid{} &&
                                    guard::no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::should_add_sep{} &&
                                    guard::sep_id_invalid{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::should_add_eos{} &&
                                    guard::eos_id_valid{} &&
                                    guard::has_capacity{}] /
            action::append_eos = sml::state<finalizing>,
        sml::state<suffix_decision>[guard::should_add_eos{} &&
                                    guard::eos_id_valid{} &&
                                    guard::no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::should_add_eos{} &&
                                    guard::eos_id_invalid{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::no_suffix{}] =
            sml::state<finalizing>,

        sml::state<finalizing> / action::finalize = sml::state<done>,

        sml::state<initialized> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<building_special_tokens> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<special_tokens_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<partitioning_raw> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<partitioning_with_specials> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<partitioning_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<selecting_backend> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<selecting_backend_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<prefix_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_ready> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_token_fragment> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_raw_fragment> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<encoding_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<suffix_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<finalizing> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<done> + sml::unexpected_event<sml::_> /
                               action::on_unexpected = sml::state<unexpected>,
        sml::state<errored> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::tokenize &ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<done>);
    const int32_t err =
        ok ? EMEL_OK
           : (context_.last_error != EMEL_OK ? context_.last_error
                                             : EMEL_ERR_BACKEND);

    if (ev.token_count_out != nullptr) {
      *ev.token_count_out = context_.token_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm,
                         events::tokenizer_done{&ev, context_.token_count});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::tokenizer_error{&ev, err});
      }
    }

    action::clear_request(context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }
  int32_t token_count() const noexcept { return context_.token_count; }

private:
  action::context context_{};
};

} // namespace emel::tokenizer
