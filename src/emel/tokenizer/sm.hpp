#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/tokenizer/actions.hpp"
#include "emel/tokenizer/events.hpp"
#include "emel/tokenizer/guards.hpp"

namespace emel::tokenizer {

struct uninitialized {};
struct binding_preprocessor {};
struct binding_preprocessor_decision {};
struct binding_encoder {};
struct binding_encoder_decision {};
struct idle {};
struct preprocessing {};
struct preprocess_decision {};
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
tokenizer architecture notes (single source of truth)

scope
- component boundary: tokenizer
- goal: tokenize text into token ids with special-token partitioning and
model-aware encoding.

state purpose
  - uninitialized: no bound vocab, awaits bind.
  - binding_preprocessor/binding_encoder: bind model-specific preprocess/encode stages.
  - idle: ready to tokenize requests.
  - preprocessing: dispatch preprocessor to build fragment list.
  - preprocess_decision: routes based on preprocess success/failure.
  - prefix_decision: applies optional BOS prefix or errors.
  - encoding_ready/encoding_*: encodes fragments in a bounded loop.
  - suffix_decision: applies optional SEP/EOS suffix or errors.
  - finalizing: marks success.
  - done: last request completed successfully.
- errored: last request failed with an error code.
- unexpected: sequencing contract violation.

key invariants
- per-request outputs are written only through the triggering event payload.
- context owns only runtime data (fragments, encoder context, counters).
- internal progress uses anonymous transitions (no self-dispatch).

guard semantics
  - can_bind: validates bind request pointers.
  - can_tokenize: validates request pointers, capacity, and bound vocab match.
  - phase_ok/phase_failed: observe errors set by actions.
  - has_capacity: checks remaining output capacity before encoding.
  - should_add_bos/sep/eos: determines prefix/suffix requirements.
  - has_more_fragments: indicates more fragments to encode.

action side effects
  - begin_bind: stores vocab and resets bind error state.
  - bind_preprocessor/bind_encoder: select backend machines for model.
  - begin_tokenize: resets request outputs and context runtime state.
  - run_preprocess: builds fragment list, honoring parse_special.
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
        *sml::state<uninitialized> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<uninitialized> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<uninitialized> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<binding_preprocessor> / action::bind_preprocessor =
            sml::state<binding_preprocessor_decision>,
        sml::state<binding_preprocessor_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<binding_preprocessor_decision>[guard::phase_ok{}] =
            sml::state<binding_encoder>,

        sml::state<binding_encoder> / action::bind_encoder =
            sml::state<binding_encoder_decision>,
        sml::state<binding_encoder_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<binding_encoder_decision>[guard::phase_ok{}] =
            sml::state<idle>,

        sml::state<idle> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<idle> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<idle> + sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<idle> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<done> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<done> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<done> + sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<done> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<errored> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<errored> + sml::event<event::bind> /
                action::reject_bind = sml::state<errored>,
        sml::state<errored> +
            sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<errored> + sml::event<event::tokenize> /
                action::reject_invalid = sml::state<errored>,

        sml::state<unexpected> + sml::event<event::bind>[guard::can_bind{}] /
                action::begin_bind = sml::state<binding_preprocessor>,
        sml::state<unexpected> + sml::event<event::bind> /
                action::reject_bind = sml::state<unexpected>,
        sml::state<unexpected> +
            sml::event<event::tokenize>[guard::can_tokenize{}] /
                action::begin_tokenize = sml::state<preprocessing>,
        sml::state<unexpected> +
            sml::event<event::tokenize> / action::reject_invalid =
            sml::state<unexpected>,

        sml::state<preprocessing> / action::run_preprocess =
            sml::state<preprocess_decision>,
        sml::state<preprocess_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<preprocess_decision>[guard::phase_ok{}] =
            sml::state<prefix_decision>,

        sml::state<prefix_decision>[guard::bos_ready{}] /
            action::append_bos = sml::state<encoding_ready>,
        sml::state<prefix_decision>[guard::bos_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<prefix_decision>[guard::bos_invalid_id{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<prefix_decision>[guard::no_prefix{}] =
            sml::state<encoding_ready>,

        sml::state<encoding_ready>[guard::no_more_fragments{}] =
            sml::state<suffix_decision>,
        sml::state<encoding_ready>[guard::more_fragments_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<encoding_ready>[guard::more_fragments_token{}] =
            sml::state<encoding_token_fragment>,
        sml::state<encoding_ready>[guard::more_fragments_raw{}] =
            sml::state<encoding_raw_fragment>,

        sml::state<encoding_token_fragment> / action::append_fragment_token =
            sml::state<encoding_decision>,
        sml::state<encoding_raw_fragment> / action::encode_raw_fragment =
            sml::state<encoding_decision>,
        sml::state<encoding_decision>[guard::phase_failed{}] =
            sml::state<errored>,
        sml::state<encoding_decision>[guard::phase_ok{}] =
            sml::state<encoding_ready>,

        sml::state<suffix_decision>[guard::sep_ready{}] /
            action::append_sep = sml::state<finalizing>,
        sml::state<suffix_decision>[guard::sep_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::sep_invalid_id{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::eos_ready{}] /
            action::append_eos = sml::state<finalizing>,
        sml::state<suffix_decision>[guard::eos_no_capacity{}] /
            action::set_capacity_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::eos_invalid_id{}] /
            action::set_invalid_id_error = sml::state<errored>,
        sml::state<suffix_decision>[guard::no_suffix{}] =
            sml::state<finalizing>,

        sml::state<finalizing> / action::finalize = sml::state<done>,

        sml::state<uninitialized> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_preprocessor> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_preprocessor_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_encoder> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<binding_encoder_decision> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<idle> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<preprocessing> +
            sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<preprocess_decision> +
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

  bool process_event(const event::bind &ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<idle>);
    const int32_t err =
        ok ? EMEL_OK
           : (context_.last_error != EMEL_OK ? context_.last_error
                                             : EMEL_ERR_BACKEND);

    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm, events::tokenizer_bind_done{&ev});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::tokenizer_bind_error{&ev, err});
      }
    }

    action::clear_request(context_);
    return accepted && ok;
  }

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
