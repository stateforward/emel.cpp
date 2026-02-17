#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/tokenizer/actions.hpp"
#include "emel/tokenizer/events.hpp"
#include "emel/tokenizer/guards.hpp"

namespace emel::tokenizer {

using Process = process_t;

/*
Tokenizer architecture notes (single source of truth)

Scope
- Component boundary: tokenizer
- Goal: tokenize text into token ids with special-token partitioning and model-aware encoding.

State purpose
- initialized: idle, accepts tokenize requests.
- building_special_tokens: builds token inventory for special-token parsing.
- partitioning_raw: treats entire text as a single raw fragment when no special tokens apply.
- partitioning_with_specials: splits raw text around special tokens when enabled.
- selecting_backend: binds the encoder context to the current vocab.
- encoding_fragments: encodes raw fragments and appends literal special tokens.
- finalizing: writes final outputs and transitions to done.
- done: last request completed successfully.
- errored: last request failed with an error code.
- unexpected: sequencing contract violation (event not valid in current state).

Key invariants
- Per-request outputs are written only through the triggering event payload.
- Context owns only runtime data (fragments, encoder context, counters).
- Internal dispatch uses boost::sml::back::process exclusively.

Guard semantics
- can_tokenize: validates request pointers and capacity.
- has_special_tokens: indicates whether special-token inventory is available.
- has_capacity: checks remaining output capacity before encoding.
- should_add_bos/sep/eos: determines prefix/suffix requirements.
- has_more_fragments: indicates more fragments to encode.

Action side effects
- begin_tokenize: resets request outputs and context runtime state.
- partition_special: builds special-token inventory.
- partition_raw/partition_with_specials: builds fragment list, honoring parse_special.
- append_bos/sep/eos: appends prefix/suffix tokens as configured by vocab.
- encode_fragment: encodes one fragment or appends a literal token.
- dispatch_capacity_error: reports insufficient output capacity.
- finalize: writes final counts and completes the request.
- dispatch_done/dispatch_error: forwards terminal outcome to owner machine.
- dispatch_unexpected: reports sequencing violations to owner.
*/
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct building_special_tokens {};
    struct partitioning_raw {};
    struct partitioning_with_specials {};
    struct selecting_backend {};
    struct encoding_fragments {};
    struct finalizing {};
    struct done {};
    struct errored {};
    struct unexpected {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::tokenize>[guard::can_tokenize{}] /
          action::begin_tokenize = sml::state<building_special_tokens>,
      sml::state<initialized> + sml::event<event::tokenize> / action::reject_invalid =
          sml::state<errored>,

      sml::state<done> + sml::event<event::tokenize>[guard::can_tokenize{}] / action::begin_tokenize =
          sml::state<building_special_tokens>,
      sml::state<done> + sml::event<event::tokenize> / action::reject_invalid =
          sml::state<errored>,

      sml::state<errored> + sml::event<event::tokenize>[guard::can_tokenize{}] / action::begin_tokenize =
          sml::state<building_special_tokens>,
      sml::state<errored> + sml::event<event::tokenize> / action::reject_invalid =
          sml::state<errored>,

      sml::state<unexpected> + sml::event<event::tokenize>[guard::can_tokenize{}] / action::begin_tokenize =
          sml::state<building_special_tokens>,
      sml::state<unexpected> + sml::event<event::tokenize> / action::reject_invalid =
          sml::state<unexpected>,

      sml::state<building_special_tokens> + sml::on_entry<event::tokenize> / action::partition_special,
      sml::state<building_special_tokens> + sml::event<event::special_tokens_ready>[guard::has_special_tokens{}] =
          sml::state<partitioning_with_specials>,
      sml::state<building_special_tokens> + sml::event<event::special_tokens_ready>[guard::no_special_tokens{}] =
          sml::state<partitioning_raw>,
      sml::state<building_special_tokens> + sml::event<event::partitioning_special_error> =
          sml::state<errored>,

      sml::state<partitioning_raw> + sml::on_entry<event::special_tokens_ready> / action::partition_raw,
      sml::state<partitioning_raw> + sml::event<event::partitioning_special_done> =
          sml::state<selecting_backend>,
      sml::state<partitioning_raw> + sml::event<event::partitioning_special_error> =
          sml::state<errored>,

      sml::state<partitioning_with_specials> + sml::on_entry<event::special_tokens_ready> /
          action::partition_with_specials,
      sml::state<partitioning_with_specials> + sml::event<event::partitioning_special_done> =
          sml::state<selecting_backend>,
      sml::state<partitioning_with_specials> + sml::event<event::partitioning_special_error> =
          sml::state<errored>,

      sml::state<selecting_backend> + sml::on_entry<event::partitioning_special_done> /
          action::select_backend,
      sml::state<selecting_backend> + sml::event<event::selecting_backend_done>
          [guard::should_add_bos{} && guard::bos_id_valid{} && guard::has_capacity{}] /
          action::append_bos = sml::state<encoding_fragments>,
      sml::state<selecting_backend> + sml::event<event::selecting_backend_done>
          [guard::should_add_bos{} && guard::bos_id_valid{} && guard::no_capacity{}] /
          action::emit_prefix_capacity_error,
      sml::state<selecting_backend> + sml::event<event::selecting_backend_done>
          [guard::should_add_bos{} && guard::bos_id_invalid{}] /
          action::emit_prefix_invalid_id_error,
      sml::state<selecting_backend> + sml::event<event::selecting_backend_done>
          [guard::no_prefix{}] = sml::state<encoding_fragments>,
      sml::state<selecting_backend> + sml::event<event::applying_special_prefix_error> =
          sml::state<errored>,
      sml::state<selecting_backend> + sml::event<event::selecting_backend_error> =
          sml::state<errored>,

      sml::state<encoding_fragments> + sml::on_entry<sml::_> / action::dispatch_next_fragment,
      sml::state<encoding_fragments> + sml::event<event::next_fragment>[guard::has_more_fragments{} && guard::has_capacity{}] /
          action::encode_fragment,
      sml::state<encoding_fragments> + sml::event<event::next_fragment>[guard::has_more_fragments{} && guard::no_capacity{}] /
          action::dispatch_capacity_error,
      sml::state<encoding_fragments> + sml::event<event::next_fragment>[guard::no_more_fragments{}] /
          action::dispatch_no_fragment_done,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>[guard::has_more_fragments{}] =
          sml::state<encoding_fragments>,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>
          [guard::no_more_fragments{} && guard::should_add_sep{} && guard::sep_id_valid{} &&
           guard::has_capacity{}] / action::append_sep = sml::state<finalizing>,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>
          [guard::no_more_fragments{} && guard::should_add_sep{} && guard::sep_id_valid{} &&
           guard::no_capacity{}] / action::emit_suffix_capacity_error,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>
          [guard::no_more_fragments{} && guard::should_add_sep{} && guard::sep_id_invalid{}] /
          action::emit_suffix_invalid_id_error,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>
          [guard::no_more_fragments{} && guard::should_add_eos{} && guard::eos_id_valid{} &&
           guard::has_capacity{}] / action::append_eos = sml::state<finalizing>,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>
          [guard::no_more_fragments{} && guard::should_add_eos{} && guard::eos_id_valid{} &&
           guard::no_capacity{}] / action::emit_suffix_capacity_error,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>
          [guard::no_more_fragments{} && guard::should_add_eos{} && guard::eos_id_invalid{}] /
          action::emit_suffix_invalid_id_error,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>
          [guard::no_more_fragments{} && guard::no_suffix{}] = sml::state<finalizing>,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_error> =
          sml::state<errored>,
      sml::state<encoding_fragments> + sml::event<event::applying_special_suffix_error> =
          sml::state<errored>,

      sml::state<finalizing> + sml::on_entry<event::encoding_fragment_done> / action::finalize,
      sml::state<finalizing> + sml::event<event::finalizing_done> = sml::state<done>,
      sml::state<finalizing> + sml::event<event::finalizing_error> = sml::state<errored>,

      sml::state<done> + sml::on_entry<event::finalizing_done> / action::dispatch_done,
      sml::state<errored> + sml::on_entry<event::tokenize> / action::dispatch_reject,
      sml::state<errored> + sml::on_entry<event::partitioning_special_error> /
          action::dispatch_error,
      sml::state<errored> + sml::on_entry<event::selecting_backend_error> /
          action::dispatch_error,
      sml::state<errored> + sml::on_entry<event::encoding_fragment_error> /
          action::dispatch_error,
      sml::state<errored> + sml::on_entry<event::applying_special_prefix_error> /
          action::dispatch_error,
      sml::state<errored> + sml::on_entry<event::applying_special_suffix_error> /
          action::dispatch_error,
      sml::state<errored> + sml::on_entry<event::finalizing_error> / action::dispatch_error,

      sml::state<initialized> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<building_special_tokens> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<partitioning_raw> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<partitioning_with_specials> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<selecting_backend> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<encoding_fragments> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<finalizing> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<done> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<errored> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>,
      sml::state<unexpected> + sml::event<sml::_> / action::dispatch_unexpected =
          sml::state<unexpected>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  explicit sm(action::context & ctx)
      : emel::detail::process_support<sm, Process>(this),
        base_type(ctx, this->process_) {}

  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;
};

}  // namespace emel::tokenizer
