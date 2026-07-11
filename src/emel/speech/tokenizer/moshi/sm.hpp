#pragma once

// benchmark: designed

#include <utility>

#include "emel/sm.hpp"
#include "emel/speech/tokenizer/moshi/actions.hpp"
#include "emel/speech/tokenizer/moshi/context.hpp"
#include "emel/speech/tokenizer/moshi/events.hpp"
#include "emel/speech/tokenizer/moshi/guards.hpp"

namespace emel::speech::tokenizer::moshi {

struct state_uninitialized {};
struct state_ready {};
struct state_prepared_full {};
struct state_prepared_generated {};
struct state_commit_full_zero {};
struct state_commit_full_generated {};
struct state_commit_generated_zero {};
struct state_commit_generated {};
struct state_output_decision {};
struct state_output_validation {};
struct state_errored {};

/*
Moshi tokenizer architecture notes (single source of truth)

state purpose
- state_ready accepts the first half of a frame and tokenizes the caller's
  audio codebooks into the delay-aligned LM input sequence.
- state_prepared_full records that all codebooks were provided, while
  state_prepared_generated records tail-only or generator-owned input. The
  state graph therefore owns cache-write policy without a persistent flag.
- commit states select initial-delay replacement and provided-cache
  preservation before executing fixed data-plane cache writes.

control invariants
- delay geometry, token sentinels, and cache storage are injected; the actor
  owns no global limits and allocates nothing during construction or dispatch.
- tokenize and detokenize must alternate. Phase-order, shape, overflow,
  initialization, and restore failures are explicit transitions.
- context contains only the persistent frame offset and injected storage.
  Per-dispatch source offsets travel through the typed detokenize_run event.

guard semantics and action side effects
- configuration guards validate every injected extent and delay before the
  cache is initialized.
- tokenize guards select full, tail, or empty input shapes; their actions write
  delayed inputs and publish one model-token frame.
- detokenize guards select generated versus zero audio and overflow outcomes;
  commit actions advance once, update the ring, and output actions reverse the
  codebook delay pattern.
*/
struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    using effect_commit_full_zero = action::effect_commit_generated<
        action::cache_write_mode::preserve_provided,
        action::audio_write_mode::zero>;
    using effect_commit_full_generated = action::effect_commit_generated<
        action::cache_write_mode::preserve_provided,
        action::audio_write_mode::generated>;
    using effect_commit_zero =
        action::effect_commit_generated<action::cache_write_mode::replace,
                                        action::audio_write_mode::zero>;
    using effect_commit_generated =
        action::effect_commit_generated<action::cache_write_mode::replace,
                                        action::audio_write_mode::generated>;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialization and recovery validate only injected geometry.
        sml::state<state_ready> <= *sml::state<state_uninitialized>
          + sml::event<event::initialize> [ guard::guard_configuration_valid{} ]
          / action::effect_initialize{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::initialize> [ guard::guard_configuration_invalid{} ]
          / action::effect_reject<error::invalid_configuration>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::initialize>
          / action::effect_reject<error::already_initialized>{}
      , sml::state<state_prepared_full> <= sml::state<state_prepared_full>
          + sml::event<event::initialize>
          / action::effect_reject<error::already_initialized>{}
      , sml::state<state_prepared_generated> <= sml::state<state_prepared_generated>
          + sml::event<event::initialize>
          / action::effect_reject<error::already_initialized>{}
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::event<event::initialize> [ guard::guard_configuration_valid{} ]
          / action::effect_initialize{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::initialize> [ guard::guard_configuration_invalid{} ]
          / action::effect_reject<error::invalid_configuration>{}

      //------------------------------------------------------------------------------//
      // Tokenization writes delayed caller input and records the cache policy
      // for the matching detokenize dispatch in state.
      , sml::state<state_prepared_full> <= sml::state<state_ready>
          + sml::event<event::tokenize> [ guard::guard_tokenize_full{} ]
          / action::effect_tokenize_full{}
      , sml::state<state_prepared_generated> <= sml::state<state_ready>
          + sml::event<event::tokenize> [ guard::guard_tokenize_tail{} ]
          / action::effect_tokenize_tail{}
      , sml::state<state_prepared_generated> <= sml::state<state_ready>
          + sml::event<event::tokenize> [ guard::guard_tokenize_empty{} ]
          / action::effect_tokenize_empty{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::tokenize> [ guard::guard_tokenize_invalid{} ]
          / action::effect_reject<error::request_shape>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::tokenize>
          [ guard::guard_tokenize_position_overflow{} ]
          / action::effect_reject<error::position_overflow>{}
      , sml::state<state_prepared_full> <= sml::state<state_prepared_full>
          + sml::event<event::tokenize>
          / action::effect_reject<error::phase_order>{}
      , sml::state<state_prepared_generated> <= sml::state<state_prepared_generated>
          + sml::event<event::tokenize>
          / action::effect_reject<error::phase_order>{}

      //------------------------------------------------------------------------------//
      // Detokenization explicitly selects full-input preservation and initial
      // delay replacement before the fixed commit action executes.
      , sml::state<state_commit_full_zero> <= sml::state<state_prepared_full>
          + sml::event<event::detokenize_run>
          [ guard::guard_detokenize_valid_replace{} ]
          / action::effect_begin_detokenize{}
      , sml::state<state_commit_full_generated> <= sml::state<state_prepared_full>
          + sml::event<event::detokenize_run>
          [ guard::guard_detokenize_valid_generated{} ]
          / action::effect_begin_detokenize{}
      , sml::state<state_commit_generated_zero> <= sml::state<state_prepared_generated>
          + sml::event<event::detokenize_run>
          [ guard::guard_detokenize_valid_replace{} ]
          / action::effect_begin_detokenize{}
      , sml::state<state_commit_generated> <= sml::state<state_prepared_generated>
          + sml::event<event::detokenize_run>
          [ guard::guard_detokenize_valid_generated{} ]
          / action::effect_begin_detokenize{}
      , sml::state<state_prepared_full> <= sml::state<state_prepared_full>
          + sml::event<event::detokenize_run>
          [ guard::guard_detokenize_shape_invalid{} ]
          / action::effect_reject_detokenize<error::request_shape>{}
      , sml::state<state_prepared_generated> <= sml::state<state_prepared_generated>
          + sml::event<event::detokenize_run>
          [ guard::guard_detokenize_shape_invalid{} ]
          / action::effect_reject_detokenize<error::request_shape>{}
      , sml::state<state_prepared_full> <= sml::state<state_prepared_full>
          + sml::event<event::detokenize_run> [ guard::guard_position_overflow{} ]
          / action::effect_reject_detokenize<error::position_overflow>{}
      , sml::state<state_prepared_generated> <= sml::state<state_prepared_generated>
          + sml::event<event::detokenize_run> [ guard::guard_position_overflow{} ]
          / action::effect_reject_detokenize<error::position_overflow>{}
      , sml::state<state_output_decision> <= sml::state<state_commit_full_zero>
          + sml::completion<event::detokenize_run> / effect_commit_full_zero{}
      , sml::state<state_output_decision> <= sml::state<state_commit_full_generated>
          + sml::completion<event::detokenize_run> / effect_commit_full_generated{}
      , sml::state<state_output_decision> <= sml::state<state_commit_generated_zero>
          + sml::completion<event::detokenize_run> / effect_commit_zero{}
      , sml::state<state_output_decision> <= sml::state<state_commit_generated>
          + sml::completion<event::detokenize_run> / effect_commit_generated{}
      , sml::state<state_ready> <= sml::state<state_output_decision>
          + sml::completion<event::detokenize_run>
          [ guard::guard_before_output_delay{} ] / action::effect_publish_no_output{}
      , sml::state<state_output_validation> <= sml::state<state_output_decision>
          + sml::completion<event::detokenize_run>
          [ guard::guard_past_output_delay{} ] / action::effect_collect_output{}
      , sml::state<state_ready> <= sml::state<state_output_validation>
          + sml::completion<event::detokenize_run> [ guard::guard_output_complete{} ]
          / action::effect_publish_output{}
      , sml::state<state_ready> <= sml::state<state_output_validation>
          + sml::completion<event::detokenize_run> [ guard::guard_output_incomplete{} ]
          / action::effect_publish_no_output{}

      //------------------------------------------------------------------------------//
      // Cache restore and sequence reset preserve injected geometry.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::restore_cache> [ guard::guard_restore_valid{} ]
          / action::effect_restore_column_major_cache{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::restore_cache> [ guard::guard_restore_invalid{} ]
          / action::effect_reject<error::request_shape>{}
      , sml::state<state_prepared_full> <= sml::state<state_prepared_full>
          + sml::event<event::restore_cache>
          / action::effect_reject<error::phase_order>{}
      , sml::state<state_prepared_generated> <= sml::state<state_prepared_generated>
          + sml::event<event::restore_cache>
          / action::effect_reject<error::phase_order>{}
      , sml::state<state_ready> <= sml::state<state_prepared_full>
          + sml::event<event::advance>
          [ guard::guard_advance_position_available{} ] / action::effect_advance{}
      , sml::state<state_prepared_full> <= sml::state<state_prepared_full>
          + sml::event<event::advance>
          [ guard::guard_advance_position_overflow{} ]
          / action::effect_reject<error::position_overflow>{}
      , sml::state<state_prepared_generated> <= sml::state<state_prepared_generated>
          + sml::event<event::advance> / action::effect_reject<error::phase_order>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::advance> / action::effect_reject<error::phase_order>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::reset> / action::effect_reset{}
      , sml::state<state_ready> <= sml::state<state_prepared_full>
          + sml::event<event::reset> / action::effect_reset{}
      , sml::state<state_ready> <= sml::state<state_prepared_generated>
          + sml::event<event::reset> / action::effect_reset{}
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::event<event::reset> / action::effect_reset{}

      //------------------------------------------------------------------------------//
      // Known operations outside their legal phase are explicit.
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::tokenize> / action::effect_reject<error::uninitialized>{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::detokenize_run>
          / action::effect_reject_detokenize<error::uninitialized>{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::restore_cache>
          / action::effect_reject<error::uninitialized>{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::advance> / action::effect_reject<error::uninitialized>{}
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::reset> / action::effect_reject<error::uninitialized>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::detokenize_run>
          / action::effect_reject_detokenize<error::phase_order>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::tokenize> / action::effect_reject<error::internal_error>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::detokenize_run>
          / action::effect_reject_detokenize<error::internal_error>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::restore_cache>
          / action::effect_reject<error::internal_error>{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::event<event::advance> / action::effect_reject<error::internal_error>{}

      //------------------------------------------------------------------------------//
      // Unknown external events make the actor observably errored.
      , sml::state<state_errored> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_prepared_full>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_prepared_generated>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  explicit sm(const dependencies &deps) : base_type(std::in_place, deps) {}

  bool process_event(const event::detokenize &ev) {
    event::detokenize_ctx ctx{};
    event::detokenize_run runtime_ev{ev, ctx};
    return base_type::process_event(runtime_ev);
  }
};

} // namespace emel::speech::tokenizer::moshi
