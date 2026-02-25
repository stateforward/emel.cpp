#pragma once

#include <cstdint>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/guards.hpp"
#include "emel/batch/planner/modes/equal/sm.hpp"
#include "emel/batch/planner/modes/sequential/sm.hpp"
#include "emel/batch/planner/modes/simple/sm.hpp"
#include "emel/sm.hpp"

namespace emel::batch::planner {

// batch planner contract (llama parity):
// - `event::request` requires `on_done` and `on_error` callbacks on every request.
// - sequence sets are provided via `seq_masks` with `seq_mask_words` 64-bit words per token.
//   the layout is `[n_tokens * seq_mask_words]`. supported width is 1..SEQ_WORDS (up to 256 seqs).
//   example (`seq_mask_words = 2`):
//     token 0 mask words at indices [0,1], token 1 at [2,3], etc.
//     uint64_t masks[] = {t0_w0, t0_w1, t1_w0, t1_w1, ...};
// - if `seq_masks` is null, `seq_primary_ids` may provide the single sequence id per token.
//   if both are null, all tokens are treated as belonging to sequence 0.
// - `equal_sequential == true` requires `seq_primary_ids` and rejects coupled sequences
//   (masks with more than one bit set), matching llama's sequential split restriction.
// - output selection:
//   - `output_all == true` marks all tokens as outputs (overrides `output_mask`).
//   - when `output_mask` is null and `output_all == false`, only the last token is output.
//   - otherwise `output_mask` marks per-token outputs.
// - `total_outputs` is derived from the output selection rules above.
// - `plan_done` includes `step_token_indices` and `step_token_offsets` so callers can
//   reconstruct per-step token ordering (matching llama's split ordering).
// - equal mode groups non-overlapping sequence sets and fills steps up to `n_steps`.
// - sequential/seq mode builds one sequence-set step at a time using subset expansion.

// ready state. invariant: no active plan in progress.
struct initialized {};
// validates input payload on current request.
struct validate_decision {};
// normalizes batch sizing parameters.
struct normalizing_batch {};
// delegates planning algorithm by request mode to mode submachines.
struct mode_decision {};
// checks output capacity and finalizes results.
struct publishing {};
// terminal success state.
struct done {};
// terminal error: invalid arguments.
struct invalid_request {};
// terminal error: split computation failed.
struct plan_failed {};
// terminal error: unexpected event.
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<validate_decision> <= *sml::state<initialized> + sml::event<event::request>
          / action::begin_plan
      //------------------------------------------------------------------------------//
      , sml::state<normalizing_batch> <= sml::state<validate_decision>
          + sml::completion<event::request> [ guard::inputs_are_valid ]
      , sml::state<invalid_request> <= sml::state<validate_decision>
          + sml::completion<event::request> / action::dispatch_invalid_request_default
      //------------------------------------------------------------------------------//
      , sml::state<mode_decision> <= sml::state<normalizing_batch>
          + sml::completion<event::request> / action::normalize_batch
      //------------------------------------------------------------------------------//
      , sml::state<modes::simple::model> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_simple ]
      , sml::state<modes::equal::model> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_equal ]
      , sml::state<modes::sequential::model> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_seq ]
      , sml::state<invalid_request> <= sml::state<mode_decision>
          + sml::completion<event::request> [ guard::mode_is_invalid ] / action::dispatch_invalid_mode
      //------------------------------------------------------------------------------//
      , sml::state<publishing> <= sml::state<modes::simple::model>
          + sml::completion<event::request> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::simple::model>
          + sml::completion<event::request> [ guard::planning_failed ]
      , sml::state<publishing> <= sml::state<modes::equal::model>
          + sml::completion<event::request> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::equal::model>
          + sml::completion<event::request> [ guard::planning_failed ]
      , sml::state<publishing> <= sml::state<modes::sequential::model>
          + sml::completion<event::request> [ guard::planning_succeeded ] / action::publish
      , sml::state<plan_failed> <= sml::state<modes::sequential::model>
          + sml::completion<event::request> [ guard::planning_failed ]
      //------------------------------------------------------------------------------//
      , sml::state<done> <= sml::state<publishing>
          + sml::completion<event::request> / action::dispatch_done
      //------------------------------------------------------------------------------//
      , sml::state<validate_decision> <= sml::state<done> + sml::event<event::request> / action::begin_plan
      , sml::state<validate_decision> <= sml::state<invalid_request>
          + sml::event<event::request> / action::begin_plan
      , sml::state<done> <= sml::state<plan_failed>
          + sml::completion<event::request> / action::dispatch_plan_failed_default
      , sml::state<validate_decision> <= sml::state<plan_failed>
          + sml::event<event::request> / action::begin_plan
      , sml::state<validate_decision> <= sml::state<unexpected_event>
          + sml::event<event::request> / action::begin_plan
      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<initialized> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<validate_decision> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<normalizing_batch> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<mode_decision> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<publishing> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<done> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<invalid_request> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<plan_failed> + sml::unexpected_event<sml::_>
      , sml::state<unexpected_event> <= sml::state<unexpected_event> + sml::unexpected_event<sml::_>
    );
    // clang-format on
  }
};

using sm = emel::sm_with_context<model, action::context>;
}  // namespace emel::batch::planner
