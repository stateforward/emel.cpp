#pragma once

#include <cstdint>

#include "emel/batch/planner/actions.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/batch/planner/guards.hpp"
#include "emel/sm.hpp"

namespace emel::batch::planner {

// batch planner contract (llama parity):
// - `event::plan` requires `on_done` and `on_error` callbacks on every request.
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
// - `plan_done` includes `ubatch_token_indices` and `ubatch_token_offsets` so callers can
//   reconstruct per-ubatch token ordering (matching llama's split ordering).
// - equal mode groups non-overlapping sequence sets and fills ubatches up to `n_ubatch`.
// - seq mode builds one sequence-set ubatch at a time using subset expansion.

// ready state. invariant: no active plan in progress.
struct initialized {};
// validates inputs copied into context by begin_plan.
struct validating {};
// normalizes batch sizing parameters.
struct normalizing_batch {};
// selects planning algorithm based on request mode.
struct selecting_mode {};
// computes micro-batch boundaries (simple mode).
struct planning_simple {};
// computes micro-batch boundaries (equal mode).
struct planning_equal {};
// computes micro-batch boundaries (equal mode, primary-id fast path).
struct planning_equal_primary {};
// computes micro-batch boundaries (seq mode).
struct planning_seq {};
// checks output capacity and finalizes results.
struct publishing {};
// terminal success state.
struct done {};
// terminal error: invalid arguments or callback contract.
struct invalid_request {};
// terminal error: split computation failed.
struct plan_failed {};
// terminal error: unexpected event.
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::plan>[guard::callbacks_are_valid] /
          action::begin_plan = sml::state<validating>,
      sml::state<initialized> + sml::event<event::plan>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<validating>[guard::inputs_are_valid] = sml::state<normalizing_batch>,
      sml::state<validating>[guard::inputs_are_invalid] = sml::state<invalid_request>,

      sml::state<normalizing_batch> + sml::on_entry<sml::_> / action::normalize_batch,
      sml::state<normalizing_batch> = sml::state<selecting_mode>,

      sml::state<selecting_mode>[guard::mode_is_simple] = sml::state<planning_simple>,
      sml::state<selecting_mode>[guard::mode_is_equal_primary_fast] =
          sml::state<planning_equal_primary>,
      sml::state<selecting_mode>[guard::mode_is_equal] = sml::state<planning_equal>,
      sml::state<selecting_mode>[guard::mode_is_seq] = sml::state<planning_seq>,
      sml::state<selecting_mode> = sml::state<invalid_request>,

      sml::state<planning_simple> + sml::on_entry<sml::_> / action::create_ubatches_simple,
      sml::state<planning_simple>[guard::plan_succeeded] = sml::state<publishing>,
      sml::state<planning_simple>[guard::plan_failed] = sml::state<plan_failed>,

      sml::state<planning_equal> + sml::on_entry<sml::_> / action::create_ubatches_equal,
      sml::state<planning_equal>[guard::plan_succeeded] = sml::state<publishing>,
      sml::state<planning_equal>[guard::plan_failed] = sml::state<plan_failed>,

      sml::state<planning_equal_primary> + sml::on_entry<sml::_> /
          action::create_ubatches_equal_primary,
      sml::state<planning_equal_primary>[guard::plan_succeeded] =
          sml::state<publishing>,
      sml::state<planning_equal_primary>[guard::plan_failed] =
          sml::state<plan_failed>,

      sml::state<planning_seq> + sml::on_entry<sml::_> / action::create_ubatches_seq,
      sml::state<planning_seq>[guard::plan_succeeded] = sml::state<publishing>,
      sml::state<planning_seq>[guard::plan_failed] = sml::state<plan_failed>,

      sml::state<publishing> + sml::on_entry<sml::_> / action::publish,
      sml::state<publishing> = sml::state<done>,

      sml::state<done> + sml::event<event::plan>[guard::callbacks_are_valid] /
          action::begin_plan = sml::state<validating>,
      sml::state<done> + sml::event<event::plan>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<invalid_request> + sml::event<event::plan>[guard::callbacks_are_valid] /
          action::begin_plan = sml::state<validating>,
      sml::state<invalid_request> + sml::event<event::plan>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<plan_failed> + sml::event<event::plan>[guard::callbacks_are_valid] /
          action::begin_plan = sml::state<validating>,
      sml::state<plan_failed> + sml::event<event::plan>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<unexpected_event> + sml::event<event::plan>[guard::callbacks_are_valid] /
          action::begin_plan = sml::state<validating>,
      sml::state<unexpected_event> + sml::event<event::plan>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<initialized> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<validating> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<normalizing_batch> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<selecting_mode> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<planning_simple> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<planning_equal> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<planning_equal_primary> + sml::unexpected_event<sml::_> =
          sml::state<unexpected_event>,
      sml::state<planning_seq> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<publishing> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<done> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<invalid_request> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<plan_failed> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<unexpected_event> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::plan & ev) {
    namespace sml = boost::sml;
    const bool accepted = base_type::process_event(ev);

    if (this->is(sml::state<done>)) {
      action::dispatch_done(ev, context_);
    } else if (this->is(sml::state<invalid_request>)) {
      action::dispatch_invalid_request(ev);
    } else if (this->is(sml::state<plan_failed>)) {
      action::dispatch_plan_failed(ev);
    } else if (this->is(sml::state<unexpected_event>)) {
      action::dispatch_unexpected(ev);
    }

    return accepted;
  }

  template <class event>
  bool process_event(const event & ev) {
    return base_type::process_event(ev);
  }

 private:
  action::context context_{};
};

}  // namespace emel::batch::planner
