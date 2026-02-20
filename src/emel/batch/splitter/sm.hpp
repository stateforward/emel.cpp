#pragma once

#include <cstdint>

#include "emel/batch/splitter/actions.hpp"
#include "emel/batch/splitter/events.hpp"
#include "emel/batch/splitter/guards.hpp"
#include "emel/sm.hpp"

namespace emel::batch::splitter {

// Batch splitter contract (llama parity):
// - `event::split` requires `on_done` and `on_error` callbacks on every request.
// - Sequence sets are provided via `seq_masks` with `seq_mask_words` 64-bit words per token.
//   The layout is `[n_tokens * seq_mask_words]`. Supported width is 1..SEQ_WORDS (up to 256 seqs).
//   Example (`seq_mask_words = 2`):
//     token 0 mask words at indices [0,1], token 1 at [2,3], etc.
//     uint64_t masks[] = {t0_w0, t0_w1, t1_w0, t1_w1, ...};
// - If `seq_masks` is null, `seq_primary_ids` may provide the single sequence id per token.
//   If both are null, all tokens are treated as belonging to sequence 0.
// - `equal_sequential == true` requires `seq_primary_ids` and rejects coupled sequences
//   (masks with more than one bit set), matching llama's sequential split restriction.
// - Output selection:
//   - `output_all == true` marks all tokens as outputs (overrides `output_mask`).
//   - When `output_mask` is null and `output_all == false`, only the last token is output.
//   - Otherwise `output_mask` marks per-token outputs.
// - `total_outputs` is derived from the output selection rules above.
// - `splitting_done` includes `ubatch_token_indices` and `ubatch_token_offsets` so callers can
//   reconstruct per-ubatch token ordering (matching llama's split ordering).
// - Equal mode groups non-overlapping sequence sets and fills ubatches up to `n_ubatch`.
// - Seq mode builds one sequence-set ubatch at a time using subset expansion.

// Ready state. Invariant: no active split in progress.
struct initialized {};
// Validates inputs copied into context by begin_split.
struct validating {};
// Normalizes batch sizing parameters.
struct normalizing_batch {};
// Selects split algorithm based on request mode.
struct selecting_mode {};
// Computes micro-batch boundaries (simple mode).
struct splitting_simple {};
// Computes micro-batch boundaries (equal mode).
struct splitting_equal {};
// Computes micro-batch boundaries (equal mode, primary-id fast path).
struct splitting_equal_primary {};
// Computes micro-batch boundaries (seq mode).
struct splitting_seq {};
// Checks output capacity and finalizes results.
struct publishing {};
// Terminal success state.
struct done {};
// Terminal error: invalid arguments or callback contract.
struct invalid_request {};
// Terminal error: split computation failed.
struct split_failed {};
// Terminal error: unexpected event.
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::split>[guard::callbacks_are_valid] /
          action::begin_split = sml::state<validating>,
      sml::state<initialized> + sml::event<event::split>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<validating>[guard::inputs_are_valid] = sml::state<normalizing_batch>,
      sml::state<validating>[guard::inputs_are_invalid] = sml::state<invalid_request>,

      sml::state<normalizing_batch> + sml::on_entry<sml::_> / action::normalize_batch,
      sml::state<normalizing_batch> = sml::state<selecting_mode>,

      sml::state<selecting_mode>[guard::mode_is_simple] = sml::state<splitting_simple>,
      sml::state<selecting_mode>[guard::mode_is_equal_primary_fast] =
          sml::state<splitting_equal_primary>,
      sml::state<selecting_mode>[guard::mode_is_equal] = sml::state<splitting_equal>,
      sml::state<selecting_mode>[guard::mode_is_seq] = sml::state<splitting_seq>,
      sml::state<selecting_mode> = sml::state<invalid_request>,

      sml::state<splitting_simple> + sml::on_entry<sml::_> / action::create_ubatches_simple,
      sml::state<splitting_simple>[guard::split_succeeded] = sml::state<publishing>,
      sml::state<splitting_simple>[guard::split_failed] = sml::state<split_failed>,

      sml::state<splitting_equal> + sml::on_entry<sml::_> / action::create_ubatches_equal,
      sml::state<splitting_equal>[guard::split_succeeded] = sml::state<publishing>,
      sml::state<splitting_equal>[guard::split_failed] = sml::state<split_failed>,

      sml::state<splitting_equal_primary> + sml::on_entry<sml::_> /
          action::create_ubatches_equal_primary,
      sml::state<splitting_equal_primary>[guard::split_succeeded] =
          sml::state<publishing>,
      sml::state<splitting_equal_primary>[guard::split_failed] =
          sml::state<split_failed>,

      sml::state<splitting_seq> + sml::on_entry<sml::_> / action::create_ubatches_seq,
      sml::state<splitting_seq>[guard::split_succeeded] = sml::state<publishing>,
      sml::state<splitting_seq>[guard::split_failed] = sml::state<split_failed>,

      sml::state<publishing> + sml::on_entry<sml::_> / action::publish,
      sml::state<publishing> = sml::state<done>,

      sml::state<done> + sml::event<event::split>[guard::callbacks_are_valid] /
          action::begin_split = sml::state<validating>,
      sml::state<done> + sml::event<event::split>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<invalid_request> + sml::event<event::split>[guard::callbacks_are_valid] /
          action::begin_split = sml::state<validating>,
      sml::state<invalid_request> + sml::event<event::split>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<split_failed> + sml::event<event::split>[guard::callbacks_are_valid] /
          action::begin_split = sml::state<validating>,
      sml::state<split_failed> + sml::event<event::split>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<unexpected_event> + sml::event<event::split>[guard::callbacks_are_valid] /
          action::begin_split = sml::state<validating>,
      sml::state<unexpected_event> + sml::event<event::split>[guard::callbacks_are_invalid] =
          sml::state<invalid_request>,

      sml::state<initialized> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<validating> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<normalizing_batch> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<selecting_mode> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<splitting_simple> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<splitting_equal> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<splitting_equal_primary> + sml::unexpected_event<sml::_> =
          sml::state<unexpected_event>,
      sml::state<splitting_seq> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<publishing> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<done> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<invalid_request> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<split_failed> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>,
      sml::state<unexpected_event> + sml::unexpected_event<sml::_> = sml::state<unexpected_event>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::split & ev) {
    namespace sml = boost::sml;
    const bool accepted = base_type::process_event(ev);

    if (this->is(sml::state<done>)) {
      action::dispatch_done(ev, context_);
    } else if (this->is(sml::state<invalid_request>)) {
      action::dispatch_invalid_request(ev);
    } else if (this->is(sml::state<split_failed>)) {
      action::dispatch_split_failed(ev);
    } else if (this->is(sml::state<unexpected_event>)) {
      action::dispatch_unexpected(ev);
    }

    return accepted;
  }

  template <class Event>
  bool process_event(const Event & ev) {
    return base_type::process_event(ev);
  }

 private:
  action::context context_{};
};

}  // namespace emel::batch::splitter
