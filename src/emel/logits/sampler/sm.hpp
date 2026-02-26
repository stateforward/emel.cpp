#pragma once

/*
design doc: docs/designs/logits/sampler.design.md
 ---
 title: logits/sampler architecture design
 status: draft
 ---
 
 # logits/sampler architecture design
 
 this document defines the logits/sampler pipeline. it takes a validated `candidate_view` for a single
 sequence and passes it through a configurable chain of sampling algorithms to select the final token.
 
 ## role
 - execute a strict, deterministic sequence of sampler algorithms (the "sampling chain").
 - select a single token ID from the `candidate_view`.
 - remain completely stateless across sequences and steps to allow pooling and reuse.
 
 ## architecture shift: sm_any and future co_sm
 to efficiently handle batch decoding and dynamic sampler configurations without virtual function
 overhead, this pipeline relies on `src/emel/sm.hpp`:
 
 1. **`emel::sm_any` for dynamic chains:** the "sampling chain" (e.g., Repetition Penalty -> Temperature
    -> Top-K -> Top-P) is implemented as a `std::vector<emel::sm_any<sampler_kind, ...>>`. `sm_any` acts
    as a fast, type-erased wrapper for the individual SML sampler actors. the chain storage is prepared
    before dispatch; the pipeline actor simply loops through it, dispatching `event::apply` synchronously.
 2. **future asynchronous execution (`emel::co_sm`):** initially, the pipeline is a standard
    synchronous `boost::sml` actor. in the future, it can be wrapped in `co_sm`, allowing the generator
    to dispatch `process_event_async` for each sequence concurrently and `co_await` completion.
    the underlying SML model requires no changes to support this transition.
 
 ## events
 - `event::sample`
   - inputs: a validated `candidate_view`, a sampling chain policy (the vector of `sm_any`
     samplers), sequence-specific state (like the `rng` seed or previous tokens), and optional
     synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: sets the final selected `token_id` in the caller-provided destination and invokes the
     appropriate callback before returning, avoiding state machine context reads.
 
 ## state model
 
 ```text
 idle ──► preparing_chain ──► applying_samplers ──► selecting_token ──► (done | errored)
   ▲                                                                        │
   └────────────────────────────────────────────────────────────────────────┘
 ```
 
 - `idle` — waiting for a `candidate_view`.
 - `preparing_chain` — binding the `sm_any` chain and sequence-specific state (RNG, previous tokens) from
   the event payload.
 - `applying_samplers` — a synchronous run-to-completion (RTC) loop that dispatches `event::apply` to
   each `sm_any` sampler in the chain sequentially. each sampler mutates the `candidate_view` in-place.
 - `selecting_token` — after the chain finishes, pick the final token. if candidates remain, the final
   sampler (usually a multinomial or greedy sampler) will have moved the winning candidate to the front.
 - `done` — selection complete, transitions back to `idle` emitting `events::sample_done`.
 
 ## responsibilities
 - **stateless application:** the pipeline and its internal samplers must not store sequence data (like
   "previously generated tokens") in their SML context. all sequence-specific state is passed ephemerally
   in the `event::sample` payload. this allows a small pool of sampler pipelines to service thousands of
   sequences interchangeably.
 - **in-place mutation:** samplers modify the `candidate_view` directly (e.g., by sorting it, truncating
   its count, or applying Softmax probabilities to the scores).
 
 ## determinism
 
 greedy sampling selects the lowest token id on tie (when multiple tokens share the maximum logit
 value). multinomial sampling accumulates the CDF in token index order. the first token whose
 cumulative probability exceeds the random threshold is selected. the PRNG is fixed and versioned.
 the default implementation uses `xoshiro256**` v1. changing the PRNG algorithm or version is a
 breaking change. given identical logits and identical PRNG seed, sampling results are identical
 across runs and platforms.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_INVALID_ARGUMENT` — the input payload contained invalid fields, such as NaN probabilities or an out-of-range sequence id.
 - `EMEL_ERR_CAPACITY` — the sampler chain length exceeds `k_max_chain_length` or the scratch buffer is too small.
 - `EMEL_ERR_EMPTY` — the candidate set is empty after all sampler transforms have been applied.
*/


#include "emel/logits/sampler/actions.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/logits/sampler/guards.hpp"
#include "emel/sm.hpp"

namespace emel::logits::sampler {

struct initialized {};
struct preparing_candidates {};
struct prepare_decision {};
struct sampling {};
struct sampling_decision {};
struct selecting_token {};
struct select_decision {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::sample> / action::begin_sample =
        sml::state<preparing_candidates>,

      sml::state<preparing_candidates> / action::run_prepare_candidates =
        sml::state<prepare_decision>,
      sml::state<prepare_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<prepare_decision> [guard::phase_ok_and_has_more_samplers{}] =
        sml::state<sampling>,
      sml::state<prepare_decision> [guard::phase_ok_and_no_more_samplers{}] =
        sml::state<selecting_token>,

      sml::state<sampling> / action::run_apply_sampling = sml::state<sampling_decision>,
      sml::state<sampling_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<sampling_decision> [guard::phase_ok_and_has_more_samplers{}] =
        sml::state<sampling>,
      sml::state<sampling_decision> [guard::phase_ok_and_no_more_samplers{}] =
        sml::state<selecting_token>,

      sml::state<selecting_token> / action::run_select_token = sml::state<select_decision>,
      sml::state<select_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<select_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::publish_done = sml::state<initialized>,
      sml::state<errored> / action::publish_error = sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<preparing_candidates> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<prepare_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<sampling> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<sampling_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<selecting_token> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<select_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t selected_token() const noexcept { return context_.selected_token; }

 private:
  action::context context_{};
};

}  // namespace emel::logits::sampler
