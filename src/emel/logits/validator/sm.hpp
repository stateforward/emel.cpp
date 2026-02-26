#pragma once

/*
design doc: docs/designs/logits/validator.design.md
 ---
 title: logits/validator architecture design
 status: draft
 ---
 
 # logits/validator architecture design
 
 this document defines the logits/validator stage. it operates on a single sequence's logits,
 validating and transforming raw model outputs into a structured candidate view for the sampler pipeline.
 
 ## role
 - act as the candidate builder for a single sequence's logits row.
 - validate candidate buffer capacities and raw logit pointers.
 - normalize the view: pass through kernel-provided candidates (if a fused kernel like `op::top_k` was used)
   or gracefully fall back to building a full-vocab candidate list from raw logits.
 - leave mathematical normalization (like Softmax) to the sampler chain.
 
 ## architecture shift: batch dispatch (future co_sm)
 because a `batch::plan` step produces multiple rows of logits (one for each sequence requiring output),
 the `generator` must process them efficiently.
 
 initially, the validator is a standard synchronous `boost::sml` actor. the generator will loop over
 the logits rows and process them sequentially.
 
 in the future, by leveraging `emel::co_sm` (defined in `src/emel/sm.hpp`), the validator can be
 dispatched asynchronously. the `generator` will be able to call `validator.process_event_async(...)`
 for each sequence, and `co_await` their completion. the core SML state machine remains identical
 in both synchronous and asynchronous contexts.
 
 ## events
 - `event::validate`
   - inputs: logits row pointer, vocab size, optional kernel-provided candidates/counts,
     a scratch buffer for candidate generation, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: builds the normalized `candidate_view` in the provided buffers and invokes the
     appropriate callback before returning, avoiding context-reading race conditions.
 
 ## state model
 
 ```text
 idle ──► validating ──► candidate_decision ──► (done | errored)
   ▲                                               │
   └───────────────────────────────────────────────┘
 ```
 
 - `idle` — waiting for a logits row.
 - `validating` — checking bounds and buffer capacities.
 - `candidate_decision` — routing based on input:
   - if kernel candidates exist: simply wrap them in a `candidate_view`.
   - if raw logits only: zip the raw floats with token IDs (0 to vocab_size-1) into the scratch buffer.
 - `done` — validation complete, transitions back to `idle` emitting `events::validate_done`.
 - unexpected events route to `unexpected`.
 
 ## responsibilities
 - **decouple generation from sampling:** ensure that whether the graph executed a highly fused `top_k`
   kernel or just dumped raw floats, the downstream sampler pipeline receives a consistent `candidate_view`.
 - **no early math:** do not apply Softmax or modify scores. merely build the structural view. specific
   samplers in the chain will apply math in-place only if their algorithm requires it.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_INVALID_ARGUMENT` — required inputs are missing or null pointers were provided.
 - `EMEL_ERR_CAPACITY` — vocab size exceeds the configured maximum or scratch buffers are too small.
*/


#include "emel/logits/validator/actions.hpp"
#include "emel/logits/validator/events.hpp"
#include "emel/logits/validator/guards.hpp"
#include "emel/sm.hpp"
#include "emel/sm.hpp"

namespace emel::logits::validator {

struct initialized {};
struct validating {};
struct validate_decision {};
struct building_candidates {};
struct build_decision {};
struct normalizing_scores {};
struct normalize_decision {};
struct done {};
struct errored {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::build> / action::begin_build =
        sml::state<validating>,

      sml::state<validating> / action::run_validate = sml::state<validate_decision>,
      sml::state<validate_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<validate_decision> [guard::phase_ok{}] = sml::state<building_candidates>,

      sml::state<building_candidates> / action::run_build_candidates =
        sml::state<build_decision>,
      sml::state<build_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<build_decision> [guard::phase_ok{}] = sml::state<normalizing_scores>,

      sml::state<normalizing_scores> / action::run_normalize_scores =
        sml::state<normalize_decision>,
      sml::state<normalize_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<normalize_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::publish_done = sml::state<initialized>,
      sml::state<errored> / action::publish_error = sml::state<initialized>,

      sml::state<initialized> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<validating> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<validate_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<building_candidates> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<build_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<normalizing_scores> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<normalize_decision> + sml::unexpected_event<sml::_> /
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

  int32_t candidate_count() const noexcept { return context_.candidate_count; }

 private:
  action::context context_{};
};

}  // namespace emel::logits::validator
