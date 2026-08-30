#pragma once

// benchmark: scaffold

#include "emel/model/needle/graph/actions.hpp"
#include "emel/model/needle/graph/context.hpp"
#include "emel/model/needle/graph/errors.hpp"
#include "emel/model/needle/graph/events.hpp"
#include "emel/model/needle/graph/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::needle::graph {

// Graph lifecycle: uninitialized until an init request validates the bound
// contract geometry, decodes the fp16 scales, and precomputes RoPE; ready
// thereafter. Each prefill/decode token is one step_run dispatch whose
// completion chain walks: begin (embed + lane broadcast) -> engram K/V (or
// skip) -> per-layer loop (engram-site/plain x window growing/full guards)
// -> logits head (or skip for non-final prefill tokens) -> finish.
// The CQ route (scalar vs AVX2) is fixed at compile time by the composed
// entry guards; both chains exist as explicit transitions.
struct state_uninitialized {};
struct state_init_decision {};
struct state_init_outcome {};
struct state_ready {};
struct state_errored {};

struct state_step_route_decision {};
struct state_step_engram_decision_scalar {};
struct state_step_engram_decision_avx2 {};
struct state_layer_loop_scalar {};
struct state_layer_loop_avx2 {};
struct state_layer_advance_scalar {};
struct state_layer_advance_avx2 {};
struct state_step_finish {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using route = action::route_kind;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Init.
        sml::state<state_init_decision> <= *sml::state<state_uninitialized>
          + sml::event<event::init_run> / action::effect_begin_init{}
      , sml::state<state_init_decision> <= sml::state<state_ready>
          + sml::event<event::init_run> / action::effect_begin_init{}
      , sml::state<state_init_decision> <= sml::state<state_errored>
          + sml::event<event::init_run> / action::effect_begin_init{}

      , sml::state<state_init_outcome> <= sml::state<state_init_decision>
          + sml::completion<event::init_run> [ guard::guard_init_supported{} ]
          / action::effect_exec_init{}
      , sml::state<state_init_outcome> <= sml::state<state_init_decision>
          + sml::completion<event::init_run> [ guard::guard_init_unsupported{} ]
          / action::effect_mark_init_unsupported{}

      , sml::state<state_ready> <= sml::state<state_init_outcome>
          + sml::completion<event::init_run> [ guard::guard_init_ok{} ]
      , sml::state<state_errored> <= sml::state<state_init_outcome>
          + sml::completion<event::init_run> [ guard::guard_init_failed{} ]

      //------------------------------------------------------------------------------//
      // Step entry: route fixed by build capability, then engram presence.
      , sml::state<state_step_route_decision> <= sml::state<state_ready>
          + sml::event<event::step_run> [ guard::guard_step_valid_scalar{} ]
          / action::effect_step_begin<route::scalar>{}
      , sml::state<state_step_route_decision> <= sml::state<state_ready>
          + sml::event<event::step_run> [ guard::guard_step_valid_avx2{} ]
          / action::effect_step_begin<route::avx2>{}
      , sml::state<state_errored> <= sml::state<state_ready>
          + sml::event<event::step_run> [ guard::guard_step_invalid{} ]
          / action::effect_mark_step_invalid{}

      , sml::state<state_step_engram_decision_scalar> <= sml::state<state_step_route_decision>
          + sml::completion<event::step_run> [ guard::guard_route_scalar{} ]
      , sml::state<state_step_engram_decision_avx2> <= sml::state<state_step_route_decision>
          + sml::completion<event::step_run> [ guard::guard_route_avx2{} ]

      , sml::state<state_layer_loop_scalar> <= sml::state<state_step_engram_decision_scalar>
          + sml::completion<event::step_run> [ guard::guard_engram_present_ok{} ]
          / action::effect_compute_engram<route::scalar>{}
      , sml::state<state_layer_loop_scalar> <= sml::state<state_step_engram_decision_scalar>
          + sml::completion<event::step_run> [ guard::guard_engram_absent_ok{} ]
      , sml::state<state_errored> <= sml::state<state_step_engram_decision_scalar>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_avx2> <= sml::state<state_step_engram_decision_avx2>
          + sml::completion<event::step_run> [ guard::guard_engram_present_ok{} ]
          / action::effect_compute_engram<route::avx2>{}
      , sml::state<state_layer_loop_avx2> <= sml::state<state_step_engram_decision_avx2>
          + sml::completion<event::step_run> [ guard::guard_engram_absent_ok{} ]
      , sml::state<state_errored> <= sml::state<state_step_engram_decision_avx2>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      //------------------------------------------------------------------------------//
      // Layer loop (scalar route): engram-site/plain x window growing/full.
      , sml::state<state_layer_advance_scalar> <= sml::state<state_layer_loop_scalar>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing{} ]
          / action::effect_run_layer<route::scalar, true, false>{}
      , sml::state<state_layer_advance_scalar> <= sml::state<state_layer_loop_scalar>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full{} ]
          / action::effect_run_layer<route::scalar, true, true>{}
      , sml::state<state_layer_advance_scalar> <= sml::state<state_layer_loop_scalar>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing{} ]
          / action::effect_run_layer<route::scalar, false, false>{}
      , sml::state<state_layer_advance_scalar> <= sml::state<state_layer_loop_scalar>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full{} ]
          / action::effect_run_layer<route::scalar, false, true>{}
      , sml::state<state_errored> <= sml::state<state_layer_loop_scalar>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      , sml::state<state_layer_loop_scalar> <= sml::state<state_layer_advance_scalar>
          + sml::completion<event::step_run> [ guard::guard_more_layers{} ]
          / action::effect_advance_layer{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_scalar>
          + sml::completion<event::step_run> [ guard::guard_layers_done_want_logits{} ]
          / action::effect_emit_logits<route::scalar>{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_scalar>
          + sml::completion<event::step_run> [ guard::guard_layers_done_no_logits{} ]
      , sml::state<state_errored> <= sml::state<state_layer_advance_scalar>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      //------------------------------------------------------------------------------//
      // Layer loop (AVX2 route).
      , sml::state<state_layer_advance_avx2> <= sml::state<state_layer_loop_avx2>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing{} ]
          / action::effect_run_layer<route::avx2, true, false>{}
      , sml::state<state_layer_advance_avx2> <= sml::state<state_layer_loop_avx2>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full{} ]
          / action::effect_run_layer<route::avx2, true, true>{}
      , sml::state<state_layer_advance_avx2> <= sml::state<state_layer_loop_avx2>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing{} ]
          / action::effect_run_layer<route::avx2, false, false>{}
      , sml::state<state_layer_advance_avx2> <= sml::state<state_layer_loop_avx2>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full{} ]
          / action::effect_run_layer<route::avx2, false, true>{}
      , sml::state<state_errored> <= sml::state<state_layer_loop_avx2>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      , sml::state<state_layer_loop_avx2> <= sml::state<state_layer_advance_avx2>
          + sml::completion<event::step_run> [ guard::guard_more_layers{} ]
          / action::effect_advance_layer{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_avx2>
          + sml::completion<event::step_run> [ guard::guard_layers_done_want_logits{} ]
          / action::effect_emit_logits<route::avx2>{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_avx2>
          + sml::completion<event::step_run> [ guard::guard_layers_done_no_logits{} ]
      , sml::state<state_errored> <= sml::state<state_layer_advance_avx2>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      //------------------------------------------------------------------------------//
      // Step finish.
      , sml::state<state_ready> <= sml::state<state_step_finish>
          + sml::completion<event::step_run> [ guard::guard_step_ok{} ]
          / action::effect_finish_step{}
      , sml::state<state_errored> <= sml::state<state_step_finish>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_errored> <= sml::state<state_uninitialized> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_errored> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_errored> <= sml::state<state_errored> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
    );
    // clang-format on
  }
};

// Public wrapper: owns the context (all storage allocated at construction
// from the bound contract) and adapts the public init/prefill/decode events
// onto the internal runtime dispatches. B=1 is the supported batch.
struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  explicit sm(const needle::contract &contract_in)
      : base_type(std::in_place, contract_in) {}
  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;

  bool process_event(const event::init &ev) {
    event::init_ctx ctx{};
    const event::init_run runtime{ev, ctx};
    const bool handled = base_type::process_event(runtime);
    return handled && ctx.err == emel::error::cast(error::none);
  }

  // Prefill: one step dispatch per prompt token; only the last token emits
  // logits (matching the reference last-position logits contract).
  bool process_event(const event::prefill &ev) {
    bool ok = !ev.tokens.empty();
    for (size_t i = 0u; ok && i < ev.tokens.size(); ++i) {
      event::step_ctx ctx{};
      ctx.token = ev.tokens[i];
      ctx.want_logits = i + 1u == ev.tokens.size();
      ctx.logits_out = ev.logits_out;
      const event::step_run runtime{ctx};
      ok = base_type::process_event(runtime) &&
           ctx.err == emel::error::cast(error::none);
    }
    return ok;
  }

  bool process_event(const event::decode &ev) {
    event::step_ctx ctx{};
    ctx.token = ev.token;
    ctx.want_logits = true;
    ctx.logits_out = ev.logits_out;
    const event::step_run runtime{ctx};
    const bool handled = base_type::process_event(runtime);
    return handled && ctx.err == emel::error::cast(error::none);
  }
};

using NeedleGraph = sm;

} // namespace emel::model::needle::graph
