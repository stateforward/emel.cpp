#pragma once

// benchmark: scaffold

#include "emel/model/needle/graph/actions.hpp"
#include "emel/model/needle/graph/context.hpp"
#include "emel/model/needle/graph/errors.hpp"
#include "emel/model/needle/graph/events.hpp"
#include "emel/model/needle/graph/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::needle::graph {

struct state_uninitialized {};
struct state_init_decision {};
struct state_init_outcome {};
struct state_ready {};
struct state_errored {};

struct state_step_route_decision {};
struct state_step_activation_decision_scalar {};
struct state_step_activation_decision_prepared_avx2 {};
struct state_step_engram_decision_scalar_a8 {};
struct state_step_engram_decision_prepared_avx2_a8 {};
struct state_step_engram_decision_scalar_f32 {};
struct state_step_engram_decision_prepared_avx2_f32 {};
struct state_layer_loop_scalar_a8 {};
struct state_layer_loop_prepared_avx2_a8 {};
struct state_layer_loop_scalar_f32 {};
struct state_layer_loop_prepared_avx2_f32 {};
struct state_layer_advance_scalar_a8 {};
struct state_layer_advance_prepared_avx2_a8 {};
struct state_layer_advance_scalar_f32 {};
struct state_layer_advance_prepared_avx2_f32 {};
struct state_step_finish {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using route = action::route_kind;
    using activation = action::activation_route_kind;
    // clang-format off
    return sml::make_transition_table(
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

      , sml::state<state_step_route_decision> <= sml::state<state_ready>
          + sml::event<event::step_run> [ guard::guard_step_valid_scalar{} ]
          / action::effect_step_begin<route::scalar>{}
      , sml::state<state_step_route_decision> <= sml::state<state_ready>
          + sml::event<event::step_run> [ guard::guard_step_valid_avx2{} ]
          / action::effect_step_begin<route::prepared_avx2>{}
      , sml::state<state_errored> <= sml::state<state_ready>
          + sml::event<event::step_run> [ guard::guard_step_invalid{} ]
          / action::effect_mark_step_invalid{}

      , sml::state<state_step_activation_decision_scalar> <= sml::state<state_step_route_decision>
          + sml::completion<event::step_run> [ guard::guard_route_scalar{} ]
      , sml::state<state_step_activation_decision_prepared_avx2> <= sml::state<state_step_route_decision>
          + sml::completion<event::step_run> [ guard::guard_route_avx2{} ]
      , sml::state<state_step_engram_decision_scalar_a8> <= sml::state<state_step_activation_decision_scalar>
          + sml::completion<event::step_run> [ guard::guard_deployment_a8{} ]
      , sml::state<state_step_engram_decision_scalar_f32> <= sml::state<state_step_activation_decision_scalar>
          + sml::completion<event::step_run> [ guard::guard_deployment_f32{} ]
      , sml::state<state_step_engram_decision_prepared_avx2_a8> <= sml::state<state_step_activation_decision_prepared_avx2>
          + sml::completion<event::step_run> [ guard::guard_deployment_a8{} ]
      , sml::state<state_step_engram_decision_prepared_avx2_f32> <= sml::state<state_step_activation_decision_prepared_avx2>
          + sml::completion<event::step_run> [ guard::guard_deployment_f32{} ]

      , sml::state<state_layer_loop_scalar_a8> <= sml::state<state_step_engram_decision_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_engram_present_ok{} ]
          / action::effect_compute_engram<route::scalar, activation::a8>{}
      , sml::state<state_layer_loop_scalar_a8> <= sml::state<state_step_engram_decision_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_engram_absent_ok{} ]
      , sml::state<state_errored> <= sml::state<state_step_engram_decision_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_scalar_f32> <= sml::state<state_step_engram_decision_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_engram_present_ok{} ]
          / action::effect_compute_engram<route::scalar, activation::f32>{}
      , sml::state<state_layer_loop_scalar_f32> <= sml::state<state_step_engram_decision_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_engram_absent_ok{} ]
      , sml::state<state_errored> <= sml::state<state_step_engram_decision_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_prepared_avx2_a8> <= sml::state<state_step_engram_decision_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_engram_present_ok{} ]
          / action::effect_compute_engram<route::prepared_avx2, activation::a8>{}
      , sml::state<state_layer_loop_prepared_avx2_a8> <= sml::state<state_step_engram_decision_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_engram_absent_ok{} ]
      , sml::state<state_errored> <= sml::state<state_step_engram_decision_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_prepared_avx2_f32> <= sml::state<state_step_engram_decision_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_engram_present_ok{} ]
          / action::effect_compute_engram<route::prepared_avx2, activation::f32>{}
      , sml::state<state_layer_loop_prepared_avx2_f32> <= sml::state<state_step_engram_decision_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_engram_absent_ok{} ]
      , sml::state<state_errored> <= sml::state<state_step_engram_decision_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing{} ]
          / action::effect_run_layer<route::scalar, activation::a8, true, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full{} ]
          / action::effect_run_layer<route::scalar, activation::a8, true, true>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing{} ]
          / action::effect_run_layer<route::scalar, activation::a8, false, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full{} ]
          / action::effect_run_layer<route::scalar, activation::a8, false, true>{}
      , sml::state<state_errored> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_scalar_a8> <= sml::state<state_layer_advance_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_more_layers{} ]
          / action::effect_advance_layer{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layers_done_want_logits{} ]
          / action::effect_emit_logits<route::scalar, activation::a8>{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layers_done_no_logits{} ]
      , sml::state<state_errored> <= sml::state<state_layer_advance_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing{} ]
          / action::effect_run_layer<route::scalar, activation::f32, true, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full{} ]
          / action::effect_run_layer<route::scalar, activation::f32, true, true>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing{} ]
          / action::effect_run_layer<route::scalar, activation::f32, false, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full{} ]
          / action::effect_run_layer<route::scalar, activation::f32, false, true>{}
      , sml::state<state_errored> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_scalar_f32> <= sml::state<state_layer_advance_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_more_layers{} ]
          / action::effect_advance_layer{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layers_done_want_logits{} ]
          / action::effect_emit_logits<route::scalar, activation::f32>{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layers_done_no_logits{} ]
      , sml::state<state_errored> <= sml::state<state_layer_advance_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, false, true>{}
      , sml::state<state_errored> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_prepared_avx2_a8> <= sml::state<state_layer_advance_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_more_layers{} ]
          / action::effect_advance_layer{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layers_done_want_logits{} ]
          / action::effect_emit_logits<route::prepared_avx2, activation::a8>{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layers_done_no_logits{} ]
      , sml::state<state_errored> <= sml::state<state_layer_advance_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, false, true>{}
      , sml::state<state_errored> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]
      , sml::state<state_layer_loop_prepared_avx2_f32> <= sml::state<state_layer_advance_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_more_layers{} ]
          / action::effect_advance_layer{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layers_done_want_logits{} ]
          / action::effect_emit_logits<route::prepared_avx2, activation::f32>{}
      , sml::state<state_step_finish> <= sml::state<state_layer_advance_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layers_done_no_logits{} ]
      , sml::state<state_errored> <= sml::state<state_layer_advance_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

      , sml::state<state_ready> <= sml::state<state_step_finish>
          + sml::completion<event::step_run> [ guard::guard_step_ok{} ]
          / action::effect_finish_step{}
      , sml::state<state_errored> <= sml::state<state_step_finish>
          + sml::completion<event::step_run> [ guard::guard_step_failed{} ]

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
    ctx.activation_quant = ev.activation_quant;
    const event::init_run runtime{ev, ctx};
    const bool handled = base_type::process_event(runtime);
    if (handled && ctx.err == emel::error::cast(error::none))
      activation_quant_ = ctx.activation_quant;
    return handled && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::prefill &ev) {
    bool ok = !ev.tokens.empty();
    for (size_t i = 0u; ok && i < ev.tokens.size(); ++i) {
      event::step_ctx ctx{};
      ctx.token = ev.tokens[i];
      ctx.want_logits = i + 1u == ev.tokens.size();
      ctx.logits_out = ev.logits_out;
      ctx.activation_quant = activation_quant_;
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
    ctx.activation_quant = activation_quant_;
    const event::step_run runtime{ctx};
    const bool handled = base_type::process_event(runtime);
    return handled && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::capture_cq_diagnostics &ev) {
    emel::kernel::cq::event::capture_prepared_diagnostics diagnostics{
        ev.prepare_calls, ev.prepared_calls};
    const bool handled = this->context_.cq.process_event(diagnostics);
    ev.prepared_index_bytes = this->context_.prepared_indices.size();
    ev.prepared_input32_bytes =
        this->context_.prepared_indices_by_input32.size();
    ev.prepared_norm_bytes =
        this->context_.prepared_norms.size() * sizeof(float);
    return handled;
  }

  bool process_event(const event::configure_cq_timing &ev) {
    return this->context_.cq.process_event(
        emel::kernel::cq::event::configure_timing{ev.enabled, ev.now});
  }

  bool process_event(const event::capture_cq_timing &ev) {
    return this->context_.cq.process_event(
        emel::kernel::cq::event::capture_timing{ev.breakdown});
  }

  bool process_event(const event::capture_a8_diagnostics &ev) {
    return this->context_.cq.process_event(
        emel::kernel::cq::event::capture_a8_diagnostics{ev.quantize_calls});
  }

private:
  bool activation_quant_ = true;
};

using NeedleGraph = sm;

} // namespace emel::model::needle::graph
