#pragma once

// benchmark: scaffold

#include "emel/model/needle/graph/actions.hpp"
#include "emel/model/needle/graph/context.hpp"
#include "emel/model/needle/graph/errors.hpp"
#include "emel/model/needle/graph/events.hpp"
#include "emel/model/needle/graph/guards.hpp"
#include <memory>
#include <new>
#include <system_error>
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

template <bool vector_exp, action::projection_route_kind projection_route>
struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using route = action::route_kind;
    using activation = action::activation_route_kind;
    using projection = action::projection_route_kind;
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
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, true, false, true, true>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, true, false, true, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing_generic{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, true, false, false, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, true, true, true, true>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, true, true, true, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full_generic{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, true, true, false, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, false, false, true, true>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, false, false, true, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing_generic{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, false, false, false, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, false, true, true, true>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, false, true, true, false>{}
      , sml::state<state_layer_advance_scalar_a8> <= sml::state<state_layer_loop_scalar_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full_generic{} ]
          / action::effect_run_layer<route::scalar, activation::a8, projection::serial, false, true, false, false>{}
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
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, true, false, true, true>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, true, false, true, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing_generic{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, true, false, false, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, true, true, true, true>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, true, true, true, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full_generic{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, true, true, false, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, false, false, true, true>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, false, false, true, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing_generic{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, false, false, false, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, false, true, true, true>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, false, true, true, false>{}
      , sml::state<state_layer_advance_scalar_f32> <= sml::state<state_layer_loop_scalar_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full_generic{} ]
          / action::effect_run_layer<route::scalar, activation::f32, projection::serial, false, true, false, false>{}
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
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, true, false, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, true, false, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, true, false, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, true, true, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, true, true, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, true, true, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, false, false, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, false, false, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, false, false, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, false, true, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, false, true, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_a8> <= sml::state<state_layer_loop_prepared_avx2_a8>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::a8, projection_route, false, true, false, false>{}
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
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, true, false, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, true, false, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_growing_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, true, false, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, true, true, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_engram_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, true, true, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_engram_full_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, true, true, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, false, false, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_growing_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, false, false, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_growing_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, false, false, false, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, false, true, true, true>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::with_exp_route<guard::guard_layer_plain_full_gqa2, !vector_exp>{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, false, true, true, false>{}
      , sml::state<state_layer_advance_prepared_avx2_f32> <= sml::state<state_layer_loop_prepared_avx2_f32>
          + sml::completion<event::step_run> [ guard::guard_layer_plain_full_generic{} ]
          / action::effect_run_layer<route::prepared_avx2, activation::f32, projection_route, false, true, false, false>{}
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

template <bool vector_exp, action::projection_route_kind projection_route>
struct basic_sm
    : public emel::sm<model<vector_exp, projection_route>, action::context> {
  using base_type =
      emel::sm<model<vector_exp, projection_route>, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  struct construction_result {
    std::unique_ptr<basic_sm> machine = {};
    emel::error::type err = emel::error::cast(error::none);
  };
  using construction_factory = basic_sm *(*)(const needle::contract &);
  static construction_result
  create(const needle::contract &contract_in,
         const construction_factory construct = &construct_machine) noexcept {
    const emel::error::type geometry_err =
        action::validate_construction(contract_in);
    if (geometry_err != emel::error::cast(error::none))
      return {.machine = {}, .err = geometry_err};
    if (construct == nullptr)
      return {.machine = {},
              .err = emel::error::cast(error::internal_error)};
    try {
      std::unique_ptr<basic_sm> machine{construct(contract_in)};
      if (machine == nullptr)
        return {.machine = {},
                .err = emel::error::cast(error::internal_error)};
      return {.machine = std::move(machine),
              .err = emel::error::cast(error::none)};
    } catch (const std::bad_alloc &) {
      return {.machine = {},
              .err = emel::error::cast(error::capacity_exceeded)};
    } catch (const std::system_error &) {
      return {.machine = {},
              .err = emel::error::cast(error::internal_error)};
    } catch (...) {
      return {.machine = {},
              .err = emel::error::cast(error::internal_error)};
    }
  }
  explicit basic_sm(const needle::contract &contract_in)
      : base_type(std::in_place, contract_in,
                  projection_route == action::projection_route_kind::parallel4) {}
  basic_sm(const basic_sm &) = delete;
  basic_sm &operator=(const basic_sm &) = delete;
  class dispatch_scope {
  public:
    explicit dispatch_scope(std::atomic_flag &gate) noexcept
        : gate_(gate), acquired_(!gate_.test_and_set(std::memory_order_acquire)) {}
    ~dispatch_scope() {
      if (acquired_)
        gate_.clear(std::memory_order_release);
    }
    explicit operator bool() const noexcept { return acquired_; }

  private:
    std::atomic_flag &gate_;
    bool acquired_ = false;
  };


  bool process_event(const event::init &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    event::init_ctx ctx{};
    ctx.activation_quant = ev.activation_quant;
    const event::init_run runtime{ev, ctx};
    const bool handled = base_type::process_event(runtime);
    if (handled && ctx.err == emel::error::cast(error::none))
      activation_quant_ = ctx.activation_quant;
    return handled && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::prefill &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
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
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
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
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    uint64_t owner_prepare = 0u;
    uint64_t owner_prepared = 0u;
    bool handled = this->context_.cq.process_event(
        emel::kernel::cq::event::capture_prepared_diagnostics{
            owner_prepare, owner_prepared});
    ev.prepare_calls = owner_prepare;
    ev.prepared_calls = owner_prepared;
    for (auto &actor : this->context_.worker_cq) {
      uint64_t worker_prepare = 0u;
      uint64_t worker_prepared = 0u;
      handled = actor.process_event(
                    emel::kernel::cq::event::capture_prepared_diagnostics{
                        worker_prepare, worker_prepared}) &&
                handled;
      ev.prepare_calls += worker_prepare;
      ev.prepared_calls += worker_prepared;
    }
    ev.prepared_index_bytes = this->context_.prepared_indices.size();
    ev.prepared_input32_bytes =
        this->context_.prepared_indices_by_input32.size();
    ev.prepared_norm_bytes =
        this->context_.prepared_norms.size() * sizeof(float);
    ev.prepared_group32_norm_bytes =
        this->context_.prepared_norms_by_group32.size() * sizeof(float);
    return handled;
  }

  bool process_event(const event::capture_projection_diagnostics &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    ev.worker_calls = this->context_.worker_projection_calls;
    ev.submitted = this->context_.projection_submitted;
    ev.joined = this->context_.projection_joined;
    ev.live = this->context_.projection_live.load(std::memory_order_acquire);
    return true;
  }

  bool process_event(const event::capture_swa_diagnostics &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    ev.gqa2_calls = this->context_.swa_gqa2_calls;
    return true;
  }

  bool process_event(const event::configure_cq_timing &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch || (ev.enabled && ev.now == nullptr))
      return false;
    if (ev.enabled && !this->context_.cq_timing_enabled)
      this->context_.projection_cq_extra_nanoseconds = 0u;
    this->context_.cq_timing_enabled = ev.enabled;
    this->context_.cq_timing_now = ev.now;
    return this->context_.cq.process_event(
        emel::kernel::cq::event::configure_timing{ev.enabled, ev.now});
  }

  bool process_event(const event::capture_cq_timing &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    const bool handled = this->context_.cq.process_event(
        emel::kernel::cq::event::capture_timing{ev.breakdown});
    ev.breakdown.dot_batch_nanoseconds +=
        this->context_.projection_cq_extra_nanoseconds;
    return handled;
  }

  bool process_event(const event::configure_timing &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch || (ev.enabled && ev.now == nullptr))
      return false;
    if (ev.enabled && !this->context_.timing_enabled)
      this->context_.timing = {};
    this->context_.timing_enabled = ev.enabled;
    this->context_.timing_now = ev.now;
    return this->context_.cq.process_event(
        emel::kernel::cq::event::configure_timing{ev.enabled, ev.now});
  }

  bool process_event(const event::reset_timing &) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    this->context_.timing = {};
    if (!this->context_.timing_enabled)
      return true;
    this->context_.projection_cq_extra_nanoseconds = 0u;
    const auto now = this->context_.timing_now;
    this->context_.cq.process_event(
        emel::kernel::cq::event::configure_timing{false, now});
    return this->context_.cq.process_event(
        emel::kernel::cq::event::configure_timing{true, now});
  }

  bool process_event(const event::capture_timing &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    ev.breakdown = this->context_.timing;
    return true;
  }

  bool process_event(const event::capture_a8_diagnostics &ev) {
    dispatch_scope dispatch{dispatch_gate_};
    if (!dispatch)
      return false;
    return this->context_.cq.process_event(
        emel::kernel::cq::event::capture_a8_diagnostics{ev.quantize_calls});
  }

private:
  static basic_sm *construct_machine(const needle::contract &contract_in) {
    return new basic_sm{contract_in};
  }

  bool activation_quant_ = true;
  std::atomic_flag dispatch_gate_ = ATOMIC_FLAG_INIT;
};
using serial_sm = basic_sm<true, action::projection_route_kind::serial>;
using parallel4_sm = basic_sm<true, action::projection_route_kind::parallel4>;
using sm = parallel4_sm;
using scalar_exp_sm =
    basic_sm<false, action::projection_route_kind::serial>;
using NeedleGraph = sm;

} // namespace emel::model::needle::graph
