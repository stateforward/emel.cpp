#pragma once
// benchmark: designed

#include "emel/sm.hpp"
#include "emel/text/generator/layer/actions.hpp"
#include "emel/text/generator/layer/guards.hpp"

namespace emel::text::generator::layer {

struct state_idle {};
struct state_residual_done {};
struct state_feed_forward_done {};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct scalar_model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Explicit scalar residual route selection.
        sml::state<state_residual_done> <= *sml::state<state_idle>
                 + sml::event<event::scalar_run>
                 [ guard::guard_scalar_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::scalar_run>
                 [ guard::guard_scalar_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::scalar_run>
                 [ guard::guard_scalar_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::scalar_run>
                 [ guard::guard_scalar_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::scalar_run>
                 [ guard::guard_scalar_shortconv_route{} ]
                 / action::effect_run_scalar_shortconv<route, lanes>{}

      , sml::state<state_idle> <= sml::state<state_idle>
                 + sml::event<event::scalar_run>
                 / action::effect_reject_unsupported_route

      //------------------------------------------------------------------------------//
      // Explicit scalar residual/feed-forward outcome progression.
      , sml::state<state_feed_forward_done> <= sml::state<state_residual_done>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_residual_ok{} ]
                 / action::effect_run_scalar_feed_forward<route, lanes>{}

      , sml::state<state_idle> <= sml::state<state_residual_done>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_residual_failed{} ]
                 / action::effect_mark_failed

      , sml::state<state_idle> <= sml::state<state_feed_forward_done>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_feed_forward_ok{} ]
                 / action::effect_mark_succeeded

      , sml::state<state_idle> <= sml::state<state_feed_forward_done>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_feed_forward_failed{} ]
                 / action::effect_mark_failed

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_idle> <= sml::state<state_idle> + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
    );
    // clang-format on
  }
};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct chunk4_model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Explicit chunk4 residual route selection.
        sml::state<state_residual_done> <= *sml::state<state_idle>
                 + sml::event<event::chunk4_run>
                 [ guard::guard_chunk4_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk4_run>
                 [ guard::guard_chunk4_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk4_run>
                 [ guard::guard_chunk4_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk4_run>
                 [ guard::guard_chunk4_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk4_run>
                 [ guard::guard_chunk4_shortconv_route{} ]
                 / action::effect_run_chunk4_shortconv<route, lanes>{}

      , sml::state<state_idle> <= sml::state<state_idle>
                 + sml::event<event::chunk4_run>
                 / action::effect_reject_unsupported_route

      //------------------------------------------------------------------------------//
      // Explicit chunk4 residual/feed-forward outcome progression.
      , sml::state<state_feed_forward_done> <= sml::state<state_residual_done>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_residual_ok{} ]
                 / action::effect_run_chunk4_feed_forward<route, lanes>{}

      , sml::state<state_idle> <= sml::state<state_residual_done>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_residual_failed{} ]
                 / action::effect_mark_failed

      , sml::state<state_idle> <= sml::state<state_feed_forward_done>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_feed_forward_ok{} ]
                 / action::effect_mark_succeeded

      , sml::state<state_idle> <= sml::state<state_feed_forward_done>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_feed_forward_failed{} ]
                 / action::effect_mark_failed

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_idle> <= sml::state<state_idle> + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
    );
    // clang-format on
  }
};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::matmul::lane_mode lanes>
struct chunk8_model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Explicit chunk8 residual route selection.
        sml::state<state_residual_done> <= *sml::state<state_idle>
                 + sml::event<event::chunk8_run>
                 [ guard::guard_chunk8_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk8_run>
                 [ guard::guard_chunk8_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk8_run>
                 [ guard::guard_chunk8_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk8_run>
                 [ guard::guard_chunk8_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_idle>
                 + sml::event<event::chunk8_run>
                 [ guard::guard_chunk8_shortconv_route{} ]
                 / action::effect_run_chunk8_shortconv<lanes>{}

      , sml::state<state_idle> <= sml::state<state_idle>
                 + sml::event<event::chunk8_run>
                 / action::effect_reject_unsupported_route

      //------------------------------------------------------------------------------//
      // Explicit chunk8 residual/feed-forward outcome progression.
      , sml::state<state_feed_forward_done> <= sml::state<state_residual_done>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_residual_ok{} ]
                 / action::effect_run_chunk8_feed_forward<lanes>{}

      , sml::state<state_idle> <= sml::state<state_residual_done>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_residual_failed{} ]
                 / action::effect_mark_failed

      , sml::state<state_idle> <= sml::state<state_feed_forward_done>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_feed_forward_ok{} ]
                 / action::effect_mark_succeeded

      , sml::state<state_idle> <= sml::state<state_feed_forward_done>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_feed_forward_failed{} ]
                 / action::effect_mark_failed

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_idle> <= sml::state<state_idle> + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
    );
    // clang-format on
  }
};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
using scalar_sm = emel::sm<scalar_model<mode, route, lanes>>;

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
using chunk4_sm = emel::sm<chunk4_model<mode, route, lanes>>;

template <emel::text::generator::attention_mode mode,
          emel::text::generator::matmul::lane_mode lanes>
using chunk8_sm = emel::sm<chunk8_model<mode, lanes>>;

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct scalar_actor {
  scalar_sm<mode, route, lanes> machine{};

  bool process_event(const event::scalar_run &ev) noexcept {
    return machine.process_event(ev);
  }
};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct chunk4_actor {
  chunk4_sm<mode, route, lanes> machine{};

  bool process_event(const event::chunk4_run &ev) noexcept {
    return machine.process_event(ev);
  }
};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::matmul::lane_mode lanes>
struct chunk8_actor {
  chunk8_sm<mode, lanes> machine{};

  bool process_event(const event::chunk8_run &ev) noexcept {
    return machine.process_event(ev);
  }
};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
inline void process_scalar(const event::scalar_run &ev) noexcept {
  scalar_actor<mode, route, lanes> actor{};
  (void)actor.process_event(ev);
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
inline void process_chunk4(const event::chunk4_run &ev) noexcept {
  chunk4_actor<mode, route, lanes> actor{};
  (void)actor.process_event(ev);
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::matmul::lane_mode lanes>
inline void process_chunk8(const event::chunk8_run &ev) noexcept {
  chunk8_actor<mode, lanes> actor{};
  (void)actor.process_event(ev);
}

} // namespace emel::text::generator::layer
