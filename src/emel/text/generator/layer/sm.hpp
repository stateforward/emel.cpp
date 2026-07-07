#pragma once
// benchmark: designed

#include "emel/sm.hpp"
#include "emel/text/generator/layer/actions.hpp"
#include "emel/text/generator/layer/guards.hpp"

namespace emel::text::generator::layer {

struct state_idle {};
struct state_input_ready {};
struct state_normalized {};
struct state_residual_done {};
struct state_feed_forward_done {};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
struct scalar_model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Explicit scalar prepare and input normalization.
        sml::state<state_input_ready> <= *sml::state<state_idle>
                 + sml::event<event::scalar_run>
                 / action::effect_prepare_scalar<wmode>{}

      , sml::state<state_normalized> <= sml::state<state_input_ready>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_stream_ready{} ]
                 / action::effect_normalize_scalar{}

      , sml::state<state_idle> <= sml::state<state_input_ready>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_stream_failed{} ]
                 / action::effect_mark_failed

      //------------------------------------------------------------------------------//
      // Explicit scalar residual route selection.
      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_scalar_normalized_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_scalar_normalized_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_scalar_normalized_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_scalar_normalized_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_scalar_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_scalar_normalized_shortconv_route{} ]
                 / action::effect_run_scalar_shortconv<route, lanes>{}

      , sml::state<state_idle> <= sml::state<state_normalized>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_normalized_ok{} ]
                 / action::effect_reject_unsupported_route

      , sml::state<state_idle> <= sml::state<state_normalized>
                 + sml::completion<event::scalar_run>
                 [ guard::guard_normalized_failed{} ]
                 / action::effect_mark_failed

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
      // Explicit chunk4 input normalization.
        sml::state<state_normalized> <= *sml::state<state_idle>
                 + sml::event<event::chunk4_run>
                 / action::effect_normalize_chunk4{}

      //------------------------------------------------------------------------------//
      // Explicit chunk4 residual route selection.
      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_chunk4_normalized_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_chunk4_normalized_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_chunk4_normalized_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_chunk4_normalized_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk4_attention<
                       mode,
                       route,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_chunk4_normalized_shortconv_route{} ]
                 / action::effect_run_chunk4_shortconv<route, lanes>{}

      , sml::state<state_idle> <= sml::state<state_normalized>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_normalized_ok{} ]
                 / action::effect_reject_unsupported_route

      , sml::state<state_idle> <= sml::state<state_normalized>
                 + sml::completion<event::chunk4_run>
                 [ guard::guard_normalized_failed{} ]
                 / action::effect_mark_failed

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
      // Explicit chunk8 input normalization.
        sml::state<state_normalized> <= *sml::state<state_idle>
                 + sml::event<event::chunk8_run>
                 / action::effect_normalize_chunk8{}

      //------------------------------------------------------------------------------//
      // Explicit chunk8 residual route selection.
      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_chunk8_normalized_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_chunk8_normalized_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::none,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_chunk8_normalized_attention_route<
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::none,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_chunk8_normalized_attention_route<
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms>{} ]
                 / action::effect_run_chunk8_attention<
                       mode,
                       event::attention_qk_norm_route::headwise_rms,
                       event::attention_v_norm_route::rms,
                       lanes>{}

      , sml::state<state_residual_done> <= sml::state<state_normalized>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_chunk8_normalized_shortconv_route{} ]
                 / action::effect_run_chunk8_shortconv<lanes>{}

      , sml::state<state_idle> <= sml::state<state_normalized>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_normalized_ok{} ]
                 / action::effect_reject_unsupported_route

      , sml::state<state_idle> <= sml::state<state_normalized>
                 + sml::completion<event::chunk8_run>
                 [ guard::guard_normalized_failed{} ]
                 / action::effect_mark_failed

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
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
using scalar_sm = emel::sm<scalar_model<mode, route, lanes, wmode>>;

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
using chunk4_sm = emel::sm<chunk4_model<mode, route, lanes>>;

template <emel::text::generator::attention_mode mode,
          emel::text::generator::matmul::lane_mode lanes>
using chunk8_sm = emel::sm<chunk8_model<mode, lanes>>;

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
struct scalar_actor {
  scalar_sm<mode, route, lanes, wmode> machine{};

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
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
inline void process_scalar(const event::scalar_run &ev) noexcept {
  scalar_actor<mode, route, lanes, wmode> actor{};
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

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
inline bool
run_layer(emel::text::generator::detail::native_backend &backend,
          const int32_t layer_index,
          const emel::text::generator::detail::kv_addressing_view &kv,
          const int32_t position, int32_t &error) noexcept {
  auto &block = backend.blocks[static_cast<size_t>(layer_index)];
  event::scalar_run ev{backend,
                       kv,
                       layer_index,
                       position,
                       block.residual_route,
                       block.qk_norm_route,
                       block.v_norm_route,
                       error};
  process_scalar<mode, route, lanes, wmode>(ev);
  return ev.succeeded;
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
inline bool run_layer(emel::text::generator::detail::native_backend &backend,
                      const int32_t layer_index, const int32_t position,
                      int32_t &error) noexcept {
  return run_layer<mode, route, lanes, wmode>(
      backend, layer_index,
      emel::text::generator::detail::identity_kv_addressing(), position, error);
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
inline bool
run_layer(emel::text::generator::detail::native_backend &backend,
          const int32_t layer_index,
          const emel::text::generator::detail::kv_addressing_view &kv,
          const int32_t position) noexcept {
  int32_t ignored_error = emel::text::generator::detail::k_error_ok;
  return run_layer<mode, route, lanes, wmode>(backend, layer_index, kv,
                                              position, ignored_error);
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes,
          emel::text::generator::detail::window_mode wmode>
inline bool run_layer(emel::text::generator::detail::native_backend &backend,
                      const int32_t layer_index,
                      const int32_t position) noexcept {
  int32_t ignored_error = emel::text::generator::detail::k_error_ok;
  return run_layer<mode, route, lanes, wmode>(
      backend, layer_index,
      emel::text::generator::detail::identity_kv_addressing(), position,
      ignored_error);
}

inline bool
run_layer_flash(emel::text::generator::detail::native_backend &backend,
                const int32_t layer_index, const int32_t position) noexcept {
  return run_layer<emel::text::generator::attention_mode::flash,
                   emel::text::generator::detail::scalar_matmul_route::kernel>(
      backend, layer_index,
      emel::text::generator::detail::identity_kv_addressing(), position);
}

inline bool
run_layer_nonflash(emel::text::generator::detail::native_backend &backend,
                   const int32_t layer_index, const int32_t position) noexcept {
  return run_layer<emel::text::generator::attention_mode::nonflash,
                   emel::text::generator::detail::scalar_matmul_route::kernel>(
      backend, layer_index,
      emel::text::generator::detail::identity_kv_addressing(), position);
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
inline bool
run_layer_chunk4(emel::text::generator::detail::native_backend &backend,
                 const emel::text::generator::detail::kv_addressing_view &kv,
                 const int32_t layer_index, const size_t token_base) noexcept {
  auto &block = backend.blocks[static_cast<size_t>(layer_index)];
  event::chunk4_run ev{backend,
                       kv,
                       layer_index,
                       token_base,
                       block.residual_route,
                       block.qk_norm_route,
                       block.v_norm_route};
  process_chunk4<mode, route, lanes>(ev);
  return ev.succeeded;
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::matmul::lane_mode lanes>
inline bool run_layer_chunk8_q8_k(
    emel::text::generator::detail::native_backend &backend,
    const emel::text::generator::detail::kv_addressing_view &kv,
    const int32_t layer_index, const size_t token_base) noexcept {
  auto &block = backend.blocks[static_cast<size_t>(layer_index)];
  event::chunk8_run ev{backend,
                       kv,
                       layer_index,
                       token_base,
                       block.residual_route,
                       block.qk_norm_route,
                       block.v_norm_route};
  process_chunk8<mode, lanes>(ev);
  return ev.succeeded;
}

} // namespace emel::text::generator::layer
