#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/text/generator/detail.hpp"
#include "emel/text/generator/events.hpp"
#include "emel/text/generator/layer/events.hpp"
#include "emel/text/generator/matmul/detail.hpp"

namespace emel::text::generator::layer::action {

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route,
          emel::text::generator::matmul::lane_mode lanes>
bool run_layer_attention_residual(
    emel::text::generator::detail::native_backend &backend,
    const int32_t layer_index,
    const emel::text::generator::detail::kv_addressing_view &kv,
    const int32_t position) noexcept {
  return emel::text::generator::detail::compute_layer_attention_residual<
      mode, route, qk_route, v_route, lanes>(backend, layer_index, kv,
                                             position);
}

template <emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
bool run_layer_shortconv_residual(
    emel::text::generator::detail::native_backend &backend,
    const int32_t layer_index,
    const emel::text::generator::detail::kv_addressing_view &kv) noexcept {
  return emel::text::generator::detail::compute_layer_shortconv_residual<route,
                                                                         lanes>(
      backend, layer_index, kv);
}

template <emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
bool run_layer_feed_forward(
    emel::text::generator::detail::native_backend &backend,
    const int32_t layer_index) noexcept {
  return emel::text::generator::detail::compute_layer_feed_forward<route,
                                                                   lanes>(
      backend, layer_index);
}

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route,
          emel::text::generator::matmul::lane_mode lanes>
bool run_layer_chunk4_attention_residual(
    emel::text::generator::detail::native_backend &backend,
    const emel::text::generator::detail::kv_addressing_view &kv,
    const int32_t layer_index, const size_t token_base) noexcept {
  return emel::text::generator::detail::compute_layer_chunk4_attention_residual<
      mode, route, qk_route, v_route, lanes>(backend, kv, layer_index,
                                             token_base);
}

template <emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
bool run_layer_chunk4_shortconv_residual(
    emel::text::generator::detail::native_backend &backend,
    const emel::text::generator::detail::kv_addressing_view &kv,
    const int32_t layer_index) noexcept {
  return emel::text::generator::detail::compute_layer_chunk4_shortconv_residual<
      route, lanes>(backend, kv, layer_index);
}

template <emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
bool run_layer_chunk4_feed_forward(
    emel::text::generator::detail::native_backend &backend,
    const int32_t layer_index) noexcept {
  return emel::text::generator::detail::compute_layer_chunk4_feed_forward<
      route, lanes>(backend, layer_index);
}

template <emel::text::generator::attention_mode mode,
          event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route,
          emel::text::generator::matmul::lane_mode lanes>
bool run_layer_chunk8_q8_k_attention_residual(
    emel::text::generator::detail::native_backend &backend,
    const emel::text::generator::detail::kv_addressing_view &kv,
    const int32_t layer_index, const size_t token_base) noexcept {
  return emel::text::generator::detail::
      compute_layer_chunk8_q8_k_attention_residual<mode, qk_route, v_route,
                                                   lanes>(
          backend, kv, layer_index, token_base);
}

template <emel::text::generator::matmul::lane_mode lanes>
bool run_layer_chunk8_q8_k_shortconv_residual(
    emel::text::generator::detail::native_backend &backend,
    const emel::text::generator::detail::kv_addressing_view &kv,
    const int32_t layer_index) noexcept {
  return emel::text::generator::detail::
      compute_layer_chunk8_q8_k_shortconv_residual<lanes>(backend, kv,
                                                          layer_index);
}

template <emel::text::generator::matmul::lane_mode lanes>
bool run_layer_chunk8_q8_k_feed_forward(
    emel::text::generator::detail::native_backend &backend,
    const int32_t layer_index) noexcept {
  return emel::text::generator::detail::compute_layer_chunk8_q8_k_feed_forward<
      lanes>(backend, layer_index);
}

template <emel::text::generator::detail::window_mode wmode>
struct effect_prepare_scalar {
  void operator()(const event::scalar_run &ev) const noexcept {
    if constexpr (wmode ==
                  emel::text::generator::detail::window_mode::streamed) {
      ev.error = emel::text::generator::detail::k_error_stream_acquire;
      ev.stream_ready = emel::text::generator::detail::acquire_streamed_layer(
          ev.backend, ev.layer_index);
    } else {
      ev.error = emel::text::generator::detail::k_error_ok;
      ev.stream_ready = true;
    }
  }
};

struct effect_normalize_scalar {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  void operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    (*this)(ev.event_);
  }

  void operator()(const event::scalar_run &ev) const noexcept {
    auto &block = ev.backend.blocks[static_cast<size_t>(ev.layer_index)];
    ev.error = emel::text::generator::detail::k_error_invalid;
    ev.normalized_ok = emel::text::generator::detail::rms_norm(
        ev.backend.hidden, block.attention_norm, ev.backend.rms_epsilon,
        ev.backend.norm);
  }
};

struct effect_normalize_chunk4 {
  void operator()(const event::chunk4_run &ev) const noexcept {
    auto &block = ev.backend.blocks[static_cast<size_t>(ev.layer_index)];
    ev.normalized_ok = emel::text::generator::detail::rms_norm_chunk4(
        ev.backend.hidden_chunk4, ev.backend.n_embd, block.attention_norm,
        ev.backend.rms_epsilon, ev.backend.norm_chunk4);
  }
};

struct effect_normalize_chunk8 {
  void operator()(const event::chunk8_run &ev) const noexcept {
    auto &block = ev.backend.blocks[static_cast<size_t>(ev.layer_index)];
    ev.normalized_ok = emel::text::generator::detail::rms_norm_chunk8(
        ev.backend.hidden_chunk8, ev.backend.n_embd, block.attention_norm,
        ev.backend.rms_epsilon, ev.backend.norm_chunk8);
  }
};

// Branch-only coverage exclusion: these action wrappers compose already-chosen
// route bodies. The transition guards remain branch-covered in guards.hpp,
// while the numeric route bodies retain line coverage in generator detail
// tests. GCOVR_EXCL_BR_START
template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::scalar_matmul_route route,
          event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route,
          emel::text::generator::matmul::lane_mode lanes>
struct effect_run_scalar_attention {
  void operator()(const event::scalar_run &ev) const noexcept {
    ev.residual_ok =
        run_layer_attention_residual<mode, route, qk_route, v_route, lanes>(
            ev.backend, ev.layer_index, ev.kv, ev.position);
  }
};

template <emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct effect_run_scalar_shortconv {
  void operator()(const event::scalar_run &ev) const noexcept {
    ev.residual_ok = run_layer_shortconv_residual<route, lanes>(
        ev.backend, ev.layer_index, ev.kv);
  }
};

template <emel::text::generator::detail::scalar_matmul_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct effect_run_scalar_feed_forward {
  void operator()(const event::scalar_run &ev) const noexcept {
    ev.feed_forward_ok =
        run_layer_feed_forward<route, lanes>(ev.backend, ev.layer_index);
  }
};

template <emel::text::generator::attention_mode mode,
          emel::text::generator::detail::chunk4_rhs_route route,
          event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route,
          emel::text::generator::matmul::lane_mode lanes>
struct effect_run_chunk4_attention {
  void operator()(const event::chunk4_run &ev) const noexcept {
    ev.residual_ok = run_layer_chunk4_attention_residual<mode, route, qk_route,
                                                         v_route, lanes>(
        ev.backend, ev.kv, ev.layer_index, ev.token_base);
  }
};

template <emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct effect_run_chunk4_shortconv {
  void operator()(const event::chunk4_run &ev) const noexcept {
    ev.residual_ok = run_layer_chunk4_shortconv_residual<route, lanes>(
        ev.backend, ev.kv, ev.layer_index);
  }
};

template <emel::text::generator::detail::chunk4_rhs_route route,
          emel::text::generator::matmul::lane_mode lanes>
struct effect_run_chunk4_feed_forward {
  void operator()(const event::chunk4_run &ev) const noexcept {
    ev.feed_forward_ok =
        run_layer_chunk4_feed_forward<route, lanes>(ev.backend, ev.layer_index);
  }
};

template <emel::text::generator::attention_mode mode,
          event::attention_qk_norm_route qk_route,
          event::attention_v_norm_route v_route,
          emel::text::generator::matmul::lane_mode lanes>
struct effect_run_chunk8_attention {
  void operator()(const event::chunk8_run &ev) const noexcept {
    ev.residual_ok = run_layer_chunk8_q8_k_attention_residual<mode, qk_route,
                                                              v_route, lanes>(
        ev.backend, ev.kv, ev.layer_index, ev.token_base);
  }
};

template <emel::text::generator::matmul::lane_mode lanes>
struct effect_run_chunk8_shortconv {
  void operator()(const event::chunk8_run &ev) const noexcept {
    ev.residual_ok = run_layer_chunk8_q8_k_shortconv_residual<lanes>(
        ev.backend, ev.kv, ev.layer_index);
  }
};

template <emel::text::generator::matmul::lane_mode lanes>
struct effect_run_chunk8_feed_forward {
  void operator()(const event::chunk8_run &ev) const noexcept {
    ev.feed_forward_ok =
        run_layer_chunk8_q8_k_feed_forward<lanes>(ev.backend, ev.layer_index);
  }
};

struct effect_reject_unsupported_route {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  void operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    (*this)(ev.event_);
  }

  void operator()(const event::scalar_run &ev) const noexcept {
    ev.succeeded = false;
    ev.failed = true;
  }

  void operator()(const event::chunk4_run &ev) const noexcept {
    ev.succeeded = false;
    ev.failed = true;
  }

  void operator()(const event::chunk8_run &ev) const noexcept {
    ev.succeeded = false;
    ev.failed = true;
  }
};

struct effect_mark_succeeded {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  void operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    (*this)(ev.event_);
  }

  void operator()(const event::scalar_run &ev) const noexcept {
    ev.succeeded = true;
    ev.failed = false;
  }

  void operator()(const event::chunk4_run &ev) const noexcept {
    ev.succeeded = true;
    ev.failed = false;
  }

  void operator()(const event::chunk8_run &ev) const noexcept {
    ev.succeeded = true;
    ev.failed = false;
  }
};

struct effect_mark_failed {
  template <class completion_type, class sm_type, class deps_type,
            class subs_type>
  void operator()(const completion_type &ev, sm_type &, deps_type &,
                  subs_type &) const noexcept {
    (*this)(ev.event_);
  }

  void operator()(const event::scalar_run &ev) const noexcept {
    ev.succeeded = false;
    ev.failed = true;
  }

  void operator()(const event::chunk4_run &ev) const noexcept {
    ev.succeeded = false;
    ev.failed = true;
  }

  void operator()(const event::chunk8_run &ev) const noexcept {
    ev.succeeded = false;
    ev.failed = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &) const noexcept {}
};
// GCOVR_EXCL_BR_STOP

inline constexpr effect_reject_unsupported_route
    effect_reject_unsupported_route{};
inline constexpr effect_mark_succeeded effect_mark_succeeded{};
inline constexpr effect_mark_failed effect_mark_failed{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::text::generator::layer::action
