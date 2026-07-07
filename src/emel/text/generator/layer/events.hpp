#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/model/data.hpp"

namespace emel::text::generator::detail {
struct native_backend;
struct kv_addressing_view;
} // namespace emel::text::generator::detail

namespace emel::text::generator::layer::event {

using residual_route = emel::model::generation_residual_route;
using attention_qk_norm_route = emel::model::generation_attention_qk_norm_route;
using attention_v_norm_route = emel::model::generation_attention_v_norm_route;

struct scalar_run {
  scalar_run(emel::text::generator::detail::native_backend &backend_ref,
             const emel::text::generator::detail::kv_addressing_view &kv_ref,
             const int32_t layer_index_ref, const int32_t position_ref,
             const residual_route residual_ref,
             const attention_qk_norm_route qk_norm_ref,
             const attention_v_norm_route v_norm_ref,
             int32_t &error_ref) noexcept
      : backend(backend_ref), kv(kv_ref), error(error_ref),
        layer_index(layer_index_ref), position(position_ref),
        residual(residual_ref), qk_norm(qk_norm_ref), v_norm(v_norm_ref) {}

  emel::text::generator::detail::native_backend &backend;
  const emel::text::generator::detail::kv_addressing_view &kv;
  int32_t &error;
  int32_t layer_index = 0;
  int32_t position = 0;
  residual_route residual = residual_route::attention;
  attention_qk_norm_route qk_norm = attention_qk_norm_route::none;
  attention_v_norm_route v_norm = attention_v_norm_route::none;
  mutable bool stream_ready = false;
  mutable bool normalized_ok = false;
  mutable bool residual_ok = false;
  mutable bool feed_forward_ok = false;
  mutable bool succeeded = false;
  mutable bool failed = false;
};

struct chunk4_run {
  chunk4_run(emel::text::generator::detail::native_backend &backend_ref,
             const emel::text::generator::detail::kv_addressing_view &kv_ref,
             const int32_t layer_index_ref, const size_t token_base_ref,
             const residual_route residual_ref,
             const attention_qk_norm_route qk_norm_ref,
             const attention_v_norm_route v_norm_ref) noexcept
      : backend(backend_ref), kv(kv_ref), layer_index(layer_index_ref),
        token_base(token_base_ref), residual(residual_ref),
        qk_norm(qk_norm_ref), v_norm(v_norm_ref) {}

  emel::text::generator::detail::native_backend &backend;
  const emel::text::generator::detail::kv_addressing_view &kv;
  int32_t layer_index = 0;
  size_t token_base = 0u;
  residual_route residual = residual_route::attention;
  attention_qk_norm_route qk_norm = attention_qk_norm_route::none;
  attention_v_norm_route v_norm = attention_v_norm_route::none;
  mutable bool normalized_ok = false;
  mutable bool residual_ok = false;
  mutable bool feed_forward_ok = false;
  mutable bool succeeded = false;
  mutable bool failed = false;
};

struct chunk8_run {
  chunk8_run(emel::text::generator::detail::native_backend &backend_ref,
             const emel::text::generator::detail::kv_addressing_view &kv_ref,
             const int32_t layer_index_ref, const size_t token_base_ref,
             const residual_route residual_ref,
             const attention_qk_norm_route qk_norm_ref,
             const attention_v_norm_route v_norm_ref) noexcept
      : backend(backend_ref), kv(kv_ref), layer_index(layer_index_ref),
        token_base(token_base_ref), residual(residual_ref),
        qk_norm(qk_norm_ref), v_norm(v_norm_ref) {}

  emel::text::generator::detail::native_backend &backend;
  const emel::text::generator::detail::kv_addressing_view &kv;
  int32_t layer_index = 0;
  size_t token_base = 0u;
  residual_route residual = residual_route::attention;
  attention_qk_norm_route qk_norm = attention_qk_norm_route::none;
  attention_v_norm_route v_norm = attention_v_norm_route::none;
  mutable bool normalized_ok = false;
  mutable bool residual_ok = false;
  mutable bool feed_forward_ok = false;
  mutable bool succeeded = false;
  mutable bool failed = false;
};

} // namespace emel::text::generator::layer::event
