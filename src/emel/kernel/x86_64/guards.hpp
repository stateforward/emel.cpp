#pragma once

#include "emel/kernel/detail.hpp"
#include "emel/kernel/x86_64/actions.hpp"
#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/events.hpp"

namespace emel::kernel::x86_64::guard {

template <class dispatch_event_type> struct simd_op {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2(ev.request,
                                                        ctx.avx2_available);
  }
};

struct guard_simd_op_mul_mat_f32_fma {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_f32_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_f32_avx2_only {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    return simd_op<::emel::kernel::x86_64::event::dispatch_op_mul_mat>{}(ev,
                                                                         ctx) &&
           !guard_simd_op_mul_mat_f32_fma{}(ev, ctx);
  }
};

struct guard_simd_op_mul_mat_q2_k_q8_k {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q2_k_q8_k_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_q3_k_q8_k {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q3_k_q8_k_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_q4_k_q8_k {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q4_k_q8_k_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_q6_k_q8_k {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q6_k_q8_k_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_q4_0_q8_0 {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q4_0_q8_0_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_q4_1_q8_0 {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q4_1_q8_0_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_q5_0_q8_0 {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q5_0_q8_0_mul_mat(
        ev.request, ctx.host_features);
  }
};

struct guard_simd_op_mul_mat_q8_0_q8_0 {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_mul_mat &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::can_use_avx2_fma_q8_0_q8_0_mul_mat(
        ev.request, ctx.host_features);
  }
};

template <class dispatch_event_type> struct valid_op {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_run_backend_request(ev.request)) {
      return false;
    }
    if constexpr (std::is_same_v<
                      dispatch_event_type,
                      ::emel::kernel::x86_64::event::dispatch_op_mul_mat>) {
      return !simd_op<dispatch_event_type>{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q2_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q3_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q4_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q6_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q4_0_q8_0{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q4_1_q8_0{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q5_0_q8_0{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q8_0_q8_0{}(ev, ctx);
    }
    return !simd_op<dispatch_event_type>{}(ev, ctx);
  }
};

struct simd_op_flash_attn_ext_f16kv_one_chunk {
  bool operator()(
      const ::emel::kernel::x86_64::event::dispatch_op_flash_attn_ext &ev,
      const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::x86_64::detail::
        can_run_avx2_fma_f16c_flash_attn_ext_f16kv_one_chunk_request(
            ev.request, ctx.host_features, ctx.flash_attn_workspace);
  }
};

struct valid_op_flash_attn_ext_shared {
  bool operator()(
      const ::emel::kernel::x86_64::event::dispatch_op_flash_attn_ext &ev,
      const action::context &ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_run_backend_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::detail::can_run_flash_attn_ext_with_workspace(
               ev.request, ctx.flash_attn_workspace) &&
           !simd_op_flash_attn_ext_f16kv_one_chunk{}(ev, ctx);
  }
};

template <class dispatch_event_type> struct invalid_op {
  bool operator()(const dispatch_event_type &ev,
                  const action::context &ctx) const noexcept {
    if constexpr (std::is_same_v<dispatch_event_type,
                                 ::emel::kernel::x86_64::event::
                                     dispatch_op_flash_attn_ext>) {
      return !simd_op_flash_attn_ext_f16kv_one_chunk{}(ev, ctx) &&
             !valid_op_flash_attn_ext_shared{}(ev, ctx);
    }
    if constexpr (std::is_same_v<
                      dispatch_event_type,
                      ::emel::kernel::x86_64::event::dispatch_op_mul_mat>) {
      return !simd_op<dispatch_event_type>{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q2_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q3_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q4_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q6_k_q8_k{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q4_0_q8_0{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q4_1_q8_0{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q5_0_q8_0{}(ev, ctx) &&
             !guard_simd_op_mul_mat_q8_0_q8_0{}(ev, ctx) &&
             !valid_op<dispatch_event_type>{}(ev, ctx);
    }
    return !simd_op<dispatch_event_type>{}(ev, ctx) &&
           !valid_op<dispatch_event_type>{}(ev, ctx);
  }
};

template <::emel::kernel::event::unary_subop subop> struct unary_subop_is {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_unary &ev,
                  const action::context &) const noexcept {
    return ev.request.subop == subop;
  }
};

template <::emel::kernel::event::unary_subop subop>
using simd_op_unary_subop = ::emel::kernel::detail::simd_unary_subop_guard<
    ::emel::kernel::x86_64::event::dispatch_op_unary, action::context,
    simd_op<::emel::kernel::x86_64::event::dispatch_op_unary>,
    unary_subop_is<subop>>;

template <::emel::kernel::event::unary_subop subop>
using valid_op_unary_subop = ::emel::kernel::detail::valid_unary_subop_guard<
    ::emel::kernel::x86_64::event::dispatch_op_unary, action::context,
    valid_op<::emel::kernel::x86_64::event::dispatch_op_unary>,
    unary_subop_is<subop>>;

using simd_op_unary_abs =
    simd_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using simd_op_unary_neg =
    simd_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using simd_op_unary_relu =
    simd_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_abs =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using valid_op_unary_neg =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using valid_op_unary_relu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_exp =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::exp>;
using valid_op_unary_tanh =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::tanh>;
using valid_op_unary_elu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::elu>;
using valid_op_unary_gelu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::gelu>;
using valid_op_unary_silu =
    valid_op_unary_subop<::emel::kernel::event::unary_subop::silu>;

// Variant predicates for the ops whose dtype/mode choice is modeled as
// explicit transition rows (op_unary pattern).
template <uint8_t src_dtype_code> struct get_rows_src_dtype_is {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_get_rows &ev,
                  const action::context &) const noexcept {
    return ::emel::kernel::detail::dtype_code(ev.request.src0.type) ==
           src_dtype_code;
  }
};

template <bool neox_mode> struct rope_mode_is {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_rope &ev,
                  const action::context &) const noexcept {
    int32_t mode = 0;
    return ::emel::kernel::detail::read_op_param_i32(
               ev.request.op_params.data(), ev.request.op_params_size, 2u,
               mode) &&
           mode == (neox_mode ? ::emel::kernel::detail::rope_mode_neox
                              : ::emel::kernel::detail::rope_mode_norm);
  }
};

template <bool f16_dst> struct im2col_dst_dtype_is {
  bool operator()(const ::emel::kernel::x86_64::event::dispatch_op_im2col &ev,
                  const action::context &) const noexcept {
    return ::emel::kernel::detail::dtype_code(ev.request.dst.type) ==
           (f16_dst ? ::emel::kernel::detail::dtype_f16
                    : ::emel::kernel::detail::dtype_f32);
  }
};

template <bool f16_weights> struct conv_transpose_1d_weight_dtype_is {
  bool operator()(
      const ::emel::kernel::x86_64::event::dispatch_op_conv_transpose_1d &ev,
      const action::context &) const noexcept {
    return ::emel::kernel::detail::dtype_code(ev.request.src0.type) ==
           (f16_weights ? ::emel::kernel::detail::dtype_f16
                        : ::emel::kernel::detail::dtype_f32);
  }
};

template <uint8_t src_dtype_code>
using valid_op_get_rows_src = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::x86_64::event::dispatch_op_get_rows, action::context,
    valid_op<::emel::kernel::x86_64::event::dispatch_op_get_rows>,
    get_rows_src_dtype_is<src_dtype_code>>;

using valid_op_get_rows_f32 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_f32>;
using valid_op_get_rows_f16 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_f16>;
using valid_op_get_rows_bf16 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_bf16>;
using valid_op_get_rows_q4_0 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_q4_0>;
using valid_op_get_rows_q8_0 =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_q8_0>;
using valid_op_get_rows_q4_k =
    valid_op_get_rows_src<::emel::kernel::detail::dtype_q4_k>;

using valid_op_rope_norm = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::x86_64::event::dispatch_op_rope, action::context,
    valid_op<::emel::kernel::x86_64::event::dispatch_op_rope>,
    rope_mode_is<false>>;
using valid_op_rope_neox = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::x86_64::event::dispatch_op_rope, action::context,
    valid_op<::emel::kernel::x86_64::event::dispatch_op_rope>,
    rope_mode_is<true>>;

using valid_op_im2col_f32 = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::x86_64::event::dispatch_op_im2col, action::context,
    valid_op<::emel::kernel::x86_64::event::dispatch_op_im2col>,
    im2col_dst_dtype_is<false>>;
using valid_op_im2col_f16 = ::emel::kernel::detail::valid_variant_guard<
    ::emel::kernel::x86_64::event::dispatch_op_im2col, action::context,
    valid_op<::emel::kernel::x86_64::event::dispatch_op_im2col>,
    im2col_dst_dtype_is<true>>;

using valid_op_conv_transpose_1d_f32 =
    ::emel::kernel::detail::valid_variant_guard<
        ::emel::kernel::x86_64::event::dispatch_op_conv_transpose_1d,
        action::context,
        valid_op<::emel::kernel::x86_64::event::dispatch_op_conv_transpose_1d>,
        conv_transpose_1d_weight_dtype_is<false>>;
using valid_op_conv_transpose_1d_f16 =
    ::emel::kernel::detail::valid_variant_guard<
        ::emel::kernel::x86_64::event::dispatch_op_conv_transpose_1d,
        action::context,
        valid_op<::emel::kernel::x86_64::event::dispatch_op_conv_transpose_1d>,
        conv_transpose_1d_weight_dtype_is<true>>;

#define EMEL_KERNEL_DECLARE_GUARD_ALIAS(op_name)                               \
  using simd_##op_name =                                                       \
      simd_op<::emel::kernel::x86_64::event::dispatch_##op_name>;              \
  using valid_##op_name =                                                      \
      valid_op<::emel::kernel::x86_64::event::dispatch_##op_name>;             \
  using invalid_##op_name =                                                    \
      invalid_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_GUARD_ALIAS)
#undef EMEL_KERNEL_DECLARE_GUARD_ALIAS

} // namespace emel::kernel::x86_64::guard
