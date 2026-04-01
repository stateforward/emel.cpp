#pragma once

#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/aarch64/context.hpp"
#include "emel/kernel/aarch64/events.hpp"

namespace emel::kernel::aarch64::guard {

template <class dispatch_event_type>
struct simd_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon(ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_vector {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_vector(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl8_tail_safe(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8_full_groups {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl8_full_groups(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl8_matrix_x4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl8_matrix_x4(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q8_0_packed_bl4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q8_0_packed_bl4(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl4(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8 {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q4_vector_packed_q8_rhs_bl8(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_packed(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_packed_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_packed_q8_rhs(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_prepared_q8_rhs(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_packed_q8_rhs {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_use_neon_mul_mat_argmax_q6_vector_packed_q8_rhs(
        ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_argmax_q6_vector_prepared_q8_rhs_i8mm(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat_argmax & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::
        can_use_neon_mul_mat_argmax_q6_vector_q8_argmax_prepared_i8mm(
            ev.request, ctx.neon_available);
  }
};

struct simd_op_mul_mat_generic {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_mul_mat & ev,
                  const action::context & ctx) const noexcept {
    return simd_op<::emel::kernel::aarch64::event::dispatch_op_mul_mat>{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl8_matrix_x4{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl8_full_groups{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl8{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_packed_bl4{}(ev, ctx) &&
        !simd_op_mul_mat_q8_0_vector{}(ev, ctx) &&
        !simd_op_mul_mat_q4_vector_packed_q8_rhs_bl8{}(ev, ctx) &&
        !simd_op_mul_mat_q4_vector_packed_q8_rhs_bl4{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_prepared_q8_rhs_i8mm{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_prepared_q8_rhs{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_packed_q8_rhs{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector_packed{}(ev, ctx) &&
        !simd_op_mul_mat_q6_vector{}(ev, ctx);
  }
};

struct simd_op_flash_attn_ext_f16kv_one_chunk {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_flash_attn_ext & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    return ::emel::kernel::aarch64::detail::can_run_neon_flash_attn_ext_f16kv_one_chunk_request(
        ev.request, ctx.neon_available, ctx.flash_attn_workspace);
  }
};

struct valid_op_flash_attn_ext_shared {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_flash_attn_ext & ev,
                  const action::context & ctx) const noexcept {
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

template <class dispatch_event_type>
struct valid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_run_backend_request(ev.request)) {
      return false;
    }
    return !simd_op<dispatch_event_type>{}(ev, ctx);
  }
};

template <class dispatch_event_type>
struct invalid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    return !simd_op<dispatch_event_type>{}(ev, ctx) &&
           !valid_op<dispatch_event_type>{}(ev, ctx);
  }
};

template <::emel::kernel::event::unary_subop subop>
struct unary_subop_is {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_unary & ev,
                  const action::context &) const noexcept {
    return ev.request.subop == subop;
  }
};

template <::emel::kernel::event::unary_subop subop>
using simd_op_unary_subop = ::emel::kernel::detail::simd_unary_subop_guard<
    ::emel::kernel::aarch64::event::dispatch_op_unary, action::context,
    simd_op<::emel::kernel::aarch64::event::dispatch_op_unary>, unary_subop_is<subop>>;

template <::emel::kernel::event::unary_subop subop>
using valid_op_unary_subop = ::emel::kernel::detail::valid_unary_subop_guard<
    ::emel::kernel::aarch64::event::dispatch_op_unary, action::context,
    valid_op<::emel::kernel::aarch64::event::dispatch_op_unary>, unary_subop_is<subop>>;

using simd_op_unary_abs = simd_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using simd_op_unary_neg = simd_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using simd_op_unary_relu = simd_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_abs = valid_op_unary_subop<::emel::kernel::event::unary_subop::abs>;
using valid_op_unary_neg = valid_op_unary_subop<::emel::kernel::event::unary_subop::neg>;
using valid_op_unary_relu = valid_op_unary_subop<::emel::kernel::event::unary_subop::relu>;
using valid_op_unary_exp = valid_op_unary_subop<::emel::kernel::event::unary_subop::exp>;

#define EMEL_KERNEL_DECLARE_GUARD_ALIAS(op_name)                                 \
  using simd_##op_name =                                                         \
      simd_op<::emel::kernel::aarch64::event::dispatch_##op_name>;               \
  using valid_##op_name =                                                        \
      valid_op<::emel::kernel::aarch64::event::dispatch_##op_name>;              \
  using invalid_##op_name =                                                      \
      invalid_op<::emel::kernel::aarch64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_GUARD_ALIAS)
#undef EMEL_KERNEL_DECLARE_GUARD_ALIAS

}  // namespace emel::kernel::aarch64::guard
