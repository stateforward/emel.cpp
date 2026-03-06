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

template <class dispatch_event_type>
struct valid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_execute_scalar(ev.request)) {
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
struct simd_op_unary_subop {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_unary & ev,
                  const action::context & ctx) const noexcept {
    return simd_op<::emel::kernel::aarch64::event::dispatch_op_unary>{}(ev, ctx) &&
           unary_subop_is<subop>{}(ev, ctx);
  }
};

template <::emel::kernel::event::unary_subop subop>
struct valid_op_unary_subop {
  bool operator()(const ::emel::kernel::aarch64::event::dispatch_op_unary & ev,
                  const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if (!::emel::kernel::detail::can_run_unary_subop(ev.request)) {
      return false;
    }
    return !simd_op<::emel::kernel::aarch64::event::dispatch_op_unary>{}(ev, ctx) &&
           unary_subop_is<subop>{}(ev, ctx);
  }
};

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
