#pragma once

#include "emel/kernel/detail.hpp"
#include "emel/kernel/x86_64/detail.hpp"
#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/events.hpp"

namespace emel::kernel::x86_64::guard {

template <class dispatch_event_type>
struct simd_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    if (!::emel::kernel::detail::validate_dispatch_request(ev.request)) {
      return false;
    }
    if constexpr (std::is_same_v<dispatch_event_type, event::dispatch_op_dup> ||
                  std::is_same_v<dispatch_event_type, event::dispatch_op_add> ||
                  std::is_same_v<dispatch_event_type, event::dispatch_op_mul>) {
      return ::emel::kernel::x86_64::detail::can_use_avx2(ev.request, ctx.avx2_available);
    }
    (void) ctx;
    return false;
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

#define EMEL_KERNEL_DECLARE_GUARD_ALIAS(op_name)                                 \
  using simd_##op_name =                                                         \
      simd_op<::emel::kernel::x86_64::event::dispatch_##op_name>;                \
  using valid_##op_name =                                                        \
      valid_op<::emel::kernel::x86_64::event::dispatch_##op_name>;               \
  using invalid_##op_name =                                                      \
      invalid_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_GUARD_ALIAS)
#undef EMEL_KERNEL_DECLARE_GUARD_ALIAS

}  // namespace emel::kernel::x86_64::guard
