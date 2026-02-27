#pragma once

#include "emel/kernel/detail.hpp"
#include "emel/kernel/wasm/context.hpp"
#include "emel/kernel/wasm/events.hpp"

namespace emel::kernel::wasm::guard {

template <class dispatch_event_type>
struct valid_op {
  bool operator()(const dispatch_event_type & ev, const action::context &) const noexcept {
    return ::emel::kernel::detail::validate_dispatch_request(ev.request);
  }
};

template <class dispatch_event_type>
struct invalid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    return !valid_op<dispatch_event_type>{}(ev, ctx);
  }
};

#define EMEL_KERNEL_DECLARE_GUARD_ALIAS(op_name)                                 \
  using valid_##op_name =                                                        \
      valid_op<::emel::kernel::wasm::event::dispatch_##op_name>;                 \
  using invalid_##op_name =                                                      \
      invalid_op<::emel::kernel::wasm::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_GUARD_ALIAS)
#undef EMEL_KERNEL_DECLARE_GUARD_ALIAS

}  // namespace emel::kernel::wasm::guard
