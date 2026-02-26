#pragma once

#include "emel/kernel/detail.hpp"
#include "emel/kernel/wasm/context.hpp"
#include "emel/kernel/wasm/events.hpp"

namespace emel::kernel::wasm::guard {

template <class dispatch_event_type>
struct valid_impl {
  static bool check(const dispatch_event_type & ev, const action::context &) noexcept {
    return ::emel::kernel::detail::validate_scaffold_request(ev.request);
  }
};

template <>
struct valid_impl<::emel::kernel::wasm::event::dispatch_op_dup> {
  static bool check(const ::emel::kernel::wasm::event::dispatch_op_dup & ev,
                    const action::context &) noexcept {
    return ev.request.src0 != nullptr && ev.request.dst != nullptr &&
           ev.request.element_count > 0;
  }
};

template <>
struct valid_impl<::emel::kernel::wasm::event::dispatch_op_add> {
  static bool check(const ::emel::kernel::wasm::event::dispatch_op_add & ev,
                    const action::context &) noexcept {
    return ev.request.src0 != nullptr && ev.request.src1 != nullptr &&
           ev.request.dst != nullptr && ev.request.element_count > 0;
  }
};

template <>
struct valid_impl<::emel::kernel::wasm::event::dispatch_op_mul> {
  static bool check(const ::emel::kernel::wasm::event::dispatch_op_mul & ev,
                    const action::context &) noexcept {
    return ev.request.src0 != nullptr && ev.request.src1 != nullptr &&
           ev.request.dst != nullptr && ev.request.element_count > 0;
  }
};

template <>
struct valid_impl<::emel::kernel::wasm::event::dispatch_op_mul_mat> {
  static bool check(const ::emel::kernel::wasm::event::dispatch_op_mul_mat & ev,
                    const action::context &) noexcept {
    return ev.request.src0 != nullptr && ev.request.src1 != nullptr &&
           ev.request.dst != nullptr && ev.request.row_count > 0 &&
           ev.request.col_count > 0;
  }
};

template <>
struct valid_impl<::emel::kernel::wasm::event::dispatch_op_rope> {
  static bool check(const ::emel::kernel::wasm::event::dispatch_op_rope & ev,
                    const action::context &) noexcept {
    return ev.request.src0 != nullptr && ev.request.dst != nullptr &&
           ev.request.token_count > 0;
  }
};

template <>
struct valid_impl<::emel::kernel::wasm::event::dispatch_op_soft_max> {
  static bool check(const ::emel::kernel::wasm::event::dispatch_op_soft_max & ev,
                    const action::context &) noexcept {
    return ev.request.src0 != nullptr && ev.request.dst != nullptr &&
           ev.request.element_count > 0;
  }
};

template <class dispatch_event_type>
struct valid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    return valid_impl<dispatch_event_type>::check(ev, ctx);
  }
};

template <class dispatch_event_type>
struct invalid_op {
  bool operator()(const dispatch_event_type & ev, const action::context & ctx) const noexcept {
    return !valid_impl<dispatch_event_type>::check(ev, ctx);
  }
};

#define EMEL_KERNEL_DECLARE_GUARD_ALIAS(op_name)                                    \
  using valid_##op_name =                                                           \
      valid_op<::emel::kernel::wasm::event::dispatch_##op_name>;            \
  using invalid_##op_name =                                                         \
      invalid_op<::emel::kernel::wasm::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_GUARD_ALIAS)
#undef EMEL_KERNEL_DECLARE_GUARD_ALIAS

}  // namespace emel::kernel::wasm::guard
