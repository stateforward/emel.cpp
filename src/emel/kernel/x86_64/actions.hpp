#pragma once

#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/events.hpp"

namespace emel::kernel::x86_64::action {

namespace detail {

template <class dispatch_event_type>
inline void mark_done(const dispatch_event_type & ev, context & ctx) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::done;
  ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::none));
}

template <class dispatch_event_type>
inline void mark_error(const dispatch_event_type & ev, context & ctx,
                       const int32_t err) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::failed;
  ev.ctx.err = err;
}

template <class dispatch_event_type>
struct exec_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type>
struct reject_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    detail::mark_error(ev, ctx, static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

}  // namespace detail

using exec_scaffold_t = detail::exec_op<::emel::kernel::x86_64::event::dispatch_scaffold>;

#define EMEL_KERNEL_DECLARE_RUN_TYPE(op_name)                                \
  using exec_##op_name##_t =                                                  \
      detail::exec_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_RUN_TYPE)
#undef EMEL_KERNEL_DECLARE_RUN_TYPE

#define EMEL_KERNEL_DECLARE_REJECT_TYPE(op_name)                                      \
  using reject_invalid_##op_name##_t =                                                \
      detail::reject_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_REJECT_TYPE)
#undef EMEL_KERNEL_DECLARE_REJECT_TYPE

struct on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.ctx; }) {
      detail::mark_error(ev, ctx, static_cast<int32_t>(emel::error::cast(error::internal_error)));
    } else {
      ++ctx.dispatch_generation;
    }
  }
};

inline constexpr exec_scaffold_t exec_scaffold{};

#define EMEL_KERNEL_DEFINE_RUN_ACTION(op_name) \
  inline constexpr exec_##op_name##_t exec_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_RUN_ACTION)
#undef EMEL_KERNEL_DEFINE_RUN_ACTION

#define EMEL_KERNEL_DEFINE_REJECT_ACTION(op_name)            \
  inline constexpr reject_invalid_##op_name##_t reject_invalid_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_REJECT_ACTION)
#undef EMEL_KERNEL_DEFINE_REJECT_ACTION

inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::kernel::x86_64::action
