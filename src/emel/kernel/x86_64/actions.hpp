#pragma once

#include "emel/emel.h"
#include "emel/kernel/op_list.hpp"
#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/events.hpp"

namespace emel::kernel::x86_64::action {

namespace detail {

template <class dispatch_event_type>
inline void mark_done(const dispatch_event_type & ev, context & ctx) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::done;
  ev.ctx.err = EMEL_OK;
}

template <class dispatch_event_type>
inline void mark_error(const dispatch_event_type & ev, context & ctx,
                       const int32_t err) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::failed;
  ev.ctx.err = err;
}

template <class dispatch_event_type>
struct run_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    detail::mark_done(ev, ctx);
  }
};

template <class dispatch_event_type>
struct reject_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    detail::mark_error(ev, ctx, EMEL_ERR_INVALID_ARGUMENT);
  }
};

}  // namespace detail

using run_scaffold_t = detail::run_op<::emel::kernel::x86_64::event::dispatch_scaffold>;

#define EMEL_KERNEL_DECLARE_RUN_TYPE(op_name)                                \
  using run_##op_name##_t =                                                  \
      detail::run_op<::emel::kernel::x86_64::event::dispatch_##op_name>;
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
      detail::mark_error(ev, ctx, EMEL_ERR_BACKEND);
    } else {
      ++ctx.dispatch_generation;
    }
  }
};

inline constexpr run_scaffold_t run_scaffold{};

#define EMEL_KERNEL_DEFINE_RUN_ACTION(op_name) \
  inline constexpr run_##op_name##_t run_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_RUN_ACTION)
#undef EMEL_KERNEL_DEFINE_RUN_ACTION

#define EMEL_KERNEL_DEFINE_REJECT_ACTION(op_name)            \
  inline constexpr reject_invalid_##op_name##_t reject_invalid_##op_name{};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_REJECT_ACTION)
#undef EMEL_KERNEL_DEFINE_REJECT_ACTION

inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::kernel::x86_64::action
