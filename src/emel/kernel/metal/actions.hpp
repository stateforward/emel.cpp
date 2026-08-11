#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/kernel/metal/context.hpp"
#include "emel/kernel/metal/detail.hpp"
#include "emel/kernel/metal/errors.hpp"
#include "emel/kernel/metal/events.hpp"

// Metal actor actions. Each transition row names one already-validated,
// already-chosen variant; the actions only stage + dispatch + read back and
// never select behavior (no runtime dtype/route branching here). Dispatch is
// a synchronous GPU round trip, so the RTC boundary holds.
namespace emel::kernel::metal::action {

namespace detail {

template <class dispatch_event_type>
inline void mark_done(const dispatch_event_type &ev, context &ctx) noexcept {
  ++ctx.dispatch_generation;
  ++ctx.metal_dispatch_count;
  ev.ctx.outcome = events::phase_outcome::done;
  ev.ctx.err = static_cast<int32_t>(emel::error::cast(error::none));
}

template <class dispatch_event_type>
inline void mark_error(const dispatch_event_type &ev, context &ctx,
                       const int32_t err) noexcept {
  ++ctx.dispatch_generation;
  ev.ctx.outcome = events::phase_outcome::failed;
  ev.ctx.err = err;
}

struct mark_done_op {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type &ev, context &ctx) const noexcept {
    mark_done(ev, ctx);
  }
};

struct mark_error_op {
  int32_t err = static_cast<int32_t>(emel::error::cast(error::internal_error));

  template <class dispatch_event_type>
  void operator()(const dispatch_event_type &ev, context &ctx) const noexcept {
    mark_error(ev, ctx, err);
  }
};

// Runs the already-guard-selected variant and maps the outcome onto the
// dispatch_ctx. The detail helpers' bool reports ONLY an unrecoverable
// runtime fault (a GPU/library failure after guard validation) - never
// input-dependent routing, which the guards own - so the accepted/done/failed
// decision stays on the action surface. This mirrors the x86_64 backend
// precedent (exec_scalar_op consumes a run_* bool identically).
template <class dispatch_event_type>
inline bool exec_or_error(dispatch_event_type &ev, context &ctx,
                          const bool ok) noexcept {
  if (ok) {
    mark_done(ev, ctx);
  } else {
    mark_error(ev, ctx,
               static_cast<int32_t>(emel::error::cast(error::internal_error)));
  }
  return ok;
}

} // namespace detail

struct exec_dispatch {
  void operator()(const ::emel::kernel::metal::event::dispatch_request &ev,
                  context &ctx) const noexcept {
    detail::mark_done(ev, ctx);
  }
};

//------------------------------------------------------------------------------//
// op_mul_mat rows.
//------------------------------------------------------------------------------//

struct exec_op_mul_mat_f32 {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(ev, ctx,
                                ::emel::kernel::metal::detail::run_mul_mat_f32(
                                    *ctx.runtime, ev.request));
  }
};

struct exec_op_mul_mat_f16 {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(ev, ctx,
                                ::emel::kernel::metal::detail::run_mul_mat_f16(
                                    *ctx.runtime, ev.request));
  }
};

struct exec_op_mul_mat_q8_0 {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(ev, ctx,
                                ::emel::kernel::metal::detail::run_mul_mat_q8_0(
                                    *ctx.runtime, ev.request));
  }
};

struct reject_invalid_op_mul_mat {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_mul_mat &ev,
                  context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

//------------------------------------------------------------------------------//
// op_add rows.
//------------------------------------------------------------------------------//

struct exec_op_add_equal {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_add &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(
        ev, ctx,
        ::emel::kernel::metal::detail::run_add(*ctx.runtime, ev.request));
  }
};

struct exec_op_add_broadcast_row {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_add &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(
        ev, ctx,
        ::emel::kernel::metal::detail::run_add_broadcast_row(*ctx.runtime,
                                                             ev.request));
  }
};

struct reject_invalid_op_add {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_add &ev,
                  context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

//------------------------------------------------------------------------------//
// op_unary rows (subop is a compile-time variant).
//------------------------------------------------------------------------------//

template <::emel::kernel::event::unary_subop subop> struct exec_op_unary_subop {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_unary &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(
        ev, ctx,
        ::emel::kernel::metal::detail::run_unary(*ctx.runtime, ev.request));
  }
};

struct reject_invalid_op_unary {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_unary &ev,
                  context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

//------------------------------------------------------------------------------//
// op_im2col rows.
//------------------------------------------------------------------------------//

struct exec_op_im2col_f32 {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_im2col &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(ev, ctx,
                                ::emel::kernel::metal::detail::run_im2col_f32(
                                    *ctx.runtime, ev.request));
  }
};

struct exec_op_im2col_f16 {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_im2col &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(ev, ctx,
                                ::emel::kernel::metal::detail::run_im2col_f16(
                                    *ctx.runtime, ev.request));
  }
};

struct reject_invalid_op_im2col {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_im2col &ev,
                  context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

//------------------------------------------------------------------------------//
// op_conv_transpose_1d rows.
//------------------------------------------------------------------------------//

struct exec_op_conv_transpose_1d_f32 {
  void operator()(
      const ::emel::kernel::metal::event::dispatch_op_conv_transpose_1d &ev,
      context &ctx) const noexcept {
    (void)detail::exec_or_error(
        ev, ctx,
        ::emel::kernel::metal::detail::run_conv_transpose_1d_f32(*ctx.runtime,
                                                                 ev.request));
  }
};

struct exec_op_conv_transpose_1d_f16 {
  void operator()(
      const ::emel::kernel::metal::event::dispatch_op_conv_transpose_1d &ev,
      context &ctx) const noexcept {
    (void)detail::exec_or_error(
        ev, ctx,
        ::emel::kernel::metal::detail::run_conv_transpose_1d_f16(*ctx.runtime,
                                                                 ev.request));
  }
};

struct reject_invalid_op_conv_transpose_1d {
  void operator()(
      const ::emel::kernel::metal::event::dispatch_op_conv_transpose_1d &ev,
      context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

//------------------------------------------------------------------------------//
// op_get_rows rows.
//------------------------------------------------------------------------------//

struct exec_op_get_rows_f32 {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_get_rows &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(ev, ctx,
                                ::emel::kernel::metal::detail::run_get_rows_f32(
                                    *ctx.runtime, ev.request));
  }
};

struct exec_op_get_rows_f16 {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_get_rows &ev,
                  context &ctx) const noexcept {
    (void)detail::exec_or_error(ev, ctx,
                                ::emel::kernel::metal::detail::run_get_rows_f16(
                                    *ctx.runtime, ev.request));
  }
};

struct reject_invalid_op_get_rows {
  void operator()(const ::emel::kernel::metal::event::dispatch_op_get_rows &ev,
                  context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::invalid_request)));
  }
};

//------------------------------------------------------------------------------//
// Unsupported op events: explicit reject instead of silent drop.
//------------------------------------------------------------------------------//

struct reject_unsupported_op {
  template <class dispatch_event_type>
  void operator()(const dispatch_event_type &ev, context &ctx) const noexcept {
    detail::mark_error(
        ev, ctx,
        static_cast<int32_t>(emel::error::cast(error::unsupported_op)));
  }
};

inline constexpr exec_dispatch exec_dispatch{};
inline constexpr exec_op_mul_mat_f32 exec_op_mul_mat_f32{};
inline constexpr exec_op_mul_mat_f16 exec_op_mul_mat_f16{};
inline constexpr exec_op_mul_mat_q8_0 exec_op_mul_mat_q8_0{};
inline constexpr reject_invalid_op_mul_mat reject_invalid_op_mul_mat{};
inline constexpr exec_op_add_equal exec_op_add_equal{};
inline constexpr exec_op_add_broadcast_row exec_op_add_broadcast_row{};
inline constexpr reject_invalid_op_add reject_invalid_op_add{};
inline constexpr reject_invalid_op_unary reject_invalid_op_unary{};
inline constexpr exec_op_im2col_f32 exec_op_im2col_f32{};
inline constexpr exec_op_im2col_f16 exec_op_im2col_f16{};
inline constexpr reject_invalid_op_im2col reject_invalid_op_im2col{};
inline constexpr exec_op_conv_transpose_1d_f32 exec_op_conv_transpose_1d_f32{};
inline constexpr exec_op_conv_transpose_1d_f16 exec_op_conv_transpose_1d_f16{};
inline constexpr reject_invalid_op_conv_transpose_1d
    reject_invalid_op_conv_transpose_1d{};
inline constexpr exec_op_get_rows_f32 exec_op_get_rows_f32{};
inline constexpr exec_op_get_rows_f16 exec_op_get_rows_f16{};
inline constexpr reject_invalid_op_get_rows reject_invalid_op_get_rows{};
inline constexpr reject_unsupported_op reject_unsupported_op{};

} // namespace emel::kernel::metal::action
