#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/kernel/matmul/context.hpp"
#include "emel/kernel/matmul/events.hpp"

namespace emel::kernel::matmul::guard {

inline bool guard_supported_lane_count(const size_t active_lanes) noexcept {
  return active_lanes == 2u || active_lanes == 4u || active_lanes == 8u;
}

inline lane_mode guard_policy_lane_mode(const size_t active_lanes) noexcept {
  return guard_supported_lane_count(active_lanes) ? lane_mode::parallel
                                                  : lane_mode::serial;
}

} // namespace emel::kernel::matmul::guard

namespace emel::kernel::matmul {

inline execution_policy
make_execution_policy(lane_pool &parallel_matmul_lanes,
                      const emel::kernel::kernel_kind kernel_kind,
                      const size_t active_lanes) noexcept {
  return execution_policy{
      .parallel_matmul_lanes = &parallel_matmul_lanes,
      .kernel_kind = kernel_kind,
      .active_lanes = active_lanes,
      .mode = guard::guard_policy_lane_mode(active_lanes),
  };
}

inline execution_policy
make_auto_execution_policy(lane_pool &parallel_matmul_lanes) noexcept {
  return make_execution_policy(parallel_matmul_lanes,
                               emel::kernel::detect_host_kind(), 8u);
}

} // namespace emel::kernel::matmul

namespace emel::kernel::matmul::guard {

inline bool
supports_parallel_execution(const execution_policy &policy) noexcept {
  const bool supported_lane_count =
      guard_supported_lane_count(policy.active_lanes);
  return policy.mode == lane_mode::parallel &&
         policy.parallel_matmul_lanes != nullptr && supported_lane_count;
}

inline bool guard_lane_storage_ready(const action::context &ctx) noexcept {
  const bool supported_lane_count =
      guard_supported_lane_count(ctx.active_lanes);
  return ctx.parallel_matmul_lanes != nullptr && supported_lane_count &&
         ctx.lanes != nullptr;
}

inline uint64_t guard_row_group_count(const event::execute_parallel &ev,
                                      const uint64_t group_rows) noexcept {
  const uint64_t rows = ev.request.src0.ne[1];
  return rows / group_rows + static_cast<uint64_t>((rows % group_rows) != 0u);
}

inline bool
guard_uses_x8_row_groups(const emel::kernel::event::dtype type) noexcept {
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  return code == emel::kernel::detail::dtype_q4_k_x8_bl4 ||
         code == emel::kernel::detail::dtype_q4_k_x8_bl8 ||
         code == emel::kernel::detail::dtype_q6_k_x8 ||
         code == emel::kernel::detail::dtype_q6_k_x8_q8_prepared ||
         code == emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared;
}

inline bool
guard_uses_x4_row_groups(const emel::kernel::event::dtype type) noexcept {
  const uint8_t code = emel::kernel::detail::dtype_code(type);
  return code == emel::kernel::detail::dtype_q8_0_x4_bl4 ||
         code == emel::kernel::detail::dtype_q8_0_x4_bl8;
}

inline bool guard_pointer_extent_valid(const void *data, const uint64_t stride,
                                       const uint64_t count) noexcept {
  if (data == nullptr || stride == 0u || count == 0u ||
      stride > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) /
                   count) {
    return false;
  }
  const uint64_t extent = stride * count;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
  return extent <= std::numeric_limits<uintptr_t>::max() - begin;
}

inline bool
guard_parallel_request_valid(const event::execute_parallel &ev) noexcept {
  const auto &request = ev.request;
  const uint64_t rows = request.src0.ne[1];
  const uint64_t group_rows = guard_uses_x8_row_groups(request.src0.type)
                                  ? emel::kernel::detail::quant::Q4_K_X8_ROWS
                              : guard_uses_x4_row_groups(request.src0.type)
                                  ? emel::kernel::detail::quant::Q8_0_X4_ROWS
                                  : 1u;
  const uint64_t storage_rows =
      rows / group_rows + static_cast<uint64_t>((rows % group_rows) != 0u);
  return rows > 0u &&
         rows <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) &&
         request.src0.data != nullptr && request.src1.data != nullptr &&
         request.dst.data != nullptr &&
         request.src0.type != emel::kernel::event::dtype::unknown &&
         request.src1.type != emel::kernel::event::dtype::unknown &&
         request.dst.type != emel::kernel::event::dtype::unknown &&
         request.src0.ne[0] > 0u && request.src1.ne[0] > 0u &&
         request.src1.ne[1] == request.src0.ne[0] &&
         request.dst.ne[0] == request.src1.ne[0] && request.dst.ne[1] == rows &&
         request.src0.ne[2] == 1u && request.src0.ne[3] == 1u &&
         request.src1.ne[2] == 1u && request.src1.ne[3] == 1u &&
         request.dst.ne[2] == 1u && request.dst.ne[3] == 1u &&
         request.op_params_size <= request.op_params.size() &&
         guard_pointer_extent_valid(request.src0.data, request.src0.nb[1],
                                    storage_rows) &&
         guard_pointer_extent_valid(request.src1.data, request.src1.nb[1],
                                    request.src1.ne[1]) &&
         guard_pointer_extent_valid(request.dst.data, request.dst.nb[1], rows);
}

inline bool guard_parallel_storage_ready(const event::execute_parallel &ev,
                                         const action::context &ctx) noexcept {
  return guard_lane_storage_ready(ctx) && guard_parallel_request_valid(ev);
}

struct guard_parallel_request_invalid {
  bool operator()(const event::execute_parallel &ev,
                  const action::context &) const noexcept {
    return !guard_parallel_request_valid(ev);
  }
};

struct guard_parallel_unavailable {
  bool operator()(const event::execute_parallel &,
                  const action::context &ctx) const noexcept {
    return ctx.parallel_matmul_lanes == nullptr ||
           !guard_lane_storage_ready(ctx);
  }
};

template <size_t lane_count>
inline bool guard_lane_count_selected(const event::execute_parallel &ev,
                                      const action::context &ctx,
                                      const uint64_t group_rows) noexcept {
  const uint64_t groups = guard_row_group_count(ev, group_rows);
  if constexpr (lane_count == 8u) {
    return ctx.active_lanes == 8u && groups >= 8u;
  } else if constexpr (lane_count == 4u) {
    return groups >= 4u &&
           (ctx.active_lanes == 4u || (ctx.active_lanes == 8u && groups < 8u));
  } else {
    return groups >= 2u &&
           (ctx.active_lanes == 2u || (ctx.active_lanes >= 4u && groups < 4u));
  }
}

template <uint64_t group_rows>
inline bool
guard_group_rows_selected(const emel::kernel::event::dtype type) noexcept {
  if constexpr (group_rows == emel::kernel::detail::quant::Q4_K_X8_ROWS) {
    return guard_uses_x8_row_groups(type);
  } else if constexpr (group_rows ==
                       emel::kernel::detail::quant::Q8_0_X4_ROWS) {
    return guard_uses_x4_row_groups(type);
  } else {
    return !guard_uses_x8_row_groups(type) && !guard_uses_x4_row_groups(type);
  }
}

template <size_t lane_count, uint64_t group_rows> struct guard_parallel_ready {
  bool operator()(const event::execute_parallel &ev,
                  const action::context &ctx) const noexcept {
    return guard_parallel_storage_ready(ev, ctx) &&
           guard_group_rows_selected<group_rows>(ev.request.src0.type) &&
           guard_lane_count_selected<lane_count>(ev, ctx, group_rows);
  }
};

struct guard_parallel_no_lane_count {
  bool operator()(const event::execute_parallel &ev,
                  const action::context &ctx) const noexcept {
    const uint64_t group_rows =
        guard_uses_x8_row_groups(ev.request.src0.type)
            ? emel::kernel::detail::quant::Q4_K_X8_ROWS
            : (guard_uses_x4_row_groups(ev.request.src0.type)
                   ? emel::kernel::detail::quant::Q8_0_X4_ROWS
                   : 1u);
    const uint64_t groups = guard_row_group_count(ev, group_rows);
    return guard_parallel_storage_ready(ev, ctx) && groups < 2u;
  }
};

struct guard_serial_accepted {
  bool operator()(const event::execute_serial &ev,
                  const action::context &) const noexcept {
    return ev.result.all_lanes_accepted;
  }
};

struct guard_serial_rejected {
  bool operator()(const event::execute_serial &ev,
                  const action::context &) const noexcept {
    return !ev.result.all_lanes_accepted;
  }
};

struct guard_parallel_submission_failed {
  bool operator()(const event::execute_parallel &ev,
                  const action::context &) const noexcept {
    return !ev.result.all_submitted;
  }
};

struct guard_parallel_lane_rejected {
  bool operator()(const event::execute_parallel &ev,
                  const action::context &) const noexcept {
    return ev.result.all_submitted && !ev.result.all_lanes_accepted;
  }
};

struct guard_parallel_all_lanes_accepted {
  bool operator()(const event::execute_parallel &ev,
                  const action::context &) const noexcept {
    return ev.result.all_submitted && ev.result.all_lanes_accepted;
  }
};

template <class event_type> struct guard_has_done_callback {
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_done);
  }
};

template <class event_type> struct guard_no_done_callback {
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback<event_type>{}(ev, ctx);
  }
};

template <class event_type> struct guard_has_error_callback {
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.on_error);
  }
};

template <class event_type> struct guard_no_error_callback {
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback<event_type>{}(ev, ctx);
  }
};

struct guard_serial_has_done_callback
    : guard_has_done_callback<event::execute_serial> {};
struct guard_serial_no_done_callback
    : guard_no_done_callback<event::execute_serial> {};
struct guard_serial_has_error_callback
    : guard_has_error_callback<event::execute_serial> {};
struct guard_serial_no_error_callback
    : guard_no_error_callback<event::execute_serial> {};
struct guard_parallel_has_done_callback
    : guard_has_done_callback<event::execute_parallel> {};
struct guard_parallel_no_done_callback
    : guard_no_done_callback<event::execute_parallel> {};
struct guard_parallel_has_error_callback
    : guard_has_error_callback<event::execute_parallel> {};
struct guard_parallel_no_error_callback
    : guard_no_error_callback<event::execute_parallel> {};

} // namespace emel::kernel::matmul::guard
