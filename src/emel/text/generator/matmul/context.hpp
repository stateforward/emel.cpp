#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

#include "emel/kernel/sm.hpp"
#include "emel/text/generator/matmul/detail.hpp"

namespace emel::text::generator::matmul::action {

using lane_pool_ref = emel::text::generator::matmul::lane_pool_ref;

struct lane_dispatch {
  emel::kernel::sm * kernel = nullptr;
  const emel::kernel::event::op_mul_mat * request = nullptr;
  bool accepted = false;
};

inline void run_lane_dispatch(void * ctx) noexcept {
  auto & dispatch = *static_cast<lane_dispatch *>(ctx);
  dispatch.accepted =
      dispatch.kernel != nullptr && dispatch.request != nullptr &&
      dispatch.kernel->process_event(*dispatch.request);
}

struct context {
  lane_pool_ref parallel_matmul_lanes = {};
  emel::kernel::kernel_kind kernel_kind = emel::kernel::kernel_kind::x86_64;
  size_t active_lanes = 1u;
  size_t lane_capacity = 0u;
  emel::kernel::sm kernel = {};
  std::unique_ptr<emel::kernel::sm[]> lane_kernels = {};
  std::unique_ptr<emel::text::generator::matmul::detail::matmul_row_slice[]>
      row_slices = {};
  std::unique_ptr<emel::kernel::event::op_mul_mat[]> lane_events = {};
  std::unique_ptr<lane_dispatch[]> lane_dispatches = {};
};

inline bool reserve_lane_storage(context & ctx,
                                 const size_t lane_capacity) noexcept {
  if (lane_capacity == 0u) {
    ctx.lane_capacity = 0u;
    ctx.lane_kernels.reset();
    ctx.row_slices.reset();
    ctx.lane_events.reset();
    ctx.lane_dispatches.reset();
    return true;
  }

  auto lane_kernels = std::unique_ptr<emel::kernel::sm[]>(
      new (std::nothrow) emel::kernel::sm[lane_capacity]);
  auto row_slices = std::unique_ptr<
      emel::text::generator::matmul::detail::matmul_row_slice[]>(
      new (std::nothrow)
          emel::text::generator::matmul::detail::matmul_row_slice[lane_capacity]);
  auto lane_events = std::unique_ptr<emel::kernel::event::op_mul_mat[]>(
      new (std::nothrow) emel::kernel::event::op_mul_mat[lane_capacity]);
  auto lane_dispatches = std::unique_ptr<lane_dispatch[]>(
      new (std::nothrow) lane_dispatch[lane_capacity]);
  if (lane_kernels == nullptr || row_slices == nullptr ||
      lane_events == nullptr || lane_dispatches == nullptr) {
    return false;
  }

  ctx.lane_capacity = lane_capacity;
  ctx.lane_kernels = std::move(lane_kernels);
  ctx.row_slices = std::move(row_slices);
  ctx.lane_events = std::move(lane_events);
  ctx.lane_dispatches = std::move(lane_dispatches);
  return true;
}

inline bool lane_storage_ready(const context & ctx) noexcept {
  return ctx.lane_capacity == ctx.parallel_matmul_lanes.lane_capacity &&
         ctx.lane_capacity > 0u && ctx.lane_kernels != nullptr &&
         ctx.row_slices != nullptr && ctx.lane_events != nullptr &&
         ctx.lane_dispatches != nullptr && ctx.active_lanes > 1u &&
         ctx.active_lanes <= ctx.lane_capacity;
}

}  // namespace emel::text::generator::matmul::action
