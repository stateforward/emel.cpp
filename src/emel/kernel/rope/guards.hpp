#pragma once

#include <cmath>

#include "emel/kernel/rope/actions.hpp"

namespace emel::kernel::rope::guard {

struct guard_execute_precompute {
  bool operator()(const event::execute_precompute &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t table =
        static_cast<uint64_t>(request.positions) * (request.head_dim / 2u);
    return std::isfinite(request.theta) && request.theta > 0.0f &&
           request.head_dim >= 2u && (request.head_dim % 2u) == 0u &&
           request.positions > 0u && request.cos_out.size() >= table &&
           request.sin_out.size() >= table;
  }
};

struct guard_execute_apply_rows {
  bool operator()(const event::execute_apply_rows &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint32_t half = request.head_dim / 2u;
    const uint64_t table_end =
        (static_cast<uint64_t>(request.position) + 1u) * half;
    return request.head_dim >= 2u && (request.head_dim % 2u) == 0u &&
           request.head_count > 0u && request.cos_table.size() >= table_end &&
           request.sin_table.size() >= table_end &&
           request.rows.size() >=
               static_cast<uint64_t>(request.head_count) * request.head_dim;
  }
};

} // namespace emel::kernel::rope::guard
