#pragma once

#include "emel/kernel/zcrms/actions.hpp"

namespace emel::kernel::zcrms::guard {

struct guard_execute_norm_rows {
  bool operator()(const event::execute_norm_rows &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t total = static_cast<uint64_t>(request.rows) * request.dim;
    return request.rows > 0u && request.dim > 0u &&
           request.input.size() >= total && request.output.size() >= total &&
           request.scale.size() >= request.dim;
  }
};

struct guard_execute_unit_rows {
  bool operator()(const event::execute_unit_rows &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t total = static_cast<uint64_t>(request.rows) * request.dim;
    return request.rows > 0u && request.dim > 0u &&
           request.input.size() >= total && request.output.size() >= total;
  }
};

} // namespace emel::kernel::zcrms::guard
