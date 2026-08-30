#pragma once

#include "emel/kernel/engram/actions.hpp"

namespace emel::kernel::engram::guard {

struct guard_execute_hash_rows {
  bool operator()(const event::execute_hash_rows &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t tables =
        static_cast<uint64_t>(request.num_orders) * request.heads;
    const uint64_t total = static_cast<uint64_t>(request.positions) * tables;
    return request.positions > 0u && request.num_orders > 0u &&
           request.num_orders <= event::k_max_orders && request.heads > 0u &&
           request.slots > 0u && request.orders.size() >= request.num_orders &&
           request.tokens.size() >= request.positions &&
           request.valid.size() >= request.positions &&
           request.indices.size() >= total && request.ngram_ok.size() >= total;
  }
};

struct guard_execute_conv_taps {
  bool operator()(const event::execute_conv_taps &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t rows =
        static_cast<uint64_t>(request.conv_taps) * request.dim;
    return request.dim > 0u && request.conv_taps > 0u &&
           request.value_rows.size() >= rows &&
           request.tap_valid.size() >= request.conv_taps &&
           request.taps.size() >= rows * 2u &&
           request.output.size() >= request.dim;
  }
};

struct guard_execute_alpha_gate {
  bool operator()(const event::execute_alpha_gate &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    return request.dim > 0u && request.u.size() >= request.dim &&
           request.key.size() >= request.dim &&
           request.value.size() >= request.dim &&
           request.output.size() >= request.dim;
  }
};

} // namespace emel::kernel::engram::guard
