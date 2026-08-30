#pragma once

#include "emel/kernel/hadamard/actions.hpp"

namespace emel::kernel::hadamard::guard {

inline bool power_of_two(const uint32_t n) noexcept {
  return n != 0u && (n & (n - 1u)) == 0u;
}

struct guard_execute_mlp_row {
  bool operator()(const event::execute_mlp_row &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t d_bytes = static_cast<uint64_t>(request.hada_n) * 2u;
    return request.d_model > 0u && power_of_two(request.hada_n) &&
           request.hada_n >= request.d_model &&
           request.input.size() >= request.d_model &&
           request.skip.size() >= request.d_model &&
           request.output.size() >= request.d_model &&
           request.workspace.size() >= request.hada_n &&
           request.d1.size() >= d_bytes && request.d2.size() >= d_bytes &&
           request.d3.size() >= d_bytes;
  }
};

} // namespace emel::kernel::hadamard::guard
