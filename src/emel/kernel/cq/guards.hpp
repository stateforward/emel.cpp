#pragma once

#include "emel/kernel/cq/actions.hpp"

namespace emel::kernel::cq::guard {

inline bool supported(const event::gemv_request &request) noexcept {
  return detail::valid_view(request.weights, request.codebook,
                            request.activation, request.output) &&
         request.workspace.size() >=
             ((request.weights.shape[1] + request.weights.group - 1u) /
              request.weights.group) * request.weights.group;
}

inline bool avx2_supported(const event::gemv_request &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  return request.weights.bits != detail::k_ternary_record_bits &&
         __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
         supported(request);
#else
  return request.weights.bits != detail::k_ternary_record_bits &&
         supported(request);
#endif
#else
  (void)request;
  return false;
#endif
}

struct guard_execute_scalar {
  bool operator()(const event::execute_scalar &ev, const action::context &) const noexcept {
    return supported(ev.request);
  }
};

struct guard_execute_avx2 {
  bool operator()(const event::execute_avx2 &ev, const action::context &) const noexcept {
    return avx2_supported(ev.request);
  }
};

} // namespace emel::kernel::cq::guard
