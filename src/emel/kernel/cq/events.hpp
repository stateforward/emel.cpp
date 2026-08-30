#pragma once

#include <cstdint>
#include <span>

#include "emel/cact/loader/events.hpp"

namespace emel::kernel::cq::event {

struct dispatch_result { bool accepted = false; };

struct gemv_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<const float> codebook;
  std::span<const float> activation;
  std::span<float> output;
  std::span<float> workspace;
};

struct execute_scalar { const gemv_request &request; dispatch_result &result; };
struct execute_avx2 { const gemv_request &request; dispatch_result &result; };
struct capture_diagnostics { uint64_t &scalar_calls; uint64_t &avx2_calls; };

} // namespace emel::kernel::cq::event
