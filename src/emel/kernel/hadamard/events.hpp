#pragma once

#include <cstdint>
#include <span>

namespace emel::kernel::hadamard::event {

struct dispatch_result {
  bool accepted = false;
};

// HadamardMLP residual for one row: with n = hada_n (power of two >= d_model),
// z = pad(input, n); z = fwht(d1 * z); z = fwht(silu(d2 * z));
// output[i] = skip[i] + (d3 * z)[i] for i < d_model. The FWHT is the
// normalized (orthonormal) transform, matching `_walsh_matrix(n)/sqrt(n)`
// composed via `(d1*z) @ H`. Diagonals are fp16 tensor payloads decoded to
// f32 on the fly. All spans borrow caller-owned memory for the complete
// dispatch. Read-only spans may overlap one another. Workspace and output must
// be mutually disjoint and each must be disjoint from every read-only range.
// A rejected event performs no workspace or output writes. Dispatch is
// single-owner and externally serialized. `workspace` holds the padded lane
// (>= n floats).
struct mlp_row_request {
  std::span<const float> input; // d_model
  std::span<const float> skip;  // d_model
  std::span<const uint8_t> d1;  // n fp16 values
  std::span<const uint8_t> d2;  // n fp16 values
  std::span<const uint8_t> d3;  // n fp16 values
  uint32_t d_model = 0u;
  uint32_t hada_n = 0u;
  std::span<float> workspace; // >= hada_n
  std::span<float> output;    // d_model
};

struct execute_mlp_row {
  const mlp_row_request &request;
  dispatch_result &result;
};

// Exact AVX2+FMA+F16C specialization for d_model == hada_n == 512. The event
// is explicit so platform/geometry routing happens before the data-plane
// action; it otherwise inherits mlp_row_request's span and alias contract.
struct execute_mlp_row_avx2 {
  const mlp_row_request &request;
  dispatch_result &result;
};

} // namespace emel::kernel::hadamard::event
