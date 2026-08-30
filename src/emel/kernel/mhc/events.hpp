#pragma once

#include <cstdint>
#include <span>

namespace emel::kernel::mhc::event {

struct dispatch_result {
  bool accepted = false;
};

inline constexpr uint32_t k_max_lanes = 8u;
inline constexpr uint32_t k_sinkhorn_iterations = 20u;

// mHC pre-mix for one token, matching the reference `_forward_cached`:
// hpre[l] = sigmoid(a * phi_dots[l] + b[l] + pre_off[l]) with
// pre_off[l] = 8 * (l == lane) - 4; u[c] = sum_l hpre[l] * lanes[l * dim + c].
// `a` is the layer's fp16 a_pre scalar and `b` the layer's fp16 b_pre row,
// both raw tensor payload bytes.
struct pre_mix_request {
  std::span<const float> lanes;    // lane_count * dim
  std::span<const float> phi_dots; // lane_count (nx @ phi_pre rows)
  std::span<const uint8_t> a;      // 1 fp16 value
  std::span<const uint8_t> b;      // lane_count fp16 values
  uint32_t lane_index = 0u;
  uint32_t lane_count = 0u;
  uint32_t dim = 0u;
  std::span<float> output; // dim
};

// mHC post-mix for one token, matching the reference:
// y = block_out - u (the mHC residual delta);
// hpost[l] = 2 * sigmoid(a_post * post_dots[l] + b_post[l] + post_off[l])
//   with post_off[l] = -4 * (1 - (l == lane));
// hres = sinkhorn(a_res * res_dots + b_res, 20 iterations, log domain);
// new_lanes[i] = sum_j hres[i, j] * lanes[j] + hpost[i] * y.
// a_post/a_res are the layer fp16 scalars; b_post the fp16 row and b_res the
// fp16 (lane_count, lane_count) block, all raw tensor payload bytes.
struct post_mix_request {
  std::span<const float> lanes;     // lane_count * dim
  std::span<const float> block_out; // dim (block output)
  std::span<const float> u;         // dim (pre-mix aggregate fed to the block)
  std::span<const float> post_dots; // lane_count (nx @ phi_post rows)
  std::span<const float> res_dots;  // lane_count^2 (nx @ phi_res rows)
  std::span<const uint8_t> a_post;  // 1 fp16 value
  std::span<const uint8_t> b_post;  // lane_count fp16 values
  std::span<const uint8_t> a_res;   // 1 fp16 value
  std::span<const uint8_t> b_res;   // lane_count^2 fp16 values
  uint32_t lane_index = 0u;
  uint32_t lane_count = 0u;
  uint32_t dim = 0u;
  std::span<float> output; // lane_count * dim
};

// Mean over lanes: output[c] = mean_l lanes[l * dim + c].
struct mean_lanes_request {
  std::span<const float> lanes; // lane_count * dim
  uint32_t lane_count = 0u;
  uint32_t dim = 0u;
  std::span<float> output; // dim
};

struct execute_pre_mix {
  const pre_mix_request &request;
  dispatch_result &result;
};

struct execute_post_mix {
  const post_mix_request &request;
  dispatch_result &result;
};

struct execute_mean_lanes {
  const mean_lanes_request &request;
  dispatch_result &result;
};

} // namespace emel::kernel::mhc::event
