#pragma once

#include <cstdint>
#include <span>

namespace emel::kernel::engram::event {

struct dispatch_result {
  bool accepted = false;
};

inline constexpr uint32_t k_max_orders = 4u;

// FNV-mix hash rows for a token window, matching the reference
// `engram_indices`: for table t = oi * heads + h,
//   seed = (0x9E3779B9 * (t + 1)) mod 2^32
//   acc  = fold over j in [0, order): acc = (acc ^ token[p - j]) * 0x01000193
//   acc ^= acc >> 15; index = acc % slots
// Tokens before the window start read as 0 (the reference's zero padding).
// `ngram_ok[p * tables + t]` is 1.0 when the n-gram's oldest token position
// (p - order + 1) is inside the window and valid, else 0.0.
struct hash_rows_request {
  std::span<const int32_t> tokens; // window positions
  std::span<const uint8_t> valid;  // 1 = valid position
  uint32_t positions = 0u;
  std::span<const uint32_t> orders; // num_orders entries
  uint32_t num_orders = 0u;
  uint32_t heads = 0u; // tables per order
  uint32_t slots = 0u;
  std::span<uint32_t> indices; // positions * (num_orders * heads)
  std::span<float> ngram_ok;   // positions * (num_orders * heads)
};

// Causal convolution taps over pre-gathered value rows, matching the
// reference tap sum `sum_j taps[j] * v[p - j*dilation] * tap_ok[j]` for the
// current output position: the caller gathers `value_rows[j]` = the value
// projection at tap j's source position and `tap_valid[j]` = that source's
// window validity. `taps` is the fp16 (conv_taps, dim) tensor payload.
struct conv_taps_request {
  std::span<const float> value_rows;  // conv_taps * dim rows
  std::span<const uint8_t> tap_valid; // conv_taps entries
  std::span<const uint8_t> taps;      // conv_taps * dim fp16 values
  uint32_t conv_taps = 0u;
  uint32_t dim = 0u;
  std::span<float> output; // dim
};

// Engram alpha gate, matching the reference site injection:
// alpha = sigmoid(dot(rms_unit(u), rms_unit(key)) / sqrt(dim));
// output = u + alpha * value.
struct alpha_gate_request {
  std::span<const float> u;     // dim
  std::span<const float> key;   // dim
  std::span<const float> value; // dim
  uint32_t dim = 0u;
  std::span<float> output; // dim
};

struct execute_hash_rows {
  const hash_rows_request &request;
  dispatch_result &result;
};

struct execute_conv_taps {
  const conv_taps_request &request;
  dispatch_result &result;
};

struct execute_alpha_gate {
  const alpha_gate_request &request;
  dispatch_result &result;
};

} // namespace emel::kernel::engram::event
