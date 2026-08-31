#pragma once

#include <cstdint>
#include <span>

namespace emel::kernel::swa::event {

struct dispatch_result {
  bool accepted = false;
};

// Sliding-window softmax attention for one query position against an f32 KV
// ring cache slice (one layer): logical key positions [window_begin, position]
// map to physical ring slots (logical % capacity). Grouped-query mapping:
// query head h reads kv head h / (heads / kv_heads). Scores are scaled by
// 1/sqrt(head_dim) and softmaxed with max-shift, matching the reference
// `_attn_cached` non-flash route. Geometry and inclusive span must be nonzero
// and representable in uint32_t, size_t, and byte-address arithmetic. Query,
// key, and value are borrowed read-only and may overlap each other. Workspace
// and output are borrowed writable ranges: they must be mutually disjoint and
// each must be disjoint from every read-only range. A rejected event performs
// no workspace or output writes.
struct attend_request {
  std::span<const float> query;       // heads * head_dim
  std::span<const float> key_cache;   // kv_heads * capacity * head_dim
  std::span<const float> value_cache; // kv_heads * capacity * head_dim
  uint32_t position = 0u;
  uint32_t window_begin = 0u;
  uint32_t capacity = 0u;
  uint32_t heads = 0u;
  uint32_t kv_heads = 0u;
  uint32_t head_dim = 0u;
  std::span<float> workspace; // generic: >= span; GQA2: >= 2 * span
  std::span<float> output;    // heads * head_dim
};

// Writes one position's K/V head rows into the ring cache slice at physical
// slot position % capacity. All spans borrow caller-owned memory for the full
// dispatch. The complete declared key/value cache ranges must be representable
// and mutually disjoint, and each cache must be disjoint from both source row
// ranges; the read-only source ranges may overlap each other. A rejected event
// performs no cache writes.
struct cache_write_request {
  std::span<const float> key_rows;   // kv_heads * head_dim
  std::span<const float> value_rows; // kv_heads * head_dim
  uint32_t position = 0u;
  uint32_t capacity = 0u;
  uint32_t kv_heads = 0u;
  uint32_t head_dim = 0u;
  std::span<float> key_cache;
  std::span<float> value_cache;
};

// In-place sigmoid gating: values[i] *= sigmoid(gate_logits[i]). The gated
// attention output `out * sigmoid(x @ gate_proj)`.
struct gate_mul_request {
  std::span<float> values;
  std::span<const float> gate_logits;
  uint32_t dim = 0u;
};

// Scalar-gated residual: output[i] = skip[i] + sigmoid(gate) * values[i].
// The block residual `skip + sigmoid(attn_gate) * attn`.
struct residual_gate_request {
  std::span<const float> skip;
  float gate = 0.0f;
  std::span<const float> values;
  uint32_t dim = 0u;
  std::span<float> output;
};

struct execute_attend {
  const attend_request &request;
  dispatch_result &result;
};

// Exact GQA reps=2 AVX2 route. The event is explicit so callers select the
// geometry/platform specialization before entering the data-plane action. It
// inherits attend_request's alias, representability, and no-write-on-rejection
// contract and requires workspace capacity for two complete score rows.
struct execute_attend_gqa2_avx2 {
  const attend_request &request;
  dispatch_result &result;
};

struct execute_cache_write {
  const cache_write_request &request;
  dispatch_result &result;
};

struct execute_gate_mul {
  const gate_mul_request &request;
  dispatch_result &result;
};

struct execute_residual_gate {
  const residual_gate_request &request;
  dispatch_result &result;
};

} // namespace emel::kernel::swa::event
