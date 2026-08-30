#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "emel/cact/loader/events.hpp"

namespace emel::kernel::cq::event {

struct dispatch_result {
  bool accepted = false;
};

// Graph/model-owned exact CQ4 lookup representation. `values` borrows the
// source codebook while `byte_planes` owns the bitwise-identical float bytes,
// duplicated across both 128-bit shuffle lanes.
struct alignas(32) prepared_codebook_q4 {
  std::span<const float> values = {};
  std::array<std::array<uint8_t, 32u>, 4u> byte_planes = {};
};

struct prepare_codebook_q4_request {
  std::span<const float> codebook;
  prepared_codebook_q4 &prepared;
};

struct prepare_codebook_q4 {
  const prepare_codebook_q4_request &request;
  dispatch_result &result;
};

// JAX-compatible signed A8 fake quantization over one full activation vector.
// `quantized` keeps the exact integer operand and `integer_values` keeps the
// same integers widened to f32 for the linear FWHT/CQ projection. The caller
// applies `scale` once to each projection output.
struct quantize_a8_request {
  std::span<const float> input;
  std::span<int8_t> quantized;
  std::span<float> integer_values;
  float &scale;
};

struct quantize_a8 {
  const quantize_a8_request &request;
  dispatch_result &result;
};
struct fwht_request {
  std::span<float> values;
};

struct execute_fwht_avx2 {
  const fwht_request &request;
  dispatch_result &result;
};


struct gemv_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<const float> codebook;
  std::span<const float> activation;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};
// Construction/init-owned CQ4 representation. Indices remain exact codebook
// selectors and norms are the exact fp16 payload decoded once to f32. The
// spans borrow caller-owned storage that must outlive every prepared dispatch.
struct prepared_q4_view {
  const uint8_t *source = nullptr;
  uint32_t out = 0u;
  uint32_t in = 0u;
  uint32_t group = 0u;
  uint32_t in_pad = 0u;
  std::span<const uint8_t> indices = {};
  // 32-row output blocks, input-major within each block. Tail rows remain in
  // row-major `indices`; the blocked layout exists only for hot full GEMV.
  std::span<const uint8_t> indices_by_input32 = {};
  std::span<const float> norms = {};
  // Complete 32-row output blocks, group-major within each block. Values are
  // copied from the exact decoded row-major norms during preparation; tail
  // rows continue to use `norms`.
  std::span<const float> norms_by_group32 = {};
};
struct prepare_q4_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<uint8_t> indices;
  std::span<uint8_t> indices_by_input32;
  std::span<float> norms;
  std::span<float> norms_by_group32;
  prepared_q4_view &prepared;
};

struct prepare_q4 {
  const prepare_q4_request &request;
  dispatch_result &result;
};

struct prepared_gemv_request {
  const prepared_q4_view &weights;
  const prepared_codebook_q4 &codebook;
  std::span<const float> activation;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

struct prepared_gemv_target {
  const prepared_q4_view *weights = nullptr;
  std::span<float> output = {};
};

// Four projections sharing one activation transform. The fixed arity keeps
// dispatch allocation-free and matches the graph's q/k/v/gate hot path.
struct prepared_gemv_batch4_request {
  std::array<prepared_gemv_target, 4u> targets = {};
  const prepared_codebook_q4 &codebook;
  std::span<const float> activation;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

struct execute_prepared_avx2_batch4_q4 {
  const prepared_gemv_batch4_request &request;
  dispatch_result &result;
};

struct execute_prepared_avx2_q4 {
  const prepared_gemv_request &request;
  dispatch_result &result;
};

template <uint32_t Bits> struct execute_scalar {
  const gemv_request &request;
  dispatch_result &result;
};
template <uint32_t Bits> struct execute_avx2 {
  const gemv_request &request;
  dispatch_result &result;
};

// Row-range GEMV over a packed CQ view: fills output[0..row_count) from packed
// rows [row_begin, row_begin + row_count). Weights stay packed; the fp16 norm
// table is addressed against the full view row count (shape[0]).
struct gemv_rows_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<const float> codebook;
  std::span<const float> activation;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

template <uint32_t Bits> struct execute_scalar_rows {
  const gemv_rows_request &request;
  dispatch_result &result;
};

struct prepared_gemv_rows_request {
  const prepared_q4_view &weights;
  const prepared_codebook_q4 &codebook;
  std::span<const float> activation;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

struct execute_prepared_avx2_rows_q4 {
  const prepared_gemv_rows_request &request;
  dispatch_result &result;
};

// Dequantizes packed CQ rows [row_begin, row_begin + row_count) to f32 exactly
// like the exporter's `_cq_unpack` (codebook value scaled by the group norm,
// then the normalized Walsh-Hadamard rotation), truncated to shape[1] columns
// and scaled by `scale`. Intended for per-row gathers (embedding rows, engram
// table rows); never a whole-tensor dequant fallback.
struct dequant_rows_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<const float> codebook;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  float scale = 1.0f;
  std::span<float> output;
};

template <uint32_t Bits> struct execute_scalar_dequant {
  const dequant_rows_request &request;
  dispatch_result &result;
};

struct prepared_dequant_rows_request {
  const prepared_q4_view &weights;
  const prepared_codebook_q4 &codebook;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  float scale = 1.0f;
  std::span<float> output;
};

struct execute_prepared_dequant_q4 {
  const prepared_dequant_rows_request &request;
  dispatch_result &result;
};

using execute_scalar_q2 = execute_scalar<2u>;
using execute_scalar_q3 = execute_scalar<3u>;
using execute_scalar_q4 = execute_scalar<4u>;
using execute_scalar_ternary = execute_scalar<5u>;
using execute_avx2_q2 = execute_avx2<2u>;
using execute_avx2_q3 = execute_avx2<3u>;
using execute_avx2_q4 = execute_avx2<4u>;
using execute_scalar_rows_q2 = execute_scalar_rows<2u>;
using execute_scalar_rows_q3 = execute_scalar_rows<3u>;
using execute_scalar_rows_q4 = execute_scalar_rows<4u>;
using execute_scalar_dequant_q2 = execute_scalar_dequant<2u>;
using execute_scalar_dequant_q3 = execute_scalar_dequant<3u>;
using execute_scalar_dequant_q4 = execute_scalar_dequant<4u>;

struct capture_diagnostics {
  uint64_t &scalar_calls;
  uint64_t &avx2_calls;
};

struct capture_prepared_diagnostics {
  uint64_t &prepare_calls;
  uint64_t &prepared_calls;
};

struct capture_a8_diagnostics {
  uint64_t &quantize_calls;
};

using timestamp_now_fn = uint64_t (*)() noexcept;

struct configure_timing {
  bool enabled = false;
  timestamp_now_fn now = nullptr;
};

struct timing_breakdown {
  uint64_t quantize_nanoseconds = 0u;
  uint64_t fwht_nanoseconds = 0u;
  uint64_t dot_full_nanoseconds = 0u;
  uint64_t dot_batch_nanoseconds = 0u;
  uint64_t dot_rows_nanoseconds = 0u;
  uint64_t dequant_nanoseconds = 0u;
};

struct capture_timing {
  timing_breakdown &breakdown;
};
} // namespace emel::kernel::cq::event
