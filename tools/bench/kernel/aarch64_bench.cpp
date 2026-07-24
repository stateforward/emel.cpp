#include "bench_cases.hpp"

#include "emel/kernel/aarch64/sm.hpp"

#include "kernel/bench_common.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace {

constexpr uint64_t Q4_MULTIROW_COLS = 4096u;
constexpr uint64_t Q4_MULTIROW_ROWS = 256u;
constexpr uint64_t Q4_MULTIROW_BLOCK_COUNT =
    Q4_MULTIROW_COLS / emel::kernel::detail::quant::QK_K;
constexpr const char *Q4_MULTIROW_CASE_NAME =
    "kernel/aarch64/op_mul_mat_q4_k_q8_k_multirow";
constexpr const char *Q4_PACKED_NATIVE_CASE_NAME =
    "kernel/aarch64/op_mul_mat_q4_k_q8_k_native_moshi";
constexpr const char *Q4_PACKED_CASE_NAME =
    "kernel/aarch64/op_mul_mat_q4_k_x8_bl8_q8_k_packed_moshi";

void initialize_q4_benchmark_inputs(
    std::vector<emel::kernel::detail::quant::block_q4_k> &weights,
    std::vector<float> &rhs_dense,
    std::vector<emel::kernel::detail::quant::block_q8_k> &rhs_q8) {
  for (uint64_t row = 0; row < Q4_MULTIROW_ROWS; ++row) {
    for (uint64_t block = 0; block < Q4_MULTIROW_BLOCK_COUNT; ++block) {
      auto &q4 = weights[row * Q4_MULTIROW_BLOCK_COUNT + block];
      q4.d = emel::kernel::detail::quant::fp32_to_fp16(
          0.0009765625f *
          static_cast<float>(((row + 3u) * (block + 5u)) % 31u + 1u));
      q4.dmin = emel::kernel::detail::quant::fp32_to_fp16(
          0.00048828125f *
          static_cast<float>(((row + 7u) * (block + 11u)) % 29u + 1u));
      for (size_t index = 0; index < q4.scales.size(); ++index) {
        q4.scales[index] = static_cast<uint8_t>(
            (row * 41u + block * 37u + index * 19u + 0x5au) & 0xffu);
      }
      for (size_t index = 0; index < q4.qs.size(); ++index) {
        q4.qs[index] = static_cast<uint8_t>(
            (row * 23u + block * 17u + index * 29u + 0x96u) & 0xffu);
      }
    }
  }
  for (uint64_t index = 0; index < Q4_MULTIROW_COLS; ++index) {
    rhs_dense[index] =
        static_cast<float>(static_cast<int32_t>((index * 17u) % 127u) - 63) *
        0.0078125f;
  }
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      rhs_dense.data(), 1u, rhs_q8.data(),
      static_cast<int64_t>(Q4_MULTIROW_COLS));
}

struct q4_multirow_case {
  std::vector<emel::kernel::detail::quant::block_q4_k> weights{
      Q4_MULTIROW_ROWS * Q4_MULTIROW_BLOCK_COUNT};
  std::vector<float> rhs_dense = std::vector<float>(Q4_MULTIROW_COLS);
  std::vector<emel::kernel::detail::quant::block_q8_k> rhs_q8{
      Q4_MULTIROW_BLOCK_COUNT};
  std::vector<float> output = std::vector<float>(Q4_MULTIROW_ROWS);
  emel::kernel::event::op_mul_mat request = {};

  q4_multirow_case() {
    initialize_q4_benchmark_inputs(weights, rhs_dense, rhs_q8);

    const uint64_t q4_row_bytes =
        emel::kernel::detail::quantized_row_storage_bytes(
            emel::kernel::detail::dtype_q4_k, Q4_MULTIROW_COLS);
    request.src0.data = weights.data();
    request.src0.type = emel::kernel::event::dtype::q4_k;
    request.src0.ne = {Q4_MULTIROW_COLS, Q4_MULTIROW_ROWS, 1u, 1u};
    request.src0.nb = {1u, q4_row_bytes, q4_row_bytes * Q4_MULTIROW_ROWS,
                       q4_row_bytes * Q4_MULTIROW_ROWS};

    const uint64_t q8_row_bytes =
        emel::kernel::detail::quantized_row_storage_bytes(
            emel::kernel::detail::dtype_q8_k, Q4_MULTIROW_COLS);
    request.src1.data = rhs_q8.data();
    request.src1.type = emel::kernel::event::dtype::q8_k;
    request.src1.ne = {1u, Q4_MULTIROW_COLS, 1u, 1u};
    request.src1.nb = {1u, q8_row_bytes, q8_row_bytes, q8_row_bytes};

    request.dst.data = output.data();
    request.dst.type = emel::kernel::event::dtype::f32;
    request.dst.ne = {1u, Q4_MULTIROW_ROWS, 1u, 1u};
    request.dst.nb = {sizeof(float), sizeof(float),
                      sizeof(float) * Q4_MULTIROW_ROWS,
                      sizeof(float) * Q4_MULTIROW_ROWS};
  }
};

struct q4_packed_case {
  const uint64_t packed_group_count =
      emel::kernel::detail::quant::packed_q4_k_x8_group_count(Q4_MULTIROW_ROWS);
  std::vector<emel::kernel::detail::quant::block_q4_k> native_weights{
      Q4_MULTIROW_ROWS * Q4_MULTIROW_BLOCK_COUNT};
  std::vector<uint8_t> packed_weights = std::vector<uint8_t>(
      packed_group_count *
      emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(
          Q4_MULTIROW_COLS));
  std::vector<float> rhs_dense = std::vector<float>(Q4_MULTIROW_COLS);
  std::vector<emel::kernel::detail::quant::block_q8_k> rhs_q8{
      Q4_MULTIROW_BLOCK_COUNT};
  std::vector<float> native_output = std::vector<float>(Q4_MULTIROW_ROWS);
  std::vector<float> packed_output = std::vector<float>(Q4_MULTIROW_ROWS);
  emel::kernel::event::op_mul_mat native_request = {};
  emel::kernel::event::op_mul_mat packed_request = {};

  q4_packed_case() {
    initialize_q4_benchmark_inputs(native_weights, rhs_dense, rhs_q8);
    if (!emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
            native_weights.data(), Q4_MULTIROW_ROWS, Q4_MULTIROW_COLS,
            packed_weights.data())) {
      std::abort();
    }

    const uint64_t native_q4_row_bytes =
        emel::kernel::detail::quantized_row_storage_bytes(
            emel::kernel::detail::dtype_q4_k, Q4_MULTIROW_COLS);
    const uint64_t packed_q4_group_bytes =
        emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(
            Q4_MULTIROW_COLS);
    const uint64_t q8_row_bytes =
        emel::kernel::detail::quantized_row_storage_bytes(
            emel::kernel::detail::dtype_q8_k, Q4_MULTIROW_COLS);

    native_request.src0.data = native_weights.data();
    native_request.src0.type = emel::kernel::event::dtype::q4_k;
    native_request.src0.ne = {Q4_MULTIROW_COLS, Q4_MULTIROW_ROWS, 1u, 1u};
    native_request.src0.nb = {1u, native_q4_row_bytes,
                              native_q4_row_bytes * Q4_MULTIROW_ROWS,
                              native_q4_row_bytes * Q4_MULTIROW_ROWS};
    packed_request.src0.data = packed_weights.data();
    packed_request.src0.type = emel::kernel::event::dtype::q4_k_x8_bl8;
    packed_request.src0.ne = {Q4_MULTIROW_COLS, Q4_MULTIROW_ROWS, 1u, 1u};
    packed_request.src0.nb = {1u, packed_q4_group_bytes,
                              packed_q4_group_bytes * packed_group_count,
                              packed_q4_group_bytes * packed_group_count};
    native_request.src1.data = rhs_q8.data();
    native_request.src1.type = emel::kernel::event::dtype::q8_k;
    native_request.src1.ne = {1u, Q4_MULTIROW_COLS, 1u, 1u};
    native_request.src1.nb = {1u, q8_row_bytes, q8_row_bytes, q8_row_bytes};
    packed_request.src1 = native_request.src1;
    native_request.dst.data = native_output.data();
    native_request.dst.type = emel::kernel::event::dtype::f32;
    native_request.dst.ne = {1u, Q4_MULTIROW_ROWS, 1u, 1u};
    native_request.dst.nb = {sizeof(float), sizeof(float),
                             sizeof(float) * Q4_MULTIROW_ROWS,
                             sizeof(float) * Q4_MULTIROW_ROWS};
    packed_request.dst = native_request.dst;
    packed_request.dst.data = packed_output.data();
  }
};

} // namespace

namespace emel::bench {

void append_emel_kernel_aarch64_cases(std::vector<result> &results,
                                      const config &cfg) {
  emel::kernel::aarch64::sm aarch_machine{};
  auto exec = [&](const auto &ev) { return aarch_machine.process_event(ev); };
  append_emel_backend_cases(results, cfg, "aarch64", exec);

  const char *q4_multirow_enabled = std::getenv("EMEL_BENCH_Q4_MULTIROW");
  const char *q4_packed_enabled = std::getenv("EMEL_BENCH_Q4_PACKED");
  const bool run_q4_multirow = q4_multirow_enabled != nullptr &&
                               std::string_view{q4_multirow_enabled} == "1";
  const bool run_q4_packed = q4_packed_enabled != nullptr &&
                             std::string_view{q4_packed_enabled} == "1";
  if (!run_q4_multirow && !run_q4_packed) {
    return;
  }

  if (run_q4_multirow) {
    q4_multirow_case q4_case{};
    const bool preflight_ok = aarch_machine.process_event(q4_case.request);
    if (!preflight_ok || aarch_machine.optimized_q4_dispatch_count() != 1u ||
        aarch_machine.optimized_q4_vector_dispatch_count() != 1u ||
        aarch_machine.shared_q4_dispatch_count() != 0u ||
        !std::isfinite(q4_case.output[0])) {
      std::fprintf(stderr,
                   "error: q4 multi-row benchmark preflight failed accepted=%d "
                   "optimized=%llu vector=%llu shared=%llu output=%g\n",
                   preflight_ok ? 1 : 0,
                   static_cast<unsigned long long>(
                       aarch_machine.optimized_q4_dispatch_count()),
                   static_cast<unsigned long long>(
                       aarch_machine.optimized_q4_vector_dispatch_count()),
                   static_cast<unsigned long long>(
                       aarch_machine.shared_q4_dispatch_count()),
                   static_cast<double>(q4_case.output[0]));
      std::abort();
    }
    volatile float sink = 0.0f;
    auto q4_fn = [&]() {
      if (!aarch_machine.process_event(q4_case.request)) {
        std::abort();
      }
      sink += q4_case.output[0];
    };
    results.push_back(measure_case(Q4_MULTIROW_CASE_NAME, cfg, q4_fn));
  }

  if (run_q4_packed) {
    q4_packed_case packed_case{};
    emel::kernel::aarch64::sm native_machine{};
    emel::kernel::aarch64::sm packed_machine{};
    const bool native_preflight_ok =
        native_machine.process_event(packed_case.native_request);
    const bool packed_preflight_ok =
        packed_machine.process_event(packed_case.packed_request);
    bool outputs_match = native_preflight_ok && packed_preflight_ok;
    for (uint64_t row = 0; outputs_match && row < Q4_MULTIROW_ROWS; ++row) {
      outputs_match = std::bit_cast<uint32_t>(packed_case.native_output[row]) ==
                      std::bit_cast<uint32_t>(packed_case.packed_output[row]);
    }
    const bool native_route_ok =
        native_machine.optimized_q4_dispatch_count() == 1u &&
        native_machine.optimized_q4_vector_dispatch_count() == 1u &&
        native_machine.optimized_q4_vector_packed_dispatch_count() == 0u &&
        native_machine.optimized_q4_vector_packed_q8_rhs_dispatch_count() ==
            0u &&
        native_machine.shared_q4_dispatch_count() == 0u;
    const bool packed_route_ok =
        packed_machine.optimized_q4_dispatch_count() == 1u &&
        packed_machine.optimized_q4_vector_dispatch_count() == 1u &&
        packed_machine.optimized_q4_vector_packed_dispatch_count() == 1u &&
        packed_machine.optimized_q4_vector_packed_q8_rhs_dispatch_count() ==
            1u &&
        packed_machine.shared_q4_dispatch_count() == 0u;
    if (!outputs_match || !native_route_ok || !packed_route_ok) {
      std::fprintf(
          stderr,
          "error: q4 packed benchmark preflight failed native_accepted=%d "
          "packed_accepted=%d outputs_match=%d "
          "native=(%llu,%llu,%llu,%llu,%llu) "
          "packed=(%llu,%llu,%llu,%llu,%llu)\n",
          native_preflight_ok ? 1 : 0, packed_preflight_ok ? 1 : 0,
          outputs_match ? 1 : 0,
          static_cast<unsigned long long>(
              native_machine.optimized_q4_dispatch_count()),
          static_cast<unsigned long long>(
              native_machine.optimized_q4_vector_dispatch_count()),
          static_cast<unsigned long long>(
              native_machine.optimized_q4_vector_packed_dispatch_count()),
          static_cast<unsigned long long>(
              native_machine
                  .optimized_q4_vector_packed_q8_rhs_dispatch_count()),
          static_cast<unsigned long long>(
              native_machine.shared_q4_dispatch_count()),
          static_cast<unsigned long long>(
              packed_machine.optimized_q4_dispatch_count()),
          static_cast<unsigned long long>(
              packed_machine.optimized_q4_vector_dispatch_count()),
          static_cast<unsigned long long>(
              packed_machine.optimized_q4_vector_packed_dispatch_count()),
          static_cast<unsigned long long>(
              packed_machine
                  .optimized_q4_vector_packed_q8_rhs_dispatch_count()),
          static_cast<unsigned long long>(
              packed_machine.shared_q4_dispatch_count()));
      std::abort();
    }
    std::printf(
        "# q4_packed_preflight: exact_output_match=1 native_route=(1,1,0,0,0) "
        "packed_route=(1,1,1,1,0) packed_before_timing=1 rows=%llu cols=%llu\n",
        static_cast<unsigned long long>(Q4_MULTIROW_ROWS),
        static_cast<unsigned long long>(Q4_MULTIROW_COLS));

    volatile float native_sink = 0.0f;
    auto native_fn = [&]() {
      if (!native_machine.process_event(packed_case.native_request)) {
        std::abort();
      }
      native_sink += packed_case.native_output[0];
    };
    volatile float packed_sink = 0.0f;
    auto packed_fn = [&]() {
      if (!packed_machine.process_event(packed_case.packed_request)) {
        std::abort();
      }
      packed_sink += packed_case.packed_output[0];
    };
    results.push_back(measure_case(Q4_PACKED_NATIVE_CASE_NAME, cfg, native_fn));
    results.push_back(measure_case(Q4_PACKED_CASE_NAME, cfg, packed_fn));
  }
}

void append_reference_kernel_aarch64_cases(std::vector<result> &results,
                                           const config &cfg) {
  append_reference_backend_cases(results, cfg, "aarch64");
}

} // namespace emel::bench
