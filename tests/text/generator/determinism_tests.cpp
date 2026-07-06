#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <doctest/doctest.h>

// Determinism contract proofs for the parallel matmul lane route
// (docs/determinism.md): the view-sliced fork/join parallel route must be
// bitwise identical to serial dispatch, bitwise invariant to the number of
// row slices (lane-count invariance), and bitwise repeatable across repeated
// dispatches. Route-level end-to-end determinism against a maintained model
// fixture is gated by scripts/check_determinism.sh; this file covers the
// slice arithmetic surface for every operand class the focused
// parallel-matmul benchmark exercises (f32/q8_0/q4_k/q6_k GEMV, f32 GEMM).

#include "emel/text/generator/detail.hpp"

namespace {

namespace gen_detail = emel::text::generator::detail;
using emel::kernel::event::dtype;
using gen_detail::matmul_lane_mode;
using gen_detail::matmul_row_slice;

inline constexpr size_t k_test_matmul_lanes = 8u;

emel::model::data::tensor_record make_tensor_record(void *data,
                                                    const int32_t type,
                                                    const int32_t cols,
                                                    const int32_t rows) {
  emel::model::data::tensor_record tensor = {};
  tensor.data = data;
  tensor.type = type;
  tensor.n_dims = 2;
  tensor.dims[0] = static_cast<int64_t>(cols);
  tensor.dims[1] = static_cast<int64_t>(rows);
  tensor.data_size =
      gen_detail::row_storage_bytes(tensor, cols) * static_cast<uint64_t>(rows);
  return tensor;
}

// Contiguous group-aligned partition of `rows` into exactly `slice_count`
// slices: the same shape the production compute_matmul_row_slices emits,
// parameterized over the slice count so the tests can sweep every effective
// lane count instead of only one fixed topology.
size_t
partition_rows(const uint64_t rows, const uint64_t group_rows,
               const uint64_t slice_count,
               std::array<matmul_row_slice, k_test_matmul_lanes> &slices) {
  const uint64_t groups = (rows + group_rows - 1u) / group_rows;
  const uint64_t lanes = std::min(slice_count, std::max<uint64_t>(groups, 1u));
  const uint64_t groups_per_lane = groups / lanes;
  const uint64_t extra_groups = groups % lanes;
  uint64_t begin_group = 0u;
  for (uint64_t lane = 0u; lane < lanes; ++lane) {
    const uint64_t lane_groups =
        groups_per_lane + static_cast<uint64_t>(lane < extra_groups);
    const uint64_t begin_row = begin_group * group_rows;
    const uint64_t end_row =
        std::min(rows, (begin_group + lane_groups) * group_rows);
    slices[lane].row_begin = static_cast<int32_t>(begin_row);
    slices[lane].row_count = static_cast<int32_t>(end_row - begin_row);
    begin_group += lane_groups;
  }
  return static_cast<size_t>(lanes);
}

struct matmul_case {
  std::vector<uint8_t> weights = {};
  std::vector<float> input = {};
  emel::model::data::tensor_record record = {};
  gen_detail::tensor_matrix matrix = {};
  emel::kernel::event::op_mul_mat ev = {};
  int32_t rows = 0;
  int32_t tokens = 0;

  matmul_case(const dtype type, const int32_t cols, const int32_t rows_in,
              const int32_t tokens_in)
      : rows(rows_in), tokens(tokens_in) {
    const uint8_t code = emel::kernel::detail::dtype_code(type);
    const uint64_t row_bytes =
        type == dtype::f32 ? sizeof(float) * static_cast<uint64_t>(cols)
                           : emel::kernel::detail::quantized_row_storage_bytes(
                                 code, static_cast<uint64_t>(cols));
    weights.assign(row_bytes * static_cast<size_t>(rows), 0u);
    for (size_t idx = 0; idx < weights.size(); ++idx) {
      weights[idx] = static_cast<uint8_t>((idx * 31u + 7u) & 0x3fu);
    }
    if (type == dtype::f32) {
      auto *values = reinterpret_cast<float *>(weights.data());
      const size_t count = weights.size() / sizeof(float);
      for (size_t idx = 0; idx < count; ++idx) {
        values[idx] = 0.25f * static_cast<float>((idx * 13u + 5u) % 17u) - 2.0f;
      }
    }
    input.assign(static_cast<size_t>(tokens) * static_cast<size_t>(cols), 0.0f);
    for (size_t idx = 0; idx < input.size(); ++idx) {
      input[idx] = 0.125f * static_cast<float>((idx * 7u + 3u) % 19u) - 1.0f;
    }
    record = make_tensor_record(weights.data(), static_cast<int32_t>(code),
                                cols, rows);
    matrix = gen_detail::tensor_matrix{&record, rows, cols};
    ev.src0 = gen_detail::make_src_view(matrix);
    ev.src1 =
        gen_detail::make_src_view(input.data(), static_cast<uint64_t>(tokens),
                                  static_cast<uint64_t>(cols));
  }

  emel::kernel::event::op_mul_mat event_for(std::vector<float> &output) const {
    output.assign(static_cast<size_t>(tokens) * static_cast<size_t>(rows),
                  -3.0f);
    emel::kernel::event::op_mul_mat bound = ev;
    bound.dst =
        gen_detail::make_dst_view(output.data(), static_cast<uint64_t>(tokens),
                                  static_cast<uint64_t>(rows));
    return bound;
  }
};

bool outputs_identical(const std::vector<float> &lhs,
                       const std::vector<float> &rhs) {
  return lhs.size() == rhs.size() &&
         std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0;
}

// Dispatch the full matmul as `slice_count` contiguous row-slice events
// through one kernel actor, exactly the event shape the parallel lane route
// forks; slice count stands in for lane count.
bool run_row_sliced(emel::kernel::sm &kernel, const matmul_case &fixture,
                    const uint64_t slice_count, std::vector<float> &output) {
  const auto full = fixture.event_for(output);
  const uint64_t group_rows =
      gen_detail::matmul_slice_group_rows(full.src0.type);
  std::array<matmul_row_slice, k_test_matmul_lanes> slices = {};
  const size_t lanes = partition_rows(static_cast<uint64_t>(fixture.rows),
                                      group_rows, slice_count, slices);
  bool all_ok = true;
  for (size_t lane = 0; lane < lanes; ++lane) {
    const auto sliced = gen_detail::compute_sliced_mul_mat_event(
        full, group_rows, slices[lane]);
    all_ok = all_ok && kernel.process_event(sliced);
  }
  return all_ok;
}

void check_slice_count_invariance(const dtype type, const int32_t cols,
                                  const int32_t rows, const int32_t tokens) {
  CAPTURE(static_cast<int>(type));
  CAPTURE(rows);
  CAPTURE(tokens);
  const matmul_case fixture(type, cols, rows, tokens);
  emel::kernel::sm kernel;
  kernel.set_kind(emel::kernel::detect_host_kind());

  std::vector<float> serial_output = {};
  const auto serial_ev = fixture.event_for(serial_output);
  REQUIRE(kernel.process_event(serial_ev));

  constexpr std::array<uint64_t, 5> k_slice_counts = {1u, 2u, 3u, 5u, 8u};
  for (const uint64_t slice_count : k_slice_counts) {
    CAPTURE(slice_count);
    std::vector<float> sliced_output = {};
    REQUIRE(run_row_sliced(kernel, fixture, slice_count, sliced_output));
    CHECK(outputs_identical(serial_output, sliced_output));
  }
}

TEST_CASE("determinism: row-sliced matmul bitwise invariant across slice "
          "counts f32 gemv") {
  check_slice_count_invariance(dtype::f32, 32, 61, 1);
}

TEST_CASE("determinism: row-sliced matmul bitwise invariant across slice "
          "counts q8_0 gemv") {
  check_slice_count_invariance(dtype::q8_0, 64, 61, 1);
}

TEST_CASE("determinism: row-sliced matmul bitwise invariant across slice "
          "counts q4_k gemv") {
  check_slice_count_invariance(dtype::q4_k, 256, 13, 1);
}

TEST_CASE("determinism: row-sliced matmul bitwise invariant across slice "
          "counts q6_k gemv") {
  check_slice_count_invariance(dtype::q6_k, 256, 13, 1);
}

TEST_CASE("determinism: row-sliced matmul bitwise invariant across slice "
          "counts f32 gemm") {
  check_slice_count_invariance(dtype::f32, 32, 61, 8);
}

TEST_CASE("determinism: parallel fork/join dispatch bitwise repeatable and "
          "serial-identical") {
  const matmul_case fixture(dtype::q8_0, 64, 61, 1);
  emel::text::generator::matmul::lane_pool<7u, 128u, 1048576u> parallel_matmul_lanes = {};
  auto policy = emel::text::generator::matmul::make_auto_execution_policy(
      parallel_matmul_lanes);
  emel::text::generator::matmul::sm matmul_actor{policy};
  gen_detail::native_backend backend = {};
  backend.kernel_kind = policy.kernel_kind;
  backend.kernel.set_kind(backend.kernel_kind);
  backend.matmul_actor = &matmul_actor;
  matmul_actor.process_event(
      emel::text::generator::matmul::event::configure_kernel_kind{backend.kernel_kind});

  std::vector<float> serial_output = {};
  const auto serial_ev = fixture.event_for(serial_output);
  REQUIRE(gen_detail::compute_mul_mat<matmul_lane_mode::serial>(backend,
                                                                serial_ev));

  constexpr int k_repeats = 5;
  for (int repeat = 0; repeat < k_repeats; ++repeat) {
    CAPTURE(repeat);
    std::vector<float> parallel_output = {};
    const auto parallel_ev = fixture.event_for(parallel_output);
    REQUIRE(gen_detail::compute_mul_mat<matmul_lane_mode::parallel>(
        backend, parallel_ev));
    CHECK(outputs_identical(serial_output, parallel_output));
  }
}

} // namespace
