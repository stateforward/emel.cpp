#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <doctest/doctest.h>

// View-sliced parallel matmul behavior: pack-group-aligned slice arithmetic
// and bit-exact parity between serial and parallel lane dispatch. Route-level
// proof against maintained model fixtures lives in the generator lifecycle
// tests; this file covers the slicing and fork/join detail surface.

#include "emel/text/generator/actions.hpp"
#include "emel/text/generator/detail.hpp"
#include "emel/text/generator/guards.hpp"
#include "emel/text/generator/matmul/actions.hpp"
#include "emel/text/generator/matmul/guards.hpp"

namespace {

namespace gen_detail = emel::text::generator::detail;
using emel::kernel::event::dtype;
using gen_detail::k_matmul_lanes;
using gen_detail::matmul_lane_mode;
using gen_detail::matmul_row_slice;

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

void check_slices_tile_rows(const uint64_t rows, const uint64_t group_rows) {
  CAPTURE(rows);
  CAPTURE(group_rows);
  std::array<matmul_row_slice, k_matmul_lanes> slices = {};
  const size_t count =
      gen_detail::compute_matmul_row_slices(rows, group_rows, slices);
  REQUIRE(count >= 1u);
  REQUIRE(count <= k_matmul_lanes);
  uint64_t expected_begin = 0u;
  for (size_t lane = 0; lane < count; ++lane) {
    CHECK(static_cast<uint64_t>(slices[lane].row_begin) == expected_begin);
    CHECK(slices[lane].row_count > 0);
    CHECK(static_cast<uint64_t>(slices[lane].row_begin) % group_rows == 0u);
    expected_begin += static_cast<uint64_t>(slices[lane].row_count);
  }
  CHECK(expected_begin == rows);
}

TEST_CASE("parallel matmul slice group rows match pack formats") {
  CHECK(gen_detail::matmul_slice_group_rows(dtype::f32) == 1u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::f16) == 1u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q8_0) == 1u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q4_k) == 1u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q6_k) == 1u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q8_0_x4_bl4) == 4u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q8_0_x4_bl8) == 4u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q4_k_x8_bl4) == 8u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q4_k_x8_bl8) == 8u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q6_k_x8) == 8u);
  CHECK(gen_detail::matmul_slice_group_rows(dtype::q6_k_x8_q8_prepared) == 8u);
  CHECK(gen_detail::matmul_slice_group_rows(
            dtype::q6_k_x8_q8_argmax_prepared) == 8u);
}

TEST_CASE("parallel matmul slices tile rows contiguously and group aligned") {
  check_slices_tile_rows(13u, 1u);
  check_slices_tile_rows(3u, 1u);
  check_slices_tile_rows(1u, 1u);
  check_slices_tile_rows(256u, 1u);
  check_slices_tile_rows(64u, 4u);
  check_slices_tile_rows(20u, 4u);
  check_slices_tile_rows(100u, 8u);
  check_slices_tile_rows(8u, 8u);
  check_slices_tile_rows(4096u, 8u);
}

TEST_CASE("parallel matmul ragged tail lands in final slice") {
  std::array<matmul_row_slice, k_matmul_lanes> slices = {};
  const size_t count = gen_detail::compute_matmul_row_slices(100u, 8u, slices);
  REQUIRE(count == k_matmul_lanes);
  const auto &tail = slices[count - 1u];
  CHECK(static_cast<uint64_t>(tail.row_begin) % 8u == 0u);
  CHECK(static_cast<uint64_t>(tail.row_begin) +
            static_cast<uint64_t>(tail.row_count) ==
        100u);
}

TEST_CASE("parallel matmul sliced event offsets views by storage groups") {
  std::array<uint8_t, 1024> src0_storage = {};
  std::array<float, 64> dst_storage = {};
  emel::kernel::event::op_mul_mat ev = {};
  ev.src0.data = src0_storage.data();
  ev.src0.type = dtype::q4_k_x8_bl8;
  ev.src0.ne = {256u, 64u, 1u, 1u};
  ev.src0.nb = {1u, 128u, 128u * 8u, 128u * 8u};
  ev.dst.data = dst_storage.data();
  ev.dst.type = dtype::f32;
  ev.dst.ne = {1u, 64u, 1u, 1u};
  ev.dst.nb = {sizeof(float), sizeof(float), sizeof(float) * 64u,
               sizeof(float) * 64u};

  const matmul_row_slice slice{16, 24};
  const auto sliced = gen_detail::compute_sliced_mul_mat_event(ev, 8u, slice);
  CHECK(sliced.src0.data == src0_storage.data() + (16u / 8u) * 128u);
  CHECK(sliced.src0.ne[1] == 24u);
  CHECK(sliced.src0.nb[1] == ev.src0.nb[1]);
  CHECK(sliced.src0.nb[2] == 128u * 3u);
  CHECK(sliced.dst.data ==
        reinterpret_cast<uint8_t *>(dst_storage.data()) + 16u * sizeof(float));
  CHECK(sliced.dst.ne[1] == 24u);
  CHECK(sliced.src1.data == ev.src1.data);
}

TEST_CASE("matmul actor guards and actions expose explicit rejection paths") {
  namespace matmul = emel::text::generator::matmul;

  matmul::action::context ctx = {};
  emel::kernel::event::op_mul_mat request = {};
  request.src0.ne[1] = 0u;
  matmul::event::dispatch_result serial_result = {};
  matmul::event::dispatch_result parallel_result = {};
  bool serial_accepted = true;
  bool parallel_accepted = true;
  const matmul::event::execute_serial serial_event{request, serial_result,
                                                   serial_accepted};
  const matmul::event::execute_parallel parallel_event{request, parallel_result,
                                                       parallel_accepted};

  CHECK(matmul::guard::guard_parallel_unavailable{}(parallel_event, ctx));
  CHECK_FALSE(matmul::guard::guard_parallel_ready{}(parallel_event, ctx));
  request.src0.ne[1] = 4u;
  CHECK(matmul::guard::guard_parallel_unavailable{}(parallel_event, ctx));

  matmul::lane_pool parallel_lanes = {};
  ctx.parallel_matmul_lanes = &parallel_lanes;
  CHECK(matmul::guard::guard_parallel_ready{}(parallel_event, ctx));
  CHECK_FALSE(matmul::guard::guard_parallel_unavailable{}(parallel_event, ctx));

  serial_result.lane_accepted[0] = false;
  CHECK(matmul::guard::guard_serial_rejected{}(serial_event, ctx));
  CHECK_FALSE(matmul::guard::guard_serial_accepted{}(serial_event, ctx));
  matmul::action::effect_reject_serial_execution(serial_event, ctx);
  CHECK_FALSE(serial_accepted);

  parallel_result.all_submitted = false;
  parallel_result.joined = true;
  CHECK(matmul::guard::guard_parallel_submission_failed{}(parallel_event, ctx));
  CHECK_FALSE(matmul::guard::guard_parallel_join_failed{}(parallel_event, ctx));

  parallel_result.all_submitted = true;
  parallel_result.joined = false;
  CHECK(matmul::guard::guard_parallel_join_failed{}(parallel_event, ctx));
  CHECK_FALSE(
      matmul::guard::guard_parallel_all_lanes_accepted{}(parallel_event, ctx));

  parallel_result.joined = true;
  parallel_result.lane_count = 0u;
  CHECK_FALSE(
      matmul::guard::guard_parallel_lane_rejected<0>{}(parallel_event, ctx));

  parallel_result.lane_count = 2u;
  parallel_result.lane_accepted[0] = false;
  parallel_result.lane_accepted[1] = false;
  CHECK_FALSE(
      matmul::guard::guard_parallel_lane_rejected<1>{}(parallel_event, ctx));

  parallel_result.lane_accepted[0] = true;
  CHECK(matmul::guard::guard_parallel_lane_rejected<1>{}(parallel_event, ctx));
  CHECK_FALSE(
      matmul::guard::guard_parallel_all_lanes_accepted{}(parallel_event, ctx));

  parallel_result.lane_accepted[1] = true;
  CHECK(
      matmul::guard::guard_parallel_all_lanes_accepted{}(parallel_event, ctx));
  matmul::action::effect_reject_parallel_execution(parallel_event, ctx);
  CHECK_FALSE(parallel_accepted);

  struct event_with_accepted {
    bool &accepted;
  };
  bool unexpected_accepted = true;
  const event_with_accepted unexpected{unexpected_accepted};
  matmul::action::effect_on_unexpected(unexpected, ctx);
  CHECK_FALSE(unexpected_accepted);

  struct event_without_accepted {};
  matmul::action::effect_on_unexpected(event_without_accepted{}, ctx);
}

struct parallel_backend_fixture {
  emel::text::generator::matmul::lane_pool parallel_matmul_lanes = {};
  emel::text::generator::matmul::sm matmul_actor{parallel_matmul_lanes};
  gen_detail::native_backend backend = {};

  parallel_backend_fixture() {
    backend.kernel_kind = gen_detail::detect_host_kernel_kind();
    backend.kernel.set_kind(backend.kernel_kind);
    backend.matmul_actor = &matmul_actor;
    matmul_actor.process_event(
        emel::text::generator::matmul::event::configure_kernel_kind{
            backend.kernel_kind});
  }
};

TEST_CASE("parallel matmul f32 gemv matches serial dispatch bit exact") {
  constexpr int32_t rows = 61;
  constexpr int32_t cols = 32;
  std::vector<float> weights(static_cast<size_t>(rows) *
                             static_cast<size_t>(cols));
  for (size_t idx = 0; idx < weights.size(); ++idx) {
    weights[idx] = 0.25f * static_cast<float>((idx * 31u + 7u) % 17u) - 2.0f;
  }
  std::vector<float> input(static_cast<size_t>(cols));
  for (size_t idx = 0; idx < input.size(); ++idx) {
    input[idx] = 0.5f * static_cast<float>((idx * 13u + 3u) % 11u) - 2.5f;
  }

  auto record = make_tensor_record(weights.data(),
                                   emel::kernel::detail::dtype_f32, cols, rows);
  gen_detail::tensor_matrix matrix{&record, rows, cols};

  parallel_backend_fixture fixture;
  std::vector<float> out_serial(static_cast<size_t>(rows), -1.0f);
  std::vector<float> out_parallel(static_cast<size_t>(rows), -2.0f);

  emel::kernel::event::op_mul_mat serial_ev{
      .src0 = gen_detail::make_src_view(matrix),
      .src1 = gen_detail::make_src_view(input.data(), static_cast<uint64_t>(1u),
                                        static_cast<uint64_t>(input.size())),
      .dst = gen_detail::make_dst_view(
          out_serial.data(), static_cast<uint64_t>(1u),
          static_cast<uint64_t>(out_serial.size())),
  };
  emel::kernel::event::op_mul_mat parallel_ev = serial_ev;
  parallel_ev.dst =
      gen_detail::make_dst_view(out_parallel.data(), static_cast<uint64_t>(1u),
                                static_cast<uint64_t>(out_parallel.size()));

  CHECK(gen_detail::compute_mul_mat<matmul_lane_mode::serial>(fixture.backend,
                                                              serial_ev));
  CHECK(gen_detail::compute_mul_mat<matmul_lane_mode::parallel>(fixture.backend,
                                                                parallel_ev));
  CHECK(std::memcmp(out_serial.data(), out_parallel.data(),
                    out_serial.size() * sizeof(float)) == 0);
}

TEST_CASE("parallel matmul q8_0 gemv matches serial dispatch bit exact") {
  constexpr int32_t rows = 61;
  constexpr int32_t cols = 64;
  constexpr size_t blocks_per_row =
      static_cast<size_t>(cols) / static_cast<size_t>(gen_detail::quant::QK8_0);
  std::vector<gen_detail::quant::block_q8_0> weights(static_cast<size_t>(rows) *
                                                     blocks_per_row);
  for (size_t block = 0; block < weights.size(); ++block) {
    weights[block].d = gen_detail::quant::fp32_to_fp16(
        0.01f + 0.001f * static_cast<float>(block % 7u));
    for (size_t idx = 0; idx < weights[block].qs.size(); ++idx) {
      weights[block].qs[idx] = static_cast<int8_t>(
          static_cast<int32_t>((block * 37u + idx * 5u) % 255u) - 127);
    }
  }
  std::vector<float> input(static_cast<size_t>(cols));
  for (size_t idx = 0; idx < input.size(); ++idx) {
    input[idx] = 0.125f * static_cast<float>((idx * 7u + 1u) % 19u) - 1.0f;
  }

  auto record = make_tensor_record(
      weights.data(), emel::kernel::detail::dtype_q8_0, cols, rows);
  gen_detail::tensor_matrix matrix{&record, rows, cols};

  parallel_backend_fixture fixture;
  std::vector<float> out_serial(static_cast<size_t>(rows), -1.0f);
  std::vector<float> out_parallel(static_cast<size_t>(rows), -2.0f);

  emel::kernel::event::op_mul_mat serial_ev{
      .src0 = gen_detail::make_src_view(matrix),
      .src1 = gen_detail::make_src_view(input.data(), static_cast<uint64_t>(1u),
                                        static_cast<uint64_t>(input.size())),
      .dst = gen_detail::make_dst_view(
          out_serial.data(), static_cast<uint64_t>(1u),
          static_cast<uint64_t>(out_serial.size())),
  };
  emel::kernel::event::op_mul_mat parallel_ev = serial_ev;
  parallel_ev.dst =
      gen_detail::make_dst_view(out_parallel.data(), static_cast<uint64_t>(1u),
                                static_cast<uint64_t>(out_parallel.size()));

  CHECK(gen_detail::compute_mul_mat<matmul_lane_mode::serial>(fixture.backend,
                                                              serial_ev));
  CHECK(gen_detail::compute_mul_mat<matmul_lane_mode::parallel>(fixture.backend,
                                                                parallel_ev));
  CHECK(std::memcmp(out_serial.data(), out_parallel.data(),
                    out_serial.size() * sizeof(float)) == 0);
}

TEST_CASE("parallel matmul route guard rejects disengaged lane pool") {
  emel::text::generator::action::context ctx = {};
  ctx.compute.backend.n_embd = gen_detail::k_parallel_min_gemv_dim;
  emel::text::generator::event::generate_ctx run_ctx = {};
  std::array<emel::text::formatter::chat_message, 1> messages = {};
  std::array<char, 8> output = {};
  size_t output_length = 0u;
  const emel::text::generator::event::generate request{
      std::span<const emel::text::formatter::chat_message>(messages), 1,
      std::span<char>(output), output_length};
  const emel::text::generator::event::generate_run run{request, run_ctx};

  CHECK_FALSE(emel::text::generator::guard::guard_decode_parallel_lanes_ready{}(
      run, ctx));

  emel::text::generator::matmul::sm serial_matmul_actor = {};
  ctx.compute.backend.matmul_actor = &serial_matmul_actor;
  CHECK_FALSE(emel::text::generator::guard::guard_decode_parallel_lanes_ready{}(
      run, ctx));

  emel::text::generator::matmul::lane_pool parallel_matmul_lanes = {};
  emel::text::generator::matmul::sm parallel_matmul_actor{
      parallel_matmul_lanes};
  ctx.compute.backend.matmul_actor = &parallel_matmul_actor;
  CHECK(emel::text::generator::guard::guard_decode_parallel_lanes_ready{}(run,
                                                                          ctx));
  ctx.compute.backend.parallel_lanes_enabled = false;
  CHECK_FALSE(emel::text::generator::guard::guard_decode_parallel_lanes_ready{}(
      run, ctx));
}

TEST_CASE(
    "benchmark lane configuration guards and actions toggle parallel lanes") {
  emel::text::generator::action::context ctx = {};
  const emel::text::generator::event::configure_benchmark_lane single{
      emel::text::generator::benchmark_lane::single};
  const emel::text::generator::event::configure_benchmark_lane multithreaded{
      emel::text::generator::benchmark_lane::multithreaded};

  CHECK(
      emel::text::generator::guard::guard_benchmark_lane_single{}(single, ctx));
  CHECK_FALSE(emel::text::generator::guard::guard_benchmark_lane_single{}(
      multithreaded, ctx));
  CHECK(emel::text::generator::guard::guard_benchmark_lane_multithreaded{}(
      multithreaded, ctx));
  CHECK_FALSE(
      emel::text::generator::guard::guard_benchmark_lane_multithreaded{}(single,
                                                                         ctx));

  ctx.compute.backend.parallel_lanes_enabled = true;
  emel::text::generator::action::effect_disable_parallel_benchmark_lanes(single,
                                                                         ctx);
  CHECK_FALSE(ctx.compute.backend.parallel_lanes_enabled);
  emel::text::generator::action::effect_enable_parallel_benchmark_lanes(
      multithreaded, ctx);
  CHECK(ctx.compute.backend.parallel_lanes_enabled);
}

} // namespace
