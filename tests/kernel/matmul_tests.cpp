#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include <doctest/doctest.h>

// View-sliced parallel matmul behavior: pack-group-aligned slice arithmetic
// and bit-exact parity between serial and parallel lane dispatch. Route-level
// proof against maintained model fixtures lives in the generator lifecycle
// tests; this file covers the slicing and fork/join detail surface.

#include "../allocation_tracker.hpp"
#include "../text/generator/generator_test_policies.hpp"
#include "emel/kernel/matmul/actions.hpp"
#include "emel/kernel/matmul/guards.hpp"
#include "emel/text/generator/actions.hpp"
#include "emel/text/generator/detail.hpp"
#include "emel/text/generator/guards.hpp"

namespace {

namespace gen_detail = emel::text::generator::detail;
namespace matmul = emel::kernel::matmul;
using emel::kernel::event::dtype;
using gen_detail::matmul_lane_mode;
using matmul::detail::matmul_row_slice;

inline constexpr size_t k_test_matmul_lanes = 8u;

struct outcome_counts {
  int done = 0;
  int error = 0;
  const matmul::event::execute_parallel *done_request = nullptr;
  const matmul::event::execute_parallel *error_request = nullptr;
};

void count_done(void *object, const matmul::events::execute_done<
                                  matmul::event::execute_parallel> &outcome) {
  auto &counts = *static_cast<outcome_counts *>(object);
  ++counts.done;
  counts.done_request = &outcome.request;
}

void count_error(void *object, const matmul::events::execute_error<
                                   matmul::event::execute_parallel> &outcome) {
  auto &counts = *static_cast<outcome_counts *>(object);
  ++counts.error;
  counts.error_request = &outcome.request;
}

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

template <size_t lane_count, uint64_t group_rows, size_t... lanes>
std::array<matmul_row_slice, lane_count>
make_production_slices(const uint64_t rows, std::index_sequence<lanes...>) {
  return {
      {matmul::action::compute_fixed_row_slice<lanes, lane_count, group_rows>(
          rows)...}};
}

template <size_t lane_count, uint64_t group_rows>
void check_slices_tile_rows(const uint64_t rows) {
  CAPTURE(rows);
  constexpr uint64_t group_rows_value = group_rows;
  const auto slices = make_production_slices<lane_count, group_rows>(
      rows, std::make_index_sequence<lane_count>{});
  uint64_t expected_begin = 0u;
  for (size_t lane = 0; lane < lane_count; ++lane) {
    CHECK(static_cast<uint64_t>(slices[lane].row_begin) == expected_begin);
    CHECK(slices[lane].row_count > 0);
    CHECK(static_cast<uint64_t>(slices[lane].row_begin) % group_rows_value ==
          0u);
    expected_begin += static_cast<uint64_t>(slices[lane].row_count);
  }
  CHECK(expected_begin == rows);
}

TEST_CASE("parallel matmul slice group rows match pack formats") {
  CHECK_FALSE(matmul::guard::guard_uses_x8_row_groups(dtype::f32));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::f32));
  CHECK_FALSE(matmul::guard::guard_uses_x8_row_groups(dtype::f16));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::f16));
  CHECK_FALSE(matmul::guard::guard_uses_x8_row_groups(dtype::q8_0));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::q8_0));
  CHECK_FALSE(matmul::guard::guard_uses_x8_row_groups(dtype::q4_k));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::q4_k));
  CHECK_FALSE(matmul::guard::guard_uses_x8_row_groups(dtype::q6_k));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::q6_k));

  CHECK_FALSE(matmul::guard::guard_uses_x8_row_groups(dtype::q8_0_x4_bl4));
  CHECK(matmul::guard::guard_uses_x4_row_groups(dtype::q8_0_x4_bl4));
  CHECK_FALSE(matmul::guard::guard_uses_x8_row_groups(dtype::q8_0_x4_bl8));
  CHECK(matmul::guard::guard_uses_x4_row_groups(dtype::q8_0_x4_bl8));

  CHECK(matmul::guard::guard_uses_x8_row_groups(dtype::q4_k_x8_bl4));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::q4_k_x8_bl4));
  CHECK(matmul::guard::guard_uses_x8_row_groups(dtype::q4_k_x8_bl8));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::q4_k_x8_bl8));
  CHECK(matmul::guard::guard_uses_x8_row_groups(dtype::q6_k_x8));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(dtype::q6_k_x8));
  CHECK(matmul::guard::guard_uses_x8_row_groups(dtype::q6_k_x8_q8_prepared));
  CHECK_FALSE(
      matmul::guard::guard_uses_x4_row_groups(dtype::q6_k_x8_q8_prepared));
  CHECK(matmul::guard::guard_uses_x8_row_groups(
      dtype::q6_k_x8_q8_argmax_prepared));
  CHECK_FALSE(matmul::guard::guard_uses_x4_row_groups(
      dtype::q6_k_x8_q8_argmax_prepared));
}

TEST_CASE("parallel matmul slices tile rows contiguously and group aligned") {
  check_slices_tile_rows<8u, 1u>(13u);
  check_slices_tile_rows<2u, 1u>(3u);
  check_slices_tile_rows<4u, 1u>(16u);
  check_slices_tile_rows<8u, 1u>(256u);
  check_slices_tile_rows<8u, 4u>(64u);
  check_slices_tile_rows<4u, 4u>(20u);
  check_slices_tile_rows<8u, 8u>(100u);
  check_slices_tile_rows<8u, 8u>(64u);
  check_slices_tile_rows<8u, 8u>(4096u);
}

TEST_CASE("parallel matmul ragged tail lands in final slice") {
  const auto slices = make_production_slices<k_test_matmul_lanes, 8u>(
      100u, std::make_index_sequence<k_test_matmul_lanes>{});
  const auto &tail = slices.back();
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
  const auto sliced =
      matmul::detail::compute_sliced_mul_mat_event(ev, 8u, slice);
  CHECK(sliced.src0.data == src0_storage.data() + (16u / 8u) * 128u);
  CHECK(sliced.src0.ne[1] == 24u);
  CHECK(sliced.src0.nb[1] == ev.src0.nb[1]);
  CHECK(sliced.src0.nb[2] == 128u * 3u);
  CHECK(sliced.dst.data ==
        reinterpret_cast<uint8_t *>(dst_storage.data()) + 16u * sizeof(float));
  CHECK(sliced.dst.ne[1] == 24u);
  CHECK(sliced.src1.data == ev.src1.data);
}

TEST_CASE("parallel matmul storage guard rejects malformed parent requests") {
  matmul::lane_pool lanes = {};
  matmul::action::context ctx = {};
  ctx.parallel_matmul_lanes = &lanes;
  ctx.active_lanes = 8u;

  emel::kernel::event::op_mul_mat request = {};
  request.src0.ne[1] = 8u;
  matmul::event::dispatch_result result = {};
  bool accepted = true;
  const matmul::event::execute_parallel run{request, result, accepted};

  CHECK_FALSE(matmul::guard::guard_parallel_storage_ready(run, ctx));
  const auto policy = matmul::make_execution_policy(
      lanes, emel::kernel::detect_host_kind(), 8u);
  matmul::sm actor{policy};
  REQUIRE(actor.process_event(run));
  CHECK_FALSE(accepted);
  CHECK(result.lane_count == 0u);
  CHECK(result.submitted_worker_lanes == 0u);

  std::array<float, 8> storage = {};
  request.src0.data = storage.data();
  request.src1.data = storage.data();
  request.dst.data = storage.data();
  request.src0.type = dtype::f32;
  request.src1.type = dtype::f32;
  request.dst.type = dtype::f32;
  request.src0.ne = {
      1u, static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1u, 1u,
      1u};
  request.src1.ne = {1u, 1u, 1u, 1u};
  request.dst.ne = {1u, request.src0.ne[1], 1u, 1u};
  request.src0.nb = {sizeof(float), sizeof(float), sizeof(float),
                     sizeof(float)};
  request.src1.nb = {sizeof(float), sizeof(float), sizeof(float),
                     sizeof(float)};
  request.dst.nb = {sizeof(float), sizeof(float), sizeof(float), sizeof(float)};
  CHECK_FALSE(matmul::guard::guard_parallel_storage_ready(run, ctx));
  result = {};
  accepted = true;
  REQUIRE(actor.process_event(run));
  CHECK_FALSE(accepted);
  CHECK(result.lane_count == 0u);
  CHECK(result.submitted_worker_lanes == 0u);
}

TEST_CASE("matmul actor guards and actions expose explicit rejection paths") {
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
  CHECK_FALSE(
      matmul::guard::guard_parallel_ready<4u, 1u>{}(parallel_event, ctx));
  request.src0.ne[1] = 4u;
  CHECK(matmul::guard::guard_parallel_unavailable{}(parallel_event, ctx));

  matmul::lane_pool parallel_lanes = {};
  ctx.parallel_matmul_lanes = &parallel_lanes;
  ctx.active_lanes = 8u;
  CHECK(matmul::guard::guard_parallel_request_invalid{}(parallel_event, ctx));
  CHECK_FALSE(
      matmul::guard::guard_parallel_ready<4u, 1u>{}(parallel_event, ctx));
  CHECK_FALSE(matmul::guard::guard_parallel_unavailable{}(parallel_event, ctx));

  std::array<float, 4> weights = {};
  std::array<float, 1> input = {};
  std::array<float, 4> output = {};
  request.src0 = gen_detail::make_src_view(weights.data(), 1u, 4u);
  request.src1 = gen_detail::make_src_view(input.data(), 1u, 1u);
  request.dst = gen_detail::make_dst_view(output.data(), 1u, 4u);
  CHECK_FALSE(
      matmul::guard::guard_parallel_request_invalid{}(parallel_event, ctx));
  CHECK(matmul::guard::guard_parallel_ready<4u, 1u>{}(parallel_event, ctx));

  matmul::action::context over_cap_ctx = {};
  over_cap_ctx.parallel_matmul_lanes = &parallel_lanes;
  over_cap_ctx.active_lanes = 3u;
  CHECK_FALSE(matmul::guard::guard_parallel_ready<4u, 1u>{}(parallel_event,
                                                            over_cap_ctx));
  CHECK(matmul::guard::guard_parallel_unavailable{}(parallel_event,
                                                    over_cap_ctx));

  serial_result.all_lanes_accepted = false;
  CHECK(matmul::guard::guard_serial_rejected{}(serial_event, ctx));
  CHECK_FALSE(matmul::guard::guard_serial_accepted{}(serial_event, ctx));
  matmul::action::effect_reject_serial_execution(serial_event, ctx);
  CHECK_FALSE(serial_accepted);

  parallel_result.all_submitted = false;
  CHECK(matmul::guard::guard_parallel_submission_failed{}(parallel_event, ctx));

  parallel_result.all_submitted = true;
  parallel_result.lane_count = 0u;
  parallel_result.all_lanes_accepted = false;
  CHECK(matmul::guard::guard_parallel_lane_rejected{}(parallel_event, ctx));
  CHECK_FALSE(
      matmul::guard::guard_parallel_all_lanes_accepted{}(parallel_event, ctx));

  parallel_result.all_lanes_accepted = true;
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

TEST_CASE("parallel matmul guards select fixed lane and packed row groups") {
  matmul::lane_pool lanes = {};
  matmul::action::context ctx = {};
  ctx.parallel_matmul_lanes = &lanes;
  ctx.active_lanes = 8u;
  emel::kernel::event::op_mul_mat request = {};
  matmul::event::dispatch_result result = {};
  bool accepted = false;
  const matmul::event::execute_parallel run{request, result, accepted};

  request.src0.type = dtype::f32;
  request.src0.ne[1] = 8u;
  CHECK(matmul::guard::guard_lane_count_selected<8u>(run, ctx, 1u));
  request.src0.ne[1] = 7u;
  CHECK_FALSE(matmul::guard::guard_lane_count_selected<8u>(run, ctx, 1u));

  ctx.active_lanes = 4u;
  request.src0.ne[1] = 8u;
  CHECK(matmul::guard::guard_lane_count_selected<4u>(run, ctx, 1u));
  ctx.active_lanes = 8u;
  request.src0.ne[1] = 4u;
  CHECK(matmul::guard::guard_lane_count_selected<4u>(run, ctx, 1u));
  request.src0.ne[1] = 8u;
  CHECK_FALSE(matmul::guard::guard_lane_count_selected<4u>(run, ctx, 1u));

  ctx.active_lanes = 2u;
  request.src0.ne[1] = 4u;
  CHECK(matmul::guard::guard_lane_count_selected<2u>(run, ctx, 1u));
  ctx.active_lanes = 4u;
  request.src0.ne[1] = 2u;
  CHECK(matmul::guard::guard_lane_count_selected<2u>(run, ctx, 1u));
  request.src0.ne[1] = 4u;
  CHECK_FALSE(matmul::guard::guard_lane_count_selected<2u>(run, ctx, 1u));

  std::array<uint8_t, 64> src0_storage = {};
  std::array<float, 1> src1_storage = {};
  std::array<float, 64> dst_storage = {};
  request.src0.data = src0_storage.data();
  request.src0.ne[0] = 1u;
  request.src0.ne[2] = 1u;
  request.src0.ne[3] = 1u;
  request.src0.nb = {1u, 1u, 64u, 64u};
  request.src1 = gen_detail::make_src_view(src1_storage.data(), 1u, 1u);
  request.dst = gen_detail::make_dst_view(dst_storage.data(), 1u, 1u);
  const auto set_request_rows = [&request](const dtype type,
                                           const uint64_t rows) noexcept {
    request.src0.type = type;
    request.src0.ne[1] = rows;
    request.dst.ne[1] = rows;
  };
  ctx.active_lanes = 8u;
  set_request_rows(dtype::f32, 1u);
  CHECK(matmul::guard::guard_parallel_no_lane_count{}(run, ctx));
  set_request_rows(dtype::q8_0_x4_bl8, 4u);
  CHECK(matmul::guard::guard_parallel_no_lane_count{}(run, ctx));
  set_request_rows(dtype::q4_k_x8_bl8, 8u);
  CHECK(matmul::guard::guard_parallel_no_lane_count{}(run, ctx));
  set_request_rows(dtype::q4_k_x8_bl8, 16u);
  CHECK_FALSE(matmul::guard::guard_parallel_no_lane_count{}(run, ctx));

  CHECK(matmul::guard::guard_group_rows_selected<1u>(dtype::f32));
  CHECK_FALSE(matmul::guard::guard_group_rows_selected<1u>(dtype::q8_0_x4_bl8));
  CHECK(matmul::guard::guard_group_rows_selected<
        emel::kernel::detail::quant::Q8_0_X4_ROWS>(dtype::q8_0_x4_bl8));
  CHECK(matmul::guard::guard_group_rows_selected<
        emel::kernel::detail::quant::Q4_K_X8_ROWS>(dtype::q4_k_x8_bl8));
}

void exercise_fixed_packed_format(const dtype type, const uint64_t rows,
                                  const size_t expected_lanes,
                                  const size_t active_lanes) {
  matmul::lane_pool pool = {};
  const auto policy = matmul::make_execution_policy(
      pool, emel::kernel::detect_host_kind(), active_lanes);
  matmul::sm actor{policy};
  std::vector<uint8_t> src0_storage(static_cast<size_t>(rows), 0u);
  std::array<float, 1> src1_storage = {};
  std::vector<float> dst_storage(static_cast<size_t>(rows), 0.0f);
  emel::kernel::event::op_mul_mat request = {};
  request.src0.data = src0_storage.data();
  request.src0.type = type;
  request.src0.ne = {1u, rows, 1u, 1u};
  request.src0.nb = {1u, 1u, rows, rows};
  request.src1 = gen_detail::make_src_view(src1_storage.data(), 1u, 1u);
  request.dst = gen_detail::make_dst_view(dst_storage.data(), 1u, rows);
  matmul::event::dispatch_result result = {};
  bool accepted = true;
  const matmul::event::execute_parallel run{request, result, accepted};
  REQUIRE(actor.process_event(run));
  CHECK(result.lane_count == expected_lanes);
  CHECK_FALSE(accepted);
}

TEST_CASE("parallel matmul dispatches explicit packed format lane bodies") {
  exercise_fixed_packed_format(dtype::q8_0_x4_bl8, 8u, 2u, 2u);
  exercise_fixed_packed_format(dtype::q8_0_x4_bl8, 16u, 4u, 4u);
  exercise_fixed_packed_format(dtype::q8_0_x4_bl8, 32u, 8u, 8u);
  exercise_fixed_packed_format(dtype::q4_k_x8_bl8, 16u, 2u, 2u);
  exercise_fixed_packed_format(dtype::q4_k_x8_bl8, 32u, 4u, 4u);
  exercise_fixed_packed_format(dtype::q4_k_x8_bl8, 64u, 8u, 8u);
}

TEST_CASE("parallel matmul uses the concrete canonical lane pool") {
  matmul::lane_pool pool = {};
  CHECK_FALSE(pool.is_current_thread_worker());
  const auto policy = matmul::make_auto_execution_policy(pool);
  CHECK(policy.parallel_matmul_lanes == &pool);
  CHECK(policy.active_lanes == 8u);
}

TEST_CASE(
    "parallel matmul auto policy stays within the runtime worker budget") {
  {
    matmul::lane_pool pool{1u};
    CHECK(matmul::make_auto_execution_policy(pool).active_lanes == 2u);
  }
  {
    matmul::lane_pool pool{2u};
    CHECK(matmul::make_auto_execution_policy(pool).active_lanes == 2u);
  }
  {
    matmul::lane_pool pool{3u};
    CHECK(matmul::make_auto_execution_policy(pool).active_lanes == 4u);
  }
  {
    matmul::lane_pool pool{6u};
    CHECK(matmul::make_auto_execution_policy(pool).active_lanes == 4u);
  }
  {
    matmul::lane_pool pool{7u};
    CHECK(matmul::make_auto_execution_policy(pool).active_lanes == 8u);
  }
}

struct parallel_backend_fixture {
  emel::kernel::matmul::lane_pool parallel_matmul_lanes = {};
  emel::kernel::matmul::execution_policy policy =
      emel::kernel::matmul::make_auto_execution_policy(parallel_matmul_lanes);
  emel::kernel::matmul::sm matmul_actor{policy};
  gen_detail::native_backend backend = {};

  parallel_backend_fixture() {
    backend.kernel_kind = policy.kernel_kind;
    backend.routes = emel::text::generator::test::k_generation_route_policy;
    backend.kernel.set_kind(backend.kernel_kind);
    backend.matmul_actor = &matmul_actor;
    backend.matmul_lane_mode = emel::kernel::matmul::lane_mode::parallel;
    matmul_actor.process_event(
        emel::kernel::matmul::event::configure_kernel_kind{
            backend.kernel_kind});
  }
};

void check_fixed_parallel_lane_count(const size_t expected_lanes) {
  constexpr int32_t rows = 16;
  constexpr int32_t cols = 8;
  std::array<float, static_cast<size_t>(rows * cols)> weights = {};
  std::array<float, static_cast<size_t>(cols)> input = {};
  std::array<float, static_cast<size_t>(rows)> output = {};
  std::fill(weights.begin(), weights.end(), 0.25f);
  std::fill(input.begin(), input.end(), 0.5f);

  auto record = make_tensor_record(weights.data(),
                                   emel::kernel::detail::dtype_f32, cols, rows);
  gen_detail::tensor_matrix matrix{&record, rows, cols};
  emel::kernel::event::op_mul_mat request{
      .src0 = gen_detail::make_src_view(matrix),
      .src1 = gen_detail::make_src_view(input.data(), 1u, input.size()),
      .dst = gen_detail::make_dst_view(output.data(), 1u, output.size()),
  };

  matmul::lane_pool pool = {};
  const auto policy = matmul::make_execution_policy(
      pool, emel::kernel::detect_host_kind(), expected_lanes);
  matmul::sm actor{policy};
  matmul::event::dispatch_result result = {};
  bool accepted = false;
  const matmul::event::execute_parallel run{request, result, accepted};
  REQUIRE(actor.process_event(run));
  REQUIRE(accepted);
  CHECK(result.lane_count == expected_lanes);
  CHECK(result.all_submitted);
  CHECK(result.submitted_worker_lanes == expected_lanes - 1u);
  CHECK(result.drained_worker_lanes == result.submitted_worker_lanes);
  CHECK(result.all_lanes_accepted);
}

TEST_CASE("parallel matmul explicitly dispatches two and four lane bodies") {
  check_fixed_parallel_lane_count(2u);
  check_fixed_parallel_lane_count(4u);
}

TEST_CASE(
    "parallel matmul partial submission rejection drains accepted workers") {
  constexpr int32_t rows = 8;
  constexpr int32_t cols = 1;
  std::array<float, static_cast<size_t>(rows * cols)> weights = {};
  std::array<float, static_cast<size_t>(cols)> input = {2.0f};
  std::array<float, static_cast<size_t>(rows)> output = {};
  weights.fill(1.0f);
  output.fill(-1.0f);

  matmul::lane_pool pool = {};
  matmul::lane_pool::join_group blocker_group{};
  std::atomic<bool> blocker_entered = false;
  std::atomic<bool> release_blocker = false;
  REQUIRE(pool.try_submit(blocker_group, [&]() noexcept {
    blocker_entered.store(true, std::memory_order_release);
    while (!release_blocker.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }));
  for (size_t attempt = 0u; attempt < 100000u; ++attempt) {
    if (blocker_entered.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  REQUIRE(blocker_entered.load(std::memory_order_acquire));

  auto record = make_tensor_record(weights.data(),
                                   emel::kernel::detail::dtype_f32, cols, rows);
  gen_detail::tensor_matrix matrix{&record, rows, cols};
  emel::kernel::event::op_mul_mat request{
      .src0 = gen_detail::make_src_view(matrix),
      .src1 = gen_detail::make_src_view(input.data(), 1u, input.size()),
      .dst = gen_detail::make_dst_view(output.data(), 1u, output.size()),
  };
  const auto policy =
      matmul::make_execution_policy(pool, emel::kernel::detect_host_kind(), 8u);
  matmul::sm actor{policy};
  matmul::event::dispatch_result result = {};
  bool accepted = true;
  const matmul::event::execute_parallel run{request, result, accepted};

  bool dispatched = false;
  size_t allocation_count = 0u;
  {
    emel::test::allocation::allocation_scope allocations{};
    dispatched = actor.process_event(run);
    allocation_count = allocations.allocations();
  }
  REQUIRE(dispatched);
  CHECK_FALSE(accepted);
  CHECK_FALSE(result.all_submitted);
  CHECK(result.submitted_worker_lanes == 6u);
  CHECK(result.drained_worker_lanes == result.submitted_worker_lanes);
  CHECK_FALSE(result.all_lanes_accepted);
  CHECK(allocation_count == 0u);
  for (size_t row = 0u; row < 7u; ++row) {
    CHECK(output[row] == doctest::Approx(2.0f));
  }
  CHECK(output[7] == doctest::Approx(-1.0f));

  release_blocker.store(true, std::memory_order_release);
  CHECK(blocker_group.wait());

  output.fill(-1.0f);
  result = {};
  accepted = false;
  REQUIRE(actor.process_event(run));
  CHECK(accepted);
  CHECK(result.all_submitted);
  CHECK(result.submitted_worker_lanes == 7u);
  CHECK(result.drained_worker_lanes == 7u);
  CHECK(result.all_lanes_accepted);
  for (const float value : output) {
    CHECK(value == doctest::Approx(2.0f));
  }
}

TEST_CASE(
    "parallel matmul capability accepts only supported fixed lane counts") {
  matmul::lane_pool two_lane_pool = {};
  auto two_lane_policy = matmul::make_execution_policy(
      two_lane_pool, emel::kernel::detect_host_kind(), 2u);
  CHECK(two_lane_policy.mode == matmul::lane_mode::parallel);
  CHECK(matmul::guard::supports_parallel_execution(two_lane_policy));
  two_lane_policy.active_lanes = 1u;
  CHECK_FALSE(matmul::guard::supports_parallel_execution(two_lane_policy));

  auto three_lane_policy = matmul::make_execution_policy(
      two_lane_pool, emel::kernel::detect_host_kind(), 3u);
  CHECK(three_lane_policy.mode == matmul::lane_mode::serial);
  CHECK_FALSE(matmul::guard::supports_parallel_execution(three_lane_policy));
  three_lane_policy.active_lanes = 2u;
  CHECK_FALSE(matmul::guard::supports_parallel_execution(three_lane_policy));

  auto disengaged_policy = matmul::make_execution_policy(
      two_lane_pool, emel::kernel::detect_host_kind(), 2u);
  disengaged_policy.parallel_matmul_lanes = nullptr;
  CHECK_FALSE(matmul::guard::supports_parallel_execution(disengaged_policy));
}

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

  std::fill(out_parallel.begin(), out_parallel.end(), -2.0f);
  matmul::event::dispatch_result result = {};
  bool accepted = false;
  outcome_counts outcomes{};
  matmul::event::execute_parallel run{parallel_ev, result, accepted};
  run.on_done = {&outcomes, count_done};
  run.on_error = {&outcomes, count_error};
  bool dispatched = false;
  size_t allocation_count = 0u;
  {
    emel::test::allocation::allocation_scope allocations{};
    dispatched = fixture.matmul_actor.process_event(run);
    allocation_count = allocations.allocations();
  }

  CHECK(dispatched);
  CHECK(accepted);
  CHECK(result.all_submitted);
  CHECK(result.submitted_worker_lanes == result.drained_worker_lanes);
  CHECK(result.all_lanes_accepted);
  CHECK(result.lane_count > 1u);
  CHECK(outcomes.done == 1);
  CHECK(outcomes.error == 0);
  CHECK(outcomes.done_request == &run);
  CHECK(allocation_count == 0u);
  CHECK(std::memcmp(out_serial.data(), out_parallel.data(),
                    out_serial.size() * sizeof(float)) == 0);

  parallel_ev.src0.ne[1] = 0u;
  matmul::event::execute_parallel rejected{parallel_ev, result, accepted};
  rejected.on_done = {&outcomes, count_done};
  rejected.on_error = {&outcomes, count_error};
  REQUIRE(fixture.matmul_actor.process_event(rejected));
  CHECK_FALSE(accepted);
  CHECK(outcomes.done == 1);
  CHECK(outcomes.error == 1);
  CHECK(outcomes.error_request == &rejected);
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
  ctx.compute.backend.n_embd =
      emel::text::generator::test::k_generation_route_policy
          .parallel_min_gemv_dim;
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

  emel::kernel::matmul::lane_pool serial_matmul_lanes = {};
  emel::kernel::matmul::execution_policy serial_policy{
      .parallel_matmul_lanes = &serial_matmul_lanes,
      .kernel_kind = emel::kernel::detect_host_kind(),
      .active_lanes = 1u,
  };
  emel::kernel::matmul::sm serial_matmul_actor{serial_policy};
  ctx.compute.backend.matmul_actor = &serial_matmul_actor;
  ctx.compute.backend.matmul_lane_mode =
      emel::kernel::matmul::lane_mode::serial;
  CHECK_FALSE(emel::text::generator::guard::guard_decode_parallel_lanes_ready{}(
      run, ctx));

  emel::kernel::matmul::lane_pool parallel_matmul_lanes = {};
  auto parallel_policy =
      emel::kernel::matmul::make_auto_execution_policy(parallel_matmul_lanes);
  emel::kernel::matmul::sm parallel_matmul_actor{parallel_policy};
  ctx.compute.backend.matmul_actor = &parallel_matmul_actor;
  ctx.compute.backend.matmul_lane_mode =
      emel::kernel::matmul::lane_mode::parallel;
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
  ctx.benchmark_parallel_lanes_enabled = true;
  emel::text::generator::action::effect_disable_parallel_benchmark_lanes(single,
                                                                         ctx);
  CHECK_FALSE(ctx.compute.backend.parallel_lanes_enabled);
  CHECK_FALSE(ctx.benchmark_parallel_lanes_enabled);
  ctx.compute.backend.parallel_lanes_enabled = true;
  emel::text::generator::action::apply_benchmark_lane_policy(ctx);
  CHECK_FALSE(ctx.compute.backend.parallel_lanes_enabled);
  emel::text::generator::action::effect_enable_parallel_benchmark_lanes(
      multithreaded, ctx);
  CHECK(ctx.compute.backend.parallel_lanes_enabled);
  CHECK(ctx.benchmark_parallel_lanes_enabled);
}

} // namespace
