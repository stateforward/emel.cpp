#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cmath>

#include "test_helpers.hpp"
#include "emel/kernel/x86_64/actions.hpp"
#include "emel/kernel/x86_64/detail.hpp"
#include "emel/kernel/x86_64/sm.hpp"

namespace {

using x86_64_sm = emel::kernel::x86_64::sm;
using emel::kernel::test::dtype;
using emel::kernel::test::make_dst;
using emel::kernel::test::make_src;

}  // namespace

TEST_CASE("kernel_x86_64_numeric_paths") {
  float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float rhs[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float out_add[4] = {};
  float out_mul[4] = {};

  const emel::kernel::event::op_add add_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(out_add, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_mul mul_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(out_mul, dtype::f32, 4),
      .nth = 1,
  };

  x86_64_sm machine{emel::kernel::x86_64::action::context{false, 0}};

  CHECK(machine.process_event(add_ev));
  CHECK(machine.process_event(mul_ev));

  CHECK(out_add[0] == doctest::Approx(6.0f));
  CHECK(out_add[1] == doctest::Approx(8.0f));
  CHECK(out_add[2] == doctest::Approx(10.0f));
  CHECK(out_add[3] == doctest::Approx(12.0f));

  CHECK(out_mul[0] == doctest::Approx(5.0f));
  CHECK(out_mul[1] == doctest::Approx(12.0f));
  CHECK(out_mul[2] == doctest::Approx(21.0f));
  CHECK(out_mul[3] == doctest::Approx(32.0f));
}

TEST_CASE("kernel_x86_64_scalar_path_honors_strides") {
  float lhs_storage[8] = {1.0f, 91.0f, 2.0f, 92.0f, 3.0f, 93.0f, 4.0f, 94.0f};
  float rhs_storage[8] = {10.0f, 81.0f, 20.0f, 82.0f, 30.0f, 83.0f, 40.0f, 84.0f};
  float dst_storage[8] = {};

  auto lhs = make_src(lhs_storage, dtype::f32, 4);
  auto rhs = make_src(rhs_storage, dtype::f32, 4);
  auto dst = make_dst(dst_storage, dtype::f32, 4);

  lhs.nb[0] = sizeof(float) * 2;
  lhs.nb[1] = lhs.nb[0] * lhs.ne[0];
  rhs.nb[0] = sizeof(float) * 2;
  rhs.nb[1] = rhs.nb[0] * rhs.ne[0];
  dst.nb[0] = sizeof(float) * 2;
  dst.nb[1] = dst.nb[0] * dst.ne[0];

  const emel::kernel::event::op_add add_ev{
      .src0 = lhs,
      .src1 = rhs,
      .dst = dst,
      .nth = 1,
  };

  x86_64_sm machine{emel::kernel::x86_64::action::context{false, 0}};
  CHECK(machine.process_event(add_ev));

  CHECK(dst_storage[0] == doctest::Approx(11.0f));
  CHECK(dst_storage[2] == doctest::Approx(22.0f));
  CHECK(dst_storage[4] == doctest::Approx(33.0f));
  CHECK(dst_storage[6] == doctest::Approx(44.0f));
}

TEST_CASE("kernel_x86_64_rejects_non_single_thread_dispatch") {
  float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float rhs[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float out[4] = {};

  emel::kernel::event::op_add invalid_nth{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(out, dtype::f32, 4),
      .nth = 2,
  };
  emel::kernel::event::op_add invalid_ith = invalid_nth;
  invalid_ith.nth = 1;
  invalid_ith.ith = 1;

  x86_64_sm machine{};
  CHECK_FALSE(machine.process_event(invalid_nth));
  CHECK_FALSE(machine.process_event(invalid_ith));
}

TEST_CASE("kernel_x86_64_forced_avx2_context_path") {
  if (!emel::kernel::x86_64::detail::avx2_intrinsics_compiled ||
      !emel::kernel::x86_64::detail::detect_avx2()) {
    return;
  }

  float lhs[4] = {2.0f, 4.0f, 6.0f, 8.0f};
  float rhs[4] = {1.0f, 3.0f, 5.0f, 7.0f};
  float out[4] = {};

  const emel::kernel::event::op_add add_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(out, dtype::f32, 4),
      .nth = 1,
  };

  x86_64_sm machine{emel::kernel::x86_64::action::context{true, 0}};
  CHECK(machine.process_event(add_ev));
  CHECK(out[0] == doctest::Approx(3.0f));
  CHECK(out[1] == doctest::Approx(7.0f));
  CHECK(out[2] == doctest::Approx(11.0f));
  CHECK(out[3] == doctest::Approx(15.0f));
}

TEST_CASE("kernel_x86_64_unary_subop_supported_and_unsupported_paths") {
  float src[4] = {-2.0f, -1.0f, 1.0f, 2.0f};
  float dst[4] = {};

  emel::kernel::event::op_unary unary_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::neg,
  };

  x86_64_sm machine{};
  CHECK(machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(2.0f));
  CHECK(dst[1] == doctest::Approx(1.0f));
  CHECK(dst[2] == doctest::Approx(-1.0f));
  CHECK(dst[3] == doctest::Approx(-2.0f));

  unary_ev.subop = emel::kernel::event::unary_subop::relu;
  CHECK(machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(0.0f));
  CHECK(dst[1] == doctest::Approx(0.0f));
  CHECK(dst[2] == doctest::Approx(1.0f));
  CHECK(dst[3] == doctest::Approx(2.0f));

  unary_ev.subop = emel::kernel::event::unary_subop::exp;
  CHECK(machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(std::exp(-2.0f)));
  CHECK(dst[1] == doctest::Approx(std::exp(-1.0f)));
  CHECK(dst[2] == doctest::Approx(std::exp(1.0f)));
  CHECK(dst[3] == doctest::Approx(std::exp(2.0f)));

  unary_ev.subop = emel::kernel::event::unary_subop::tanh;
  CHECK_FALSE(machine.process_event(unary_ev));

  x86_64_sm scalar_machine{emel::kernel::x86_64::action::context{false, 0}};

  unary_ev.subop = emel::kernel::event::unary_subop::abs;
  CHECK(scalar_machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(2.0f));
  CHECK(dst[1] == doctest::Approx(1.0f));
  CHECK(dst[2] == doctest::Approx(1.0f));
  CHECK(dst[3] == doctest::Approx(2.0f));

  unary_ev.subop = emel::kernel::event::unary_subop::neg;
  CHECK(scalar_machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(2.0f));
  CHECK(dst[1] == doctest::Approx(1.0f));
  CHECK(dst[2] == doctest::Approx(-1.0f));
  CHECK(dst[3] == doctest::Approx(-2.0f));

  unary_ev.subop = emel::kernel::event::unary_subop::relu;
  CHECK(scalar_machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(0.0f));
  CHECK(dst[1] == doctest::Approx(0.0f));
  CHECK(dst[2] == doctest::Approx(1.0f));
  CHECK(dst[3] == doctest::Approx(2.0f));

  unary_ev.subop = emel::kernel::event::unary_subop::exp;
  CHECK(scalar_machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(std::exp(-2.0f)));
  CHECK(dst[1] == doctest::Approx(std::exp(-1.0f)));
  CHECK(dst[2] == doctest::Approx(std::exp(1.0f)));
  CHECK(dst[3] == doctest::Approx(std::exp(2.0f)));

  unary_ev.subop = emel::kernel::event::unary_subop::tanh;
  CHECK_FALSE(scalar_machine.process_event(unary_ev));
}

TEST_CASE("kernel_x86_64_rejects_unimplemented_ops") {
  float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dst[4] = {};

  const emel::kernel::event::op_sum sum_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };

  x86_64_sm machine{};
  CHECK_FALSE(machine.process_event(sum_ev));
}

TEST_CASE("kernel_x86_64_mul_mat_simd_matches_scalar_tiled_edges") {
  const bool host_avx2 = emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
                         emel::kernel::x86_64::detail::detect_avx2();
  if (!host_avx2) {
    return;
  }

  constexpr uint64_t k = 131;
  constexpr uint64_t m = 7;
  constexpr uint64_t n = 73;

  std::array<float, k * m> src0{};
  std::array<float, k * n> src1{};
  std::array<float, n * m> dst_simd{};
  std::array<float, n * m> dst_scalar{};

  for (uint64_t i = 0; i < src0.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 17u) - 8;
    src0[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.0625f;
  }
  for (uint64_t i = 0; i < src1.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 23u) - 11;
    src1[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.03125f;
  }

  const emel::kernel::event::op_mul_mat simd_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_simd.data(), dtype::f32, n, m),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_scalar.data(), dtype::f32, n, m),
      .nth = 1,
  };

  CHECK(emel::kernel::x86_64::detail::execute_avx2_mul_mat(simd_ev));
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  for (uint64_t idx = 0; idx < dst_simd.size(); ++idx) {
    CHECK(dst_simd[static_cast<size_t>(idx)] ==
          doctest::Approx(dst_scalar[static_cast<size_t>(idx)]).epsilon(1e-5f));
  }
}

TEST_CASE("kernel_x86_64_detail_branch_paths") {
  const bool host_avx2 = emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
                         emel::kernel::x86_64::detail::detect_avx2();

  float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float rhs[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  float dst[4] = {};

  emel::kernel::event::op_add add_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };

  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(add_ev, false));
  CHECK(emel::kernel::x86_64::detail::can_use_avx2(add_ev, host_avx2) == host_avx2);
  CHECK(emel::kernel::x86_64::detail::execute_request(
      add_ev, emel::kernel::x86_64::action::context{host_avx2, 0}));

  add_ev.dst.nb[0] = add_ev.dst.nb[0] * 2;
  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(add_ev, host_avx2));

  add_ev.dst.nb[0] = emel::kernel::detail::dtype_size_bytes(
      emel::kernel::detail::dtype_code(add_ev.dst.type));
  add_ev.src1.type = dtype::q4_0;
  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(add_ev, host_avx2));

  add_ev.src1.type = dtype::f32;
  add_ev.src1.nb[0] = add_ev.src1.nb[0] * 2;
  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(add_ev, host_avx2));

  add_ev.src1 = make_src(rhs, dtype::f32, 4);
  add_ev.src0.type = dtype::q4_0;
  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(add_ev, host_avx2));

  emel::kernel::event::op_unary unary_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::relu,
  };
  CHECK(emel::kernel::x86_64::detail::can_use_avx2(unary_ev, host_avx2) == host_avx2);
  unary_ev.subop = emel::kernel::event::unary_subop::exp;
  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(unary_ev, host_avx2));
}

TEST_CASE("kernel_x86_64_detail_helper_edge_paths") {
  const bool host_avx2 = emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
                         emel::kernel::x86_64::detail::detect_avx2();

  float src0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dst0[4] = {};

  auto src = make_src(src0, dtype::f32, 4);

  CHECK(emel::kernel::x86_64::detail::is_dense_contiguous(src));
  src.nb[0] = 0;
  CHECK(emel::kernel::x86_64::detail::is_dense_contiguous(src));
  src.nb[0] = 8;
  CHECK_FALSE(emel::kernel::x86_64::detail::is_dense_contiguous(src));

  src = make_src(src0, dtype::q4_0, 4);
  CHECK_FALSE(emel::kernel::x86_64::detail::is_dense_contiguous(src));

  const emel::kernel::event::op_dup dup_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_add add_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_mul mul_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_div div_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_sqr sqr_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_sqrt sqrt_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };
  const emel::kernel::event::op_sub sub_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };
  float src_mm0[8] = {1.0f, 0.5f, -1.0f, 2.0f, 0.0f, -0.5f, 3.0f, 1.0f};
  float src_mm1[32] = {
      1.0f,  0.0f,  0.5f, -1.0f, 0.5f, 1.0f, -0.5f, 2.0f,
      0.0f,  1.0f,  1.0f,  0.0f, 2.0f, 0.5f,  0.0f, 1.0f,
      -1.0f, 2.0f,  0.0f,  1.0f, 1.5f, 0.0f,  2.0f, -0.5f,
      2.0f,  -1.0f, 1.0f,  0.5f, 0.0f, 1.0f, -1.0f, 1.0f,
  };
  float dst_mm[16] = {};
  const emel::kernel::event::op_mul_mat mul_mat_ev{
      .src0 = make_src(src_mm0, dtype::f32, 4, 2),
      .src1 = make_src(src_mm1, dtype::f32, 8, 4),
      .dst = make_dst(dst_mm, dtype::f32, 8, 2),
      .nth = 1,
  };
  emel::kernel::event::op_unary unary_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::relu,
  };

  if (host_avx2) {
    CHECK(emel::kernel::x86_64::detail::execute_avx2_dup(dup_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_add(add_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_sub(sub_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_mul(mul_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_div(div_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_sqr(sqr_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_sqrt(sqrt_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_mul_mat(mul_mat_ev));
    CHECK(emel::kernel::x86_64::detail::execute_avx2_unary(unary_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(dup_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(add_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(sub_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(mul_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(div_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(sqr_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(sqrt_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(mul_mat_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(unary_ev));
  }
#if !(defined(__x86_64__) || defined(_M_X64))
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_dup(dup_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_add(add_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_sub(sub_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_mul(mul_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_div(div_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_sqr(sqr_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_sqrt(sqrt_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_mul_mat(mul_mat_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_avx2_unary(unary_ev));
  emel::kernel::x86_64::detail::execute_simd_unchecked(dup_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(add_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(sub_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(mul_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(div_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(sqr_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(sqrt_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(mul_mat_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(unary_ev);
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(dup_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(add_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(sub_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(mul_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(div_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(sqr_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(sqrt_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(mul_mat_ev));
  CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(unary_ev));
#endif

  unary_ev.subop = emel::kernel::event::unary_subop::exp;
  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(unary_ev, host_avx2));
}

TEST_CASE("kernel_x86_64_simd_action_exec_marks_done") {
  float src0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float src1[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  float dst[4] = {};

  const emel::kernel::event::op_add add_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src1, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };

  emel::kernel::x86_64::event::dispatch_ctx dispatch_ctx{};
  emel::kernel::x86_64::action::context ctx{false, 0};
  const emel::kernel::x86_64::event::dispatch_op_add dispatch_ev{add_ev, dispatch_ctx};

  emel::kernel::x86_64::action::exec_simd_op_add(dispatch_ev, ctx);

  CHECK(dispatch_ctx.outcome == emel::kernel::x86_64::events::phase_outcome::done);
  CHECK(dispatch_ctx.err ==
        static_cast<int32_t>(emel::error::cast(emel::kernel::x86_64::error::none)));
}
