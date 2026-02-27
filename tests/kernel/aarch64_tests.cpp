#include <doctest/doctest.h>

#include "test_helpers.hpp"
#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/aarch64/detail.hpp"
#include "emel/kernel/aarch64/sm.hpp"

namespace {

using aarch64_sm = emel::kernel::aarch64::sm;
using emel::kernel::test::dtype;
using emel::kernel::test::make_dst;
using emel::kernel::test::make_src;

}  // namespace

TEST_CASE("kernel_aarch64_numeric_paths") {
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

  aarch64_sm machine{emel::kernel::aarch64::action::context{false, 0}};

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

TEST_CASE("kernel_aarch64_forced_neon_context_path") {
  float lhs[4] = {2.0f, 4.0f, 6.0f, 8.0f};
  float rhs[4] = {1.0f, 3.0f, 5.0f, 7.0f};
  float out[4] = {};

  const emel::kernel::event::op_add add_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(out, dtype::f32, 4),
      .nth = 1,
  };

  aarch64_sm machine{emel::kernel::aarch64::action::context{true, 0}};
  CHECK(machine.process_event(add_ev));
  CHECK(out[0] == doctest::Approx(3.0f));
  CHECK(out[1] == doctest::Approx(7.0f));
  CHECK(out[2] == doctest::Approx(11.0f));
  CHECK(out[3] == doctest::Approx(15.0f));
}

TEST_CASE("kernel_aarch64_detail_branch_paths") {
  float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float rhs[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  float dst[4] = {};

  emel::kernel::event::op_add add_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };

  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(add_ev, false));
#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(emel::kernel::aarch64::detail::can_use_neon(add_ev, true));
#else
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(add_ev, true));
#endif
  CHECK(emel::kernel::aarch64::detail::execute_request(
      add_ev, emel::kernel::aarch64::action::context{true, 0}));

  add_ev.dst.nb[0] = add_ev.dst.nb[0] * 2;
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(add_ev, true));

  add_ev.dst.nb[0] = emel::kernel::detail::dtype_size_bytes(
      emel::kernel::detail::dtype_code(add_ev.dst.type));
  add_ev.src1.type = dtype::q4_0;
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(add_ev, true));

  add_ev.src1.type = dtype::f32;
  add_ev.src1.nb[0] = add_ev.src1.nb[0] * 2;
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(add_ev, true));

  add_ev.src1 = make_src(rhs, dtype::f32, 4);
  add_ev.src0.type = dtype::q4_0;
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(add_ev, true));
}

TEST_CASE("kernel_aarch64_detail_helper_edge_paths") {
  float src0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dst0[4] = {};

  auto src = make_src(src0, dtype::f32, 4);

  CHECK(emel::kernel::aarch64::detail::is_dense_contiguous(src));
  src.nb[0] = 0;
  CHECK(emel::kernel::aarch64::detail::is_dense_contiguous(src));
  src.nb[0] = 8;
  CHECK_FALSE(emel::kernel::aarch64::detail::is_dense_contiguous(src));

  src = make_src(src0, dtype::q4_0, 4);
  CHECK_FALSE(emel::kernel::aarch64::detail::is_dense_contiguous(src));

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
  const emel::kernel::event::op_sub sub_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
      .nth = 1,
  };

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(emel::kernel::aarch64::detail::execute_neon_dup(dup_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_add(add_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_mul(mul_ev));
#else
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_dup(dup_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_add(add_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_mul(mul_ev));
#endif

  const bool simd_dup = emel::kernel::aarch64::detail::execute_simd(dup_ev);
  const bool simd_add = emel::kernel::aarch64::detail::execute_simd(add_ev);
  const bool simd_mul = emel::kernel::aarch64::detail::execute_simd(mul_ev);
  (void) simd_dup;
  (void) simd_add;
  (void) simd_mul;
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_simd(sub_ev));
}
