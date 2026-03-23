#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cmath>
#include <vector>

#include "../allocation_tracker.hpp"
#include "test_helpers.hpp"
#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/aarch64/detail.hpp"
#include "emel/kernel/aarch64/events.hpp"
#include "emel/kernel/aarch64/sm.hpp"

namespace {

using aarch64_sm = emel::kernel::aarch64::sm;
using allocation_scope = emel::test::allocation::allocation_scope;
using emel::kernel::test::dtype;
using emel::kernel::test::flash_attn_ext_fixture;
using emel::kernel::test::make_dst;
using emel::kernel::test::make_flash_attn_ext_event;
using emel::kernel::test::make_quantized_src;
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

  aarch64_sm machine{emel::kernel::aarch64::action::context{false, {}, 0}};

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

TEST_CASE("kernel_aarch64_scalar_path_honors_strides") {
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

  aarch64_sm machine{emel::kernel::aarch64::action::context{false, {}, 0}};
  CHECK(machine.process_event(add_ev));

  CHECK(dst_storage[0] == doctest::Approx(11.0f));
  CHECK(dst_storage[2] == doctest::Approx(22.0f));
  CHECK(dst_storage[4] == doctest::Approx(33.0f));
  CHECK(dst_storage[6] == doctest::Approx(44.0f));
}

TEST_CASE("kernel_aarch64_rejects_non_single_thread_dispatch") {
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

  aarch64_sm machine{};
  CHECK_FALSE(machine.process_event(invalid_nth));
  CHECK_FALSE(machine.process_event(invalid_ith));
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

  aarch64_sm machine{emel::kernel::aarch64::action::context{true, {}, 0}};
  CHECK(machine.process_event(add_ev));
  CHECK(out[0] == doctest::Approx(3.0f));
  CHECK(out[1] == doctest::Approx(7.0f));
  CHECK(out[2] == doctest::Approx(11.0f));
  CHECK(out[3] == doctest::Approx(15.0f));
}

TEST_CASE("kernel_aarch64_mul_mat_simd_matches_scalar_tiled_edges") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t k = 131;
  constexpr uint64_t m = 7;
  constexpr uint64_t n = 73;

  std::array<float, k * m> src0{};
  std::array<float, k * n> src1{};
  std::array<float, n * m> dst_simd{};
  std::array<float, n * m> dst_scalar{};

  for (uint64_t i = 0; i < src0.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 19u) - 9;
    src0[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.0625f;
  }
  for (uint64_t i = 0; i < src1.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 29u) - 14;
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

  CHECK(emel::kernel::aarch64::detail::execute_neon_mul_mat(simd_ev));
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  for (uint64_t idx = 0; idx < dst_simd.size(); ++idx) {
    CHECK(dst_simd[static_cast<size_t>(idx)] ==
          doctest::Approx(dst_scalar[static_cast<size_t>(idx)]).epsilon(1e-5f));
  }
#endif
}

TEST_CASE("kernel_aarch64_quantized_mul_mat_simd_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q3_k;
  using emel::kernel::detail::quant::block_q6_k;

  const std::array<float, QK_K> src1 = [] {
    std::array<float, QK_K> values = {};
    for (size_t i = 0; i < values.size(); ++i) {
      const int32_t centered = static_cast<int32_t>(i % 17u) - 8;
      values[i] = static_cast<float>(centered) * 0.125f;
    }
    return values;
  }();

  block_q2_k q2 = {};
  q2.d = 0x3c00u;
  q2.dmin = 0x3c00u;
  for (size_t i = 0; i < q2.scales.size(); ++i) {
    q2.scales[i] = static_cast<uint8_t>(((i % 11u) << 4u) | ((i * 3u) % 13u));
  }
  for (size_t i = 0; i < q2.qs.size(); ++i) {
    q2.qs[i] = static_cast<uint8_t>((i * 29u) ^ (i >> 1u));
  }

  block_q3_k q3 = {};
  q3.d = 0x3c00u;
  for (size_t i = 0; i < q3.scales.size(); ++i) {
    q3.scales[i] = static_cast<uint8_t>((i * 17u) ^ 0x5au);
  }
  for (size_t i = 0; i < q3.hmask.size(); ++i) {
    q3.hmask[i] = static_cast<uint8_t>((i * 9u) ^ 0xa5u);
  }
  for (size_t i = 0; i < q3.qs.size(); ++i) {
    q3.qs[i] = static_cast<uint8_t>((i * 13u) ^ 0x33u);
  }

  block_q6_k q6 = {};
  q6.d = 0x3c00u;
  for (size_t i = 0; i < q6.scales.size(); ++i) {
    q6.scales[i] = static_cast<int8_t>((static_cast<int>(i % 7u) - 3) * 9);
  }
  for (size_t i = 0; i < q6.ql.size(); ++i) {
    q6.ql[i] = static_cast<uint8_t>((i * 7u) ^ 0x96u);
  }
  for (size_t i = 0; i < q6.qh.size(); ++i) {
    q6.qh[i] = static_cast<uint8_t>((i * 5u) ^ 0x69u);
  }

  std::array<float, 1> dst_simd = {};
  std::array<float, 1> dst_scalar = {};

  auto run_case = [&](const auto & block, const dtype type) {
    dst_simd.fill(0.0f);
    dst_scalar.fill(0.0f);
    const emel::kernel::event::op_mul_mat simd_ev{
        .src0 = make_quantized_src(&block, type, QK_K, 1),
        .src1 = make_src(src1.data(), dtype::f32, 1, QK_K),
        .dst = make_dst(dst_simd.data(), dtype::f32, 1, 1),
        .nth = 1,
    };
    const emel::kernel::event::op_mul_mat scalar_ev{
        .src0 = make_quantized_src(&block, type, QK_K, 1),
        .src1 = make_src(src1.data(), dtype::f32, 1, QK_K),
        .dst = make_dst(dst_scalar.data(), dtype::f32, 1, 1),
        .nth = 1,
    };

    CHECK(emel::kernel::aarch64::detail::can_use_neon(simd_ev, true));
    CHECK(emel::kernel::aarch64::detail::execute_neon_mul_mat(simd_ev));
    CHECK(emel::kernel::detail::execute_scalar(scalar_ev));
    CHECK(dst_simd[0] == doctest::Approx(dst_scalar[0]).epsilon(1e-5f));
  };

  run_case(q2, dtype::q2_k);
  run_case(q3, dtype::q3_k);
  run_case(q6, dtype::q6_k);
#endif
}

TEST_CASE("kernel_aarch64_q2_row_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q8_k;

  block_q2_k q2 = {};
  q2.d = 0x3c00u;
  q2.dmin = 0x3c00u;
  for (size_t i = 0; i < q2.scales.size(); ++i) {
    q2.scales[i] = static_cast<uint8_t>(((i % 9u) << 4u) | ((i * 5u) % 13u));
  }
  for (size_t i = 0; i < q2.qs.size(); ++i) {
    q2.qs[i] = static_cast<uint8_t>((i * 23u) ^ (i >> 2u));
  }

  std::array<float, QK_K> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>(i % 19u) - 9;
    src1[i] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<block_q8_k, 1> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), 1, &q8_blocks[0], emel::kernel::detail::quant::QK_K);

  const float scalar =
      emel::kernel::detail::dot_q2_k_q8_k_row_scalar(&q2, q8_blocks.data(), q8_blocks.size());
  const float neon =
      emel::kernel::aarch64::detail::dot_q2_k_q8_k_row_neon(&q2, q8_blocks.data(),
                                                             q8_blocks.size());

  CHECK(neon == doctest::Approx(scalar).epsilon(1e-5f));
#endif
}

TEST_CASE("kernel_aarch64_q3_row_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q3_k;
  using emel::kernel::detail::quant::block_q8_k;

  block_q3_k q3 = {};
  q3.d = 0x3c00u;
  for (size_t i = 0; i < q3.scales.size(); ++i) {
    q3.scales[i] = static_cast<uint8_t>((i * 17u) ^ 0x5au);
  }
  for (size_t i = 0; i < q3.hmask.size(); ++i) {
    q3.hmask[i] = static_cast<uint8_t>((i * 9u) ^ 0xa5u);
  }
  for (size_t i = 0; i < q3.qs.size(); ++i) {
    q3.qs[i] = static_cast<uint8_t>((i * 13u) ^ 0x33u);
  }

  std::array<float, QK_K> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>(i % 23u) - 11;
    src1[i] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<block_q8_k, 1> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), 1, &q8_blocks[0], emel::kernel::detail::quant::QK_K);

  const float scalar =
      emel::kernel::detail::dot_q3_k_q8_k_row_scalar(&q3, q8_blocks.data(), q8_blocks.size());
  const float neon =
      emel::kernel::aarch64::detail::dot_q3_k_q8_k_row_neon(&q3, q8_blocks.data(),
                                                             q8_blocks.size());

  CHECK(neon == doctest::Approx(scalar).epsilon(1e-5f));
#endif
}

TEST_CASE("kernel_aarch64_q6_row_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q6_k;
  using emel::kernel::detail::quant::block_q8_k;

  block_q6_k q6 = {};
  q6.d = 0x3c00u;
  for (size_t i = 0; i < q6.scales.size(); ++i) {
    q6.scales[i] = static_cast<int8_t>(static_cast<int32_t>(i % 15u) - 7);
  }
  for (size_t i = 0; i < q6.ql.size(); ++i) {
    q6.ql[i] = static_cast<uint8_t>((i * 19u) ^ 0x6cu);
  }
  for (size_t i = 0; i < q6.qh.size(); ++i) {
    q6.qh[i] = static_cast<uint8_t>((i * 7u) ^ 0x95u);
  }

  std::array<float, QK_K> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>(i % 29u) - 14;
    src1[i] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<block_q8_k, 1> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), 1, &q8_blocks[0], emel::kernel::detail::quant::QK_K);

  const float scalar =
      emel::kernel::detail::dot_q6_k_q8_k_row_scalar(&q6, q8_blocks.data(), q8_blocks.size());
  const float neon =
      emel::kernel::aarch64::detail::dot_q6_k_q8_k_row_neon(&q6, q8_blocks.data(),
                                                             q8_blocks.size());

  CHECK(neon == doctest::Approx(scalar).epsilon(1e-5f));
#endif
}

TEST_CASE("kernel_aarch64_sm_reports_q2_vectorized_dispatch_at_kernel_seam") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q3_k;

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    for (size_t i = 0; i < values.size(); ++i) {
      const int32_t centered = static_cast<int32_t>(i % 13u) - 6;
      values[i] = static_cast<float>(centered) * 0.125f;
    }
    return values;
  }();

  block_q2_k q2 = {};
  q2.d = 0x3c00u;
  q2.dmin = 0x3c00u;
  for (size_t i = 0; i < q2.scales.size(); ++i) {
    q2.scales[i] = static_cast<uint8_t>(((i % 7u) << 4u) | ((i * 3u) % 11u));
  }
  for (size_t i = 0; i < q2.qs.size(); ++i) {
    q2.qs[i] = static_cast<uint8_t>((i * 17u) ^ (i >> 1u));
  }

  block_q3_k q3 = {};
  q3.d = 0x3c00u;
  for (size_t i = 0; i < q3.scales.size(); ++i) {
    q3.scales[i] = static_cast<uint8_t>((i * 11u) ^ 0x4du);
  }
  for (size_t i = 0; i < q3.hmask.size(); ++i) {
    q3.hmask[i] = static_cast<uint8_t>((i * 5u) ^ 0xb2u);
  }
  for (size_t i = 0; i < q3.qs.size(); ++i) {
    q3.qs[i] = static_cast<uint8_t>((i * 7u) ^ 0x39u);
  }

  float q2_out[1] = {};
  float q3_out[1] = {};
  const emel::kernel::event::op_mul_mat q2_ev{
      .src0 = make_quantized_src(&q2, dtype::q2_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q2_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat q3_ev{
      .src0 = make_quantized_src(&q3, dtype::q3_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q3_out, dtype::f32, 1, 1),
      .nth = 1,
  };

  aarch64_sm machine{};
  CHECK(machine.process_event(q2_ev));
  CHECK(machine.process_event(q3_ev));

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(machine.optimized_q2_dispatch_count() == 1u);
  CHECK(machine.shared_q2_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q2_dispatch_count() == 0u);
  CHECK(machine.shared_q2_dispatch_count() == 1u);
#endif
}

TEST_CASE("kernel_aarch64_sm_reports_q3_vectorized_dispatch_at_kernel_seam") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q3_k;

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    for (size_t i = 0; i < values.size(); ++i) {
      const int32_t centered = static_cast<int32_t>(i % 17u) - 8;
      values[i] = static_cast<float>(centered) * 0.125f;
    }
    return values;
  }();

  block_q2_k q2 = {};
  q2.d = 0x3c00u;
  q2.dmin = 0x3c00u;
  for (size_t i = 0; i < q2.scales.size(); ++i) {
    q2.scales[i] = static_cast<uint8_t>(((i % 7u) << 4u) | ((i * 3u) % 11u));
  }
  for (size_t i = 0; i < q2.qs.size(); ++i) {
    q2.qs[i] = static_cast<uint8_t>((i * 17u) ^ (i >> 1u));
  }

  block_q3_k q3 = {};
  q3.d = 0x3c00u;
  for (size_t i = 0; i < q3.scales.size(); ++i) {
    q3.scales[i] = static_cast<uint8_t>((i * 11u) ^ 0x4du);
  }
  for (size_t i = 0; i < q3.hmask.size(); ++i) {
    q3.hmask[i] = static_cast<uint8_t>((i * 5u) ^ 0xb2u);
  }
  for (size_t i = 0; i < q3.qs.size(); ++i) {
    q3.qs[i] = static_cast<uint8_t>((i * 7u) ^ 0x39u);
  }

  float q2_out[1] = {};
  float q3_out[1] = {};
  const emel::kernel::event::op_mul_mat q2_ev{
      .src0 = make_quantized_src(&q2, dtype::q2_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q2_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat q3_ev{
      .src0 = make_quantized_src(&q3, dtype::q3_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q3_out, dtype::f32, 1, 1),
      .nth = 1,
  };

  aarch64_sm machine{};
  CHECK(machine.process_event(q2_ev));
  CHECK(machine.process_event(q3_ev));

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(machine.optimized_q3_dispatch_count() == 1u);
  CHECK(machine.shared_q3_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q3_dispatch_count() == 0u);
  CHECK(machine.shared_q3_dispatch_count() == 1u);
#endif
}

TEST_CASE("kernel_aarch64_sm_reports_q6_vectorized_dispatch_at_kernel_seam") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q3_k;
  using emel::kernel::detail::quant::block_q6_k;

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    for (size_t i = 0; i < values.size(); ++i) {
      const int32_t centered = static_cast<int32_t>(i % 19u) - 9;
      values[i] = static_cast<float>(centered) * 0.125f;
    }
    return values;
  }();

  block_q2_k q2 = {};
  q2.d = 0x3c00u;
  q2.dmin = 0x3c00u;
  for (size_t i = 0; i < q2.scales.size(); ++i) {
    q2.scales[i] = static_cast<uint8_t>(((i % 7u) << 4u) | ((i * 3u) % 11u));
  }
  for (size_t i = 0; i < q2.qs.size(); ++i) {
    q2.qs[i] = static_cast<uint8_t>((i * 17u) ^ (i >> 1u));
  }

  block_q3_k q3 = {};
  q3.d = 0x3c00u;
  for (size_t i = 0; i < q3.scales.size(); ++i) {
    q3.scales[i] = static_cast<uint8_t>((i * 11u) ^ 0x4du);
  }
  for (size_t i = 0; i < q3.hmask.size(); ++i) {
    q3.hmask[i] = static_cast<uint8_t>((i * 5u) ^ 0xb2u);
  }
  for (size_t i = 0; i < q3.qs.size(); ++i) {
    q3.qs[i] = static_cast<uint8_t>((i * 7u) ^ 0x39u);
  }

  block_q6_k q6 = {};
  q6.d = 0x3c00u;
  for (size_t i = 0; i < q6.scales.size(); ++i) {
    q6.scales[i] = static_cast<int8_t>(static_cast<int32_t>(i % 15u) - 7);
  }
  for (size_t i = 0; i < q6.ql.size(); ++i) {
    q6.ql[i] = static_cast<uint8_t>((i * 19u) ^ 0x6cu);
  }
  for (size_t i = 0; i < q6.qh.size(); ++i) {
    q6.qh[i] = static_cast<uint8_t>((i * 7u) ^ 0x95u);
  }

  float q2_out[1] = {};
  float q3_out[1] = {};
  float q6_out[1] = {};
  const emel::kernel::event::op_mul_mat q2_ev{
      .src0 = make_quantized_src(&q2, dtype::q2_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q2_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat q3_ev{
      .src0 = make_quantized_src(&q3, dtype::q3_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q3_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat q6_ev{
      .src0 = make_quantized_src(&q6, dtype::q6_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q6_out, dtype::f32, 1, 1),
      .nth = 1,
  };

  aarch64_sm machine{};
  CHECK(machine.process_event(q2_ev));
  CHECK(machine.process_event(q3_ev));
  CHECK(machine.process_event(q6_ev));

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(machine.optimized_q6_dispatch_count() == 1u);
  CHECK(machine.shared_q6_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q6_dispatch_count() == 0u);
  CHECK(machine.shared_q6_dispatch_count() == 1u);
#endif
}

TEST_CASE("kernel_aarch64_supported_quantized_dispatch_is_alloc_free") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q3_k;
  using emel::kernel::detail::quant::block_q6_k;

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    for (size_t i = 0; i < values.size(); ++i) {
      const int32_t centered = static_cast<int32_t>(i % 21u) - 10;
      values[i] = static_cast<float>(centered) * 0.125f;
    }
    return values;
  }();

  block_q2_k q2 = {};
  q2.d = 0x3c00u;
  q2.dmin = 0x3c00u;
  std::fill(q2.scales.begin(), q2.scales.end(), static_cast<uint8_t>(0x11u));
  std::fill(q2.qs.begin(), q2.qs.end(), static_cast<uint8_t>(0x22u));

  block_q3_k q3 = {};
  q3.d = 0x3c00u;
  std::fill(q3.scales.begin(), q3.scales.end(), static_cast<uint8_t>(0x33u));
  std::fill(q3.hmask.begin(), q3.hmask.end(), static_cast<uint8_t>(0x44u));
  std::fill(q3.qs.begin(), q3.qs.end(), static_cast<uint8_t>(0x55u));

  block_q6_k q6 = {};
  q6.d = 0x3c00u;
  std::fill(q6.scales.begin(), q6.scales.end(), static_cast<int8_t>(3));
  std::fill(q6.ql.begin(), q6.ql.end(), static_cast<uint8_t>(0x66u));
  std::fill(q6.qh.begin(), q6.qh.end(), static_cast<uint8_t>(0x77u));

  float q2_out[1] = {};
  float q3_out[1] = {};
  float q6_out[1] = {};
  const emel::kernel::event::op_mul_mat q2_ev{
      .src0 = make_quantized_src(&q2, dtype::q2_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q2_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat q3_ev{
      .src0 = make_quantized_src(&q3, dtype::q3_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q3_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat q6_ev{
      .src0 = make_quantized_src(&q6, dtype::q6_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(q6_out, dtype::f32, 1, 1),
      .nth = 1,
  };

  aarch64_sm machine{};
  allocation_scope allocations{};
  CHECK(machine.process_event(q2_ev));
  CHECK(machine.process_event(q3_ev));
  CHECK(machine.process_event(q6_ev));
  CHECK(allocations.allocations() == 0u);

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(machine.optimized_q2_dispatch_count() == 1u);
  CHECK(machine.shared_q2_dispatch_count() == 0u);
  CHECK(machine.optimized_q3_dispatch_count() == 1u);
  CHECK(machine.shared_q3_dispatch_count() == 0u);
  CHECK(machine.optimized_q6_dispatch_count() == 1u);
  CHECK(machine.shared_q6_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q2_dispatch_count() == 0u);
  CHECK(machine.shared_q2_dispatch_count() == 1u);
  CHECK(machine.optimized_q3_dispatch_count() == 0u);
  CHECK(machine.shared_q3_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_dispatch_count() == 0u);
  CHECK(machine.shared_q6_dispatch_count() == 1u);
#endif
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
      add_ev, emel::kernel::aarch64::action::context{true, {}, 0}));

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

  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  block_q2_k q2 = {};
  q2.d = 0x3c00u;
  q2.dmin = 0x3c00u;
  std::fill(q2.scales.begin(), q2.scales.end(), static_cast<uint8_t>(0x11u));
  std::fill(q2.qs.begin(), q2.qs.end(), static_cast<uint8_t>(0x00u));
  float quant_dst[1] = {};
  const float quant_rhs[QK_K] = {0.0f};
  const emel::kernel::event::op_mul_mat quant_mul_mat_ev{
      .src0 = make_quantized_src(&q2, dtype::q2_k, QK_K, 1),
      .src1 = make_src(quant_rhs, dtype::f32, 1, QK_K),
      .dst = make_dst(quant_dst, dtype::f32, 1, 1),
      .nth = 1,
  };
#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(emel::kernel::aarch64::detail::can_use_neon(quant_mul_mat_ev, true));
#else
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(quant_mul_mat_ev, true));
#endif

  emel::kernel::event::op_unary unary_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::relu,
  };
#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(emel::kernel::aarch64::detail::can_use_neon(unary_ev, true));
#else
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(unary_ev, true));
#endif
  unary_ev.subop = emel::kernel::event::unary_subop::exp;
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(unary_ev, true));
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

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(emel::kernel::aarch64::detail::execute_neon_dup(dup_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_add(add_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_sub(sub_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_mul(mul_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_div(div_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_sqr(sqr_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_sqrt(sqrt_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_mul_mat(mul_mat_ev));
  CHECK(emel::kernel::aarch64::detail::execute_neon_unary(unary_ev));
#else
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_dup(dup_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_add(add_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_sub(sub_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_mul(mul_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_div(div_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_sqr(sqr_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_sqrt(sqrt_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_mul_mat(mul_mat_ev));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_unary(unary_ev));
#endif

  const bool simd_dup = emel::kernel::aarch64::detail::execute_simd(dup_ev);
  const bool simd_add = emel::kernel::aarch64::detail::execute_simd(add_ev);
  const bool simd_sub = emel::kernel::aarch64::detail::execute_simd(sub_ev);
  const bool simd_mul = emel::kernel::aarch64::detail::execute_simd(mul_ev);
  const bool simd_div = emel::kernel::aarch64::detail::execute_simd(div_ev);
  const bool simd_sqr = emel::kernel::aarch64::detail::execute_simd(sqr_ev);
  const bool simd_sqrt = emel::kernel::aarch64::detail::execute_simd(sqrt_ev);
  const bool simd_mul_mat = emel::kernel::aarch64::detail::execute_simd(mul_mat_ev);
  const bool simd_unary = emel::kernel::aarch64::detail::execute_simd(unary_ev);
  (void) simd_dup;
  (void) simd_add;
  (void) simd_sub;
  (void) simd_mul;
  (void) simd_div;
  (void) simd_sqr;
  (void) simd_sqrt;
  (void) simd_mul_mat;
  (void) simd_unary;

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(simd_dup);
  CHECK(simd_add);
  CHECK(simd_sub);
  CHECK(simd_mul);
  CHECK(simd_div);
  CHECK(simd_sqr);
  CHECK(simd_sqrt);
  CHECK(simd_mul_mat);
  CHECK(simd_unary);
#else
  CHECK_FALSE(simd_dup);
  CHECK_FALSE(simd_add);
  CHECK_FALSE(simd_sub);
  CHECK_FALSE(simd_mul);
  CHECK_FALSE(simd_div);
  CHECK_FALSE(simd_sqr);
  CHECK_FALSE(simd_sqrt);
  CHECK_FALSE(simd_mul_mat);
  CHECK_FALSE(simd_unary);
#endif

  unary_ev.subop = emel::kernel::event::unary_subop::exp;
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(unary_ev, true));
}

TEST_CASE("kernel_aarch64_rejects_unimplemented_ops") {
  float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dst[4] = {};

  const emel::kernel::event::op_sum sum_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
  };

  aarch64_sm machine{};
  CHECK_FALSE(machine.process_event(sum_ev));
}

TEST_CASE("kernel_aarch64_unary_subop_scalar_paths") {
  float src[4] = {-2.0f, -1.0f, 1.0f, 2.0f};
  float dst[4] = {};

  emel::kernel::event::op_unary unary_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
      .nth = 1,
      .subop = emel::kernel::event::unary_subop::abs,
  };

  aarch64_sm scalar_machine{emel::kernel::aarch64::action::context{false, {}, 0}};

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

TEST_CASE("kernel_aarch64_flash_attn_ext_reuses_persistent_workspace") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  flash_attn_ext_fixture fixture{};
  const auto request = make_flash_attn_ext_event(fixture);

  emel::kernel::aarch64::action::context ctx{};
  emel::kernel::aarch64::event::dispatch_ctx dispatch_ctx0{};
  const emel::kernel::aarch64::event::dispatch_op_flash_attn_ext dispatch0{request, dispatch_ctx0};
  const float * scratch = ctx.flash_attn_workspace.score_buffer.data();

  emel::kernel::aarch64::action::exec_op_flash_attn_ext(dispatch0, ctx);
  CHECK(dispatch_ctx0.outcome == emel::kernel::aarch64::events::phase_outcome::done);
  CHECK(ctx.optimized_flash_dispatch_count == 1u);
  CHECK(ctx.shared_flash_dispatch_count == 0u);
  CHECK(ctx.flash_attn_workspace.prepared_tokens == 2u);
  CHECK(ctx.flash_attn_workspace.reuse_count == 0u);
  CHECK(ctx.flash_attn_workspace.score_buffer.data() == scratch);

  fixture.dst[0] = 0.0f;
  fixture.dst[1] = 0.0f;
  fixture.dst[2] = 0.0f;
  fixture.dst[3] = 0.0f;

  emel::kernel::aarch64::event::dispatch_ctx dispatch_ctx1{};
  const emel::kernel::aarch64::event::dispatch_op_flash_attn_ext dispatch1{request, dispatch_ctx1};

  emel::kernel::aarch64::action::exec_op_flash_attn_ext(dispatch1, ctx);
  CHECK(dispatch_ctx1.outcome == emel::kernel::aarch64::events::phase_outcome::done);
  CHECK(ctx.optimized_flash_dispatch_count == 2u);
  CHECK(ctx.shared_flash_dispatch_count == 0u);
  CHECK(ctx.flash_attn_workspace.prepared_tokens == 2u);
  CHECK(ctx.flash_attn_workspace.reuse_count == 1u);
  CHECK(ctx.flash_attn_workspace.score_buffer.data() == scratch);
#endif
}

TEST_CASE("kernel_aarch64_flash_attn_ext_uses_optimized_backend_path") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  flash_attn_ext_fixture fixture{};
  const auto request = make_flash_attn_ext_event(fixture);

  emel::kernel::aarch64::action::context ctx{};
  emel::kernel::aarch64::event::dispatch_ctx dispatch_ctx{};
  const emel::kernel::aarch64::event::dispatch_op_flash_attn_ext dispatch{request, dispatch_ctx};

  emel::kernel::aarch64::action::exec_op_flash_attn_ext(dispatch, ctx);

  CHECK(dispatch_ctx.outcome == emel::kernel::aarch64::events::phase_outcome::done);
  CHECK(ctx.optimized_flash_dispatch_count == 1u);
  CHECK(ctx.shared_flash_dispatch_count == 0u);
  CHECK(fixture.dst[0] == doctest::Approx(1.4621172f).epsilon(1e-5f));
  CHECK(fixture.dst[1] == doctest::Approx(1.0757657f).epsilon(1e-5f));
  CHECK(fixture.dst[2] == doctest::Approx(0.0f));
  CHECK(fixture.dst[3] == doctest::Approx(0.0f));
#endif
}

TEST_CASE("kernel_aarch64_flash_attn_ext_matches_shared_workspace_on_long_kv_spans") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t head_dim = 64u;
  constexpr uint64_t head_count = 12u;
  constexpr uint64_t kv_head_count = 12u;
  constexpr uint64_t kv_tokens = 128u;
  const uint64_t kv_dim = head_dim * kv_head_count;

  std::vector<float> q(head_dim * head_count);
  std::vector<float> k(kv_dim * kv_tokens);
  std::vector<float> v(kv_dim * kv_tokens);
  std::vector<float> dst_neon(head_dim * head_count, -1.0f);
  std::vector<float> dst_shared(head_dim * head_count, -1.0f);

  for (uint64_t head = 0; head < head_count; ++head) {
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      const double angle = static_cast<double>((head + 1u) * (dim + 3u));
      q[head * head_dim + dim] = static_cast<float>(std::sin(angle * 0.03125));
    }
  }

  for (uint64_t token = 0; token < kv_tokens; ++token) {
    for (uint64_t head = 0; head < kv_head_count; ++head) {
      for (uint64_t dim = 0; dim < head_dim; ++dim) {
        const uint64_t offset = token * kv_dim + head * head_dim + dim;
        const double base = static_cast<double>((token + 1u) * (head + 3u) * (dim + 5u));
        k[offset] = static_cast<float>(std::cos(base * 0.0078125));
        v[offset] = static_cast<float>(std::sin(base * 0.01171875));
      }
    }
  }

  emel::kernel::event::op_flash_attn_ext request{};
  request.src0 = make_src(q.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1 = make_src(k.data(), dtype::f32, head_dim, kv_tokens, kv_head_count);
  request.src2 = make_src(v.data(), dtype::f32, head_dim, kv_tokens, kv_head_count);
  request.dst = make_dst(dst_neon.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1.nb[1] = sizeof(float) * kv_dim;
  request.src1.nb[2] = sizeof(float) * head_dim;
  request.src2.nb[1] = sizeof(float) * kv_dim;
  request.src2.nb[2] = sizeof(float) * head_dim;
  request.nth = 1;

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  request.op_params_size = sizeof(scale);

  emel::kernel::detail::flash_attn_workspace neon_workspace{};
  emel::kernel::detail::flash_attn_workspace shared_workspace{};

  REQUIRE(emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
      request, true, neon_workspace));

  request.dst = make_dst(dst_shared.data(), dtype::f32, head_dim, 1u, head_count);
  REQUIRE(emel::kernel::detail::run_flash_attn_ext_with_workspace(request, shared_workspace));

  for (size_t idx = 0; idx < dst_neon.size(); ++idx) {
    CHECK(dst_neon[idx] == doctest::Approx(dst_shared[idx]).epsilon(1e-5f));
  }
#endif
}
