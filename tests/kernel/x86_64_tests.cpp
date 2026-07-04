#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include "../allocation_tracker.hpp"
#include "emel/kernel/x86_64/actions.hpp"
#include "emel/kernel/x86_64/detail.hpp"
#include "emel/kernel/x86_64/sm.hpp"
#include "test_helpers.hpp"

namespace {

using x86_64_sm = emel::kernel::x86_64::sm;
using allocation_scope = emel::test::allocation::allocation_scope;
using emel::kernel::test::dtype;
using emel::kernel::test::flash_attn_ext_fixture;
using emel::kernel::test::flash_attn_reference_f16_scores;
using emel::kernel::test::flash_attn_reference_masked_total_tokens;
using emel::kernel::test::k_flash_online_f16_abs_tolerance;
using emel::kernel::test::make_dst;
using emel::kernel::test::make_flash_attn_ext_event;
using emel::kernel::test::make_quantized_src;
using emel::kernel::test::make_src;
using emel::kernel::test::to_fp16_storage;
using emel::kernel::test::within_flash_online_f16_tolerance;

using emel::kernel::detail::quant::block_q2_k;
using emel::kernel::detail::quant::block_q3_k;
using emel::kernel::detail::quant::block_q4_0;
using emel::kernel::detail::quant::block_q4_1;
using emel::kernel::detail::quant::block_q4_k;
using emel::kernel::detail::quant::block_q5_0;
using emel::kernel::detail::quant::block_q6_k;
using emel::kernel::detail::quant::block_q8_0;
using emel::kernel::detail::quant::block_q8_k;
using emel::kernel::detail::quant::QK4_0;
using emel::kernel::detail::quant::QK4_1;
using emel::kernel::detail::quant::QK5_0;
using emel::kernel::detail::quant::QK8_0;
using emel::kernel::detail::quant::QK_K;

template <size_t block_count>
std::array<float, QK_K * block_count>
make_quantized_rhs_values(const uint32_t salt) {
  std::array<float, QK_K * block_count> values = {};
  for (size_t i = 0; i < values.size(); ++i) {
    const int32_t centered =
        static_cast<int32_t>(((i + salt) * (7u + salt)) % 41u) - 20;
    values[i] = static_cast<float>(centered) * 0.03125f;
  }
  return values;
}

void fill_q2_block(block_q2_k &q2, const uint32_t salt) {
  q2.d = static_cast<uint16_t>(0x3c00u + (salt % 17u));
  q2.dmin = static_cast<uint16_t>(0x3800u + (salt % 11u));
  for (size_t i = 0; i < q2.scales.size(); ++i) {
    q2.scales[i] = static_cast<uint8_t>((((i + salt) % 13u) << 4u) |
                                        (((i * 5u) + salt) % 15u));
  }
  for (size_t i = 0; i < q2.qs.size(); ++i) {
    q2.qs[i] = static_cast<uint8_t>((i * (23u + salt)) ^ ((i + salt) >> 1u));
  }
}

void fill_q3_block(block_q3_k &q3, const uint32_t salt) {
  q3.d = static_cast<uint16_t>(0x3c00u + (salt % 19u));
  for (size_t i = 0; i < q3.scales.size(); ++i) {
    q3.scales[i] = static_cast<uint8_t>((i * (17u + salt)) ^ (0x5au + salt));
  }
  for (size_t i = 0; i < q3.hmask.size(); ++i) {
    q3.hmask[i] = static_cast<uint8_t>((i * (9u + salt)) ^ (0xa5u - salt));
  }
  for (size_t i = 0; i < q3.qs.size(); ++i) {
    q3.qs[i] = static_cast<uint8_t>((i * (13u + salt)) ^ (0x33u + salt * 7u));
  }
}

void fill_q4_block(block_q4_k &q4, const uint32_t salt) {
  q4.d = static_cast<uint16_t>(0x3c00u + (salt % 17u));
  q4.dmin = static_cast<uint16_t>(0x3800u + (salt % 13u));
  for (size_t i = 0; i < q4.scales.size(); ++i) {
    q4.scales[i] = static_cast<uint8_t>((i * (11u + salt)) ^ (0x47u + salt));
  }
  for (size_t i = 0; i < q4.qs.size(); ++i) {
    q4.qs[i] = static_cast<uint8_t>((i * (5u + salt)) ^ (0x9du - salt));
  }
}

void fill_q6_block(block_q6_k &q6, const uint32_t salt) {
  q6.d = static_cast<uint16_t>(0x3c00u + (salt % 23u));
  for (size_t i = 0; i < q6.scales.size(); ++i) {
    const int32_t scale_value =
        static_cast<int32_t>(((i + salt) * 3u) % 31u) - 15;
    q6.scales[i] = static_cast<int8_t>(scale_value);
  }
  for (size_t i = 0; i < q6.ql.size(); ++i) {
    q6.ql[i] = static_cast<uint8_t>((i * (19u + salt)) ^ (0x6cu + salt));
  }
  for (size_t i = 0; i < q6.qh.size(); ++i) {
    q6.qh[i] = static_cast<uint8_t>((i * (7u + salt)) ^ (0x95u - salt));
  }
}

void fill_q4_0_block(block_q4_0 &q4, const uint32_t salt) {
  q4.d = static_cast<uint16_t>(0x3c00u + (salt % 17u));
  for (size_t i = 0; i < q4.qs.size(); ++i) {
    q4.qs[i] = static_cast<uint8_t>((i * (7u + salt)) ^ (0x53u + salt));
  }
}

void fill_q4_1_block(block_q4_1 &q4, const uint32_t salt) {
  q4.d = static_cast<uint16_t>(0x3c00u + (salt % 19u));
  q4.m = static_cast<uint16_t>(0x3800u + (salt % 11u));
  for (size_t i = 0; i < q4.qs.size(); ++i) {
    q4.qs[i] = static_cast<uint8_t>((i * (11u + salt)) ^ (0x2eu + salt));
  }
}

void fill_q5_0_block(block_q5_0 &q5, const uint32_t salt) {
  q5.d = static_cast<uint16_t>(0x3c00u + (salt % 13u));
  for (size_t i = 0; i < q5.qh.size(); ++i) {
    q5.qh[i] = static_cast<uint8_t>((i * (29u + salt)) ^ (0xb4u - salt));
  }
  for (size_t i = 0; i < q5.qs.size(); ++i) {
    q5.qs[i] = static_cast<uint8_t>((i * (13u + salt)) ^ (0x71u + salt));
  }
}

void fill_q8_0_block(block_q8_0 &q8, const uint32_t salt) {
  q8.d = static_cast<uint16_t>(0x3c00u + (salt % 23u));
  for (size_t i = 0; i < q8.qs.size(); ++i) {
    const int32_t centered =
        static_cast<int32_t>(((i + salt) * (5u + salt)) % 251u) - 125;
    q8.qs[i] = static_cast<int8_t>(centered);
  }
}

emel::kernel::x86_64::detail::host_feature_contract
avx2_fma_contract(const bool enabled) {
  return emel::kernel::x86_64::detail::host_feature_contract{
      .avx2_available = enabled,
      .fma_available = enabled,
      .f16c_available = enabled,
  };
}

bool host_has_avx2_fma() {
  return emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
         emel::kernel::x86_64::detail::detect_avx2() &&
         emel::kernel::x86_64::detail::detect_fma();
}

bool host_has_avx2_fma_f16c() {
  return host_has_avx2_fma() && emel::kernel::x86_64::detail::detect_f16c();
}

} // namespace

TEST_CASE("kernel_x86_64_numeric_paths") {
  float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float rhs[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float out_add[4] = {};
  float out_mul[4] = {};

  const emel::kernel::event::op_add add_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(out_add, dtype::f32, 4),
  };
  const emel::kernel::event::op_mul mul_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(out_mul, dtype::f32, 4),
  };

  x86_64_sm machine{emel::kernel::x86_64::action::context{false, {}, 0}};

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
  float rhs_storage[8] = {10.0f, 81.0f, 20.0f, 82.0f,
                          30.0f, 83.0f, 40.0f, 84.0f};
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
  };

  x86_64_sm machine{emel::kernel::x86_64::action::context{false, {}, 0}};
  CHECK(machine.process_event(add_ev));

  CHECK(dst_storage[0] == doctest::Approx(11.0f));
  CHECK(dst_storage[2] == doctest::Approx(22.0f));
  CHECK(dst_storage[4] == doctest::Approx(33.0f));
  CHECK(dst_storage[6] == doctest::Approx(44.0f));
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
  };

  x86_64_sm machine{emel::kernel::x86_64::action::context{true, {}, 0}};
  CHECK(machine.process_event(add_ev));
  CHECK(out[0] == doctest::Approx(3.0f));
  CHECK(out[1] == doctest::Approx(7.0f));
  CHECK(out[2] == doctest::Approx(11.0f));
  CHECK(out[3] == doctest::Approx(15.0f));
}

TEST_CASE("kernel_x86_64_host_feature_contract_is_published") {
  const emel::kernel::x86_64::detail::host_feature_contract contract{
      .avx2_available = true,
      .fma_available = true,
      .f16c_available = true,
  };

  const x86_64_sm machine{
      emel::kernel::x86_64::action::context{contract, {}, 0}};

  CHECK(machine.avx2_available());
  CHECK(machine.fma_available());
  CHECK(machine.f16c_available());
  CHECK(machine.avx2_fma_f16c_available());
  CHECK_FALSE(machine.avx512_claimed());
  CHECK_FALSE(machine.avx_vnni_claimed());
  CHECK_FALSE(machine.amx_claimed());
  CHECK_FALSE(machine.bf16_claimed());
  CHECK_FALSE(machine.native_fp16_claimed());
}

TEST_CASE("kernel_x86_64_q2_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 4u;
  std::array<block_q2_k, block_count> q2_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q2_block(q2_blocks[block], static_cast<uint32_t>(block + 1u));
  }

  const auto rhs_values = make_quantized_rhs_values<block_count>(3u);
  std::array<block_q8_k, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      rhs_values.data(), 1u, q8_blocks.data(),
      static_cast<int64_t>(QK_K * block_count));

  const float scalar = emel::kernel::detail::dot_q2_k_q8_k_row_scalar(
      q2_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q2_k_q8_k_row_avx2_fma(
          q2_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-6f));
}

TEST_CASE("kernel_x86_64_q3_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 4u;
  std::array<block_q3_k, block_count> q3_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q3_block(q3_blocks[block], static_cast<uint32_t>(block + 5u));
  }

  const auto rhs_values = make_quantized_rhs_values<block_count>(7u);
  std::array<block_q8_k, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      rhs_values.data(), 1u, q8_blocks.data(),
      static_cast<int64_t>(QK_K * block_count));

  const float scalar = emel::kernel::detail::dot_q3_k_q8_k_row_scalar(
      q3_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q3_k_q8_k_row_avx2_fma(
          q3_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-6f));
}

TEST_CASE("kernel_x86_64_q4_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 4u;
  std::array<block_q4_k, block_count> q4_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q4_block(q4_blocks[block], static_cast<uint32_t>(block + 9u));
  }

  const auto rhs_values = make_quantized_rhs_values<block_count>(13u);
  std::array<block_q8_k, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      rhs_values.data(), 1u, q8_blocks.data(),
      static_cast<int64_t>(QK_K * block_count));

  const float scalar = emel::kernel::detail::dot_q4_k_q8_k_row_scalar(
      q4_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q4_k_q8_k_row_avx2_fma(
          q4_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-6f));
}

TEST_CASE("kernel_x86_64_q2_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 2u;
  constexpr uint64_t k = QK_K * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q2_k, rows * block_count> q2_rows = {};
  for (size_t idx = 0; idx < q2_rows.size(); ++idx) {
    fill_q2_block(q2_rows[idx], static_cast<uint32_t>(idx + 11u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 5u) % 31u) - 15;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q2_rows.data(), dtype::q2_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q2_rows.data(), dtype::q2_k, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q2_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q2_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q2_rows.data(), dtype::q2_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q2_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q2_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
  }
}

TEST_CASE("kernel_x86_64_q3_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 2u;
  constexpr uint64_t k = QK_K * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q3_k, rows * block_count> q3_rows = {};
  for (size_t idx = 0; idx < q3_rows.size(); ++idx) {
    fill_q3_block(q3_rows[idx], static_cast<uint32_t>(idx + 19u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 7u) % 37u) - 18;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q3_rows.data(), dtype::q3_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q3_rows.data(), dtype::q3_k, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q3_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q3_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q3_rows.data(), dtype::q3_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q3_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q3_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
  }
}

TEST_CASE("kernel_x86_64_q4_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 2u;
  constexpr uint64_t k = QK_K * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q4_k, rows * block_count> q4_rows = {};
  for (size_t idx = 0; idx < q4_rows.size(); ++idx) {
    fill_q4_block(q4_rows[idx], static_cast<uint32_t>(idx + 29u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 7u) % 37u) - 18;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q4_rows.data(), dtype::q4_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q4_rows.data(), dtype::q4_k, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q4_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q4_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q4_rows.data(), dtype::q4_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q4_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q4_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
  }
}

TEST_CASE("kernel_x86_64_q4_0_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 8u;
  constexpr size_t k = QK4_0 * block_count;
  std::array<block_q4_0, block_count> q4_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q4_0_block(q4_blocks[block], static_cast<uint32_t>(block + 3u));
  }

  std::array<float, k> rhs_values = {};
  for (size_t i = 0; i < rhs_values.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 5u) % 43u) - 21;
    rhs_values[i] = static_cast<float>(centered) * 0.03125f;
  }
  std::array<block_q8_0, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      rhs_values.data(), 1u, q8_blocks.data(), static_cast<int64_t>(k));

  const float scalar = emel::kernel::detail::dot_q4_0_q8_0_row_scalar(
      q4_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q4_0_q8_0_row_avx2_fma(
          q4_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-6f));
}

TEST_CASE("kernel_x86_64_q4_1_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 8u;
  constexpr size_t k = QK4_1 * block_count;
  std::array<block_q4_1, block_count> q4_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q4_1_block(q4_blocks[block], static_cast<uint32_t>(block + 5u));
  }

  std::array<float, k> rhs_values = {};
  for (size_t i = 0; i < rhs_values.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 7u) % 39u) - 19;
    rhs_values[i] = static_cast<float>(centered) * 0.03125f;
  }
  std::array<block_q8_0, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      rhs_values.data(), 1u, q8_blocks.data(), static_cast<int64_t>(k));

  const float scalar = emel::kernel::detail::dot_q4_1_q8_0_row_scalar(
      q4_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q4_1_q8_0_row_avx2_fma(
          q4_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-6f));
}

TEST_CASE("kernel_x86_64_q5_0_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 8u;
  constexpr size_t k = QK5_0 * block_count;
  std::array<block_q5_0, block_count> q5_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q5_0_block(q5_blocks[block], static_cast<uint32_t>(block + 7u));
  }

  std::array<float, k> rhs_values = {};
  for (size_t i = 0; i < rhs_values.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 11u) % 47u) - 23;
    rhs_values[i] = static_cast<float>(centered) * 0.03125f;
  }
  std::array<block_q8_0, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      rhs_values.data(), 1u, q8_blocks.data(), static_cast<int64_t>(k));

  const float scalar = emel::kernel::detail::dot_q5_0_q8_0_row_scalar(
      q5_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q5_0_q8_0_row_avx2_fma(
          q5_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-6f));
}

TEST_CASE("kernel_x86_64_q8_0_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 8u;
  constexpr size_t k = QK8_0 * block_count;
  std::array<block_q8_0, block_count> lhs_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q8_0_block(lhs_blocks[block], static_cast<uint32_t>(block + 9u));
  }

  std::array<float, k> rhs_values = {};
  for (size_t i = 0; i < rhs_values.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 13u) % 51u) - 25;
    rhs_values[i] = static_cast<float>(centered) * 0.03125f;
  }
  std::array<block_q8_0, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      rhs_values.data(), 1u, q8_blocks.data(), static_cast<int64_t>(k));

  const float scalar = emel::kernel::detail::dot_q8_0_q8_0_row_scalar(
      lhs_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q8_0_q8_0_row_avx2_fma(
          lhs_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-6f));
}

TEST_CASE("kernel_x86_64_q4_0_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 4u;
  constexpr uint64_t k = QK4_0 * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q4_0, rows * block_count> q4_rows = {};
  for (size_t idx = 0; idx < q4_rows.size(); ++idx) {
    fill_q4_0_block(q4_rows[idx], static_cast<uint32_t>(idx + 31u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 7u) % 37u) - 18;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q4_rows.data(), dtype::q4_0, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q4_rows.data(), dtype::q4_0, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q4_0_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q4_0_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q4_rows.data(), dtype::q4_0, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q4_0_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q4_0_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
  }
}

TEST_CASE("kernel_x86_64_q4_1_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 4u;
  constexpr uint64_t k = QK4_1 * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q4_1, rows * block_count> q4_rows = {};
  for (size_t idx = 0; idx < q4_rows.size(); ++idx) {
    fill_q4_1_block(q4_rows[idx], static_cast<uint32_t>(idx + 37u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 5u) % 33u) - 16;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q4_rows.data(), dtype::q4_1, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q4_rows.data(), dtype::q4_1, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q4_1_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q4_1_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q4_rows.data(), dtype::q4_1, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q4_1_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q4_1_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
  }
}

TEST_CASE("kernel_x86_64_q5_0_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 4u;
  constexpr uint64_t k = QK5_0 * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q5_0, rows * block_count> q5_rows = {};
  for (size_t idx = 0; idx < q5_rows.size(); ++idx) {
    fill_q5_0_block(q5_rows[idx], static_cast<uint32_t>(idx + 41u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 11u) % 41u) - 20;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q5_rows.data(), dtype::q5_0, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q5_rows.data(), dtype::q5_0, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q5_0_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q5_0_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q5_rows.data(), dtype::q5_0, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q5_0_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q5_0_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
  }
}

TEST_CASE("kernel_x86_64_q8_0_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 4u;
  constexpr uint64_t k = QK8_0 * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q8_0, rows * block_count> q8_rows = {};
  for (size_t idx = 0; idx < q8_rows.size(); ++idx) {
    fill_q8_0_block(q8_rows[idx], static_cast<uint32_t>(idx + 43u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 13u) % 45u) - 22;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q8_rows.data(), dtype::q8_0, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q8_rows.data(), dtype::q8_0, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q8_0_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q8_0_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q8_rows.data(), dtype::q8_0, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q8_0_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q8_0_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-6f));
  }
}

TEST_CASE("kernel_x86_64_q6_row_avx2_fma_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 4u;
  std::array<block_q6_k, block_count> q6_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    fill_q6_block(q6_blocks[block], static_cast<uint32_t>(block + 29u));
  }

  const auto rhs_values = make_quantized_rhs_values<block_count>(11u);
  std::array<block_q8_k, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      rhs_values.data(), 1u, q8_blocks.data(),
      static_cast<int64_t>(QK_K * block_count));

  const float scalar = emel::kernel::detail::dot_q6_k_q8_k_row_scalar(
      q6_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::x86_64::detail::dot_q6_k_q8_k_row_avx2_fma(
          q6_blocks.data(), q8_blocks.data(), block_count);

  CHECK(optimized == doctest::Approx(scalar).epsilon(1e-5f));
}

TEST_CASE("kernel_x86_64_q6_mul_mat_uses_optimized_and_shared_routes") {
  constexpr size_t block_count = 2u;
  constexpr uint64_t k = QK_K * block_count;
  constexpr uint64_t rows = 2u;
  constexpr uint64_t cols = 2u;

  std::array<block_q6_k, rows * block_count> q6_rows = {};
  for (size_t idx = 0; idx < q6_rows.size(); ++idx) {
    fill_q6_block(q6_rows[idx], static_cast<uint32_t>(idx + 37u));
  }

  std::array<float, k * cols> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 11u) % 43u) - 21;
    rhs[i] = static_cast<float>(centered) * 0.0625f;
  }

  float expected[rows * cols] = {};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q6_rows.data(), dtype::q6_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(expected, dtype::f32, cols, rows),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    float optimized_out[rows * cols] = {};
    const emel::kernel::event::op_mul_mat optimized_ev{
        .src0 = make_quantized_src(q6_rows.data(), dtype::q6_k, k, rows),
        .src1 = make_src(rhs.data(), dtype::f32, cols, k),
        .dst = make_dst(optimized_out, dtype::f32, cols, rows),
    };
    x86_64_sm optimized_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(optimized_machine.process_event(optimized_ev));
    CHECK(optimized_machine.optimized_q6_dispatch_count() == 1u);
    CHECK(optimized_machine.shared_q6_dispatch_count() == 0u);
    for (size_t i = 0; i < std::size(optimized_out); ++i) {
      CHECK(optimized_out[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
    }
  }

  float shared_out[rows * cols] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(q6_rows.data(), dtype::q6_k, k, rows),
      .src1 = make_src(rhs.data(), dtype::f32, cols, k),
      .dst = make_dst(shared_out, dtype::f32, cols, rows),
  };
  x86_64_sm shared_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(false), {}, 0}};

  CHECK(shared_machine.process_event(shared_ev));
  CHECK(shared_machine.optimized_q6_dispatch_count() == 0u);
  CHECK(shared_machine.shared_q6_dispatch_count() == 1u);
  for (size_t i = 0; i < std::size(shared_out); ++i) {
    CHECK(shared_out[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
  }
}

TEST_CASE("kernel_x86_64_quantized_hot_path_dispatches_without_allocation") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr size_t block_count = 1u;
  constexpr uint64_t k = QK_K * block_count;
  std::array<float, k> rhs = {};
  for (size_t i = 0; i < rhs.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 13u) % 47u) - 23;
    rhs[i] = static_cast<float>(centered) * 0.03125f;
  }

  block_q2_k q2 = {};
  block_q3_k q3 = {};
  block_q6_k q6 = {};
  fill_q2_block(q2, 43u);
  fill_q3_block(q3, 47u);
  fill_q6_block(q6, 53u);

  float q2_out[1] = {};
  float q3_out[1] = {};
  float q6_out[1] = {};
  const emel::kernel::event::op_mul_mat q2_ev{
      .src0 = make_quantized_src(&q2, dtype::q2_k, k, 1u),
      .src1 = make_src(rhs.data(), dtype::f32, 1u, k),
      .dst = make_dst(q2_out, dtype::f32, 1u, 1u),
  };
  const emel::kernel::event::op_mul_mat q3_ev{
      .src0 = make_quantized_src(&q3, dtype::q3_k, k, 1u),
      .src1 = make_src(rhs.data(), dtype::f32, 1u, k),
      .dst = make_dst(q3_out, dtype::f32, 1u, 1u),
  };
  const emel::kernel::event::op_mul_mat q6_ev{
      .src0 = make_quantized_src(&q6, dtype::q6_k, k, 1u),
      .src1 = make_src(rhs.data(), dtype::f32, 1u, k),
      .dst = make_dst(q6_out, dtype::f32, 1u, 1u),
  };

  x86_64_sm machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};
  allocation_scope allocations{};
  CHECK(machine.process_event(q2_ev));
  CHECK(machine.process_event(q3_ev));
  CHECK(machine.process_event(q6_ev));
  CHECK(allocations.allocations() == 0u);
  CHECK(machine.optimized_q2_dispatch_count() == 1u);
  CHECK(machine.shared_q2_dispatch_count() == 0u);
  CHECK(machine.optimized_q3_dispatch_count() == 1u);
  CHECK(machine.shared_q3_dispatch_count() == 0u);
  CHECK(machine.optimized_q6_dispatch_count() == 1u);
  CHECK(machine.shared_q6_dispatch_count() == 0u);
}

TEST_CASE("kernel_x86_64_flash_attn_ext_uses_optimized_backend_path") {
  const bool host_flash =
      emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
      emel::kernel::x86_64::detail::detect_avx2() &&
      emel::kernel::x86_64::detail::detect_fma() &&
      emel::kernel::x86_64::detail::detect_f16c();
  if (!host_flash) {
    return;
  }

  flash_attn_ext_fixture fixture{};
  const auto request = make_flash_attn_ext_event(fixture);
  const emel::kernel::x86_64::detail::host_feature_contract contract{
      .avx2_available = true,
      .fma_available = true,
      .f16c_available = true,
  };
  x86_64_sm machine{emel::kernel::x86_64::action::context{contract, {}, 0}};

  CHECK(machine.process_event(request));

  const auto q = std::span<const float>(fixture.q, request.src0.ne[0]);
  const auto k = std::span<const uint16_t>(fixture.k, request.src0.ne[0] *
                                                          request.src1.ne[1]);
  const auto v = std::span<const uint16_t>(fixture.v, request.src0.ne[0] *
                                                          request.src2.ne[1]);
  const std::vector<float> expected = flash_attn_reference_f16_scores(
      q, k, v, request.src0.ne[0], request.src1.ne[1], 1.0f);
  CHECK(machine.optimized_flash_dispatch_count() == 1u);
  CHECK(machine.shared_flash_dispatch_count() == 0u);
  CHECK(within_flash_online_f16_tolerance(fixture.dst[0], expected[0]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[1], expected[1]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[2], expected[2]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[3], expected[3]));
}

TEST_CASE(
    "kernel_x86_64_flash_attn_ext_falls_back_when_feature_contract_disabled") {
  flash_attn_ext_fixture fixture{};
  const auto request = make_flash_attn_ext_event(fixture);
  const emel::kernel::x86_64::detail::host_feature_contract contract{
      .avx2_available = false,
      .fma_available = false,
      .f16c_available = false,
  };
  x86_64_sm machine{emel::kernel::x86_64::action::context{contract, {}, 0}};

  CHECK(machine.process_event(request));

  const auto q = std::span<const float>(fixture.q, request.src0.ne[0]);
  const auto k = std::span<const uint16_t>(fixture.k, request.src0.ne[0] *
                                                          request.src1.ne[1]);
  const auto v = std::span<const uint16_t>(fixture.v, request.src0.ne[0] *
                                                          request.src2.ne[1]);
  const std::vector<float> expected = flash_attn_reference_f16_scores(
      q, k, v, request.src0.ne[0], request.src1.ne[1], 1.0f);
  CHECK(machine.optimized_flash_dispatch_count() == 0u);
  CHECK(machine.shared_flash_dispatch_count() == 1u);
  CHECK(within_flash_online_f16_tolerance(fixture.dst[0], expected[0]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[1], expected[1]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[2], expected[2]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[3], expected[3]));
}

TEST_CASE("kernel_x86_64_flash_attn_ext_reuses_persistent_workspace_on_"
          "optimized_path") {
  const bool host_flash =
      emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
      emel::kernel::x86_64::detail::detect_avx2() &&
      emel::kernel::x86_64::detail::detect_fma() &&
      emel::kernel::x86_64::detail::detect_f16c();
  if (!host_flash) {
    return;
  }

  flash_attn_ext_fixture fixture{};
  const auto request = make_flash_attn_ext_event(fixture);
  const emel::kernel::x86_64::detail::host_feature_contract contract{
      .avx2_available = true,
      .fma_available = true,
      .f16c_available = true,
  };
  x86_64_sm machine{emel::kernel::x86_64::action::context{contract, {}, 0}};

  CHECK(machine.process_event(request));
  CHECK(machine.flash_attn_workspace_prepared_tokens() == 2u);
  CHECK(machine.flash_attn_workspace_reuse_count() == 0u);

  std::fill_n(fixture.dst, request.dst.ne[0], 0.0f);
  CHECK(machine.process_event(request));
  CHECK(machine.optimized_flash_dispatch_count() == 2u);
  CHECK(machine.shared_flash_dispatch_count() == 0u);
  CHECK(machine.flash_attn_workspace_prepared_tokens() == 2u);
  CHECK(machine.flash_attn_workspace_reuse_count() == 1u);
}

TEST_CASE("kernel_x86_64_flash_attn_ext_matches_masked_total_token_reference") {
  if (!host_has_avx2_fma_f16c()) {
    return;
  }

  flash_attn_ext_fixture fixture{};
  auto request = make_flash_attn_ext_event(fixture);
  const uint32_t masked_total_tokens = 8u;
  const float scale = 1.0f;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  std::memcpy(request.op_params.data() + sizeof(scale), &masked_total_tokens,
              sizeof(masked_total_tokens));
  request.op_params_size = sizeof(scale) + sizeof(masked_total_tokens);
  const emel::kernel::x86_64::detail::host_feature_contract contract{
      .avx2_available = true,
      .fma_available = true,
      .f16c_available = true,
  };
  x86_64_sm machine{emel::kernel::x86_64::action::context{contract, {}, 0}};

  CHECK(machine.process_event(request));

  const auto q = std::span<const float>(fixture.q, request.src0.ne[0]);
  const auto k = std::span<const uint16_t>(fixture.k, request.src0.ne[0] *
                                                          request.src1.ne[1]);
  const auto v = std::span<const uint16_t>(fixture.v, request.src0.ne[0] *
                                                          request.src2.ne[1]);
  const std::vector<float> expected = flash_attn_reference_masked_total_tokens(
      q, k, v, request.src0.ne[0], request.src1.ne[1], masked_total_tokens,
      scale);
  CHECK(machine.optimized_flash_dispatch_count() == 1u);
  CHECK(machine.shared_flash_dispatch_count() == 0u);
  CHECK(fixture.dst[0] ==
        doctest::Approx(expected[0]).epsilon(k_flash_online_f16_abs_tolerance));
  CHECK(fixture.dst[1] ==
        doctest::Approx(expected[1]).epsilon(k_flash_online_f16_abs_tolerance));
  CHECK(fixture.dst[2] ==
        doctest::Approx(expected[2]).epsilon(k_flash_online_f16_abs_tolerance));
  CHECK(fixture.dst[3] ==
        doctest::Approx(expected[3]).epsilon(k_flash_online_f16_abs_tolerance));
}

TEST_CASE(
    "kernel_x86_64_flash_attn_ext_matches_masked_total_token_reference_on_"
    "long_multihead_kv") {
  if (!host_has_avx2_fma_f16c()) {
    return;
  }

  constexpr uint64_t head_dim = 64u;
  constexpr uint64_t head_count = 12u;
  constexpr uint64_t kv_head_count = 12u;
  constexpr uint64_t kv_tokens = 769u;
  constexpr uint32_t total_tokens = 2048u;
  const uint64_t kv_dim = head_dim * kv_head_count;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> q(head_dim * head_count);
  std::vector<float> k(kv_dim * kv_tokens);
  std::vector<float> v(kv_dim * kv_tokens);
  std::vector<float> dst(head_dim * head_count, 0.0f);

  for (uint64_t head = 0; head < head_count; ++head) {
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      const double angle = static_cast<double>((head + 1u) * (dim + 3u));
      q[head * head_dim + dim] = emel::kernel::detail::quant::fp16_to_fp32(
          emel::kernel::detail::quant::fp32_to_fp16(
              static_cast<float>(std::sin(angle * 0.03125))));
    }
  }

  for (uint64_t token = 0; token < kv_tokens; ++token) {
    for (uint64_t head = 0; head < kv_head_count; ++head) {
      for (uint64_t dim = 0; dim < head_dim; ++dim) {
        const uint64_t offset = token * kv_dim + head * head_dim + dim;
        const double base =
            static_cast<double>((token + 1u) * (head + 3u) * (dim + 5u));
        k[offset] = emel::kernel::detail::quant::fp16_to_fp32(
            emel::kernel::detail::quant::fp32_to_fp16(
                static_cast<float>(std::cos(base * 0.0078125))));
        v[offset] = emel::kernel::detail::quant::fp16_to_fp32(
            emel::kernel::detail::quant::fp32_to_fp16(
                static_cast<float>(std::sin(base * 0.01171875))));
      }
    }
  }
  const auto k_fp16 = to_fp16_storage(k);
  const auto v_fp16 = to_fp16_storage(v);

  emel::kernel::event::op_flash_attn_ext request{};
  request.src0 = make_src(q.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1 =
      make_src(k_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.src2 =
      make_src(v_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.dst = make_dst(dst.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src1.nb[2] = sizeof(uint16_t) * head_dim;
  request.src2.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src2.nb[2] = sizeof(uint16_t) * head_dim;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  std::memcpy(request.op_params.data() + sizeof(scale), &total_tokens,
              sizeof(total_tokens));
  request.op_params_size = sizeof(scale) + sizeof(total_tokens);

  x86_64_sm machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};
  REQUIRE(machine.process_event(request));
  CHECK(machine.optimized_flash_dispatch_count() == 1u);
  CHECK(machine.shared_flash_dispatch_count() == 0u);

  std::vector<float> expected(head_dim * head_count, 0.0f);
  for (uint64_t head = 0; head < head_count; ++head) {
    std::vector<float> k_head(kv_tokens * head_dim);
    std::vector<float> v_head(kv_tokens * head_dim);
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const uint64_t src_offset = token * kv_dim + head * head_dim;
      const uint64_t dst_offset = token * head_dim;
      std::memcpy(k_head.data() + dst_offset, k.data() + src_offset,
                  sizeof(float) * head_dim);
      std::memcpy(v_head.data() + dst_offset, v.data() + src_offset,
                  sizeof(float) * head_dim);
    }
    const auto expected_head = flash_attn_reference_masked_total_tokens(
        std::span<const float>(q.data() + head * head_dim, head_dim),
        std::span<const float>(k_head.data(), k_head.size()),
        std::span<const float>(v_head.data(), v_head.size()), head_dim,
        kv_tokens, total_tokens, scale);
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      expected[head * head_dim + dim] = expected_head[static_cast<size_t>(dim)];
    }
  }

  for (size_t idx = 0; idx < dst.size(); ++idx) {
    CHECK(within_flash_online_f16_tolerance(dst[idx], expected[idx]));
  }
}

TEST_CASE("kernel_x86_64_host_feature_contract_can_fail_closed") {
  const emel::kernel::x86_64::detail::host_feature_contract contract{};
  const x86_64_sm machine{
      emel::kernel::x86_64::action::context{contract, {}, 0}};

  CHECK_FALSE(machine.avx2_available());
  CHECK_FALSE(machine.fma_available());
  CHECK_FALSE(machine.f16c_available());
  CHECK_FALSE(machine.avx2_fma_f16c_available());
  CHECK_FALSE(machine.avx512_claimed());
  CHECK_FALSE(machine.avx_vnni_claimed());
  CHECK_FALSE(machine.amx_claimed());
  CHECK_FALSE(machine.bf16_claimed());
  CHECK_FALSE(machine.native_fp16_claimed());
}

TEST_CASE("kernel_x86_64_detects_host_feature_contract") {
  const auto contract =
      emel::kernel::x86_64::detail::detect_host_feature_contract();

  CHECK(contract.avx2_available == emel::kernel::x86_64::detail::detect_avx2());
  CHECK(contract.fma_available == emel::kernel::x86_64::detail::detect_fma());
  CHECK(contract.f16c_available == emel::kernel::x86_64::detail::detect_f16c());
  CHECK(contract.avx2_fma_f16c_available() ==
        (contract.avx2_available && contract.fma_available &&
         contract.f16c_available));
  CHECK_FALSE(contract.avx512_claimed);
  CHECK_FALSE(contract.avx_vnni_claimed);
  CHECK_FALSE(contract.amx_claimed);
  CHECK_FALSE(contract.bf16_claimed);
  CHECK_FALSE(contract.native_fp16_claimed);
}

TEST_CASE("kernel_x86_64_unary_subop_supported_and_unsupported_paths") {
  float src[4] = {-2.0f, -1.0f, 1.0f, 2.0f};
  float dst[4] = {};

  emel::kernel::event::op_unary unary_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
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
  CHECK(machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(std::tanh(-2.0f)));
  CHECK(dst[3] == doctest::Approx(std::tanh(2.0f)));

  unary_ev.subop = emel::kernel::event::unary_subop::elu;
  CHECK(machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(std::expm1(-2.0f)));
  CHECK(dst[3] == doctest::Approx(2.0f));

  unary_ev.subop = emel::kernel::event::unary_subop::silu;
  CHECK(machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(-2.0f / (1.0f + std::exp(2.0f))));
  CHECK(dst[3] == doctest::Approx(2.0f / (1.0f + std::exp(-2.0f))));

  unary_ev.subop = emel::kernel::event::unary_subop::gelu;
  CHECK(machine.process_event(unary_ev));
  CHECK(dst[3] > 1.9f);

  unary_ev.subop = emel::kernel::event::unary_subop::sigmoid;
  CHECK_FALSE(machine.process_event(unary_ev));

  x86_64_sm scalar_machine{emel::kernel::x86_64::action::context{false, {}, 0}};

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
  CHECK(scalar_machine.process_event(unary_ev));
  CHECK(dst[0] == doctest::Approx(std::tanh(-2.0f)));
  CHECK(dst[3] == doctest::Approx(std::tanh(2.0f)));

  unary_ev.subop = emel::kernel::event::unary_subop::sigmoid;
  CHECK_FALSE(scalar_machine.process_event(unary_ev));
}

TEST_CASE("kernel_x86_64_rejects_unimplemented_ops") {
  float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dst[4] = {};

  const emel::kernel::event::op_sum sum_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
  };

  x86_64_sm machine{};
  CHECK_FALSE(machine.process_event(sum_ev));
}

TEST_CASE("kernel_x86_64_mul_mat_simd_matches_scalar_tiled_edges") {
  const bool host_avx2 =
      emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
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
  };
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_scalar.data(), dtype::f32, n, m),
  };

  CHECK(emel::kernel::x86_64::detail::execute_avx2_mul_mat(simd_ev));
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  for (uint64_t idx = 0; idx < dst_simd.size(); ++idx) {
    CHECK(dst_simd[static_cast<size_t>(idx)] ==
          doctest::Approx(dst_scalar[static_cast<size_t>(idx)]).epsilon(1e-5f));
  }
}

TEST_CASE("kernel_x86_64_f32_mul_mat_fma_simd_matches_scalar_tiled_edges") {
  if (!host_has_avx2_fma()) {
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
  };
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_scalar.data(), dtype::f32, n, m),
  };

  CHECK(emel::kernel::x86_64::detail::execute_avx2_fma_mul_mat(simd_ev));
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  for (uint64_t idx = 0; idx < dst_simd.size(); ++idx) {
    CHECK(dst_simd[static_cast<size_t>(idx)] ==
          doctest::Approx(dst_scalar[static_cast<size_t>(idx)]).epsilon(1e-5f));
  }
}

TEST_CASE("kernel_x86_64_f32_mul_mat_uses_fma_and_avx2_only_routes") {
  const bool host_avx2 =
      emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
      emel::kernel::x86_64::detail::detect_avx2();
  if (!host_avx2) {
    return;
  }

  constexpr uint64_t k = 16;
  constexpr uint64_t m = 4;
  constexpr uint64_t n = 8;

  std::array<float, k * m> src0{};
  std::array<float, k * n> src1{};
  for (uint64_t i = 0; i < src0.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 13u) - 6;
    src0[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.125f;
  }
  for (uint64_t i = 0; i < src1.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 19u) - 9;
    src1[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<float, n * m> expected{};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(expected.data(), dtype::f32, n, m),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  if (host_has_avx2_fma()) {
    std::array<float, n * m> fma_out{};
    const emel::kernel::event::op_mul_mat fma_ev{
        .src0 = make_src(src0.data(), dtype::f32, k, m),
        .src1 = make_src(src1.data(), dtype::f32, n, k),
        .dst = make_dst(fma_out.data(), dtype::f32, n, m),
    };
    x86_64_sm fma_machine{
        emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

    CHECK(fma_machine.process_event(fma_ev));
    CHECK(fma_machine.optimized_f32_fma_dispatch_count() == 1u);
    for (size_t i = 0; i < expected.size(); ++i) {
      CHECK(fma_out[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
    }
  }

  std::array<float, n * m> avx2_out{};
  const emel::kernel::event::op_mul_mat avx2_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(avx2_out.data(), dtype::f32, n, m),
  };
  const emel::kernel::x86_64::detail::host_feature_contract avx2_only_contract{
      .avx2_available = true,
      .fma_available = false,
      .f16c_available = false,
  };
  x86_64_sm avx2_machine{
      emel::kernel::x86_64::action::context{avx2_only_contract, {}, 0}};

  CHECK(avx2_machine.process_event(avx2_ev));
  CHECK(avx2_machine.optimized_f32_fma_dispatch_count() == 0u);
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(avx2_out[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
  }
}

TEST_CASE("kernel_x86_64_f32_mul_mat_vector_fma_route_matches_scalar") {
  if (!host_has_avx2_fma()) {
    return;
  }

  constexpr uint64_t k = 133;
  constexpr uint64_t m = 5;

  std::array<float, k * m> src0{};
  std::array<float, k> src1{};
  for (uint64_t i = 0; i < src0.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 29u) - 14;
    src0[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.0625f;
  }
  for (uint64_t i = 0; i < src1.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 31u) - 15;
    src1[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.03125f;
  }

  std::array<float, m> expected{};
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, 1u, k),
      .dst = make_dst(expected.data(), dtype::f32, 1u, m),
  };
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  std::array<float, m> vector_out{};
  const emel::kernel::event::op_mul_mat vector_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, 1u, k),
      .dst = make_dst(vector_out.data(), dtype::f32, 1u, m),
  };
  x86_64_sm vector_machine{
      emel::kernel::x86_64::action::context{avx2_fma_contract(true), {}, 0}};

  CHECK(vector_machine.process_event(vector_ev));
  CHECK(vector_machine.optimized_f32_fma_vector_dispatch_count() == 1u);
  CHECK(vector_machine.optimized_f32_fma_dispatch_count() == 0u);
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(vector_out[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
  }
}

TEST_CASE("kernel_x86_64_mul_mat_tail_resets_nan_dst_on_first_depth_block") {
  const bool host_avx2 =
      emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
      emel::kernel::x86_64::detail::detect_avx2();
  if (!host_avx2) {
    return;
  }

  constexpr uint64_t k = 5;
  constexpr uint64_t m = 3;
  constexpr uint64_t n = 9;

  std::array<float, k * m> src0{};
  std::array<float, k * n> src1{};
  std::array<float, n * m> dst_simd{};
  std::array<float, n * m> dst_scalar{};

  for (uint64_t i = 0; i < src0.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 7u) - 3;
    src0[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.125f;
  }
  for (uint64_t i = 0; i < src1.size(); ++i) {
    const int64_t centered = static_cast<int64_t>(i % 11u) - 5;
    src1[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.0625f;
  }

  std::fill(dst_simd.begin(), dst_simd.end(),
            std::numeric_limits<float>::quiet_NaN());
  dst_scalar.fill(0.0f);

  const emel::kernel::event::op_mul_mat simd_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_simd.data(), dtype::f32, n, m),
  };
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_scalar.data(), dtype::f32, n, m),
  };

  CHECK(emel::kernel::x86_64::detail::execute_avx2_mul_mat(simd_ev));
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  for (uint64_t idx = 0; idx < dst_simd.size(); ++idx) {
    CHECK(std::isfinite(dst_simd[static_cast<size_t>(idx)]));
    CHECK(dst_simd[static_cast<size_t>(idx)] ==
          doctest::Approx(dst_scalar[static_cast<size_t>(idx)]).epsilon(1e-5f));
  }
}

TEST_CASE("kernel_x86_64_detail_branch_paths") {
  const bool host_avx2 =
      emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
      emel::kernel::x86_64::detail::detect_avx2();

  float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float rhs[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  float dst[4] = {};

  emel::kernel::event::op_add add_ev{
      .src0 = make_src(lhs, dtype::f32, 4),
      .src1 = make_src(rhs, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
  };

  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(add_ev, false));
  CHECK(emel::kernel::x86_64::detail::can_use_avx2(add_ev, host_avx2) ==
        host_avx2);
  CHECK(emel::kernel::x86_64::detail::execute_request(
      add_ev, emel::kernel::x86_64::action::context{host_avx2, {}, 0}));

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
      .subop = emel::kernel::event::unary_subop::relu,
  };
  CHECK(emel::kernel::x86_64::detail::can_use_avx2(unary_ev, host_avx2) ==
        host_avx2);
  unary_ev.subop = emel::kernel::event::unary_subop::exp;
  CHECK_FALSE(emel::kernel::x86_64::detail::can_use_avx2(unary_ev, host_avx2));
}

TEST_CASE("kernel_x86_64_detail_helper_edge_paths") {
  const bool host_avx2 =
      emel::kernel::x86_64::detail::avx2_intrinsics_compiled &&
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
  };
  const emel::kernel::event::op_add add_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
  };
  const emel::kernel::event::op_mul mul_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
  };
  const emel::kernel::event::op_div div_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
  };
  const emel::kernel::event::op_sqr sqr_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
  };
  const emel::kernel::event::op_sqrt sqrt_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
  };
  const emel::kernel::event::op_sub sub_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .src1 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
  };
  float src_mm0[8] = {1.0f, 0.5f, -1.0f, 2.0f, 0.0f, -0.5f, 3.0f, 1.0f};
  float src_mm1[32] = {
      1.0f, 0.0f,  0.5f, -1.0f, 0.5f, 1.0f,  -0.5f, 2.0f, 0.0f,  1.0f, 1.0f,
      0.0f, 2.0f,  0.5f, 0.0f,  1.0f, -1.0f, 2.0f,  0.0f, 1.0f,  1.5f, 0.0f,
      2.0f, -0.5f, 2.0f, -1.0f, 1.0f, 0.5f,  0.0f,  1.0f, -1.0f, 1.0f,
  };
  float dst_mm[16] = {};
  const emel::kernel::event::op_mul_mat mul_mat_ev{
      .src0 = make_src(src_mm0, dtype::f32, 4, 2),
      .src1 = make_src(src_mm1, dtype::f32, 8, 4),
      .dst = make_dst(dst_mm, dtype::f32, 8, 2),
  };
  emel::kernel::event::op_unary unary_ev{
      .src0 = make_src(src0, dtype::f32, 4),
      .dst = make_dst(dst0, dtype::f32, 4),
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
    emel::kernel::x86_64::detail::execute_simd_unary_subop_unchecked<
        emel::kernel::event::unary_subop::relu>(unary_ev);
    CHECK(emel::kernel::x86_64::detail::execute_simd(dup_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(add_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(sub_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(mul_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(div_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(sqr_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(sqrt_ev));
    CHECK(emel::kernel::x86_64::detail::execute_simd(mul_mat_ev));
    CHECK_FALSE(emel::kernel::x86_64::detail::execute_simd(unary_ev));
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
  emel::kernel::x86_64::detail::execute_simd_unchecked(dup_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(add_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(sub_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(mul_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(div_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(sqr_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(sqrt_ev);
  emel::kernel::x86_64::detail::execute_simd_unchecked(mul_mat_ev);
  emel::kernel::x86_64::detail::execute_simd_unary_subop_unchecked<
      emel::kernel::event::unary_subop::relu>(unary_ev);
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
  };

  emel::kernel::x86_64::event::dispatch_ctx dispatch_ctx{};
  emel::kernel::x86_64::action::context ctx{false, {}, 0};
  const emel::kernel::x86_64::event::dispatch_op_add dispatch_ev{add_ev,
                                                                 dispatch_ctx};

  emel::kernel::x86_64::action::exec_simd_op_add(dispatch_ev, ctx);

  CHECK(dispatch_ctx.outcome ==
        emel::kernel::x86_64::events::phase_outcome::done);
  CHECK(dispatch_ctx.err == static_cast<int32_t>(emel::error::cast(
                                emel::kernel::x86_64::error::none)));
}

namespace {

template <class event_type>
void set_op_param_i32(event_type &ev, const uint32_t slot,
                      const int32_t value) {
  std::memcpy(ev.op_params.data() + slot * sizeof(int32_t), &value,
              sizeof(value));
  ev.op_params_size =
      std::max<uint32_t>(ev.op_params_size, (slot + 1u) * sizeof(int32_t));
}

template <class event_type>
void set_op_param_f32(event_type &ev, const uint32_t slot, const float value) {
  std::memcpy(ev.op_params.data() + slot * sizeof(float), &value,
              sizeof(value));
  ev.op_params_size =
      std::max<uint32_t>(ev.op_params_size, (slot + 1u) * sizeof(float));
}

} // namespace

TEST_CASE("kernel_x86_64_get_rows_gathers_f32_and_q8_0_rows") {
  x86_64_sm machine{emel::kernel::x86_64::action::context{false, {}, 0}};

  float table[12] = {0.0f,  1.0f,  2.0f,  3.0f,  10.0f, 11.0f,
                     12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f};
  int32_t indices[2] = {2, 0};
  float gathered[8] = {};

  emel::kernel::event::op_get_rows gather_ev{
      .src0 = make_src(table, dtype::f32, 4, 3),
      .src1 = make_src(indices, dtype::i32, 2),
      .dst = make_dst(gathered, dtype::f32, 4, 2),
  };
  CHECK(machine.process_event(gather_ev));
  CHECK(gathered[0] == doctest::Approx(20.0f));
  CHECK(gathered[3] == doctest::Approx(23.0f));
  CHECK(gathered[4] == doctest::Approx(0.0f));
  CHECK(gathered[7] == doctest::Approx(3.0f));

  int32_t bad_indices[2] = {3, 0};
  gather_ev.src1 = make_src(bad_indices, dtype::i32, 2);
  CHECK_FALSE(machine.process_event(gather_ev));

  // Index metadata without bound storage rejects instead of dereferencing a
  // null tensor inside the validation scan.
  auto null_indices = make_src(bad_indices, dtype::i32, 2);
  null_indices.data = nullptr;
  gather_ev.src1 = null_indices;
  CHECK_FALSE(machine.process_event(gather_ev));

  constexpr int64_t k_cols = 32;
  float source_rows[2 * k_cols] = {};
  for (int64_t i = 0; i < 2 * k_cols; ++i) {
    source_rows[i] = 0.25f * static_cast<float>(i - k_cols);
  }
  emel::kernel::detail::quant::block_q8_0 blocks[2] = {};
  emel::kernel::detail::quant::quantize_row_q8_0_strided(source_rows, 1u,
                                                         &blocks[0], k_cols);
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      source_rows + k_cols, 1u, &blocks[1], k_cols);
  float expected[k_cols] = {};
  emel::kernel::detail::quant::dequantize_row_q8_0(&blocks[1], expected,
                                                   k_cols);

  int32_t quant_index[1] = {1};
  float quant_out[k_cols] = {};
  emel::kernel::event::op_get_rows quant_ev{
      .src0 = emel::kernel::test::make_quantized_src(blocks, dtype::q8_0,
                                                     k_cols, 2),
      .src1 = make_src(quant_index, dtype::i32, 1),
      .dst = make_dst(quant_out, dtype::f32, k_cols, 1),
  };
  CHECK(machine.process_event(quant_ev));
  for (int64_t i = 0; i < k_cols; ++i) {
    CHECK(quant_out[i] == doctest::Approx(expected[i]));
  }

  // bf16 sources must reach the explicit bf16 conversion row: the generic
  // src0 dtype fallback previously rejected them before the row could match.
  uint16_t bf16_table[12] = {};
  for (int64_t i = 0; i < 12; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &table[i], sizeof(bits));
    bf16_table[i] = static_cast<uint16_t>(bits >> 16);
  }
  int32_t bf16_indices[2] = {2, 0};
  float bf16_out[8] = {};
  emel::kernel::event::op_get_rows bf16_ev{
      .src0 = make_src(bf16_table, dtype::bf16, 4, 3),
      .src1 = make_src(bf16_indices, dtype::i32, 2),
      .dst = make_dst(bf16_out, dtype::f32, 4, 2),
  };
  CHECK(machine.process_event(bf16_ev));
  CHECK(bf16_out[0] == doctest::Approx(20.0f));
  CHECK(bf16_out[3] == doctest::Approx(23.0f));
  CHECK(bf16_out[4] == doctest::Approx(0.0f));
  CHECK(bf16_out[7] == doctest::Approx(3.0f));
}

TEST_CASE("kernel_x86_64_rms_norm_and_norm_rows") {
  x86_64_sm machine{emel::kernel::x86_64::action::context{false, {}, 0}};

  float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dst[4] = {};
  constexpr float k_eps = 1e-5f;

  emel::kernel::event::op_rms_norm rms_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
  };
  set_op_param_f32(rms_ev, 0u, k_eps);
  CHECK(machine.process_event(rms_ev));
  const float mean_sq = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
  const float rms_scale = 1.0f / std::sqrt(mean_sq + k_eps);
  CHECK(dst[0] == doctest::Approx(1.0f * rms_scale));
  CHECK(dst[3] == doctest::Approx(4.0f * rms_scale));

  emel::kernel::event::op_norm norm_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
  };
  set_op_param_f32(norm_ev, 0u, k_eps);
  CHECK(machine.process_event(norm_ev));
  const float mean = 2.5f;
  const float variance = (2.25f + 0.25f + 0.25f + 2.25f) / 4.0f;
  const float norm_scale = 1.0f / std::sqrt(variance + k_eps);
  CHECK(dst[0] == doctest::Approx((1.0f - mean) * norm_scale));
  CHECK(dst[3] == doctest::Approx((4.0f - mean) * norm_scale));

  emel::kernel::event::op_rms_norm missing_eps_ev{
      .src0 = make_src(src, dtype::f32, 4),
      .dst = make_dst(dst, dtype::f32, 4),
  };
  CHECK_FALSE(machine.process_event(missing_eps_ev));
}

TEST_CASE("kernel_x86_64_rope_rotates_norm_and_neox_pairs") {
  x86_64_sm machine{emel::kernel::x86_64::action::context{false, {}, 0}};

  // one head, head_dim 4, two tokens with positions {0, 1}
  float src[8] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
  int32_t positions[2] = {0, 1};
  float dst[8] = {};

  emel::kernel::event::op_rope rope_ev{
      .src0 = make_src(src, dtype::f32, 4, 1, 2),
      .src1 = make_src(positions, dtype::i32, 2),
      .dst = make_dst(dst, dtype::f32, 4, 1, 2),
  };
  set_op_param_i32(rope_ev, 1u, 4);        // n_dims
  set_op_param_i32(rope_ev, 2u, 0);        // mode = norm
  set_op_param_f32(rope_ev, 5u, 10000.0f); // freq_base
  set_op_param_f32(rope_ev, 6u, 1.0f);     // freq_scale
  set_op_param_f32(rope_ev, 7u, 0.0f);     // ext_factor
  set_op_param_f32(rope_ev, 8u, 1.0f);     // attn_factor
  set_op_param_f32(rope_ev, 9u, 32.0f);    // beta_fast
  set_op_param_f32(rope_ev, 10u, 1.0f);    // beta_slow

  // Metadata-only positions reject in the guard instead of dereferencing
  // null inside the exec.
  auto null_positions_ev = rope_ev;
  auto null_positions = make_src(positions, dtype::i32, 2);
  null_positions.data = nullptr;
  null_positions_ev.src1 = null_positions;
  CHECK_FALSE(machine.process_event(null_positions_ev));

  // Non-finite frequency parameters pass a bare positivity check and would
  // propagate NaNs through sin/cos while the kernel reports success.
  auto inf_freq_scale_ev = rope_ev;
  set_op_param_f32(inf_freq_scale_ev, 6u, std::numeric_limits<float>::infinity());
  CHECK_FALSE(machine.process_event(inf_freq_scale_ev));

  auto inf_freq_base_ev = rope_ev;
  set_op_param_f32(inf_freq_base_ev, 5u, std::numeric_limits<float>::infinity());
  CHECK_FALSE(machine.process_event(inf_freq_base_ev));

  CHECK(machine.process_event(rope_ev));

  // position 0 is the identity rotation
  CHECK(dst[0] == doctest::Approx(1.0f));
  CHECK(dst[1] == doctest::Approx(0.0f));
  // position 1, pair 0: theta = 1, pair 1: theta = 10000^(-2/4)
  CHECK(dst[4] == doctest::Approx(std::cos(1.0f)));
  CHECK(dst[5] == doctest::Approx(std::sin(1.0f)));
  const float theta1 = std::pow(10000.0f, -2.0f / 4.0f);
  CHECK(dst[6] == doctest::Approx(std::cos(theta1)));
  CHECK(dst[7] == doctest::Approx(std::sin(theta1)));

  // neox pairing rotates (x[0], x[2]) and (x[1], x[3])
  float neox_src[4] = {1.0f, 2.0f, 0.0f, 0.0f};
  float neox_dst[4] = {};
  int32_t neox_pos[1] = {1};
  emel::kernel::event::op_rope neox_ev{
      .src0 = make_src(neox_src, dtype::f32, 4, 1, 1),
      .src1 = make_src(neox_pos, dtype::i32, 1),
      .dst = make_dst(neox_dst, dtype::f32, 4, 1, 1),
  };
  set_op_param_i32(neox_ev, 1u, 4);
  set_op_param_i32(neox_ev, 2u, 2); // mode = neox
  set_op_param_f32(neox_ev, 5u, 10000.0f);
  set_op_param_f32(neox_ev, 6u, 1.0f);
  set_op_param_f32(neox_ev, 7u, 0.0f);
  set_op_param_f32(neox_ev, 8u, 1.0f);
  set_op_param_f32(neox_ev, 9u, 32.0f);
  set_op_param_f32(neox_ev, 10u, 1.0f);
  CHECK(machine.process_event(neox_ev));
  CHECK(neox_dst[0] == doctest::Approx(std::cos(1.0f)));
  CHECK(neox_dst[2] == doctest::Approx(std::sin(1.0f)));
  CHECK(neox_dst[1] == doctest::Approx(2.0f * std::cos(theta1)));
  CHECK(neox_dst[3] == doctest::Approx(2.0f * std::sin(theta1)));
}

TEST_CASE("kernel_x86_64_im2col_and_conv_transpose_1d") {
  x86_64_sm machine{emel::kernel::x86_64::action::context{false, {}, 0}};

  // conv1d frontend: L=4, IC=1, K=3, s=1, p=1, d=1 -> OW=4 columns of taps
  float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float kernel_shape_only[3] = {};
  float columns[12] = {};
  emel::kernel::event::op_im2col im2col_ev{
      .src0 = make_src(kernel_shape_only, dtype::f32, 3, 1),
      .src1 = make_src(input, dtype::f32, 4, 1),
      .dst = make_dst(columns, dtype::f32, 3, 4),
  };
  set_op_param_i32(im2col_ev, 0u, 1); // s0
  set_op_param_i32(im2col_ev, 1u, 0); // s1
  set_op_param_i32(im2col_ev, 2u, 1); // p0
  set_op_param_i32(im2col_ev, 3u, 0); // p1
  set_op_param_i32(im2col_ev, 4u, 1); // d0
  set_op_param_i32(im2col_ev, 5u, 0); // d1
  set_op_param_i32(im2col_ev, 6u, 0); // is_2D
  CHECK(machine.process_event(im2col_ev));
  const float expected_columns[12] = {0.0f, 1.0f, 2.0f, 1.0f, 2.0f, 3.0f,
                                      2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 0.0f};
  for (int i = 0; i < 12; ++i) {
    CHECK(columns[i] == doctest::Approx(expected_columns[i]));
  }

  // transposed conv: K=2, s=2, IC=1, OC=1, L=2 -> OL=4
  float up_input[2] = {1.0f, 2.0f};
  float up_kernel[2] = {3.0f, 5.0f};
  float up_out[4] = {};
  emel::kernel::event::op_conv_transpose_1d convtr_ev{
      .src0 = make_src(up_kernel, dtype::f32, 2, 1, 1),
      .src1 = make_src(up_input, dtype::f32, 2, 1),
      .dst = make_dst(up_out, dtype::f32, 4, 1),
  };
  set_op_param_i32(convtr_ev, 0u, 2); // s0
  set_op_param_i32(convtr_ev, 1u, 0); // p0
  set_op_param_i32(convtr_ev, 2u, 1); // d0
  CHECK(machine.process_event(convtr_ev));
  CHECK(up_out[0] == doctest::Approx(3.0f));
  CHECK(up_out[1] == doctest::Approx(5.0f));
  CHECK(up_out[2] == doctest::Approx(6.0f));
  CHECK(up_out[3] == doctest::Approx(10.0f));

  // Extents past the signed-arithmetic cap reject before the output-length
  // formulas can overflow.
  auto huge_im2col = im2col_ev;
  auto huge_im2col_input = im2col_ev.src1;
  huge_im2col_input.ne[0] = uint64_t{1} << 62;
  huge_im2col.src1 = huge_im2col_input;
  CHECK_FALSE(machine.process_event(huge_im2col));

  auto huge_convtr = convtr_ev;
  auto huge_convtr_input = convtr_ev.src1;
  huge_convtr_input.ne[0] = uint64_t{1} << 62;
  huge_convtr.src1 = huge_convtr_input;
  CHECK_FALSE(machine.process_event(huge_convtr));

  // A small kernel with a huge nonnegative padding scales the im2col
  // out_length past any real destination: with a matching dst shape the exec
  // would write ~2^31 column elements into a 12-float buffer, so the
  // total-output bound must reject before dispatch.
  float pad_out[12] = {};
  emel::kernel::event::op_im2col padded_im2col_ev{
      .src0 = make_src(kernel_shape_only, dtype::f32, 3, 1),
      .src1 = make_src(input, dtype::f32, 4, 1),
      .dst = make_dst(pad_out, dtype::f32, 3, 720000002u),
  };
  set_op_param_i32(padded_im2col_ev, 0u, 1);         // s0
  set_op_param_i32(padded_im2col_ev, 1u, 0);         // s1
  set_op_param_i32(padded_im2col_ev, 2u, 360000000); // p0 -> out_length ~7.2e8
  set_op_param_i32(padded_im2col_ev, 3u, 0);         // p1
  set_op_param_i32(padded_im2col_ev, 4u, 1);         // d0
  set_op_param_i32(padded_im2col_ev, 5u, 0);         // d1
  set_op_param_i32(padded_im2col_ev, 6u, 0);         // is_2D
  CHECK_FALSE(machine.process_event(padded_im2col_ev));

  // Per-extent caps alone still admit an s0-scaled out_length whose product
  // with out_channels exceeds the guard cap: with a matching destination
  // shape the exec would fill far past the destination allocation, so the
  // total-output bound must reject before dispatch.
  float wide_kernel[2 * 16] = {};
  float wide_input[8] = {};
  float wide_out[16] = {};
  emel::kernel::event::op_conv_transpose_1d wide_convtr_ev{
      .src0 = make_src(wide_kernel, dtype::f32, 2, 16, 1),
      .src1 = make_src(wide_input, dtype::f32, 65536, 1),
      .dst = make_dst(wide_out, dtype::f32, 536862722u, 16),
  };
  set_op_param_i32(wide_convtr_ev, 0u, 8192); // s0 -> out_length ~2^29
  set_op_param_i32(wide_convtr_ev, 1u, 0);    // p0
  set_op_param_i32(wide_convtr_ev, 2u, 1);    // d0
  CHECK_FALSE(machine.process_event(wide_convtr_ev));

  // Metadata-only inputs reject in the guard instead of dereferencing null
  // inside the exec.
  auto null_conv_input = make_src(up_input, dtype::f32, 2, 1);
  null_conv_input.data = nullptr;
  convtr_ev.src1 = null_conv_input;
  CHECK_FALSE(machine.process_event(convtr_ev));

  auto null_im2col_input = make_src(input, dtype::f32, 4, 1);
  null_im2col_input.data = nullptr;
  im2col_ev.src1 = null_im2col_input;
  CHECK_FALSE(machine.process_event(im2col_ev));
}

TEST_CASE("kernel_x86_64_add_and_mul_broadcast_row_variants") {
  x86_64_sm machine{emel::kernel::x86_64::action::context{false, {}, 0}};

  float rows[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // 2 rows of 3
  float bias[3] = {10.0f, 20.0f, 30.0f};
  float out[6] = {};

  emel::kernel::event::op_add add_ev{
      .src0 = make_src(rows, dtype::f32, 3, 2),
      .src1 = make_src(bias, dtype::f32, 3, 1),
      .dst = make_dst(out, dtype::f32, 3, 2),
  };
  CHECK(machine.process_event(add_ev));
  CHECK(out[0] == doctest::Approx(11.0f));
  CHECK(out[2] == doctest::Approx(33.0f));
  CHECK(out[3] == doctest::Approx(14.0f));
  CHECK(out[5] == doctest::Approx(36.0f));

  emel::kernel::event::op_mul mul_ev{
      .src0 = make_src(rows, dtype::f32, 3, 2),
      .src1 = make_src(bias, dtype::f32, 3, 1),
      .dst = make_dst(out, dtype::f32, 3, 2),
  };
  CHECK(machine.process_event(mul_ev));
  CHECK(out[0] == doctest::Approx(10.0f));
  CHECK(out[5] == doctest::Approx(180.0f));

  // row length mismatch is rejected, not silently broadcast
  emel::kernel::event::op_add bad_ev{
      .src0 = make_src(rows, dtype::f32, 3, 2),
      .src1 = make_src(bias, dtype::f32, 2, 1),
      .dst = make_dst(out, dtype::f32, 3, 2),
  };
  CHECK_FALSE(machine.process_event(bad_ev));
}
