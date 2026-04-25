#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <span>
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
using emel::kernel::test::flash_attn_reference_f16_scores;
using emel::kernel::test::flash_attn_reference_masked_total_tokens;
using emel::kernel::test::flash_attn_reference_online_softmax_f16_values;
using emel::kernel::test::k_flash_online_f16_abs_tolerance;
using emel::kernel::test::make_batch_major_dst;
using emel::kernel::test::make_packed_q4_k_x8_bl4_src;
using emel::kernel::test::make_packed_q4_k_x8_bl8_src;
using emel::kernel::test::make_dst;
using emel::kernel::test::make_packed_q8_0_x4_bl4_src;
using emel::kernel::test::make_packed_q8_0_x4_bl8_src;
using emel::kernel::test::make_packed_q8_0_x4_bl8_rhs_src;
using emel::kernel::test::make_argmax_prepared_q6_k_x8_q8_src;
using emel::kernel::test::make_flash_attn_ext_event;
using emel::kernel::test::make_packed_q6_k_x8_src;
using emel::kernel::test::make_prepared_q6_k_x8_q8_src;
using emel::kernel::test::make_q8_0_vector_src;
using emel::kernel::test::make_q8_k_vector_src;
using emel::kernel::test::make_q8_k_x4_rhs_src;
using emel::kernel::test::make_q8_k_x8_rhs_src;
using emel::kernel::test::make_quantized_src;
using emel::kernel::test::make_src;
using emel::kernel::test::to_fp16_storage;
using emel::kernel::test::within_flash_online_f16_tolerance;

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

TEST_CASE("kernel_aarch64_mul_mat_tail_resets_nan_dst_on_first_depth_block") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t k = 5;
  constexpr uint64_t m = 3;
  constexpr uint64_t n = 5;

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

  std::fill(dst_simd.begin(), dst_simd.end(), std::numeric_limits<float>::quiet_NaN());
  dst_scalar.fill(0.0f);

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
    CHECK(std::isfinite(dst_simd[static_cast<size_t>(idx)]));
    CHECK(dst_simd[static_cast<size_t>(idx)] ==
          doctest::Approx(dst_scalar[static_cast<size_t>(idx)]).epsilon(1e-5f));
  }
#endif
}

TEST_CASE("kernel_aarch64_mul_mat_simd_matches_scalar_when_matrix_has_eight_rows") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t k = 96;
  constexpr uint64_t m = 8;
  constexpr uint64_t n = 64;

  std::array<float, k * m> src0{};
  std::array<float, k * n> src1{};
  std::array<float, n * m> dst_simd{};
  std::array<float, n * m> dst_scalar{};

  for (uint64_t i = 0; i < src0.size(); ++i) {
    const int64_t centered = static_cast<int64_t>((i * 5u) % 23u) - 11;
    src0[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.03125f;
  }
  for (uint64_t i = 0; i < src1.size(); ++i) {
    const int64_t centered = static_cast<int64_t>((i * 7u) % 31u) - 15;
    src1[static_cast<size_t>(i)] = static_cast<float>(centered) * 0.015625f;
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

TEST_CASE("kernel_aarch64_prepared_f32_lhs_4row_matches_scalar_with_depth_and_tails") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t k = 263;
  constexpr uint64_t m = 9;
  constexpr uint64_t n = 13;

  std::array<float, k * m> src0{};
  std::array<float, k * n> src1{};
  std::array<float, n * m> dst_prepared{};
  std::array<float, n * m> dst_scalar{};
  std::vector<float> prepared(
      static_cast<size_t>(
          emel::kernel::aarch64::detail::prepared_f32_lhs_4row_value_count(k, m)),
      0.0f);

  for (uint64_t index = 0; index < src0.size(); ++index) {
    const int64_t centered = static_cast<int64_t>((index * 5u) % 37u) - 18;
    src0[static_cast<size_t>(index)] = static_cast<float>(centered) * 0.015625f;
  }
  for (uint64_t index = 0; index < src1.size(); ++index) {
    const int64_t centered = static_cast<int64_t>((index * 7u) % 43u) - 21;
    src1[static_cast<size_t>(index)] = static_cast<float>(centered) * 0.0078125f;
  }
  std::fill(dst_prepared.begin(), dst_prepared.end(), std::numeric_limits<float>::quiet_NaN());

  REQUIRE(emel::kernel::aarch64::detail::prepare_neon_mul_mat_f32_lhs_4row(
      src0.data(),
      k,
      m,
      prepared.data(),
      static_cast<uint64_t>(prepared.size())));

  const emel::kernel::event::op_mul_mat prepared_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_prepared.data(), dtype::f32, n, m),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst_scalar.data(), dtype::f32, n, m),
      .nth = 1,
  };

  CHECK(emel::kernel::aarch64::detail::execute_neon_mul_mat_prepared_f32_lhs_4row(
      prepared_ev,
      prepared.data(),
      static_cast<uint64_t>(prepared.size())));
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  for (uint64_t index = 0; index < dst_prepared.size(); ++index) {
    CHECK(std::isfinite(dst_prepared[static_cast<size_t>(index)]));
    CHECK(dst_prepared[static_cast<size_t>(index)] ==
          doctest::Approx(dst_scalar[static_cast<size_t>(index)]).epsilon(1e-5f));
  }
#endif
}

TEST_CASE("kernel_aarch64_prepared_f32_lhs_4row_rejects_invalid_contract") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t k = 4;
  constexpr uint64_t m = 2;
  constexpr uint64_t n = 3;

  std::array<float, k * m> src0{};
  std::array<float, k * n> src1{};
  std::array<float, n * m> dst{};
  std::vector<float> prepared(
      static_cast<size_t>(
          emel::kernel::aarch64::detail::prepared_f32_lhs_4row_value_count(k, m)),
      0.0f);

  CHECK_FALSE(emel::kernel::aarch64::detail::prepare_neon_mul_mat_f32_lhs_4row(
      nullptr,
      k,
      m,
      prepared.data(),
      static_cast<uint64_t>(prepared.size())));
  CHECK_FALSE(emel::kernel::aarch64::detail::prepare_neon_mul_mat_f32_lhs_4row(
      src0.data(),
      k,
      m,
      prepared.data(),
      static_cast<uint64_t>(prepared.size() - 1u)));

  emel::kernel::event::op_mul_mat ev{
      .src0 = make_src(src0.data(), dtype::f32, k, m),
      .src1 = make_src(src1.data(), dtype::f32, n, k),
      .dst = make_dst(dst.data(), dtype::f32, n, m),
      .nth = 1,
  };

  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_mul_mat_prepared_f32_lhs_4row(
      ev,
      nullptr,
      static_cast<uint64_t>(prepared.size())));
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_mul_mat_prepared_f32_lhs_4row(
      ev,
      prepared.data(),
      static_cast<uint64_t>(prepared.size() - 1u)));

  ev.src1.ne[1] = k - 1u;
  CHECK_FALSE(emel::kernel::aarch64::detail::execute_neon_mul_mat_prepared_f32_lhs_4row(
      ev,
      prepared.data(),
      static_cast<uint64_t>(prepared.size())));
#endif
}

TEST_CASE("kernel_aarch64_image_pointwise_f32_matches_scalar_reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t output_channels = 77;
  constexpr int32_t input_channels = 48;
  constexpr int32_t pixel_count = 9;
  constexpr int32_t panel_cols = 64;

  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(input_channels)> input = {};
  std::array<float, static_cast<size_t>(input_channels) * static_cast<size_t>(output_channels)>
      transposed_weights = {};
  std::array<float, static_cast<size_t>(input_channels) * static_cast<size_t>(output_channels)>
      packed_weights = {};
  std::array<float, static_cast<size_t>(output_channels)> scale = {};
  std::array<float, static_cast<size_t>(output_channels)> shift = {};
  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      output_kernel = {};
  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      output_scalar = {};

  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 5u) % 29u) - 14;
    input[idx] = static_cast<float>(centered) * 0.0625f;
  }
  for (int32_t output_channel = 0; output_channel < output_channels; ++output_channel) {
    scale[static_cast<size_t>(output_channel)] =
        0.75f + static_cast<float>((output_channel % 7) + 1) * 0.0625f;
    shift[static_cast<size_t>(output_channel)] =
        static_cast<float>((output_channel % 9) - 4) * 0.03125f;
    for (int32_t input_channel = 0; input_channel < input_channels; ++input_channel) {
      const size_t idx =
          static_cast<size_t>(input_channel) * static_cast<size_t>(output_channels) +
          static_cast<size_t>(output_channel);
      const int32_t centered =
          static_cast<int32_t>(((static_cast<size_t>(output_channel) * 13u) +
                                (static_cast<size_t>(input_channel) * 7u)) %
                                   41u) -
          20;
      transposed_weights[idx] = static_cast<float>(centered) * 0.015625f;
    }
  }
  for (int32_t panel_col = 0; panel_col < output_channels; panel_col += panel_cols) {
    const int32_t cols_this_panel = std::min(panel_cols, output_channels - panel_col);
    float * panel_dst =
        packed_weights.data() + static_cast<size_t>(panel_col) * static_cast<size_t>(input_channels);
    for (int32_t input_channel = 0; input_channel < input_channels; ++input_channel) {
      for (int32_t local_col = 0; local_col < cols_this_panel; ++local_col) {
        panel_dst[static_cast<size_t>(input_channel) * static_cast<size_t>(cols_this_panel) +
                  static_cast<size_t>(local_col)] =
            transposed_weights[static_cast<size_t>(input_channel) *
                                   static_cast<size_t>(output_channels) +
                               static_cast<size_t>(panel_col + local_col)];
      }
    }
  }

  CHECK(emel::kernel::aarch64::detail::execute_neon_image_pointwise_f32<true, true>({
      .input = input.data(),
      .packed_rhs = packed_weights.data(),
      .batch_norm_scale = scale.data(),
      .batch_norm_shift = shift.data(),
      .output = output_kernel.data(),
      .pixel_count = pixel_count,
      .input_channels = input_channels,
      .output_channels = output_channels,
      .packed_rhs_cols = output_channels,
  }));

  for (int32_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
    const float * input_pixel =
        input.data() + static_cast<size_t>(pixel_index) * static_cast<size_t>(input_channels);
    float * output_pixel =
        output_scalar.data() + static_cast<size_t>(pixel_index) * static_cast<size_t>(output_channels);
    for (int32_t output_channel = 0; output_channel < output_channels; ++output_channel) {
      float acc = 0.0f;
      for (int32_t input_channel = 0; input_channel < input_channels; ++input_channel) {
        acc += input_pixel[static_cast<size_t>(input_channel)] *
            transposed_weights[static_cast<size_t>(input_channel) *
                                   static_cast<size_t>(output_channels) +
                               static_cast<size_t>(output_channel)];
      }
      acc = acc * scale[static_cast<size_t>(output_channel)] +
          shift[static_cast<size_t>(output_channel)];
      output_pixel[static_cast<size_t>(output_channel)] = std::max(acc, 0.0f);
    }
  }

  for (size_t idx = 0; idx < output_kernel.size(); ++idx) {
    CHECK(output_kernel[idx] == doctest::Approx(output_scalar[idx]).epsilon(1.0e-5f));
  }
#endif
}

TEST_CASE("kernel_aarch64_image_depthwise_f32_matches_scalar_reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t channels = 29;
  constexpr int32_t input_spatial = 7;
  constexpr int32_t kernel_h = 3;
  constexpr int32_t kernel_w = 3;
  constexpr int32_t stride = 2;
  constexpr int32_t pad = ((stride - 1) + (kernel_h - 1)) / 2;
  constexpr int32_t output_spatial = (input_spatial + 2 * pad - kernel_h) / stride + 1;

  std::array<float, static_cast<size_t>(input_spatial) * static_cast<size_t>(input_spatial) *
                        static_cast<size_t>(channels)>
      input = {};
  std::array<float, static_cast<size_t>(kernel_h) * static_cast<size_t>(kernel_w) *
                        static_cast<size_t>(channels)>
      kernel_major = {};
  std::array<float, static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
                        static_cast<size_t>(channels)>
      output_kernel = {};
  std::array<float, static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
                        static_cast<size_t>(channels)>
      output_scalar = {};

  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 7u) % 23u) - 11;
    input[idx] = static_cast<float>(centered) * 0.03125f;
  }
  for (size_t idx = 0; idx < kernel_major.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 11u) % 19u) - 9;
    kernel_major[idx] = static_cast<float>(centered) * 0.0625f;
  }

  CHECK(emel::kernel::aarch64::detail::execute_neon_image_depthwise_f32({
      .input = input.data(),
      .kernel_major = kernel_major.data(),
      .output = output_kernel.data(),
      .input_spatial = input_spatial,
      .output_spatial = output_spatial,
      .output_channels = channels,
      .kernel_h = kernel_h,
      .kernel_w = kernel_w,
      .stride = stride,
      .pad_h = pad,
      .pad_w = pad,
  }));

  for (int32_t oy = 0; oy < output_spatial; ++oy) {
    for (int32_t ox = 0; ox < output_spatial; ++ox) {
      float * output_pixel =
          output_scalar.data() +
          static_cast<size_t>(oy * output_spatial + ox) * static_cast<size_t>(channels);
      std::fill(output_pixel, output_pixel + channels, 0.0f);
      for (int32_t ky = 0; ky < kernel_h; ++ky) {
        const int32_t iy = oy * stride + ky - pad;
        if (iy < 0 || iy >= input_spatial) {
          continue;
        }
        for (int32_t kx = 0; kx < kernel_w; ++kx) {
          const int32_t ix = ox * stride + kx - pad;
          if (ix < 0 || ix >= input_spatial) {
            continue;
          }
          const float * input_pixel =
              input.data() +
              static_cast<size_t>(iy * input_spatial + ix) * static_cast<size_t>(channels);
          const float * kernel_weights =
              kernel_major.data() +
              static_cast<size_t>(ky * kernel_w + kx) * static_cast<size_t>(channels);
          for (int32_t channel = 0; channel < channels; ++channel) {
            output_pixel[static_cast<size_t>(channel)] +=
                kernel_weights[static_cast<size_t>(channel)] *
                input_pixel[static_cast<size_t>(channel)];
          }
        }
      }
    }
  }

  for (size_t idx = 0; idx < output_kernel.size(); ++idx) {
    CHECK(output_kernel[idx] == doctest::Approx(output_scalar[idx]).epsilon(1.0e-5f));
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

TEST_CASE("kernel_aarch64_q5_0_vector_route_is_explicit_and_numeric_match") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  if (!emel::kernel::aarch64::detail::neon_q5_0_vector_supported()) {
    return;
  }

  using emel::kernel::detail::quant::QK5_0;
  using emel::kernel::detail::quant::block_q5_0;
  using emel::kernel::detail::quant::block_q8_0;
  using emel::kernel::detail::quant::quantize_row_q5_0_ref;
  using emel::kernel::detail::quant::quantize_row_q8_0_strided;

  constexpr uint64_t row_count = 4u;
  std::array<float, QK5_0 * row_count> dense_rows = {};
  for (size_t index = 0; index < dense_rows.size(); ++index) {
    dense_rows[index] = static_cast<float>(static_cast<int32_t>((index * 5u) % 21u) - 10) * 0.125f;
  }

  std::array<block_q5_0, row_count> q5_rows = {};
  for (uint64_t row = 0; row < row_count; ++row) {
    quantize_row_q5_0_ref(
        dense_rows.data() + row * QK5_0,
        q5_rows.data() + row,
        static_cast<int64_t>(QK5_0));
  }

  std::array<float, QK5_0> input = {};
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int32_t>((index * 7u) % 19u) - 9) * 0.25f;
  }

  std::array<block_q8_0, 1> q8_input = {};
  quantize_row_q8_0_strided(input.data(), 1u, q8_input.data(), static_cast<int64_t>(QK5_0));

  std::array<float, row_count> reference = {};
  for (uint64_t row = 0; row < row_count; ++row) {
    reference[row] =
        emel::kernel::detail::dot_q5_0_q8_0_row_scalar(q5_rows.data() + row, q8_input.data(), 1u);
  }

  std::array<float, row_count> simd_out = {};
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_quantized_src(q5_rows.data(), dtype::q5_0, QK5_0, row_count),
      .src1 = make_src(input.data(), dtype::f32, 1u, QK5_0),
      .dst = make_dst(simd_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };

  CHECK(emel::kernel::aarch64::detail::can_use_neon_mul_mat_q5_0_vector(ev, true));
  CHECK(emel::kernel::aarch64::detail::execute_neon_mul_mat_q5_0_vector(ev));
  for (uint64_t row = 0; row < row_count; ++row) {
    CHECK(simd_out[row] == doctest::Approx(reference[row]).epsilon(1.0e-5f));
  }

  aarch64_sm machine{};
  REQUIRE(machine.process_event(ev));
  CHECK(machine.optimized_q5_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q5_0_vector_dispatch_count() == 1u);
  CHECK(machine.shared_q5_0_dispatch_count() == 0u);
#endif
}

TEST_CASE("kernel_aarch64_q2_row_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q8_k;

  constexpr size_t block_count = 4u;
  std::array<block_q2_k, block_count> q2_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    auto & q2 = q2_blocks[block];
    q2.d = static_cast<uint16_t>(0x3c00u + block);
    q2.dmin = static_cast<uint16_t>(0x3c00u + (block % 2u));
    for (size_t i = 0; i < q2.scales.size(); ++i) {
      q2.scales[i] = static_cast<uint8_t>(((i + block) % 9u) << 4u |
                                          (((i * 5u) + block * 3u) % 13u));
    }
    for (size_t i = 0; i < q2.qs.size(); ++i) {
      q2.qs[i] = static_cast<uint8_t>((i * (23u + block)) ^ ((i + block) >> 2u));
    }
  }

  std::array<float, QK_K * block_count> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 7u) % 29u) - 14;
    src1[i] = static_cast<float>(centered) * 0.03125f;
  }

  std::array<block_q8_k, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), static_cast<int32_t>(block_count), q8_blocks.data(),
      emel::kernel::detail::quant::QK_K);

  const float scalar =
      emel::kernel::detail::dot_q2_k_q8_k_row_scalar(q2_blocks.data(), q8_blocks.data(),
                                                     q8_blocks.size());
  const float neon =
      emel::kernel::aarch64::detail::dot_q2_k_q8_k_row_neon(q2_blocks.data(), q8_blocks.data(),
                                                             q8_blocks.size());

  CHECK(neon == doctest::Approx(scalar).epsilon(1e-7f));
#endif
}

TEST_CASE("kernel_aarch64_q2_row_neon_matches_shared_accumulation_order") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q8_k;

  constexpr size_t block_count = 64u;
  std::array<block_q2_k, block_count> q2_blocks = {};
  std::array<block_q8_k, block_count> q8_blocks = {};
  std::mt19937 rng(0u);
  std::uniform_int_distribution<int> bits_dist(0, 255);
  std::uniform_int_distribution<int> half_dist(0x0400, 0x7bff);
  std::uniform_real_distribution<float> value_dist(-4.0f, 4.0f);

  for (size_t block = 0; block < block_count; ++block) {
    auto & q2 = q2_blocks[block];
    q2.d = static_cast<uint16_t>(half_dist(rng));
    q2.dmin = static_cast<uint16_t>(half_dist(rng));
    for (auto & value : q2.scales) {
      value = static_cast<uint8_t>(bits_dist(rng));
    }
    for (auto & value : q2.qs) {
      value = static_cast<uint8_t>(bits_dist(rng));
    }

    std::array<float, QK_K> src = {};
    for (float & value : src) {
      value = value_dist(rng);
    }
    emel::kernel::detail::quant::quantize_row_q8_k_strided(
        src.data(), 1, &q8_blocks[block], emel::kernel::detail::quant::QK_K);
  }

  const float shared =
      emel::kernel::detail::dot_q2_k_q8_k_row_scalar(q2_blocks.data(), q8_blocks.data(), block_count);
  const float optimized =
      emel::kernel::aarch64::detail::dot_q2_k_q8_k_row_neon(q2_blocks.data(),
                                                             q8_blocks.data(),
                                                             block_count);

  CHECK(optimized == doctest::Approx(shared).epsilon(1e-8f));
#endif
}

TEST_CASE("kernel_aarch64_q3_row_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q3_k;
  using emel::kernel::detail::quant::block_q8_k;

  constexpr size_t block_count = 4u;
  std::array<block_q3_k, block_count> q3_blocks = {};
  for (size_t block = 0; block < block_count; ++block) {
    auto & q3 = q3_blocks[block];
    q3.d = static_cast<uint16_t>(0x3c00u + block);
    for (size_t i = 0; i < q3.scales.size(); ++i) {
      q3.scales[i] = static_cast<uint8_t>((i * (17u + block)) ^ (0x5au + block));
    }
    for (size_t i = 0; i < q3.hmask.size(); ++i) {
      q3.hmask[i] = static_cast<uint8_t>((i * (9u + block)) ^ (0xa5u - block));
    }
    for (size_t i = 0; i < q3.qs.size(); ++i) {
      q3.qs[i] = static_cast<uint8_t>((i * (13u + block)) ^ (0x33u + block * 7u));
    }
  }

  std::array<float, QK_K * block_count> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 11u) % 37u) - 18;
    src1[i] = static_cast<float>(centered) * 0.03125f;
  }

  std::array<block_q8_k, block_count> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), static_cast<int32_t>(block_count), q8_blocks.data(),
      emel::kernel::detail::quant::QK_K);

  const float scalar =
      emel::kernel::detail::dot_q3_k_q8_k_row_scalar(q3_blocks.data(), q8_blocks.data(),
                                                     q8_blocks.size());
  const float neon =
      emel::kernel::aarch64::detail::dot_q3_k_q8_k_row_neon(q3_blocks.data(), q8_blocks.data(),
                                                             q8_blocks.size());

  CHECK(neon == doctest::Approx(scalar).epsilon(1e-7f));
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

TEST_CASE("kernel_aarch64_q6_4rows_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q6_k;
  using emel::kernel::detail::quant::block_q8_k;

  std::array<float, QK_K> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 13u) % 37u) - 18;
    src1[i] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<block_q6_k, 4> q6_rows = {};
  for (size_t row = 0; row < q6_rows.size(); ++row) {
    q6_rows[row].d = 0x3c00u;
    for (size_t i = 0; i < q6_rows[row].scales.size(); ++i) {
      const int32_t scale_value = static_cast<int32_t>((i + row * 5u) % 15u) - 7;
      q6_rows[row].scales[i] = static_cast<int8_t>(scale_value);
    }
    for (size_t i = 0; i < q6_rows[row].ql.size(); ++i) {
      q6_rows[row].ql[i] = static_cast<uint8_t>((i * (11u + row)) ^ (0x35u + row));
    }
    for (size_t i = 0; i < q6_rows[row].qh.size(); ++i) {
      q6_rows[row].qh[i] = static_cast<uint8_t>((i * (7u + row)) ^ (0x92u - row));
    }
  }

  std::array<block_q8_k, 1> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), 1, &q8_blocks[0], emel::kernel::detail::quant::QK_K);

  float scalar[4] = {};
  for (size_t row = 0; row < q6_rows.size(); ++row) {
    scalar[row] = emel::kernel::detail::dot_q6_k_q8_k_row_scalar(
        &q6_rows[row], q8_blocks.data(), q8_blocks.size());
  }

  float neon[4] = {};
  emel::kernel::aarch64::detail::dot_q6_k_q8_k_4rows_neon(
      &q6_rows[0], &q6_rows[1], &q6_rows[2], &q6_rows[3], q8_blocks.data(),
      q8_blocks.size(), neon);

  CHECK(neon[0] == doctest::Approx(scalar[0]).epsilon(1e-5f));
  CHECK(neon[1] == doctest::Approx(scalar[1]).epsilon(1e-5f));
  CHECK(neon[2] == doctest::Approx(scalar[2]).epsilon(1e-5f));
  CHECK(neon[3] == doctest::Approx(scalar[3]).epsilon(1e-5f));
#endif
}

TEST_CASE("kernel_aarch64_q4_k_row_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  SUCCEED();
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q4_k;
  using emel::kernel::detail::quant::block_q8_k;

  block_q4_k q4 = {};
  q4.d = 0x3c00u;
  q4.dmin = 0x3800u;
  for (size_t i = 0; i < q4.scales.size(); ++i) {
    q4.scales[i] = static_cast<uint8_t>((i * 11u) ^ 0x5au);
  }
  for (size_t i = 0; i < q4.qs.size(); ++i) {
    q4.qs[i] = static_cast<uint8_t>((i * 9u) ^ 0x96u);
  }

  std::array<float, QK_K> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>(i % 31u) - 15;
    src1[i] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<block_q8_k, 1> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), 1, &q8_blocks[0], emel::kernel::detail::quant::QK_K);

  const float scalar =
      emel::kernel::detail::dot_q4_k_q8_k_row_scalar(&q4, q8_blocks.data(), q8_blocks.size());
  const float neon =
      emel::kernel::aarch64::detail::dot_q4_k_q8_k_row_neon(&q4, q8_blocks.data(),
                                                             q8_blocks.size());

  CHECK(neon == doctest::Approx(scalar).epsilon(1e-5f));
#endif
}

TEST_CASE("kernel_aarch64_q4_k_2rows_neon_matches_scalar") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  SUCCEED();
  return;
#else
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q4_k;
  using emel::kernel::detail::quant::block_q8_k;

  std::array<block_q4_k, 2> q4_rows = {};
  for (size_t row = 0; row < q4_rows.size(); ++row) {
    q4_rows[row].d = 0x3c00u;
    q4_rows[row].dmin = static_cast<uint16_t>(0x3800u + row * 0x0100u);
    for (size_t i = 0; i < q4_rows[row].scales.size(); ++i) {
      q4_rows[row].scales[i] = static_cast<uint8_t>((i * (11u + row)) ^ (0x5au + row));
    }
    for (size_t i = 0; i < q4_rows[row].qs.size(); ++i) {
      q4_rows[row].qs[i] = static_cast<uint8_t>((i * (9u + row)) ^ (0x96u - row));
    }
  }

  std::array<float, QK_K> src1 = {};
  for (size_t i = 0; i < src1.size(); ++i) {
    const int32_t centered = static_cast<int32_t>((i * 7u) % 31u) - 15;
    src1[i] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<block_q8_k, 1> q8_blocks = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      src1.data(), 1, &q8_blocks[0], emel::kernel::detail::quant::QK_K);

  float scalar[2] = {};
  for (size_t row = 0; row < q4_rows.size(); ++row) {
    scalar[row] = emel::kernel::detail::dot_q4_k_q8_k_row_scalar(
        &q4_rows[row], q8_blocks.data(), q8_blocks.size());
  }

  float neon[2] = {};
  emel::kernel::aarch64::detail::dot_q4_k_q8_k_2rows_neon(
      &q4_rows[0], &q4_rows[1], q8_blocks.data(), q8_blocks.size(), neon);

  CHECK(neon[0] == doctest::Approx(scalar[0]).epsilon(1e-5f));
  CHECK(neon[1] == doctest::Approx(scalar[1]).epsilon(1e-5f));
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

TEST_CASE("kernel_aarch64_sm_reports_q8_0_vectorized_dispatch_at_kernel_seam") {
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q8_0;

  std::array<float, QK8_0> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>(idx % 23u) - 11;
    input[idx] = static_cast<float>(centered) * 0.0625f;
  }

  block_q8_0 q8 = {};
  q8.d = emel::kernel::detail::quant::fp32_to_fp16(1.0f / 16.0f);
  for (size_t idx = 0; idx < q8.qs.size(); ++idx) {
    q8.qs[idx] = static_cast<int8_t>(static_cast<int32_t>(idx % 17u) - 8);
  }

  float q8_out[1] = {};
  const emel::kernel::event::op_mul_mat q8_ev{
      .src0 = make_quantized_src(&q8, dtype::q8_0, QK8_0, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK8_0),
      .dst = make_dst(q8_out, dtype::f32, 1, 1),
      .nth = 1,
  };

  aarch64_sm machine{};
  CHECK(machine.process_event(q8_ev));

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 1u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q8_0_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 1u);
#endif
}

TEST_CASE("kernel_aarch64_q4_k_uses_neon_dispatch_when_dotprod_is_available") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q4_k;

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    values.fill(1.0f);
    return values;
  }();

  block_q4_k q4 = {};
  q4.d = 0x3c00u;
  q4.dmin = 0x3c00u;
  q4.scales.fill(0u);
  q4.scales[0] = 0x01u;
  q4.scales[1] = 0x01u;
  q4.scales[2] = 0x01u;
  q4.scales[3] = 0x01u;
  q4.scales[4] = 0x01u;
  q4.scales[5] = 0x01u;
  q4.scales[6] = 0x01u;
  q4.scales[7] = 0x01u;
  q4.scales[8] = 0x11u;
  q4.scales[9] = 0x11u;
  q4.scales[10] = 0x11u;
  q4.scales[11] = 0x11u;
  std::fill(q4.qs.begin(), q4.qs.end(), static_cast<uint8_t>(0x00u));

  float shared_out[1] = {};
  float scalar_out[1] = {};
  const emel::kernel::event::op_mul_mat shared_ev{
      .src0 = make_quantized_src(&q4, dtype::q4_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(shared_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(&q4, dtype::q4_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 1, QK_K),
      .dst = make_dst(scalar_out, dtype::f32, 1, 1),
      .nth = 1,
  };

  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

#if defined(__ARM_FEATURE_DOTPROD)
  CHECK(emel::kernel::aarch64::detail::can_use_neon(shared_ev, true));
#else
  CHECK_FALSE(emel::kernel::aarch64::detail::can_use_neon(shared_ev, true));
#endif

  aarch64_sm machine{};
  CHECK(machine.process_event(shared_ev));
  CHECK(shared_out[0] == doctest::Approx(scalar_out[0]).epsilon(1.0e-6f));
  CHECK(shared_out[0] == doctest::Approx(-256.0f).epsilon(1.0e-6f));
}

TEST_CASE("kernel_aarch64_supported_quantized_dispatch_is_alloc_free") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q2_k;
  using emel::kernel::detail::quant::block_q3_k;
  using emel::kernel::detail::quant::block_q6_k;
  using emel::kernel::detail::quant::block_q8_0;

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

  block_q8_0 q8 = {};
  q8.d = emel::kernel::detail::quant::fp32_to_fp16(1.0f / 16.0f);
  for (size_t idx = 0; idx < q8.qs.size(); ++idx) {
    q8.qs[idx] = static_cast<int8_t>(static_cast<int32_t>(idx % 15u) - 7);
  }

  float q2_out[1] = {};
  float q3_out[1] = {};
  float q6_out[1] = {};
  float q8_out[1] = {};
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
  const std::array<float, QK8_0> q8_input = [] {
    std::array<float, QK8_0> values = {};
    for (size_t idx = 0; idx < values.size(); ++idx) {
      const int32_t centered = static_cast<int32_t>(idx % 19u) - 9;
      values[idx] = static_cast<float>(centered) * 0.0625f;
    }
    return values;
  }();
  const emel::kernel::event::op_mul_mat q8_ev{
      .src0 = make_quantized_src(&q8, dtype::q8_0, QK8_0, 1),
      .src1 = make_src(q8_input.data(), dtype::f32, 1, QK8_0),
      .dst = make_dst(q8_out, dtype::f32, 1, 1),
      .nth = 1,
  };
  std::array<block_q8_0, 4> q8_rows = {};
  for (size_t row = 0; row < q8_rows.size(); ++row) {
    q8_rows[row] = q8;
    q8_rows[row].d = emel::kernel::detail::quant::fp32_to_fp16(
        0.03125f * static_cast<float>(row + 1u));
  }
  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q8_0x4) *
                 emel::kernel::detail::quant::packed_q8_0_x4_group_count(4u)>
      q8_packed_storage = {};
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl4(
      q8_rows.data(), 4u, QK8_0, q8_packed_storage.data()));
  const emel::kernel::event::op_mul_mat q8_packed_ev{
      .src0 = make_packed_q8_0_x4_bl4_src(q8_packed_storage.data(), QK8_0, 4u),
      .src1 = make_q8_0_vector_src(q8_input.data(), QK8_0),
      .dst = make_dst(q8_out, dtype::f32, 1, 4u),
      .nth = 1,
  };

  aarch64_sm machine{};
  allocation_scope allocations{};
  CHECK(machine.process_event(q2_ev));
  CHECK(machine.process_event(q3_ev));
  CHECK(machine.process_event(q6_ev));
  CHECK(machine.process_event(q8_ev));
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  CHECK(machine.process_event(q8_packed_ev));
#else
  CHECK_FALSE(machine.process_event(q8_packed_ev));
#endif
  CHECK(allocations.allocations() == 0u);

#if defined(__aarch64__) || defined(__ARM_NEON)
  constexpr uint64_t expected_packed_q8_dispatches =
#if defined(__ARM_FEATURE_DOTPROD)
      1u;
#else
      0u;
#endif
  CHECK(machine.optimized_q2_dispatch_count() == 1u);
  CHECK(machine.shared_q2_dispatch_count() == 0u);
  CHECK(machine.optimized_q3_dispatch_count() == 1u);
  CHECK(machine.shared_q3_dispatch_count() == 0u);
  CHECK(machine.optimized_q6_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_dispatch_count() == 1u);
  CHECK(machine.shared_q6_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u + expected_packed_q8_dispatches);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == expected_packed_q8_dispatches);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q2_dispatch_count() == 0u);
  CHECK(machine.shared_q2_dispatch_count() == 1u);
  CHECK(machine.optimized_q3_dispatch_count() == 0u);
  CHECK(machine.shared_q3_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_dispatch_count() == 0u);
  CHECK(machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q6_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 1u);
#endif
}

TEST_CASE("kernel_aarch64_q6_matrix_dispatch_does_not_claim_vector_path") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q6_k;

  std::array<float, QK_K * 2u> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>(idx % 17u) - 8;
    input[idx] = static_cast<float>(centered) * 0.125f;
  }

  block_q6_k q6 = {};
  q6.d = 0x3c00u;
  std::fill(q6.scales.begin(), q6.scales.end(), static_cast<int8_t>(3));
  std::fill(q6.ql.begin(), q6.ql.end(), static_cast<uint8_t>(0x66u));
  std::fill(q6.qh.begin(), q6.qh.end(), static_cast<uint8_t>(0x77u));

  std::array<float, 2u> out = {};
  const emel::kernel::event::op_mul_mat q6_ev{
      .src0 = make_quantized_src(&q6, dtype::q6_k, QK_K, 1),
      .src1 = make_src(input.data(), dtype::f32, 2u, QK_K),
      .dst = make_dst(out.data(), dtype::f32, 2u, 1u),
      .nth = 1,
  };

  aarch64_sm machine{};
  CHECK(machine.process_event(q6_ev));

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(machine.optimized_q6_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q6_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q6_dispatch_count() == 0u);
  CHECK(machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q6_dispatch_count() == 1u);
#endif
}

TEST_CASE("kernel_aarch64_q6_packed_vector_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q6_k;

  constexpr uint64_t k_rows = 8u;
  std::array<block_q6_k, k_rows> native_rows = {};
  for (size_t row = 0; row < native_rows.size(); ++row) {
    native_rows[row].d = 0x3c00u;
    for (size_t idx = 0; idx < native_rows[row].scales.size(); ++idx) {
      native_rows[row].scales[idx] =
          static_cast<int8_t>((static_cast<int32_t>((row + idx) % 13u)) - 6);
    }
    for (size_t idx = 0; idx < native_rows[row].ql.size(); ++idx) {
      native_rows[row].ql[idx] = static_cast<uint8_t>(((row + 1u) * 17u + idx * 7u) & 0xffu);
    }
    for (size_t idx = 0; idx < native_rows[row].qh.size(); ++idx) {
      native_rows[row].qh[idx] = static_cast<uint8_t>(((row + 3u) * 11u + idx * 5u) & 0xffu);
    }
  }

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    for (size_t idx = 0; idx < values.size(); ++idx) {
      const int32_t centered = static_cast<int32_t>(idx % 29u) - 14;
      values[idx] = static_cast<float>(centered) * 0.0625f;
    }
    return values;
  }();

  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q6_kx8) * (QK_K / QK_K)>
      packed_storage = {};
  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q6_kx8_q8_argmax_prepared) * (QK_K / QK_K)>
      argmax_prepared_storage = {};
  REQUIRE(emel::kernel::detail::quant::pack_q6_k_rows_x8(
      native_rows.data(),
      k_rows,
      QK_K,
      packed_storage.data()));
  REQUIRE(emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_argmax_prepared(
      native_rows.data(),
      k_rows,
      QK_K,
      argmax_prepared_storage.data()));
  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q6_kx8_q8_prepared) * (QK_K / QK_K)>
      prepared_storage = {};
  REQUIRE(emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_prepared(
      native_rows.data(),
      k_rows,
      QK_K,
      prepared_storage.data()));

  std::array<emel::kernel::detail::quant::block_q8_k, QK_K / QK_K> q8_input = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      input.data(),
      1u,
      q8_input.data(),
      static_cast<int64_t>(QK_K));

  std::array<float, k_rows> native_out = {};
  std::array<float, k_rows> packed_out = {};
  std::array<float, k_rows> prepared_out = {};
  const emel::kernel::event::op_mul_mat native_ev{
      .src0 = make_quantized_src(native_rows.data(), dtype::q6_k, QK_K, k_rows),
      .src1 = make_src(input.data(), dtype::f32, 1u, QK_K),
      .dst = make_dst(native_out.data(), dtype::f32, 1u, k_rows),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat packed_ev{
      .src0 = make_packed_q6_k_x8_src(packed_storage.data(), QK_K, k_rows),
      .src1 = make_q8_k_vector_src(q8_input.data(), QK_K),
      .dst = make_dst(packed_out.data(), dtype::f32, 1u, k_rows),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat prepared_ev{
      .src0 = make_prepared_q6_k_x8_q8_src(prepared_storage.data(), QK_K, k_rows),
      .src1 = make_q8_k_vector_src(q8_input.data(), QK_K),
      .dst = make_dst(prepared_out.data(), dtype::f32, 1u, k_rows),
      .nth = 1,
  };

  aarch64_sm native_machine{};
  aarch64_sm packed_machine{};
  aarch64_sm prepared_machine{};
  CHECK(native_machine.process_event(native_ev));
  CHECK(packed_machine.process_event(packed_ev));
  CHECK(prepared_machine.process_event(prepared_ev));

  for (size_t row = 0; row < native_out.size(); ++row) {
    CHECK(packed_out[row] == doctest::Approx(native_out[row]).epsilon(1.0e-6f));
    CHECK(prepared_out[row] == doctest::Approx(native_out[row]).epsilon(1.0e-6f));
  }

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  CHECK(packed_machine.optimized_q6_dispatch_count() == 1u);
  CHECK(packed_machine.optimized_q6_vector_dispatch_count() == 1u);
  CHECK(packed_machine.optimized_q6_vector_packed_dispatch_count() == 1u);
  CHECK(packed_machine.optimized_q6_vector_packed_q8_rhs_dispatch_count() == 1u);
  CHECK(packed_machine.shared_q6_dispatch_count() == 0u);
  CHECK(prepared_machine.optimized_q6_dispatch_count() == 1u);
  CHECK(prepared_machine.optimized_q6_vector_dispatch_count() == 1u);
  CHECK(prepared_machine.optimized_q6_vector_packed_dispatch_count() == 1u);
  CHECK(prepared_machine.optimized_q6_vector_packed_q8_rhs_dispatch_count() == 1u);
  CHECK(prepared_machine.optimized_q6_vector_prepared_q8_rhs_dispatch_count() == 1u);
#if defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(prepared_machine.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count() == 1u);
#else
  CHECK(prepared_machine.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count() == 0u);
#endif
  CHECK(prepared_machine.shared_q6_dispatch_count() == 0u);
#else
  CHECK(packed_machine.optimized_q6_dispatch_count() == 0u);
  CHECK(packed_machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(packed_machine.optimized_q6_vector_packed_dispatch_count() == 0u);
  CHECK(packed_machine.optimized_q6_vector_packed_q8_rhs_dispatch_count() == 0u);
  CHECK(packed_machine.shared_q6_dispatch_count() == 0u);
  CHECK(prepared_machine.optimized_q6_dispatch_count() == 0u);
  CHECK(prepared_machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(prepared_machine.optimized_q6_vector_packed_dispatch_count() == 0u);
  CHECK(prepared_machine.optimized_q6_vector_packed_q8_rhs_dispatch_count() == 0u);
  CHECK(prepared_machine.optimized_q6_vector_prepared_q8_rhs_dispatch_count() == 0u);
  CHECK(prepared_machine.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count() == 0u);
  CHECK(prepared_machine.shared_q6_dispatch_count() == 0u);
#endif
}

TEST_CASE("kernel_aarch64_q4_packed_vector_route_is_explicit_and_numeric_match") {
#if !(defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD))
  SUCCEED();
#else
  using emel::kernel::detail::quant::Q4_K_X8_ROWS;
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q4_k;

  std::array<block_q4_k, Q4_K_X8_ROWS> native_rows = {};
  for (size_t row = 0; row < native_rows.size(); ++row) {
    native_rows[row].d = 0x3800u;
    native_rows[row].dmin = 0x3400u;
    for (size_t idx = 0; idx < native_rows[row].scales.size(); ++idx) {
      native_rows[row].scales[idx] =
          static_cast<uint8_t>(((row + 5u) * 23u + idx * 11u) & 0xffu);
    }
    for (size_t idx = 0; idx < native_rows[row].qs.size(); ++idx) {
      native_rows[row].qs[idx] = static_cast<uint8_t>(((row + 1u) * 29u + idx * 3u) & 0xffu);
    }
  }

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    for (size_t idx = 0; idx < values.size(); ++idx) {
      const int32_t centered = static_cast<int32_t>(idx % 31u) - 15;
      values[idx] = static_cast<float>(centered) * 0.0625f;
    }
    return values;
  }();

  std::array<uint8_t, sizeof(emel::kernel::detail::quant::block_q4_kx8)> packed_bl4_storage = {};
  std::array<uint8_t, sizeof(emel::kernel::detail::quant::block_q4_kx8)> packed_bl8_storage = {};
  REQUIRE(emel::kernel::detail::quant::pack_q4_k_rows_x8_bl4(
      native_rows.data(), Q4_K_X8_ROWS, QK_K, packed_bl4_storage.data()));
  REQUIRE(emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
      native_rows.data(), Q4_K_X8_ROWS, QK_K, packed_bl8_storage.data()));

  std::array<emel::kernel::detail::quant::block_q8_k, QK_K / QK_K> q8_input = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      input.data(), 1u, q8_input.data(), static_cast<int64_t>(QK_K));

  std::array<float, Q4_K_X8_ROWS> native_out = {};
  std::array<float, Q4_K_X8_ROWS> packed_bl4_out = {};
  std::array<float, Q4_K_X8_ROWS> packed_bl8_out = {};
  const emel::kernel::event::op_mul_mat native_ev{
      .src0 = make_quantized_src(native_rows.data(), dtype::q4_k, QK_K, Q4_K_X8_ROWS),
      .src1 = make_src(input.data(), dtype::f32, 1u, QK_K),
      .dst = make_dst(native_out.data(), dtype::f32, 1u, Q4_K_X8_ROWS),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat packed_bl4_ev{
      .src0 = make_packed_q4_k_x8_bl4_src(packed_bl4_storage.data(), QK_K, Q4_K_X8_ROWS),
      .src1 = make_q8_k_vector_src(q8_input.data(), QK_K),
      .dst = make_dst(packed_bl4_out.data(), dtype::f32, 1u, Q4_K_X8_ROWS),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat packed_bl8_ev{
      .src0 = make_packed_q4_k_x8_bl8_src(packed_bl8_storage.data(), QK_K, Q4_K_X8_ROWS),
      .src1 = make_q8_k_vector_src(q8_input.data(), QK_K),
      .dst = make_dst(packed_bl8_out.data(), dtype::f32, 1u, Q4_K_X8_ROWS),
      .nth = 1,
  };

  aarch64_sm native_machine{};
  aarch64_sm packed_bl4_machine{};
  aarch64_sm packed_bl8_machine{};
  CHECK(native_machine.process_event(native_ev));
  CHECK(packed_bl4_machine.process_event(packed_bl4_ev));
  CHECK(packed_bl8_machine.process_event(packed_bl8_ev));

  for (size_t row = 0; row < native_out.size(); ++row) {
    CHECK(packed_bl4_out[row] == doctest::Approx(native_out[row]).epsilon(1.0e-5f));
    CHECK(packed_bl8_out[row] == doctest::Approx(native_out[row]).epsilon(1.0e-5f));
  }

  CHECK(packed_bl4_machine.optimized_q4_dispatch_count() == 1u);
  CHECK(packed_bl4_machine.optimized_q4_vector_dispatch_count() == 1u);
  CHECK(packed_bl4_machine.optimized_q4_vector_packed_dispatch_count() == 1u);
  CHECK(packed_bl4_machine.optimized_q4_vector_packed_q8_rhs_dispatch_count() == 1u);
  CHECK(packed_bl4_machine.shared_q4_dispatch_count() == 0u);
  CHECK(packed_bl8_machine.optimized_q4_dispatch_count() == 1u);
  CHECK(packed_bl8_machine.optimized_q4_vector_dispatch_count() == 1u);
  CHECK(packed_bl8_machine.optimized_q4_vector_packed_dispatch_count() == 1u);
  CHECK(packed_bl8_machine.optimized_q4_vector_packed_q8_rhs_dispatch_count() == 1u);
  CHECK(packed_bl8_machine.shared_q4_dispatch_count() == 0u);
#endif
}

TEST_CASE("kernel_aarch64_sm_reports_q8_0_packed_dispatch_at_kernel_seam") {
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q8_0;

  constexpr uint64_t row_count = 4u;
  std::array<block_q8_0, row_count> native_rows = {};
  for (size_t row = 0; row < native_rows.size(); ++row) {
    native_rows[row].d = emel::kernel::detail::quant::fp32_to_fp16(
        0.0625f * static_cast<float>(row + 1u));
    for (size_t idx = 0; idx < native_rows[row].qs.size(); ++idx) {
      native_rows[row].qs[idx] = static_cast<int8_t>(
          static_cast<int32_t>(((row + 1u) * 9u + idx * 3u) % 29u) - 14);
    }
  }

  const std::array<float, QK8_0> input = [] {
    std::array<float, QK8_0> values = {};
    for (size_t idx = 0; idx < values.size(); ++idx) {
      const int32_t centered = static_cast<int32_t>(idx % 17u) - 8;
      values[idx] = static_cast<float>(centered) * 0.0625f;
    }
    return values;
  }();
  std::array<block_q8_0, QK8_0 / QK8_0> q8_input = {};
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      input.data(), 1u, q8_input.data(), static_cast<int64_t>(QK8_0));

  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q8_0x4) *
                 emel::kernel::detail::quant::packed_q8_0_x4_group_count(row_count)>
      packed_bl4 = {};
  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q8_0x4) *
                 emel::kernel::detail::quant::packed_q8_0_x4_group_count(row_count)>
      packed_bl8 = {};
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl4(
      native_rows.data(), row_count, QK8_0, packed_bl4.data()));
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      native_rows.data(), row_count, QK8_0, packed_bl8.data()));

  float out[4] = {};
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl8_src(packed_bl8.data(), QK8_0, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), QK8_0),
      .dst = make_dst(out, dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl4_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl4_src(packed_bl4.data(), QK8_0, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), QK8_0),
      .dst = make_dst(out, dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl4_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl4_src(packed_bl4.data(), QK8_0, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), QK8_0),
      .dst = make_dst(out, dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.optimized_q8_0_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl4_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#endif
}

TEST_CASE("kernel_aarch64_q8_0_packed_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q8_0;

  constexpr uint64_t row_count = 8u;
  std::array<block_q8_0, row_count> native_rows = {};
  for (size_t row = 0; row < native_rows.size(); ++row) {
    native_rows[row].d = emel::kernel::detail::quant::fp32_to_fp16(
        0.03125f * static_cast<float>((row % 5u) + 1u));
    for (size_t idx = 0; idx < native_rows[row].qs.size(); ++idx) {
      native_rows[row].qs[idx] = static_cast<int8_t>(
          static_cast<int32_t>(((row + 2u) * 11u + idx * 5u) % 31u) - 15);
    }
  }

  std::array<block_q8_0, QK8_0 / QK8_0> q8_input = {};
  const std::array<float, QK8_0> input = [] {
    std::array<float, QK8_0> values = {};
    for (size_t idx = 0; idx < values.size(); ++idx) {
      const int32_t centered = static_cast<int32_t>(idx % 21u) - 10;
      values[idx] = static_cast<float>(centered) * 0.03125f;
    }
    return values;
  }();
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      input.data(), 1u, q8_input.data(), static_cast<int64_t>(QK8_0));

  std::array<float, row_count> reference = {};
  for (size_t row = 0; row < row_count; ++row) {
    reference[row] = emel::kernel::detail::dot_q8_0_q8_0_row_scalar(
        native_rows.data() + row, q8_input.data(), 1u);
  }

  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q8_0x4) *
                 emel::kernel::detail::quant::packed_q8_0_x4_group_count(row_count)>
      packed_bl4 = {};
  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q8_0x4) *
                 emel::kernel::detail::quant::packed_q8_0_x4_group_count(row_count)>
      packed_bl8 = {};
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl4(
      native_rows.data(), row_count, QK8_0, packed_bl4.data()));
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      native_rows.data(), row_count, QK8_0, packed_bl8.data()));

  std::array<float, row_count> packed_out = {};
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl8_src(packed_bl8.data(), QK8_0, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), QK8_0),
      .dst = make_dst(packed_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t row = 0; row < packed_out.size(); ++row) {
    CHECK(packed_out[row] == doctest::Approx(reference[row]).epsilon(1.0e-6f));
  }
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl4_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl4_src(packed_bl4.data(), QK8_0, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), QK8_0),
      .dst = make_dst(packed_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t row = 0; row < packed_out.size(); ++row) {
    CHECK(packed_out[row] == doctest::Approx(reference[row]).epsilon(1.0e-6f));
  }
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl4_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl4_src(packed_bl4.data(), QK8_0, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), QK8_0),
      .dst = make_dst(packed_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.optimized_q8_0_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl4_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#endif
}

TEST_CASE("kernel_aarch64_q8_0_packed_route_matches_multi_block_native_reference") {
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q8_0;

  constexpr uint64_t row_count = 10u;
  constexpr uint64_t col_count = QK8_0 * 8u;
  constexpr uint64_t block_count = col_count / QK8_0;

  std::vector<block_q8_0> native_rows(static_cast<size_t>(row_count * block_count));
  for (size_t row = 0; row < static_cast<size_t>(row_count); ++row) {
    for (size_t block = 0; block < static_cast<size_t>(block_count); ++block) {
      auto & cell = native_rows[row * static_cast<size_t>(block_count) + block];
      cell.d = emel::kernel::detail::quant::fp32_to_fp16(
          0.0078125f * static_cast<float>(((row + 3u) * (block + 5u)) % 23u + 1u));
      for (size_t idx = 0; idx < cell.qs.size(); ++idx) {
        const int32_t centered =
            static_cast<int32_t>(((row + 7u) * 17u + (block + 11u) * 13u + idx * 5u) % 255u) -
            127;
        cell.qs[idx] = static_cast<int8_t>(std::clamp(centered, -127, 127));
      }
    }
  }

  std::array<float, col_count> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 9u + 17u) % 63u) - 31;
    input[idx] = static_cast<float>(centered) * 0.015625f;
  }

  std::vector<block_q8_0> q8_input(static_cast<size_t>(block_count));
  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      input.data(), 1u, q8_input.data(), static_cast<int64_t>(col_count));

  std::array<float, row_count> reference = {};
  for (size_t row = 0; row < static_cast<size_t>(row_count); ++row) {
    reference[row] = emel::kernel::detail::dot_q8_0_q8_0_row_scalar(
        native_rows.data() + row * static_cast<size_t>(block_count),
        q8_input.data(),
        block_count);
  }

  std::vector<uint8_t> packed_bl4(
      sizeof(emel::kernel::detail::quant::block_q8_0x4) *
      emel::kernel::detail::quant::packed_q8_0_x4_group_count(row_count) * block_count);
  std::vector<uint8_t> packed_bl8(
      sizeof(emel::kernel::detail::quant::block_q8_0x4) *
      emel::kernel::detail::quant::packed_q8_0_x4_group_count(row_count) * block_count);
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl4(
      native_rows.data(), row_count, col_count, packed_bl4.data()));
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      native_rows.data(), row_count, col_count, packed_bl8.data()));

  std::array<float, row_count> packed_out = {};
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl8_src(packed_bl8.data(), col_count, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), col_count),
      .dst = make_dst(packed_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t row = 0; row < packed_out.size(); ++row) {
    CHECK(packed_out[row] == doctest::Approx(reference[row]).epsilon(1.0e-6f));
  }
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 0u);
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl4_src(packed_bl4.data(), col_count, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), col_count),
      .dst = make_dst(packed_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t row = 0; row < packed_out.size(); ++row) {
    CHECK(packed_out[row] == doctest::Approx(reference[row]).epsilon(1.0e-6f));
  }
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl4_src(packed_bl4.data(), col_count, row_count),
      .src1 = make_q8_0_vector_src(q8_input.data(), col_count),
      .dst = make_dst(packed_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
#endif
}

TEST_CASE("kernel_aarch64_q8_0_packed_bl8_matrix_x4_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::Q8_0_X4_ROWS;
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q8_0;

  constexpr uint64_t row_count = 8u;
  constexpr uint64_t rhs_rows = Q8_0_X4_ROWS;
  constexpr uint64_t col_count = QK8_0 * 8u;
  constexpr uint64_t block_count = col_count / QK8_0;

  std::vector<block_q8_0> native_rows(static_cast<size_t>(row_count * block_count));
  for (size_t row = 0; row < static_cast<size_t>(row_count); ++row) {
    for (size_t block = 0; block < static_cast<size_t>(block_count); ++block) {
      auto & cell = native_rows[row * static_cast<size_t>(block_count) + block];
      cell.d = emel::kernel::detail::quant::fp32_to_fp16(
          0.0078125f * static_cast<float>(((row + 3u) * (block + 5u)) % 23u + 1u));
      for (size_t idx = 0; idx < cell.qs.size(); ++idx) {
        const int32_t centered =
            static_cast<int32_t>(((row + 7u) * 17u + (block + 11u) * 13u + idx * 5u) % 255u) -
            127;
        cell.qs[idx] = static_cast<int8_t>(std::clamp(centered, -127, 127));
      }
    }
  }

  std::vector<float> rhs_dense(static_cast<size_t>(rhs_rows * col_count), 0.0f);
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    for (size_t col = 0; col < static_cast<size_t>(col_count); ++col) {
      const int32_t centered =
          static_cast<int32_t>(((row + 5u) * 19u + col * 7u) % 63u) - 31;
      rhs_dense[row * static_cast<size_t>(col_count) + col] =
          static_cast<float>(centered) * 0.015625f;
    }
  }

  std::vector<block_q8_0> rhs_q8(static_cast<size_t>(rhs_rows * block_count));
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    emel::kernel::detail::quant::quantize_row_q8_0_strided(
        rhs_dense.data() + row * static_cast<size_t>(col_count),
        1u,
        rhs_q8.data() + row * static_cast<size_t>(block_count),
        static_cast<int64_t>(col_count));
  }

  std::vector<float> reference(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  for (size_t rhs_row = 0; rhs_row < static_cast<size_t>(rhs_rows); ++rhs_row) {
    for (size_t lhs_row = 0; lhs_row < static_cast<size_t>(row_count); ++lhs_row) {
      reference[rhs_row * static_cast<size_t>(row_count) + lhs_row] =
          emel::kernel::detail::dot_q8_0_q8_0_row_scalar(
              native_rows.data() + lhs_row * static_cast<size_t>(block_count),
              rhs_q8.data() + rhs_row * static_cast<size_t>(block_count),
              block_count);
    }
  }

  std::vector<uint8_t> packed_bl8(
      sizeof(emel::kernel::detail::quant::block_q8_0x4) *
      emel::kernel::detail::quant::packed_q8_0_x4_group_count(row_count) * block_count);
  std::vector<uint8_t> rhs_packed_bl8(
      sizeof(emel::kernel::detail::quant::block_q8_0x4) * block_count);
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      native_rows.data(), row_count, col_count, packed_bl8.data()));
  REQUIRE(emel::kernel::detail::quant::pack_q8_0_rows_x4_bl8(
      rhs_q8.data(), rhs_rows, col_count, rhs_packed_bl8.data()));

  std::vector<float> packed_out(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl8_src(packed_bl8.data(), col_count, row_count),
      .src1 = make_packed_q8_0_x4_bl8_rhs_src(rhs_packed_bl8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(packed_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t idx = 0; idx < packed_out.size(); ++idx) {
    CHECK(packed_out[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-6f));
  }
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_packed_bl8_full_groups_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_packed_bl8_matrix_x4_dispatch_count() == 1u);
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q8_0_x4_bl8_src(packed_bl8.data(), col_count, row_count),
      .src1 = make_packed_q8_0_x4_bl8_rhs_src(rhs_packed_bl8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(packed_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.optimized_q8_0_packed_bl8_matrix_x4_dispatch_count() == 0u);
#endif
}

TEST_CASE("kernel_aarch64_q4_k_packed_bl8_matrix_x4_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::Q8_0_X4_ROWS;
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q4_k;
  using emel::kernel::detail::quant::block_q8_k;

  constexpr uint64_t row_count = 8u;
  constexpr uint64_t rhs_rows = Q8_0_X4_ROWS;
  constexpr uint64_t col_count = QK_K;
  constexpr uint64_t block_count = col_count / QK_K;

  std::vector<block_q4_k> native_rows(static_cast<size_t>(row_count * block_count));
  for (size_t row = 0; row < static_cast<size_t>(row_count); ++row) {
    auto & q4 = native_rows[row];
    q4.d = 0x3c00u;
    q4.dmin = static_cast<uint16_t>(0x3800u + row * 0x0040u);
    for (size_t idx = 0; idx < q4.scales.size(); ++idx) {
      q4.scales[idx] = static_cast<uint8_t>((idx * (11u + row)) ^ (0x5au + row));
    }
    for (size_t idx = 0; idx < q4.qs.size(); ++idx) {
      q4.qs[idx] = static_cast<uint8_t>((idx * (9u + row)) ^ (0x96u - row));
    }
  }

  std::vector<float> rhs_dense(static_cast<size_t>(rhs_rows * col_count), 0.0f);
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    for (size_t col = 0; col < static_cast<size_t>(col_count); ++col) {
      const int32_t centered =
          static_cast<int32_t>(((row + 5u) * 19u + col * 7u) % 63u) - 31;
      rhs_dense[row * static_cast<size_t>(col_count) + col] =
          static_cast<float>(centered) * 0.015625f;
    }
  }

  std::vector<block_q8_k> rhs_q8(static_cast<size_t>(rhs_rows * block_count));
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    emel::kernel::detail::quant::quantize_row_q8_k_strided(
        rhs_dense.data() + row * static_cast<size_t>(col_count),
        1u,
        rhs_q8.data() + row * static_cast<size_t>(block_count),
        static_cast<int64_t>(col_count));
  }

  std::vector<float> reference(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  for (size_t rhs_row = 0; rhs_row < static_cast<size_t>(rhs_rows); ++rhs_row) {
    for (size_t lhs_row = 0; lhs_row < static_cast<size_t>(row_count); ++lhs_row) {
      reference[rhs_row * static_cast<size_t>(row_count) + lhs_row] =
          emel::kernel::detail::dot_q4_k_q8_k_row_scalar(
              native_rows.data() + lhs_row * static_cast<size_t>(block_count),
              rhs_q8.data() + rhs_row * static_cast<size_t>(block_count),
              block_count);
    }
  }

  std::vector<uint8_t> packed_bl8(
      sizeof(emel::kernel::detail::quant::block_q4_kx8) *
      emel::kernel::detail::quant::packed_q4_k_x8_group_count(row_count) * block_count);
  REQUIRE(emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
      native_rows.data(), row_count, col_count, packed_bl8.data()));

  std::vector<float> packed_out(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q4_k_x8_bl8_src(packed_bl8.data(), col_count, row_count),
      .src1 = make_q8_k_x4_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(packed_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t idx = 0; idx < packed_out.size(); ++idx) {
    CHECK(packed_out[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-4f));
  }
  CHECK(machine.optimized_q4_dispatch_count() == 1u);
  CHECK(machine.optimized_q4_vector_dispatch_count() == 1u);
  CHECK(machine.optimized_q4_vector_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q4_vector_packed_q8_rhs_dispatch_count() == 1u);
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q4_k_x8_bl8_src(packed_bl8.data(), col_count, row_count),
      .src1 = make_q8_k_x4_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(packed_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
#endif
}

TEST_CASE("kernel_aarch64_q4_k_packed_bl8_matrix_x8_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::Q4_K_X8_ROWS;
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q4_k;
  using emel::kernel::detail::quant::block_q8_k;

  constexpr uint64_t row_count = 8u;
  constexpr uint64_t rhs_rows = Q4_K_X8_ROWS;
  constexpr uint64_t col_count = QK_K;
  constexpr uint64_t block_count = col_count / QK_K;

  std::vector<block_q4_k> native_rows(static_cast<size_t>(row_count * block_count));
  for (size_t row = 0; row < static_cast<size_t>(row_count); ++row) {
    auto & q4 = native_rows[row];
    q4.d = 0x3c00u;
    q4.dmin = static_cast<uint16_t>(0x3600u + row * 0x0020u);
    for (size_t idx = 0; idx < q4.scales.size(); ++idx) {
      q4.scales[idx] = static_cast<uint8_t>((idx * (13u + row)) ^ (0x66u - row));
    }
    for (size_t idx = 0; idx < q4.qs.size(); ++idx) {
      q4.qs[idx] = static_cast<uint8_t>((idx * (7u + row)) ^ (0xa5u + row));
    }
  }

  std::vector<float> rhs_dense(static_cast<size_t>(rhs_rows * col_count), 0.0f);
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    for (size_t col = 0; col < static_cast<size_t>(col_count); ++col) {
      const int32_t centered =
          static_cast<int32_t>(((row + 3u) * 17u + col * 5u) % 67u) - 33;
      rhs_dense[row * static_cast<size_t>(col_count) + col] =
          static_cast<float>(centered) * 0.015625f;
    }
  }

  std::vector<block_q8_k> rhs_q8(static_cast<size_t>(rhs_rows * block_count));
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    emel::kernel::detail::quant::quantize_row_q8_k_strided(
        rhs_dense.data() + row * static_cast<size_t>(col_count),
        1u,
        rhs_q8.data() + row * static_cast<size_t>(block_count),
        static_cast<int64_t>(col_count));
  }

  std::vector<float> reference(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  for (size_t rhs_row = 0; rhs_row < static_cast<size_t>(rhs_rows); ++rhs_row) {
    for (size_t lhs_row = 0; lhs_row < static_cast<size_t>(row_count); ++lhs_row) {
      reference[rhs_row * static_cast<size_t>(row_count) + lhs_row] =
          emel::kernel::detail::dot_q4_k_q8_k_row_scalar(
              native_rows.data() + lhs_row * static_cast<size_t>(block_count),
              rhs_q8.data() + rhs_row * static_cast<size_t>(block_count),
              block_count);
    }
  }

  std::vector<uint8_t> packed_bl8(
      sizeof(emel::kernel::detail::quant::block_q4_kx8) *
      emel::kernel::detail::quant::packed_q4_k_x8_group_count(row_count) * block_count);
  REQUIRE(emel::kernel::detail::quant::pack_q4_k_rows_x8_bl8(
      native_rows.data(), row_count, col_count, packed_bl8.data()));

  std::vector<float> packed_out(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q4_k_x8_bl8_src(packed_bl8.data(), col_count, row_count),
      .src1 = make_q8_k_x8_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(packed_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t idx = 0; idx < packed_out.size(); ++idx) {
    CHECK(packed_out[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-4f));
  }
  CHECK(machine.optimized_q4_dispatch_count() == 1u);
  CHECK(machine.optimized_q4_vector_dispatch_count() == 1u);
  CHECK(machine.optimized_q4_vector_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q4_vector_packed_q8_rhs_dispatch_count() == 1u);
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_packed_q4_k_x8_bl8_src(packed_bl8.data(), col_count, row_count),
      .src1 = make_q8_k_x8_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(packed_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
#endif
}

TEST_CASE(
    "kernel_aarch64_q6_k_prepared_q8_rhs_i8mm_matrix_x4_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::Q8_0_X4_ROWS;
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q6_k;
  using emel::kernel::detail::quant::block_q8_k;

  constexpr uint64_t row_count = 8u;
  constexpr uint64_t rhs_rows = Q8_0_X4_ROWS;
  constexpr uint64_t col_count = QK_K;
  constexpr uint64_t block_count = col_count / QK_K;

  std::vector<block_q6_k> native_rows(static_cast<size_t>(row_count * block_count));
  for (size_t row = 0; row < static_cast<size_t>(row_count); ++row) {
    auto & q6 = native_rows[row];
    q6.d = 0x3c00u;
    for (size_t idx = 0; idx < q6.scales.size(); ++idx) {
      const int32_t scale_value = static_cast<int32_t>((idx + row * 5u) % 15u) - 7;
      q6.scales[idx] = static_cast<int8_t>(scale_value);
    }
    for (size_t idx = 0; idx < q6.ql.size(); ++idx) {
      q6.ql[idx] = static_cast<uint8_t>((idx * (11u + row)) ^ (0x35u + row));
    }
    for (size_t idx = 0; idx < q6.qh.size(); ++idx) {
      q6.qh[idx] = static_cast<uint8_t>((idx * (7u + row)) ^ (0x92u - row));
    }
  }

  std::vector<float> rhs_dense(static_cast<size_t>(rhs_rows * col_count), 0.0f);
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    for (size_t col = 0; col < static_cast<size_t>(col_count); ++col) {
      const int32_t centered =
          static_cast<int32_t>(((row + 7u) * 23u + col * 3u) % 59u) - 29;
      rhs_dense[row * static_cast<size_t>(col_count) + col] =
          static_cast<float>(centered) * 0.015625f;
    }
  }

  std::vector<block_q8_k> rhs_q8(static_cast<size_t>(rhs_rows * block_count));
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    emel::kernel::detail::quant::quantize_row_q8_k_strided(
        rhs_dense.data() + row * static_cast<size_t>(col_count),
        1u,
        rhs_q8.data() + row * static_cast<size_t>(block_count),
        static_cast<int64_t>(col_count));
  }

  std::vector<float> reference(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  for (size_t rhs_row = 0; rhs_row < static_cast<size_t>(rhs_rows); ++rhs_row) {
    for (size_t lhs_row = 0; lhs_row < static_cast<size_t>(row_count); ++lhs_row) {
      reference[rhs_row * static_cast<size_t>(row_count) + lhs_row] =
          emel::kernel::detail::dot_q6_k_q8_k_row_scalar(
              native_rows.data() + lhs_row * static_cast<size_t>(block_count),
              rhs_q8.data() + rhs_row * static_cast<size_t>(block_count),
              block_count);
    }
  }

  std::vector<uint8_t> prepared_storage(
      emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(col_count) *
      emel::kernel::detail::quant::packed_q6_k_x8_group_count(row_count));
  REQUIRE(emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_prepared(
      native_rows.data(), row_count, col_count, prepared_storage.data()));

  std::vector<float> prepared_out(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_prepared_q6_k_x8_q8_src(prepared_storage.data(), col_count, row_count),
      .src1 = make_q8_k_x4_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(prepared_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t idx = 0; idx < prepared_out.size(); ++idx) {
    CHECK(prepared_out[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-5f));
  }
  CHECK(machine.optimized_q6_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_packed_q8_rhs_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_prepared_q8_rhs_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count() == 1u);
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_prepared_q6_k_x8_q8_src(prepared_storage.data(), col_count, row_count),
      .src1 = make_q8_k_x4_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(prepared_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
#endif
}

TEST_CASE(
    "kernel_aarch64_q6_k_prepared_q8_rhs_i8mm_matrix_x8_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::Q6_K_X8_ROWS;
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q6_k;
  using emel::kernel::detail::quant::block_q8_k;

  constexpr uint64_t row_count = 8u;
  constexpr uint64_t rhs_rows = Q6_K_X8_ROWS;
  constexpr uint64_t col_count = QK_K;
  constexpr uint64_t block_count = col_count / QK_K;

  std::vector<block_q6_k> native_rows(static_cast<size_t>(row_count * block_count));
  for (size_t row = 0; row < static_cast<size_t>(row_count); ++row) {
    auto & q6 = native_rows[row];
    q6.d = 0x3c00u;
    for (size_t idx = 0; idx < q6.scales.size(); ++idx) {
      const int32_t scale_value = static_cast<int32_t>((idx + row * 7u) % 13u) - 6;
      q6.scales[idx] = static_cast<int8_t>(scale_value);
    }
    for (size_t idx = 0; idx < q6.ql.size(); ++idx) {
      q6.ql[idx] = static_cast<uint8_t>((idx * (9u + row)) ^ (0x41u + row));
    }
    for (size_t idx = 0; idx < q6.qh.size(); ++idx) {
      q6.qh[idx] = static_cast<uint8_t>((idx * (5u + row)) ^ (0x8eu - row));
    }
  }

  std::vector<float> rhs_dense(static_cast<size_t>(rhs_rows * col_count), 0.0f);
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    for (size_t col = 0; col < static_cast<size_t>(col_count); ++col) {
      const int32_t centered =
          static_cast<int32_t>(((row + 11u) * 21u + col * 7u) % 61u) - 30;
      rhs_dense[row * static_cast<size_t>(col_count) + col] =
          static_cast<float>(centered) * 0.015625f;
    }
  }

  std::vector<block_q8_k> rhs_q8(static_cast<size_t>(rhs_rows * block_count));
  for (size_t row = 0; row < static_cast<size_t>(rhs_rows); ++row) {
    emel::kernel::detail::quant::quantize_row_q8_k_strided(
        rhs_dense.data() + row * static_cast<size_t>(col_count),
        1u,
        rhs_q8.data() + row * static_cast<size_t>(block_count),
        static_cast<int64_t>(col_count));
  }

  std::vector<float> reference(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  for (size_t rhs_row = 0; rhs_row < static_cast<size_t>(rhs_rows); ++rhs_row) {
    for (size_t lhs_row = 0; lhs_row < static_cast<size_t>(row_count); ++lhs_row) {
      reference[rhs_row * static_cast<size_t>(row_count) + lhs_row] =
          emel::kernel::detail::dot_q6_k_q8_k_row_scalar(
              native_rows.data() + lhs_row * static_cast<size_t>(block_count),
              rhs_q8.data() + rhs_row * static_cast<size_t>(block_count),
              block_count);
    }
  }

  std::vector<uint8_t> prepared_storage(
      emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(col_count) *
      emel::kernel::detail::quant::packed_q6_k_x8_group_count(row_count));
  REQUIRE(emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_prepared(
      native_rows.data(), row_count, col_count, prepared_storage.data()));

  std::vector<float> prepared_out(static_cast<size_t>(rhs_rows * row_count), 0.0f);
  aarch64_sm machine{};

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_prepared_q6_k_x8_q8_src(prepared_storage.data(), col_count, row_count),
      .src1 = make_q8_k_x8_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(prepared_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK(machine.process_event(ev));
  for (size_t idx = 0; idx < prepared_out.size(); ++idx) {
    CHECK(prepared_out[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-5f));
  }
  CHECK(machine.optimized_q6_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_packed_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_packed_q8_rhs_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_prepared_q8_rhs_dispatch_count() == 1u);
  CHECK(machine.optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count() == 1u);
#else
  const emel::kernel::event::op_mul_mat ev{
      .src0 = make_prepared_q6_k_x8_q8_src(prepared_storage.data(), col_count, row_count),
      .src1 = make_q8_k_x8_rhs_src(rhs_q8.data(), rhs_rows, col_count),
      .dst = make_batch_major_dst(prepared_out.data(), dtype::f32, rhs_rows, row_count),
      .nth = 1,
  };
  CHECK_FALSE(machine.process_event(ev));
#endif
}

TEST_CASE("kernel_aarch64_q8_0_vector_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::QK8_0;
  using emel::kernel::detail::quant::block_q8_0;

  constexpr uint64_t row_count = 8u;
  std::array<block_q8_0, row_count> q8_rows = {};
  for (size_t row = 0; row < q8_rows.size(); ++row) {
    q8_rows[row].d = emel::kernel::detail::quant::fp32_to_fp16(
        0.03125f * static_cast<float>((row % 5u) + 1u));
    for (size_t idx = 0; idx < q8_rows[row].qs.size(); ++idx) {
      q8_rows[row].qs[idx] = static_cast<int8_t>(
          static_cast<int32_t>(((row + 1u) * 13u + idx * 5u) % 31u) - 15);
    }
  }

  const std::array<float, QK8_0> input = [] {
    std::array<float, QK8_0> values = {};
    for (size_t idx = 0; idx < values.size(); ++idx) {
      const int32_t centered = static_cast<int32_t>(idx % 27u) - 13;
      values[idx] = static_cast<float>(centered) * 0.03125f;
    }
    return values;
  }();

  std::array<float, row_count> optimized_out = {};
  std::array<float, row_count> scalar_out = {};
  const emel::kernel::event::op_mul_mat optimized_ev{
      .src0 = make_quantized_src(q8_rows.data(), dtype::q8_0, QK8_0, row_count),
      .src1 = make_src(input.data(), dtype::f32, 1u, QK8_0),
      .dst = make_dst(optimized_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };
  const emel::kernel::event::op_mul_mat scalar_ev{
      .src0 = make_quantized_src(q8_rows.data(), dtype::q8_0, QK8_0, row_count),
      .src1 = make_src(input.data(), dtype::f32, 1u, QK8_0),
      .dst = make_dst(scalar_out.data(), dtype::f32, 1u, row_count),
      .nth = 1,
  };

  aarch64_sm machine{};
  CHECK(machine.process_event(optimized_ev));
  CHECK(emel::kernel::detail::execute_scalar(scalar_ev));

  for (size_t row = 0; row < optimized_out.size(); ++row) {
    CHECK(optimized_out[row] == doctest::Approx(scalar_out[row]).epsilon(1.0e-6f));
  }

#if defined(__aarch64__) || defined(__ARM_NEON)
  CHECK(machine.optimized_q8_0_dispatch_count() == 1u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 1u);
  CHECK(machine.shared_q8_0_dispatch_count() == 0u);
#else
  CHECK(machine.optimized_q8_0_dispatch_count() == 0u);
  CHECK(machine.optimized_q8_0_vector_dispatch_count() == 0u);
  CHECK(machine.shared_q8_0_dispatch_count() == 1u);
#endif
}

TEST_CASE("kernel_aarch64_q6_packed_vector_argmax_route_is_explicit_and_numeric_match") {
  using emel::kernel::detail::quant::QK_K;
  using emel::kernel::detail::quant::block_q6_k;

  constexpr uint64_t k_rows = 8u;
  std::array<block_q6_k, k_rows> native_rows = {};
  for (size_t row = 0; row < native_rows.size(); ++row) {
    native_rows[row].d = 0x3c00u;
    for (size_t idx = 0; idx < native_rows[row].scales.size(); ++idx) {
      native_rows[row].scales[idx] =
          static_cast<int8_t>((static_cast<int32_t>((row + idx) % 13u)) - 6);
    }
    for (size_t idx = 0; idx < native_rows[row].ql.size(); ++idx) {
      native_rows[row].ql[idx] = static_cast<uint8_t>(((row + 1u) * 17u + idx * 7u) & 0xffu);
    }
    for (size_t idx = 0; idx < native_rows[row].qh.size(); ++idx) {
      native_rows[row].qh[idx] = static_cast<uint8_t>(((row + 3u) * 11u + idx * 5u) & 0xffu);
    }
  }

  const std::array<float, QK_K> input = [] {
    std::array<float, QK_K> values = {};
    for (size_t idx = 0; idx < values.size(); ++idx) {
      const int32_t centered = static_cast<int32_t>(idx % 29u) - 14;
      values[idx] = static_cast<float>(centered) * 0.0625f;
    }
    return values;
  }();

  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q6_kx8) * (QK_K / QK_K)>
      packed_storage = {};
  std::array<uint8_t,
             sizeof(emel::kernel::detail::quant::block_q6_kx8_q8_argmax_prepared) *
                 (QK_K / QK_K)>
      argmax_prepared_storage = {};
  REQUIRE(emel::kernel::detail::quant::pack_q6_k_rows_x8(
      native_rows.data(),
      k_rows,
      QK_K,
      packed_storage.data()));
  REQUIRE(emel::kernel::detail::quant::pack_q6_k_rows_x8_q8_argmax_prepared(
      native_rows.data(),
      k_rows,
      QK_K,
      argmax_prepared_storage.data()));

  std::array<emel::kernel::detail::quant::block_q8_k, QK_K / QK_K> q8_input = {};
  emel::kernel::detail::quant::quantize_row_q8_k_strided(
      input.data(),
      1u,
      q8_input.data(),
      static_cast<int64_t>(QK_K));

  std::array<float, k_rows> native_out = {};
  const emel::kernel::event::op_mul_mat native_ev{
      .src0 = make_quantized_src(native_rows.data(), dtype::q6_k, QK_K, k_rows),
      .src1 = make_src(input.data(), dtype::f32, 1u, QK_K),
      .dst = make_dst(native_out.data(), dtype::f32, 1u, k_rows),
      .nth = 1,
  };
  float argmax_score = 0.0f;
  int32_t argmax_index = -1;
  float argmax_prepared_score = 0.0f;
  int32_t argmax_prepared_index = -1;
  const emel::kernel::event::op_mul_mat_argmax packed_argmax_ev{
      .src0 = make_packed_q6_k_x8_src(packed_storage.data(), QK_K, k_rows),
      .src1 = make_q8_k_vector_src(q8_input.data(), QK_K),
      .dst = make_dst(&argmax_score, dtype::f32, 1u, 1u),
      .nth = 1,
      .index_out = &argmax_index,
  };
  const emel::kernel::event::op_mul_mat_argmax argmax_prepared_ev{
      .src0 = make_argmax_prepared_q6_k_x8_q8_src(argmax_prepared_storage.data(), QK_K, k_rows),
      .src1 = make_q8_k_vector_src(q8_input.data(), QK_K),
      .dst = make_dst(&argmax_prepared_score, dtype::f32, 1u, 1u),
      .nth = 1,
      .index_out = &argmax_prepared_index,
  };

  aarch64_sm native_machine{};
  aarch64_sm packed_machine{};
  aarch64_sm argmax_prepared_machine{};
  CHECK(native_machine.process_event(native_ev));
  CHECK(packed_machine.process_event(packed_argmax_ev));
  CHECK(argmax_prepared_machine.process_event(argmax_prepared_ev));

  int32_t expected_index = 0;
  float expected_score = native_out[0];
  for (size_t row = 1; row < native_out.size(); ++row) {
    if (native_out[row] > expected_score) {
      expected_score = native_out[row];
      expected_index = static_cast<int32_t>(row);
    }
  }

  CHECK(argmax_index == expected_index);
  CHECK(argmax_score == doctest::Approx(expected_score).epsilon(1.0e-6f));
  CHECK(argmax_prepared_index == expected_index);
  CHECK(argmax_prepared_score == doctest::Approx(expected_score).epsilon(1.0e-6f));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  CHECK(packed_machine.optimized_q6_dispatch_count() == 1u);
  CHECK(packed_machine.optimized_q6_vector_dispatch_count() == 1u);
  CHECK(packed_machine.optimized_q6_vector_argmax_dispatch_count() == 1u);
  CHECK(packed_machine.optimized_q6_vector_packed_dispatch_count() == 1u);
  CHECK(packed_machine.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count() == 1u);
  CHECK(packed_machine.shared_q6_dispatch_count() == 0u);
#if defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(argmax_prepared_machine.optimized_q6_dispatch_count() == 1u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_dispatch_count() == 1u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_argmax_dispatch_count() == 1u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count() ==
        1u);
  CHECK(argmax_prepared_machine.shared_q6_dispatch_count() == 0u);
#else
  CHECK(argmax_prepared_machine.optimized_q6_dispatch_count() == 0u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_argmax_dispatch_count() == 0u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count() ==
        0u);
  CHECK(argmax_prepared_machine.shared_q6_dispatch_count() == 0u);
#endif
#else
  CHECK(packed_machine.optimized_q6_dispatch_count() == 0u);
  CHECK(packed_machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(packed_machine.optimized_q6_vector_argmax_dispatch_count() == 0u);
  CHECK(packed_machine.optimized_q6_vector_packed_dispatch_count() == 0u);
  CHECK(packed_machine.optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count() == 0u);
  CHECK(packed_machine.shared_q6_dispatch_count() == 0u);
  CHECK(argmax_prepared_machine.optimized_q6_dispatch_count() == 0u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_dispatch_count() == 0u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_argmax_dispatch_count() == 0u);
  CHECK(argmax_prepared_machine.optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count() ==
        0u);
  CHECK(argmax_prepared_machine.shared_q6_dispatch_count() == 0u);
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

  emel::kernel::aarch64::action::exec_simd_op_flash_attn_ext_f16kv_one_chunk(dispatch0, ctx);
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

  emel::kernel::aarch64::action::exec_simd_op_flash_attn_ext_f16kv_one_chunk(dispatch1, ctx);
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

  emel::kernel::aarch64::action::exec_simd_op_flash_attn_ext_f16kv_one_chunk(dispatch, ctx);

  const std::vector<float> expected = flash_attn_reference_f16_scores(
      std::span<const float>(fixture.q, 4u),
      std::span<const uint16_t>(fixture.k, 8u),
      std::span<const uint16_t>(fixture.v, 8u),
      4u,
      2u,
      1.0f);

  CHECK(dispatch_ctx.outcome == emel::kernel::aarch64::events::phase_outcome::done);
  CHECK(ctx.optimized_flash_dispatch_count == 1u);
  CHECK(ctx.shared_flash_dispatch_count == 0u);
  CHECK(within_flash_online_f16_tolerance(fixture.dst[0], expected[0]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[1], expected[1]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[2], expected[2]));
  CHECK(within_flash_online_f16_tolerance(fixture.dst[3], expected[3]));
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
  const auto k_fp16 = to_fp16_storage(k);
  const auto v_fp16 = to_fp16_storage(v);

  emel::kernel::event::op_flash_attn_ext request{};
  request.src0 = make_src(q.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1 = make_src(k_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.src2 = make_src(v_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.dst = make_dst(dst_neon.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src1.nb[2] = sizeof(uint16_t) * head_dim;
  request.src2.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src2.nb[2] = sizeof(uint16_t) * head_dim;
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
    CHECK(within_flash_online_f16_tolerance(dst_neon[idx], dst_shared[idx]));
  }
#endif
}

TEST_CASE("kernel_aarch64_flash_attn_ext_matches_online_softmax_f16_reference_on_long_multihead_kv") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t head_dim = 64u;
  constexpr uint64_t head_count = 12u;
  constexpr uint64_t kv_head_count = 12u;
  constexpr uint64_t kv_tokens = 104u;
  const uint64_t kv_dim = head_dim * kv_head_count;

  std::vector<float> q(head_dim * head_count);
  std::vector<float> k(kv_dim * kv_tokens);
  std::vector<float> v(kv_dim * kv_tokens);
  std::vector<float> neon_dst(head_dim * head_count, 0.0f);
  std::vector<float> shared_dst(head_dim * head_count, 0.0f);

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
  request.src1 = make_src(k_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.src2 = make_src(v_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.dst = make_dst(neon_dst.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src1.nb[2] = sizeof(uint16_t) * head_dim;
  request.src2.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src2.nb[2] = sizeof(uint16_t) * head_dim;
  request.nth = 1;

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  request.op_params_size = sizeof(scale);

  auto shared_request = request;
  shared_request.dst = make_dst(shared_dst.data(), dtype::f32, head_dim, 1u, head_count);

  emel::kernel::detail::flash_attn_workspace neon_workspace{};
  emel::kernel::detail::flash_attn_workspace shared_workspace{};
  REQUIRE(emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
      request, true, neon_workspace));
  REQUIRE(emel::kernel::detail::run_flash_attn_ext_with_workspace(
      shared_request, shared_workspace));

  for (uint64_t head = 0; head < head_count; ++head) {
    std::vector<float> k_head(kv_tokens * head_dim);
    std::vector<float> v_head(kv_tokens * head_dim);
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const uint64_t src_offset = token * kv_dim + head * head_dim;
      const uint64_t dst_offset = token * head_dim;
      std::memcpy(k_head.data() + dst_offset, k.data() + src_offset, sizeof(float) * head_dim);
      std::memcpy(v_head.data() + dst_offset, v.data() + src_offset, sizeof(float) * head_dim);
    }
    const auto k_head_fp16 = to_fp16_storage(k_head);
    const auto v_head_fp16 = to_fp16_storage(v_head);
    const std::vector<float> expected =
        flash_attn_reference_online_softmax_f16_values(
        std::span<const float>(q.data() + head * head_dim, head_dim),
        std::span<const uint16_t>(k_head_fp16.data(), k_head_fp16.size()),
        std::span<const uint16_t>(v_head_fp16.data(), v_head_fp16.size()),
        head_dim,
        kv_tokens,
        scale);

    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      const size_t idx = static_cast<size_t>(head * head_dim + dim);
      CHECK(within_flash_online_f16_tolerance(
          neon_dst[idx], expected[static_cast<size_t>(dim)]));
      CHECK(within_flash_online_f16_tolerance(
          shared_dst[idx], expected[static_cast<size_t>(dim)]));
    }
  }
#endif
}

TEST_CASE("kernel_aarch64_flash_attn_ext_matches_masked_total_token_reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t head_dim = 64u;
  constexpr uint64_t kv_tokens = 257u;
  constexpr uint32_t total_tokens = 2048u;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> q(head_dim);
  std::vector<float> k(head_dim * kv_tokens);
  std::vector<float> v(head_dim * kv_tokens);
  std::vector<float> neon_dst(head_dim, 0.0f);
  std::vector<float> shared_dst(head_dim, 0.0f);

  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    const double angle = static_cast<double>((dim + 3u) * 11u);
    q[dim] = emel::kernel::detail::quant::fp16_to_fp32(
        emel::kernel::detail::quant::fp32_to_fp16(
            static_cast<float>(std::sin(angle * 0.02197265625))));
  }

  for (uint64_t token = 0; token < kv_tokens; ++token) {
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      const size_t offset = static_cast<size_t>(token * head_dim + dim);
      const double base = static_cast<double>((token + 1u) * (dim + 5u));
      k[offset] = emel::kernel::detail::quant::fp16_to_fp32(
          emel::kernel::detail::quant::fp32_to_fp16(
              static_cast<float>(std::cos(base * 0.00390625))));
      v[offset] = emel::kernel::detail::quant::fp16_to_fp32(
          emel::kernel::detail::quant::fp32_to_fp16(
              static_cast<float>(std::sin(base * 0.005859375))));
    }
  }
  const auto k_fp16 = to_fp16_storage(k);
  const auto v_fp16 = to_fp16_storage(v);

  emel::kernel::event::op_flash_attn_ext request{};
  request.src0 = make_src(q.data(), dtype::f32, head_dim, 1u, 1u, 1u);
  request.src1 = make_src(k_fp16.data(), dtype::f16, head_dim, kv_tokens, 1u, 1u);
  request.src2 = make_src(v_fp16.data(), dtype::f16, head_dim, kv_tokens, 1u, 1u);
  request.dst = make_dst(neon_dst.data(), dtype::f32, head_dim, 1u, 1u, 1u);
  request.nth = 1;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  std::memcpy(request.op_params.data() + sizeof(scale), &total_tokens, sizeof(total_tokens));
  request.op_params_size = sizeof(scale) + sizeof(total_tokens);

  auto shared_request = request;
  shared_request.dst = make_dst(shared_dst.data(), dtype::f32, head_dim, 1u, 1u, 1u);

  emel::kernel::detail::flash_attn_workspace neon_workspace{};
  emel::kernel::detail::flash_attn_workspace shared_workspace{};
  REQUIRE(emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
      request, true, neon_workspace));
  REQUIRE(emel::kernel::detail::run_flash_attn_ext_with_workspace(
      shared_request, shared_workspace));

  const std::vector<float> expected = flash_attn_reference_masked_total_tokens(
      std::span<const float>(q.data(), q.size()),
      std::span<const uint16_t>(k_fp16.data(), k_fp16.size()),
      std::span<const uint16_t>(v_fp16.data(), v_fp16.size()),
      head_dim,
      kv_tokens,
      total_tokens,
      scale);
  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    CHECK(within_flash_online_f16_tolerance(
        neon_dst[static_cast<size_t>(dim)], expected[static_cast<size_t>(dim)]));
    CHECK(within_flash_online_f16_tolerance(
        shared_dst[static_cast<size_t>(dim)], expected[static_cast<size_t>(dim)]));
  }
#endif
}

TEST_CASE("kernel_aarch64_flash_attn_ext_matches_masked_total_token_reference_on_long_multihead_kv") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
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
  std::vector<float> neon_dst(head_dim * head_count, 0.0f);
  std::vector<float> shared_dst(head_dim * head_count, 0.0f);

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
        const double base = static_cast<double>((token + 1u) * (head + 3u) * (dim + 5u));
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
  request.src1 = make_src(k_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.src2 = make_src(v_fp16.data(), dtype::f16, head_dim, kv_tokens, kv_head_count);
  request.dst = make_dst(neon_dst.data(), dtype::f32, head_dim, 1u, head_count);
  request.src1.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src1.nb[2] = sizeof(uint16_t) * head_dim;
  request.src2.nb[1] = sizeof(uint16_t) * kv_dim;
  request.src2.nb[2] = sizeof(uint16_t) * head_dim;
  request.nth = 1;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  std::memcpy(request.op_params.data() + sizeof(scale), &total_tokens, sizeof(total_tokens));
  request.op_params_size = sizeof(scale) + sizeof(total_tokens);

  auto shared_request = request;
  shared_request.dst = make_dst(shared_dst.data(), dtype::f32, head_dim, 1u, head_count);

  emel::kernel::detail::flash_attn_workspace neon_workspace{};
  emel::kernel::detail::flash_attn_workspace shared_workspace{};
  REQUIRE(emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
      request, true, neon_workspace));
  REQUIRE(emel::kernel::detail::run_flash_attn_ext_with_workspace(
      shared_request, shared_workspace));

  std::vector<float> expected(head_dim * head_count, 0.0f);
  for (uint64_t head = 0; head < head_count; ++head) {
    std::vector<float> k_head(kv_tokens * head_dim);
    std::vector<float> v_head(kv_tokens * head_dim);
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const uint64_t src_offset = token * kv_dim + head * head_dim;
      const uint64_t dst_offset = token * head_dim;
      std::memcpy(k_head.data() + dst_offset, k.data() + src_offset, sizeof(float) * head_dim);
      std::memcpy(v_head.data() + dst_offset, v.data() + src_offset, sizeof(float) * head_dim);
    }
    const auto expected_head = flash_attn_reference_masked_total_tokens(
        std::span<const float>(q.data() + static_cast<std::ptrdiff_t>(head * head_dim), head_dim),
        std::span<const float>(k_head.data(), k_head.size()),
        std::span<const float>(v_head.data(), v_head.size()),
        head_dim,
        kv_tokens,
        total_tokens,
        scale);
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      expected[head * head_dim + dim] = expected_head[static_cast<size_t>(dim)];
    }
  }

  for (size_t idx = 0; idx < neon_dst.size(); ++idx) {
    CHECK(within_flash_online_f16_tolerance(neon_dst[idx], expected[idx]));
    CHECK(within_flash_online_f16_tolerance(shared_dst[idx], expected[idx]));
  }
#endif
}

TEST_CASE("kernel_aarch64_flash_attn_ext_does_not_materialize_masked_tail_into_workspace_requirements") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  flash_attn_ext_fixture fixture{};
  auto request = make_flash_attn_ext_event(fixture);
  constexpr uint32_t total_tokens =
      static_cast<uint32_t>(emel::kernel::detail::flash_attn_workspace_token_capacity + 1024u);
  const float scale = 1.0f;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  std::memcpy(request.op_params.data() + sizeof(scale), &total_tokens, sizeof(total_tokens));
  request.op_params_size = sizeof(scale) + sizeof(total_tokens);

  std::array<float, 4u> shared_dst = {};
  auto shared_request = request;
  shared_request.dst = make_dst(shared_dst.data(), dtype::f32, 4u, 1u, 1u, 1u);

  const std::vector<float> expected = flash_attn_reference_f16_scores(
      std::span<const float>(fixture.q, 4u),
      std::span<const uint16_t>(fixture.k, 8u),
      std::span<const uint16_t>(fixture.v, 8u),
      4u,
      2u,
      scale);

  emel::kernel::detail::flash_attn_workspace neon_workspace{};
  emel::kernel::detail::flash_attn_workspace shared_workspace{};
  REQUIRE(emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
      request, true, neon_workspace));
  REQUIRE(emel::kernel::detail::run_flash_attn_ext_with_workspace(
      shared_request, shared_workspace));

  for (uint64_t dim = 0; dim < 4u; ++dim) {
    CHECK(within_flash_online_f16_tolerance(
        fixture.dst[static_cast<size_t>(dim)], expected[static_cast<size_t>(dim)]));
    CHECK(within_flash_online_f16_tolerance(
        shared_dst[static_cast<size_t>(dim)], expected[static_cast<size_t>(dim)]));
  }
#endif
}

TEST_CASE("kernel_aarch64_flash_attn_ext_matches_online_softmax_f16_reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr uint64_t head_dim = 32u;
  constexpr uint64_t kv_tokens = 19u;
  constexpr float scale = 0.125f;

  std::array<float, head_dim> q = {};
  std::array<float, head_dim * kv_tokens> k = {};
  std::array<float, head_dim * kv_tokens> v = {};
  std::array<float, head_dim> neon_dst = {};
  std::array<float, head_dim> shared_dst = {};

  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    const int32_t centered = static_cast<int32_t>((dim * 7u) % 23u) - 11;
    q[static_cast<size_t>(dim)] = static_cast<float>(centered) * 0.09375f;
  }

  for (uint64_t token = 0; token < kv_tokens; ++token) {
    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim) +
                            static_cast<size_t>(dim);
      const int32_t key_centered =
          static_cast<int32_t>(((token + 3u) * (dim + 5u)) % 29u) - 14;
      const int32_t value_centered =
          static_cast<int32_t>(((token + 11u) * (dim + 7u)) % 31u) - 15;
      k[offset] = static_cast<float>(key_centered) * 0.0625f;
      const float raw_value = static_cast<float>(value_centered) * 0.28125f;
      v[offset] = emel::kernel::detail::quant::fp16_to_fp32(
          emel::kernel::detail::quant::fp32_to_fp16(raw_value));
    }
  }
  const auto k_fp16 = to_fp16_storage(k);
  const auto v_fp16 = to_fp16_storage(v);

  emel::kernel::event::op_flash_attn_ext request{};
  request.src0 = make_src(q.data(), dtype::f32, head_dim, 1, 1, 1);
  request.src1 = make_src(k_fp16.data(), dtype::f16, head_dim, kv_tokens, 1, 1);
  request.src2 = make_src(v_fp16.data(), dtype::f16, head_dim, kv_tokens, 1, 1);
  request.dst = make_dst(neon_dst.data(), dtype::f32, head_dim, 1, 1, 1);
  request.nth = 1;
  std::memcpy(request.op_params.data(), &scale, sizeof(scale));
  request.op_params_size = sizeof(scale);

  auto shared_request = request;
  shared_request.dst = make_dst(shared_dst.data(), dtype::f32, head_dim, 1, 1, 1);

  emel::kernel::detail::flash_attn_workspace neon_workspace{};
  emel::kernel::detail::flash_attn_workspace shared_workspace{};
  REQUIRE(emel::kernel::aarch64::detail::run_flash_attn_ext_neon(
      request, true, neon_workspace));
  REQUIRE(emel::kernel::detail::run_flash_attn_ext_with_workspace(
      shared_request, shared_workspace));

  const std::vector<float> expected =
      flash_attn_reference_online_softmax_f16_values(
          std::span<const float>(q.data(), q.size()),
          std::span<const uint16_t>(k_fp16.data(), k_fp16.size()),
          std::span<const uint16_t>(v_fp16.data(), v_fp16.size()),
          head_dim,
          kv_tokens,
          scale);
  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    CHECK(within_flash_online_f16_tolerance(
        neon_dst[static_cast<size_t>(dim)], expected[static_cast<size_t>(dim)]));
    CHECK(within_flash_online_f16_tolerance(
        shared_dst[static_cast<size_t>(dim)], expected[static_cast<size_t>(dim)]));
  }
#endif
}
