#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <doctest/doctest.h>

#include "emel/generator/detail.hpp"

namespace {

using emel::generator::detail::quant::QK_K;
using emel::generator::detail::quant::block_q2_k;
using emel::generator::detail::quant::block_q3_k;
using emel::generator::detail::quant::block_q6_k;

void apply_rope_reference(std::span<float> vector,
                          const int32_t head_count,
                          const int32_t head_dim,
                          const int32_t n_rot,
                          const int32_t position,
                          const float rope_freq_base) {
  const int32_t rot_dim = std::min(n_rot, head_dim);
  if (head_count <= 0 || head_dim <= 1 || rot_dim <= 1) {
    return;
  }

  const float theta_scale = ::powf(rope_freq_base, -2.0f / static_cast<float>(rot_dim));
  for (int32_t head = 0; head < head_count; ++head) {
    float * head_ptr =
        vector.data() + (static_cast<size_t>(head) * static_cast<size_t>(head_dim));
    float theta = static_cast<float>(position);
    for (int32_t dim = 0; dim + 1 < rot_dim; dim += 2) {
      const float cos_theta = ::cosf(theta);
      const float sin_theta = ::sinf(theta);
      const float x0 = head_ptr[dim];
      const float x1 = head_ptr[dim + 1];
      head_ptr[dim] = x0 * cos_theta - x1 * sin_theta;
      head_ptr[dim + 1] = x0 * sin_theta + x1 * cos_theta;
      theta *= theta_scale;
    }
  }
}

}  // namespace

TEST_CASE("generator_detail_fp16_to_fp32_handles_normal_special_and_subnormal_values") {
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x3c00u) == doctest::Approx(1.0f));
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x3800u) == doctest::Approx(0.5f));
  CHECK(std::isinf(emel::generator::detail::quant::fp16_to_fp32(0x7c00u)));
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x0001u) > 0.0f);
}

TEST_CASE("generator_detail_fp16_conversion_matches_native_arm_fp16_rounding") {
#if defined(__ARM_NEON) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && \
    !defined(__MUSACC__)
  constexpr std::array<float, 8> samples = {
      0.0f,
      0.5f,
      -0.934325f,
      0.0345459f,
      -36.4516f,
      65504.0f,
      6.1035156e-05f,
      -6.1035156e-05f,
  };

  for (const float sample : samples) {
    uint16_t native_bits = 0u;
    const __fp16 native_value = sample;
    std::memcpy(&native_bits, &native_value, sizeof(native_bits));

    CHECK(emel::generator::detail::quant::fp32_to_fp16(sample) == native_bits);
    CHECK(emel::generator::detail::quant::fp16_to_fp32(native_bits) ==
          doctest::Approx(static_cast<float>(native_value)));
  }
#else
  CHECK(true);
#endif
}

TEST_CASE("generator_detail_apply_rope_matches_ggml_float_recurrence") {
  std::array<float, 64> actual = {};
  std::array<float, 64> reference = {};
  for (size_t idx = 0; idx < actual.size(); ++idx) {
    actual[idx] = std::sin(static_cast<float>(idx) * 0.03125f) * 3.0f;
  }
  reference = actual;

  emel::generator::detail::apply_rope(actual, 1, 64, 64, 103, 10000.0f);
  apply_rope_reference(reference, 1, 64, 64, 103, 10000.0f);

  for (size_t idx = 0; idx < actual.size(); ++idx) {
    CHECK(actual[idx] == doctest::Approx(reference[idx]).epsilon(1.0e-8));
  }
}

TEST_CASE("generator_detail_dequantizes_q2_k_blocks") {
  block_q2_k block = {};
  block.d = 0x3c00u;
  block.dmin = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(), static_cast<uint8_t>(0x11u));
  std::fill(block.qs.begin(), block.qs.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::generator::detail::quant::dequantize_row_q2_k(&block, out.data(), QK_K);

  CHECK(out.front() == doctest::Approx(-1.0f));
  CHECK(out[127] == doctest::Approx(-1.0f));
  CHECK(out.back() == doctest::Approx(-1.0f));
}

TEST_CASE("generator_detail_dequantizes_q3_k_blocks_without_unsigned_wrap") {
  block_q3_k block = {};
  block.d = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(), static_cast<uint8_t>(0x00u));
  std::fill(block.hmask.begin(), block.hmask.end(), static_cast<uint8_t>(0x00u));
  std::fill(block.qs.begin(), block.qs.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::generator::detail::quant::dequantize_row_q3_k(&block, out.data(), QK_K);

  CHECK(std::isfinite(out.front()));
  CHECK(out.front() == doctest::Approx(128.0f));
  CHECK(out[127] == doctest::Approx(128.0f));
  CHECK(out.back() == doctest::Approx(128.0f));
}

TEST_CASE("generator_detail_dequantizes_q6_k_blocks") {
  block_q6_k block = {};
  block.d = 0x3c00u;
  std::fill(block.scales.begin(), block.scales.end(), static_cast<int8_t>(1));
  std::fill(block.ql.begin(), block.ql.end(), static_cast<uint8_t>(0x00u));
  std::fill(block.qh.begin(), block.qh.end(), static_cast<uint8_t>(0x00u));

  std::array<float, QK_K> out = {};
  emel::generator::detail::quant::dequantize_row_q6_k(&block, out.data(), QK_K);

  CHECK(out.front() == doctest::Approx(-32.0f));
  CHECK(out[127] == doctest::Approx(-32.0f));
  CHECK(out.back() == doctest::Approx(-32.0f));
}

TEST_CASE("generator_detail_builds_flash_request_over_position_major_kv_cache") {
  emel::generator::detail::native_backend backend{};
  backend.n_head = 2;
  backend.n_head_kv = 2;
  backend.head_dim = 2;
  backend.head_dim_kv = 2;
  backend.n_ctx = 2;
  backend.q = {1.0f, 0.1f, 0.2f, 1.0f};
  backend.q_attn = {9.0f, 9.0f, 9.0f, 9.0f};
  backend.key_cache = {
      1.0f, 0.0f, 0.5f, 0.5f,
      0.0f, 1.0f, 0.25f, 0.75f,
  };
  backend.value_cache = {
      2.0f, 0.0f, 1.0f, 1.0f,
      0.0f, 4.0f, 0.5f, 1.5f,
  };
  backend.attn_ctx.resize(4);

  const auto request = emel::generator::detail::make_flash_attn_request(backend, 0, 1);
  float scale = 0.0f;
  uint32_t total_tokens = 0u;
  std::memcpy(&scale, request.op_params.data(), sizeof(scale));
  std::memcpy(&total_tokens, request.op_params.data() + sizeof(scale), sizeof(total_tokens));

  CHECK(request.src0.ne[0] == 2u);
  CHECK(request.src0.ne[2] == 2u);
  CHECK(request.src0.data == backend.q.data());
  CHECK(request.src1.ne[0] == 2u);
  CHECK(request.src1.ne[1] == 2u);
  CHECK(request.src1.ne[2] == 2u);
  CHECK(request.src1.nb[1] == sizeof(float) * 4u);
  CHECK(request.src1.nb[2] == sizeof(float) * 2u);
  CHECK(request.dst.ne[0] == 2u);
  CHECK(request.dst.ne[2] == 2u);
  CHECK(request.op_params_size == sizeof(float) + sizeof(uint32_t));
  CHECK(scale == doctest::Approx(1.0f / std::sqrt(2.0f)));
  CHECK(total_tokens == 2u);
  CHECK(emel::kernel::detail::can_run_flash_attn_ext(request));
}
