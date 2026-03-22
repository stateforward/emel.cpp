#include <algorithm>
#include <array>
#include <cmath>

#include <doctest/doctest.h>

#include "emel/generator/detail.hpp"

namespace {

using emel::generator::detail::quant::QK_K;
using emel::generator::detail::quant::block_q2_k;
using emel::generator::detail::quant::block_q3_k;
using emel::generator::detail::quant::block_q6_k;

}  // namespace

TEST_CASE("generator_detail_fp16_to_fp32_handles_normal_special_and_subnormal_values") {
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x3c00u) == doctest::Approx(1.0f));
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x3800u) == doctest::Approx(0.5f));
  CHECK(std::isinf(emel::generator::detail::quant::fp16_to_fp32(0x7c00u)));
  CHECK(emel::generator::detail::quant::fp16_to_fp32(0x0001u) > 0.0f);
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
  backend.q = {1.0f, 0.0f, 0.0f, 1.0f};
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
  std::memcpy(&scale, request.op_params.data(), sizeof(scale));

  CHECK(request.src0.ne[0] == 2u);
  CHECK(request.src0.ne[2] == 2u);
  CHECK(request.src1.ne[0] == 2u);
  CHECK(request.src1.ne[1] == 2u);
  CHECK(request.src1.ne[2] == 2u);
  CHECK(request.src1.nb[1] == sizeof(float) * 4u);
  CHECK(request.src1.nb[2] == sizeof(float) * 2u);
  CHECK(request.dst.ne[0] == 2u);
  CHECK(request.dst.ne[2] == 2u);
  CHECK(request.op_params_size == sizeof(float));
  CHECK(scale == doctest::Approx(1.0f / std::sqrt(2.0f)));
  CHECK(emel::kernel::detail::can_run_flash_attn_ext(request));
}
