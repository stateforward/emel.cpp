#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <doctest/doctest.h>

#include "emel/generator/detail.hpp"
#include "../kernel/test_helpers.hpp"

namespace {

using emel::generator::detail::quant::QK_K;
using emel::generator::detail::quant::block_q2_k;
using emel::generator::detail::quant::block_q3_k;
using emel::generator::detail::quant::block_q6_k;
using emel::kernel::test::flash_attn_reference_online_softmax_f16_values;
using emel::kernel::test::k_flash_online_f16_abs_tolerance;
using emel::kernel::test::within_flash_online_f16_tolerance;

uint16_t fp16_bits(const float value) {
  return emel::generator::detail::quant::fp32_to_fp16(value);
}

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

std::vector<float> flash_attention_online_reference(
    const emel::generator::detail::native_backend & backend,
    const int32_t layer_index,
    const int32_t position,
    const std::span<const float> q_vector) {
  const int32_t head_count = backend.n_head;
  const int32_t head_dim = backend.head_dim;
  const int32_t kv_head_dim = backend.head_dim_kv;
  const uint64_t kv_tokens = static_cast<uint64_t>(position + 1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> out(static_cast<size_t>(head_count) * static_cast<size_t>(head_dim), 0.0f);
  std::vector<uint16_t> head_k(
      static_cast<size_t>(head_dim) * static_cast<size_t>(kv_tokens), 0u);
  std::vector<uint16_t> head_v(
      static_cast<size_t>(head_dim) * static_cast<size_t>(kv_tokens), 0u);

  for (int32_t head = 0; head < head_count; ++head) {
    const int32_t kv_head = head / backend.n_rep;
    const size_t q_offset = static_cast<size_t>(head) * static_cast<size_t>(head_dim);

    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const size_t src_offset = emel::generator::detail::flash_layer_cache_head_position_offset(
          backend, layer_index, kv_head, static_cast<int32_t>(token), kv_head_dim);
      const size_t dst_offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
      for (int32_t dim = 0; dim < head_dim; ++dim) {
        const size_t dim_offset = static_cast<size_t>(dim);
        head_k[dst_offset + dim_offset] = backend.flash_key_cache[src_offset + dim_offset];
        head_v[dst_offset + dim_offset] = backend.flash_value_cache[src_offset + dim_offset];
      }
    }

    const std::vector<float> expected_head = flash_attn_reference_online_softmax_f16_values(
        q_vector.subspan(q_offset, static_cast<size_t>(head_dim)),
        std::span<const uint16_t>(head_k.data(), head_k.size()),
        std::span<const uint16_t>(head_v.data(), head_v.size()),
        static_cast<uint64_t>(head_dim),
        kv_tokens,
        scale);

    for (int32_t dim = 0; dim < head_dim; ++dim) {
      out[q_offset + static_cast<size_t>(dim)] = expected_head[static_cast<size_t>(dim)];
    }
  }

  return out;
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

TEST_CASE("generator_detail_builds_flash_request_over_head_major_kv_cache") {
  emel::generator::detail::native_backend backend{};
  backend.n_head = 2;
  backend.n_head_kv = 2;
  backend.head_dim = 2;
  backend.head_dim_kv = 2;
  backend.n_ctx = 2;
  backend.q = {1.0f, 0.1f, 0.2f, 1.0f};
  backend.q_attn = {9.0f, 9.0f, 9.0f, 9.0f};
  backend.flash_key_cache = {
      fp16_bits(1.0f), fp16_bits(0.0f), fp16_bits(0.0f), fp16_bits(1.0f),
      fp16_bits(0.5f), fp16_bits(0.5f), fp16_bits(0.25f), fp16_bits(0.75f),
  };
  backend.flash_value_cache = {
      fp16_bits(2.0f), fp16_bits(0.0f), fp16_bits(0.0f), fp16_bits(4.0f),
      fp16_bits(1.0f), fp16_bits(1.0f), fp16_bits(0.5f), fp16_bits(1.5f),
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
  CHECK(request.src1.nb[1] == sizeof(uint16_t) * 2u);
  CHECK(request.src1.nb[2] == sizeof(uint16_t) * 4u);
  CHECK(request.dst.ne[0] == 2u);
  CHECK(request.dst.ne[2] == 2u);
  CHECK(request.op_params_size == sizeof(float) + sizeof(uint32_t));
  CHECK(scale == doctest::Approx(1.0f / std::sqrt(2.0f)));
  CHECK(total_tokens == 2u);
  CHECK(request.src1.type == emel::kernel::event::dtype::f16);
  CHECK(request.src2.type == emel::kernel::event::dtype::f16);
  CHECK(emel::kernel::detail::can_run_flash_attn_ext(request));
}

TEST_CASE("generator_detail_flash_dispatch_matches_online_softmax_reference_on_same_backend_state") {
  emel::generator::detail::native_backend backend{};
  backend.n_head = 12;
  backend.n_head_kv = 12;
  backend.n_rep = 1;
  backend.head_dim = 64;
  backend.head_dim_kv = 64;
  backend.n_ctx = 256;
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;

  const size_t n_embd =
      static_cast<size_t>(backend.n_head) * static_cast<size_t>(backend.head_dim);
  const size_t kv_dim =
      static_cast<size_t>(backend.n_head_kv) * static_cast<size_t>(backend.head_dim_kv);
  const int32_t position = 255;
  const int32_t position_limit = position + 1;

  backend.q.resize(n_embd);
  backend.q_attn.resize(n_embd);
  backend.flash_key_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.flash_value_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs_rounded.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_value_column.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(n_embd);

  for (size_t idx = 0; idx < backend.q.size(); ++idx) {
    const float raw = static_cast<float>(std::sin(static_cast<double>(idx + 1u) * 0.03125));
    backend.q[idx] = raw;
  }

  for (int32_t token = 0; token < position_limit; ++token) {
    for (int32_t head = 0; head < backend.n_head_kv; ++head) {
      for (int32_t dim = 0; dim < backend.head_dim_kv; ++dim) {
        const size_t offset = emel::generator::detail::flash_layer_cache_head_position_offset(
            backend, 0, head, token, backend.head_dim_kv) + static_cast<size_t>(dim);
        const double base =
            static_cast<double>((token + 1) * (head + 3) * (dim + 5));
        backend.flash_key_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::cos(base * 0.0078125)));
        backend.flash_value_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::sin(base * 0.01171875)));
      }
    }
  }

  std::vector<float> flash_ctx(backend.attn_ctx.size(), 0.0f);
  std::vector<float> expected_ctx(backend.attn_ctx.size(), 0.0f);
  backend.attn_ctx = flash_ctx;
  REQUIRE(emel::generator::detail::dispatch_flash_attention(backend, 0, position));
  flash_ctx = backend.attn_ctx;
  expected_ctx = flash_attention_online_reference(backend, 0, position, backend.q);

  for (size_t idx = 0; idx < flash_ctx.size(); ++idx) {
    CHECK(within_flash_online_f16_tolerance(flash_ctx[idx], expected_ctx[idx]));
  }
}

TEST_CASE("generator_detail_flash_dispatch_matches_online_softmax_reference_across_long_reuse") {
  emel::generator::detail::native_backend backend{};
  backend.n_head = 12;
  backend.n_head_kv = 12;
  backend.n_rep = 1;
  backend.head_dim = 64;
  backend.head_dim_kv = 64;
  backend.n_ctx = 1024;
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;

  const size_t n_embd =
      static_cast<size_t>(backend.n_head) * static_cast<size_t>(backend.head_dim);
  const size_t kv_dim =
      static_cast<size_t>(backend.n_head_kv) * static_cast<size_t>(backend.head_dim_kv);

  backend.q.resize(n_embd);
  backend.q_attn.resize(n_embd);
  backend.flash_key_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.flash_value_cache.resize(static_cast<size_t>(backend.n_ctx) * kv_dim);
  backend.attn_scores.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_probs_rounded.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_value_column.resize(static_cast<size_t>(backend.n_ctx));
  backend.attn_ctx.resize(n_embd);

  int32_t first_position = -1;
  size_t first_index = 0u;
  float first_flash = 0.0f;
  float first_expected = 0.0f;
  float max_abs = 0.0f;

  for (int32_t position = 0; position < backend.n_ctx; ++position) {
    for (size_t idx = 0; idx < backend.q.size(); ++idx) {
      const double q_base = static_cast<double>((position + 1) * (idx + 1u));
      const float raw = static_cast<float>(std::sin(q_base * 0.0009765625));
      backend.q[idx] = raw;
    }

    for (int32_t head = 0; head < backend.n_head_kv; ++head) {
      for (int32_t dim = 0; dim < backend.head_dim_kv; ++dim) {
        const size_t offset = emel::generator::detail::flash_layer_cache_head_position_offset(
            backend, 0, head, position, backend.head_dim_kv) + static_cast<size_t>(dim);
        const double kv_base =
            static_cast<double>((position + 1) * (head + 3) * (dim + 5));
        backend.flash_key_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::cos(kv_base * 0.0078125)));
        backend.flash_value_cache[offset] = emel::generator::detail::quant::fp32_to_fp16(
            static_cast<float>(std::sin(kv_base * 0.01171875)));
      }
    }

    std::vector<float> flash_ctx(backend.attn_ctx.size(), 0.0f);
    std::vector<float> expected_ctx(backend.attn_ctx.size(), 0.0f);
    backend.attn_ctx = flash_ctx;
    REQUIRE(emel::generator::detail::dispatch_flash_attention(backend, 0, position));
    flash_ctx = backend.attn_ctx;
    expected_ctx = flash_attention_online_reference(backend, 0, position, backend.q);

    for (size_t idx = 0; idx < flash_ctx.size(); ++idx) {
      const float diff = std::fabs(flash_ctx[idx] - expected_ctx[idx]);
      if (diff > max_abs) {
        max_abs = diff;
      }
      if (first_position < 0 &&
          !within_flash_online_f16_tolerance(flash_ctx[idx], expected_ctx[idx])) {
        first_position = position;
        first_index = idx;
        first_flash = flash_ctx[idx];
        first_expected = expected_ctx[idx];
      }
    }
  }

  INFO("first_position=" << first_position << " first_index=" << first_index
                         << " first_flash=" << first_flash
                         << " first_expected=" << first_expected
                         << " max_abs=" << max_abs);
  CHECK(max_abs <= k_flash_online_f16_abs_tolerance);
}
