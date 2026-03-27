#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <doctest/doctest.h>

#include "emel/generator/detail.hpp"
#include "../kernel/test_helpers.hpp"

namespace {

using emel::generator::detail::quant::QK_K;
using emel::generator::detail::quant::Q6_K_X8_ROWS;
using emel::generator::detail::quant::block_q2_k;
using emel::generator::detail::quant::block_q3_k;
using emel::generator::detail::quant::block_q6_k;
using emel::kernel::test::flash_attn_reference_online_softmax_f16_values;
using emel::kernel::test::k_flash_online_f16_abs_tolerance;
using emel::kernel::test::within_flash_online_f16_tolerance;

uint16_t fp16_bits(const float value) {
  return emel::generator::detail::quant::fp32_to_fp16(value);
}

std::array<block_q6_k, Q6_K_X8_ROWS> make_q6_rows() {
  std::array<block_q6_k, Q6_K_X8_ROWS> rows = {};
  for (size_t row = 0; row < rows.size(); ++row) {
    rows[row].d = 0x3c00u;
    for (size_t idx = 0; idx < rows[row].scales.size(); ++idx) {
      rows[row].scales[idx] =
          static_cast<int8_t>((static_cast<int32_t>((row + idx) % 13u)) - 6);
    }
    for (size_t idx = 0; idx < rows[row].ql.size(); ++idx) {
      rows[row].ql[idx] = static_cast<uint8_t>(((row + 1u) * 17u + idx * 7u) & 0xffu);
    }
    for (size_t idx = 0; idx < rows[row].qh.size(); ++idx) {
      rows[row].qh[idx] = static_cast<uint8_t>(((row + 3u) * 11u + idx * 5u) & 0xffu);
    }
  }
  return rows;
}

emel::model::data::tensor_record make_tensor_record(void * data,
                                                    const int32_t type,
                                                    const int32_t cols,
                                                    const int32_t rows) {
  emel::model::data::tensor_record tensor = {};
  tensor.data = data;
  tensor.type = type;
  tensor.n_dims = 2;
  tensor.dims[0] = static_cast<uint64_t>(cols);
  tensor.dims[1] = static_cast<uint64_t>(rows);
  tensor.data_size =
      emel::generator::detail::row_storage_bytes(tensor, cols) * static_cast<uint64_t>(rows);
  return tensor;
}

struct runtime_request_fixture {
  emel::model::data model = {};
  emel::model::llama::detail::execution_view execution = {};
  emel::model::llama::detail::topology topology = {};
  emel::model::llama::detail::step_plan plan = {};
  emel::generator::detail::native_backend backend = {};
  std::array<int32_t, 1> token_ids = {0};
  std::array<int32_t, 1> positions = {0};
  std::array<float, 6> logits = {};
  int32_t selected_token = -1;
  float selected_score = 0.0f;
  emel::generator::compute_io io = {};
  emel::graph::processor::event::execute request = {};

  explicit runtime_request_fixture(
      const emel::model::llama::detail::step_kind kind =
          emel::model::llama::detail::step_kind::prefill) {
    topology.execution = &execution;
    plan.graph = &topology;
    plan.kind = kind;
    plan.expected_outputs = 1;

    backend.model = &model;
    backend.n_embd = 4;
    backend.n_head = 1;
    backend.n_head_kv = 1;
    backend.n_layer = 1;
    backend.n_vocab = 4;
    backend.n_ctx = 8;
    backend.head_dim = 4;
    backend.head_dim_kv = 4;
    backend.blocks.resize(1u);
    backend.bound_tokens.resize(1u);
    backend.bound_positions.resize(1u);
    backend.bound_logits = {0.25f, 0.5f, 0.75f, 1.0f};

    io.backend_ctx = &backend;
    io.token_ids = token_ids.data();
    io.token_count = static_cast<int32_t>(token_ids.size());
    io.logits = logits.data();
    io.logits_capacity = static_cast<int32_t>(logits.size());
    io.selected_token_out = &selected_token;
    io.selected_score_out = &selected_score;

    request.step_plan = &plan;
    request.expected_outputs = plan.expected_outputs;
    request.compute_ctx = &io;
    request.positions = positions.data();
    request.positions_count = static_cast<int32_t>(positions.size());
    request.kv_tokens = 0;
  }
};

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

TEST_CASE("generator_detail_prepares_explicit_logits_routes_and_support_predicates") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.n_embd = static_cast<int32_t>(QK_K);
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

  CHECK(emel::generator::detail::packed_q6_k_x8_logits_supported(backend));
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  CHECK(emel::generator::detail::prepared_q6_k_x8_q8_logits_supported(backend));
#endif
  CHECK_FALSE(emel::generator::detail::q8_logits_input_path_supported({}));
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(backend));

  REQUIRE(emel::generator::detail::prepare_output_logits(backend));
  REQUIRE(emel::generator::detail::prepare_logits_input_q8_workspace(backend));
  CHECK(backend.logits_input_q8_storage.size() == 1u);
  CHECK(emel::generator::detail::q8_logits_input_path_supported(backend.output));
  CHECK(emel::generator::detail::q8_logits_input_path_supported(backend.output_argmax));

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_prepared);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8_q8_argmax_prepared);
  CHECK(emel::generator::detail::row_storage_bytes(*backend.output.tensor,
                                                   static_cast<int32_t>(QK_K)) ==
        emel::kernel::detail::quant::prepared_q6_k_x8_q8_group_storage_bytes(QK_K));
  CHECK(emel::generator::detail::row_storage_bytes(*backend.output_argmax.tensor,
                                                   static_cast<int32_t>(QK_K)) ==
        emel::kernel::detail::quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(QK_K));
  CHECK(emel::generator::detail::preselected_argmax_direct_supported(backend));
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) == emel::kernel::detail::dtype_q6_k_x8);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::generator::detail::preselected_argmax_direct_supported(backend));
#else
  CHECK(backend.output.tensor == &q6_tensor);
  CHECK(backend.output_argmax.tensor == &q6_tensor);
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(backend));
#endif
}

TEST_CASE("generator_detail_explicit_logits_routes_cover_packed_and_passthrough_helpers") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::native_backend backend{};
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;

  emel::generator::detail::reset_output_logits(backend);
  CHECK(backend.output.tensor == &q6_tensor);
  CHECK(backend.output_argmax.tensor == &q6_tensor);
  CHECK(backend.output_packed_storage.empty());
  CHECK(backend.output_prepared_storage.empty());
  CHECK(backend.output_argmax_prepared_storage.empty());

  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  REQUIRE(emel::generator::detail::prepare_packed_output_logits(backend));
  REQUIRE(backend.output.tensor != nullptr);
  REQUIRE(backend.output_argmax.tensor != nullptr);
  CHECK(static_cast<uint8_t>(backend.output.tensor->type) == emel::kernel::detail::dtype_q6_k_x8);
  CHECK(static_cast<uint8_t>(backend.output_argmax.tensor->type) ==
        emel::kernel::detail::dtype_q6_k_x8);
  CHECK(emel::generator::detail::matrix_buffer_bytes(backend.output) ==
        emel::kernel::detail::quant::packed_q6_k_x8_group_storage_bytes(QK_K));

  std::array<float, 8> f32_rows = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
  auto f32_tensor =
      make_tensor_record(f32_rows.data(), emel::kernel::detail::dtype_f32, 4, 2);
  emel::generator::detail::native_backend passthrough{};
  passthrough.output_native.tensor = &f32_tensor;
  passthrough.output_native.cols = 4;
  passthrough.output_native.rows = 2;
  passthrough.output = passthrough.output_native;
  passthrough.output_argmax = passthrough.output_native;
  REQUIRE(emel::generator::detail::prepare_output_logits(passthrough));
  CHECK(passthrough.output.tensor == &f32_tensor);
  CHECK(passthrough.output_argmax.tensor == &f32_tensor);
}

TEST_CASE("generator_detail_tensor_binding_and_copy_helpers_accept_explicit_quantized_routes") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::tensor_matrix q6_matrix = {};
  REQUIRE(emel::generator::detail::bind_tensor_rows(q6_tensor, q6_matrix));
  CHECK(q6_matrix.cols == static_cast<int32_t>(QK_K));
  CHECK(q6_matrix.rows == static_cast<int32_t>(Q6_K_X8_ROWS));

  auto invalid_q6_tensor =
      make_tensor_record(q6_rows.data(), emel::kernel::detail::dtype_q6_k, 8, 2);
  emel::generator::detail::tensor_matrix invalid_matrix = {};
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(invalid_q6_tensor, invalid_matrix));

  block_q3_k q3_block = {};
  q3_block.d = 0x3c00u;
  std::fill(q3_block.scales.begin(), q3_block.scales.end(), static_cast<uint8_t>(0x00u));
  std::fill(q3_block.hmask.begin(), q3_block.hmask.end(), static_cast<uint8_t>(0x00u));
  std::fill(q3_block.qs.begin(), q3_block.qs.end(), static_cast<uint8_t>(0x00u));
  auto q3_tensor = make_tensor_record(&q3_block, emel::kernel::detail::dtype_q3_k,
                                      static_cast<int32_t>(QK_K), 1);
  std::vector<float> q3_out(QK_K, 0.0f);
  REQUIRE(emel::generator::detail::copy_tensor_row(q3_tensor, 0, q3_out));
  CHECK(q3_out.front() == doctest::Approx(128.0f));
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(q3_tensor, 1, q3_out));

  std::vector<float> q6_out = {};
  REQUIRE(emel::generator::detail::dequantize_tensor_vector(q6_tensor, q6_out) == false);
  auto q6_vector_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K), 1);
  REQUIRE(emel::generator::detail::dequantize_tensor_vector(q6_vector_tensor, q6_out));
  std::vector<float> q6_expected(QK_K, 0.0f);
  emel::generator::detail::quant::dequantize_row_q6_k(q6_rows.data(), q6_expected.data(), QK_K);
  CHECK(q6_out.size() == QK_K);
  CHECK(q6_out.front() == doctest::Approx(q6_expected.front()));
  CHECK(q6_out.back() == doctest::Approx(q6_expected.back()));
}

TEST_CASE("generator_detail_logits_route_helpers_cover_explicit_failure_edges") {
  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));

  emel::generator::detail::native_backend backend{};
  backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  backend.output_native.tensor = &q6_tensor;
  backend.output_native.cols = static_cast<int32_t>(QK_K);
  backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  backend.output = backend.output_native;
  backend.output_argmax = backend.output_native;
  backend.logits_input_q8_storage.resize(1u);

  emel::model::data::tensor_record packed_tensor = q6_tensor;
  packed_tensor.type = emel::kernel::detail::dtype_q6_k_x8;
  backend.output_argmax.tensor = &packed_tensor;
  CHECK(emel::generator::detail::preselected_argmax_direct_supported(backend));

  emel::model::data::tensor_record unsupported_tensor = q6_tensor;
  unsupported_tensor.type = emel::kernel::detail::dtype_f32;
  backend.output_argmax.tensor = &unsupported_tensor;
  CHECK_FALSE(emel::generator::detail::preselected_argmax_direct_supported(backend));

  emel::generator::detail::native_backend null_backend{};
  CHECK_FALSE(emel::generator::detail::prepare_output_logits(null_backend));
  CHECK_FALSE(emel::generator::detail::prepare_prepared_output_logits(null_backend));
  CHECK_FALSE(emel::generator::detail::prepare_packed_output_logits(null_backend));

  auto invalid_q6_tensor =
      make_tensor_record(q6_rows.data(), emel::kernel::detail::dtype_q6_k, 8, 2);
  emel::generator::detail::native_backend invalid_backend{};
  invalid_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  invalid_backend.output_native.tensor = &invalid_q6_tensor;
  invalid_backend.output_native.cols = 8;
  invalid_backend.output_native.rows = 2;
  invalid_backend.output = invalid_backend.output_native;
  invalid_backend.output_argmax = invalid_backend.output_native;
  CHECK_FALSE(emel::generator::detail::prepare_prepared_output_logits(invalid_backend));
  CHECK_FALSE(emel::generator::detail::prepare_packed_output_logits(invalid_backend));

  emel::generator::detail::native_backend no_simd_backend{};
  no_simd_backend.kernel_kind = emel::kernel::kernel_kind::x86_64;
  no_simd_backend.output_native.tensor = &q6_tensor;
  no_simd_backend.output_native.cols = static_cast<int32_t>(QK_K);
  no_simd_backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  no_simd_backend.output = no_simd_backend.output_native;
  no_simd_backend.output_argmax = no_simd_backend.output_native;
  REQUIRE(emel::generator::detail::prepare_output_logits(no_simd_backend));
  CHECK(no_simd_backend.output.tensor == &q6_tensor);
  CHECK(no_simd_backend.output_argmax.tensor == &q6_tensor);

  emel::generator::detail::native_backend huge_backend{};
  huge_backend.output.tensor = &packed_tensor;
  huge_backend.output_argmax.tensor = &packed_tensor;
  huge_backend.n_embd = static_cast<int32_t>(
      (emel::generator::detail::quant::MAX_Q8_K_BLOCKS + 1u) * QK_K);
  CHECK_FALSE(emel::generator::detail::prepare_logits_input_q8_workspace(huge_backend));
}

TEST_CASE("generator_detail_tensor_helpers_reject_invalid_records_explicitly") {
  emel::generator::detail::tensor_matrix out = {};
  emel::model::data::tensor_record empty_tensor = {};
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(empty_tensor, out));

  std::array<float, 4> f32_data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto f32_tensor =
      make_tensor_record(f32_data.data(), emel::kernel::detail::dtype_f32, 2, 2);
  std::vector<float> copy_out(2u, 0.0f);
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(f32_tensor, -1, copy_out));

  emel::model::data::tensor_record bad_dims = f32_tensor;
  bad_dims.dims[1] = 0u;
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(bad_dims, out));

  emel::model::data::tensor_record unsupported = f32_tensor;
  unsupported.type = 255;
  CHECK_FALSE(emel::generator::detail::bind_tensor_rows(unsupported, out));
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(unsupported, 0, copy_out));

  std::vector<float> wrong_size(3u, 0.0f);
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(f32_tensor, 0, wrong_size));
  CHECK_FALSE(emel::generator::detail::copy_tensor_row(f32_tensor, 3, copy_out));
}

TEST_CASE("generator_detail_numeric_helpers_reject_invalid_shapes_explicitly") {
  emel::generator::detail::native_backend backend{};
  emel::generator::detail::tensor_matrix empty_matrix = {};
  std::array<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 2> output = {};
  int32_t selected_index = -1;
  float selected_score = 0.0f;
  std::array<emel::kernel::detail::quant::block_q8_k, 1> q8_input = {};

  CHECK_FALSE(emel::generator::detail::matmul_vector(
      backend, empty_matrix, std::span<const float>(input.data(), input.size()),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::generator::detail::matmul_vector_argmax(
      backend,
      empty_matrix,
      std::span<const float>(input.data(), input.size()),
      selected_index,
      selected_score));
  CHECK_FALSE(emel::generator::detail::quantize_vector_q8_k(
      std::span<const float>(input.data(), 3u), std::span(q8_input.data(), q8_input.size())));
  CHECK_FALSE(emel::generator::detail::matmul_vector_q8_input(
      backend,
      empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_k>(q8_input.data(), q8_input.size()),
      static_cast<int32_t>(QK_K),
      std::span<float>(output.data(), output.size())));
  CHECK_FALSE(emel::generator::detail::matmul_vector_q8_input_argmax(
      backend,
      empty_matrix,
      std::span<const emel::kernel::detail::quant::block_q8_k>(q8_input.data(), q8_input.size()),
      static_cast<int32_t>(QK_K),
      selected_index,
      selected_score));

  std::array<float, 2> weight = {1.0f, 1.0f};
  CHECK_FALSE(emel::generator::detail::rms_norm(
      std::span<const float>(input.data(), 0u),
      std::span<const float>(weight.data(), weight.size()),
      1.0e-5f,
      std::span<float>(output.data(), output.size())));

  auto q6_rows = make_q6_rows();
  auto q6_tensor = make_tensor_record(
      q6_rows.data(), emel::kernel::detail::dtype_q6_k, static_cast<int32_t>(QK_K),
      static_cast<int32_t>(Q6_K_X8_ROWS));
  emel::generator::detail::native_backend packed_backend{};
  packed_backend.kernel_kind = emel::kernel::kernel_kind::aarch64;
  packed_backend.output_native.tensor = &q6_tensor;
  packed_backend.output_native.cols = static_cast<int32_t>(QK_K);
  packed_backend.output_native.rows = static_cast<int32_t>(Q6_K_X8_ROWS);
  packed_backend.output = packed_backend.output_native;
  packed_backend.output_argmax = packed_backend.output_native;
  REQUIRE(emel::generator::detail::prepare_packed_output_logits(packed_backend));
  std::vector<float> packed_copy(QK_K, 0.0f);
  CHECK_FALSE(
      emel::generator::detail::copy_tensor_row(*packed_backend.output.tensor, 0, packed_copy));

  std::array<float, 4> rope_identity = {1.0f, 2.0f, 3.0f, 4.0f};
  const auto rope_before = rope_identity;
  emel::generator::detail::apply_rope(rope_identity, 1, 1, 1, 3, 10000.0f);
  CHECK(rope_identity == rope_before);
}

TEST_CASE("generator_detail_request_and_backend_validators_reject_invalid_inputs_explicitly") {
  int32_t err = 123;
  CHECK_FALSE(emel::generator::detail::check_backend(nullptr, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  emel::generator::detail::native_backend backend{};
  backend.model = reinterpret_cast<const emel::model::data *>(0x1);
  backend.n_embd = 4;
  backend.n_head = 1;
  backend.n_head_kv = 1;
  backend.n_layer = 1;
  backend.n_vocab = 8;
  backend.n_ctx = 4;
  backend.head_dim = 4;
  backend.head_dim_kv = 4;
  CHECK_FALSE(emel::generator::detail::check_backend(&backend, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  emel::graph::processor::event::execute request = {};
  CHECK(emel::generator::detail::request_plan(request, &err) == nullptr);
  CHECK(err == emel::generator::detail::k_error_invalid);

  backend.bound_tokens.resize(2u);
  backend.bound_positions.resize(2u);
  CHECK_FALSE(emel::generator::detail::store_bound_request(backend, request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  std::array<float, 4> probs = {1.0f, 1.0f, 1.0f, 1.0f};
  std::array<float, 4> rounded = {1.0f, 1.0f, 1.0f, 1.0f};
  emel::generator::detail::fill_masked_softmax_probs_ggml(
      std::span<const float>(), 0, probs, rounded);
  CHECK(std::all_of(probs.begin(), probs.end(), [](const float value) { return value == 0.0f; }));
  CHECK(std::all_of(
      rounded.begin(), rounded.end(), [](const float value) { return value == 0.0f; }));
}

TEST_CASE("generator_detail_runtime_wrappers_validate_requests_explicitly") {
  runtime_request_fixture fixture{};
  int32_t err = -1;
  bool reused = true;

  CHECK(emel::generator::detail::validate(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.request.expected_outputs = 2;
  CHECK_FALSE(emel::generator::detail::validate(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  fixture.request.expected_outputs = fixture.plan.expected_outputs;

  CHECK(emel::generator::detail::validate_preselected_argmax(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.io.selected_score_out = nullptr;
  CHECK_FALSE(emel::generator::detail::validate_preselected_argmax(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  fixture.io.selected_score_out = &fixture.selected_score;

  CHECK(emel::generator::detail::prepare_graph(fixture.request, &reused, &err));
  CHECK_FALSE(reused);
  CHECK(err == emel::generator::detail::k_error_ok);

  CHECK(emel::generator::detail::alloc_graph(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.request.compute_ctx = nullptr;
  CHECK_FALSE(emel::generator::detail::bind_inputs(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
}

TEST_CASE("generator_detail_runtime_wrappers_bind_run_and_extract_explicitly") {
  runtime_request_fixture fixture{};
  int32_t err = -1;
  int32_t outputs = 0;

  CHECK(emel::generator::detail::bind_inputs(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture.backend.bound_ready);
  CHECK(fixture.backend.bound_token_count == 1);
  CHECK(fixture.backend.bound_position_count == 1);

  fixture.backend.bound_ready = false;
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_preselected_argmax(
      fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  fixture.backend.bound_ready = true;
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash(fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  CHECK_FALSE(emel::generator::detail::run_kernel_nonflash_preselected_argmax(
      fixture.request, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);

  CHECK(emel::generator::detail::extract_outputs(fixture.request, &outputs, &err));
  CHECK(outputs == 1);
  CHECK(err == emel::generator::detail::k_error_ok);
  CHECK(fixture.logits[0] == doctest::Approx(0.25f));
  CHECK(fixture.logits[1] == doctest::Approx(0.5f));
  CHECK(fixture.logits[2] == doctest::Approx(0.75f));
  CHECK(fixture.logits[3] == doctest::Approx(1.0f));
  CHECK(fixture.logits[4] == doctest::Approx(-1.0f));
  CHECK(fixture.logits[5] == doctest::Approx(-1.0f));

  fixture.io.logits = nullptr;
  CHECK_FALSE(emel::generator::detail::extract_outputs(fixture.request, &outputs, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
  fixture.io.logits = fixture.logits.data();

  CHECK(emel::generator::detail::extract_preselected_argmax(fixture.request, &outputs, &err));
  CHECK(outputs == 1);
  CHECK(err == emel::generator::detail::k_error_ok);

  fixture.io.selected_score_out = nullptr;
  CHECK_FALSE(emel::generator::detail::extract_preselected_argmax(fixture.request, &outputs, &err));
  CHECK(err == emel::generator::detail::k_error_invalid);
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
