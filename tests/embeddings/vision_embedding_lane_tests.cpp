#include <array>
#include <cstdint>
#include <vector>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/text/conditioner/detail.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "te_fixture.hpp"

namespace {

namespace embedding_detail = emel::embeddings::generator::detail;
namespace te_fixture = emel::tests::embeddings::te_fixture;

using te_fixture::cached_te_fixture;
using te_fixture::initialize_embedding_generator;
using te_fixture::l2_norm;
using te_fixture::make_rgba_square;
using te_fixture::max_abs_difference;
using te_fixture::te_assets_present;

struct image_embed_callback_probe {
  bool done_called = false;
  bool error_called = false;
  const emel::embeddings::generator::event::embed_image * request = nullptr;
  int32_t output_dimension = 0;
  emel::error::type err = emel::error::cast(emel::embeddings::generator::error::none);

  void on_done(const emel::embeddings::generator::events::image_embedding_done & ev) noexcept {
    done_called = true;
    request = ev.request;
    output_dimension = ev.output_dimension;
  }

  void on_error(const emel::embeddings::generator::events::image_embedding_error & ev) noexcept {
    error_called = true;
    request = ev.request;
    err = ev.err;
  }
};

}  // namespace

TEST_CASE("embeddings vision lane returns normalized TE embeddings when fixture present") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE vision-lane embedding test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  REQUIRE(emel::model::architecture_name_view(*fixture.model) == "omniembed");

  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm embedding_generator{
    *fixture.model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(embedding_generator, initialize_error, tokenizer);

  const std::vector<uint8_t> red_square = make_rgba_square(255u, 0u, 0u, 32, 32);
  const std::vector<uint8_t> blue_square = make_rgba_square(0u, 0u, 255u, 32, 32);

  std::array<float, 1280> red_embedding = {};
  std::array<float, 1280> blue_embedding = {};
  int32_t red_dimension = -1;
  int32_t blue_dimension = -1;
  emel::error::type red_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type blue_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_image red_request{
    red_square,
    32,
    32,
    red_embedding,
    red_dimension,
  };
  red_request.error_out = &red_error;
  REQUIRE(embedding_generator.process_event(red_request));
  CHECK(red_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(red_dimension == 1280);
  CHECK(l2_norm(std::span<const float>{
            red_embedding.data(),
            static_cast<size_t>(red_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));

  emel::embeddings::generator::event::embed_image blue_request{
    blue_square,
    32,
    32,
    blue_embedding,
    blue_dimension,
  };
  blue_request.error_out = &blue_error;
  REQUIRE(embedding_generator.process_event(blue_request));
  CHECK(blue_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(blue_dimension == 1280);
  CHECK(l2_norm(std::span<const float>{
            blue_embedding.data(),
            static_cast<size_t>(blue_dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
  CHECK(max_abs_difference(std::span<const float>{
            red_embedding.data(),
            static_cast<size_t>(red_dimension)},
        std::span<const float>{
            blue_embedding.data(),
            static_cast<size_t>(blue_dimension)}) > 1.0e-5f);
}

TEST_CASE("embeddings vision runtime sizes feature buffers for expansion tensors") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE vision buffer sizing test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();

  emel::embeddings::generator::action::context context = {};
  context.model = fixture.model.get();
  REQUIRE(embedding_detail::reserve_scratch(context, *fixture.model));

  int32_t spatial = context.image.input_size / 4;
  REQUIRE(context.image.feature_buffer_elements >= spatial * spatial * context.image.stage0.output_channels);

  for (int32_t index = 0; index < context.image.block_count; ++index) {
    const auto & block = context.image.blocks[static_cast<size_t>(index)];
    int32_t expansion_spatial = spatial;
    if (block.has_dw_start) {
      const int32_t dw_start_stride = block.has_dw_mid ? 1 : block.stride;
      expansion_spatial =
          embedding_detail::output_dim_same(spatial, block.dw_start.kernel_h, dw_start_stride);
    }
    CHECK(context.image.feature_buffer_elements >=
          expansion_spatial * expansion_spatial * block.expanded_channels);

    int32_t output_spatial = expansion_spatial;
    if (block.has_dw_mid) {
      output_spatial =
          embedding_detail::output_dim_same(expansion_spatial, block.dw_mid.kernel_h, block.stride);
    }
    CHECK(context.image.feature_buffer_elements >=
          output_spatial * output_spatial * block.output_channels);
    spatial = output_spatial;
  }

  CHECK(context.image.feature_buffer_elements >= spatial * spatial * context.image.stage4.output_channels);
  CHECK(context.image.feature_buffer_elements >= spatial * spatial * context.image.head.output_channels);
}

TEST_CASE("embeddings vision lane supports truncation and rejects malformed image payloads") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE vision truncation test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm embedding_generator{
    *fixture.model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(embedding_generator, initialize_error, tokenizer);

  const std::vector<uint8_t> red_square = make_rgba_square(255u, 0u, 0u, 32, 32);
  std::array<float, 256> truncated_output = {};
  int32_t truncated_dimension = -1;
  emel::error::type truncated_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_image truncate_request{
    red_square,
    32,
    32,
    truncated_output,
    truncated_dimension,
  };
  truncate_request.truncate_dimension = 256;
  truncate_request.error_out = &truncated_error;

  REQUIRE(embedding_generator.process_event(truncate_request));
  CHECK(truncated_error == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(truncated_dimension == 256);
  CHECK(l2_norm(std::span<const float>{truncated_output.data(), 256u}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));

  std::array<float, 1280> invalid_output = {};
  int32_t invalid_dimension = -1;
  emel::error::type invalid_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_image invalid_request{
    std::span<const uint8_t>{red_square.data(), red_square.size() - 1u},
    32,
    32,
    invalid_output,
    invalid_dimension,
  };
  invalid_request.error_out = &invalid_error;

  CHECK_FALSE(embedding_generator.process_event(invalid_request));
  CHECK(invalid_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(invalid_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));
}

TEST_CASE("embeddings vision helper paths cover image request callbacks and validation") {
  CHECK(embedding_detail::is_valid_image_payload(std::span<const uint8_t>{}, 0, 0) == false);

  std::vector<uint8_t> rgba = make_rgba_square(255u, 0u, 0u, 8, 8);
  std::array<float, 1280> output = {};
  int32_t output_dimension = -1;
  emel::error::type embed_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  image_embed_callback_probe probe = {};
  emel::embeddings::generator::event::embed_image request{
    rgba,
    8,
    8,
    output,
    output_dimension,
  };
  request.error_out = &embed_error;
  request.on_done =
      emel::callback<void(const emel::embeddings::generator::events::image_embedding_done &)>::from<
          image_embed_callback_probe,
          &image_embed_callback_probe::on_done>(&probe);
  request.on_error =
      emel::callback<void(const emel::embeddings::generator::events::image_embedding_error &)>::from<
          image_embed_callback_probe,
          &image_embed_callback_probe::on_error>(&probe);

  emel::embeddings::generator::event::embed_image_ctx runtime_ctx = {};
  emel::embeddings::generator::event::embed_image_run runtime_ev{request, runtime_ctx};
  CHECK(embedding_detail::has_embed_callbacks(runtime_ev));
  CHECK(embedding_detail::has_embed_error_callback(runtime_ev));
  runtime_ctx.output_dimension = 321;
  embedding_detail::set_error(runtime_ev, emel::embeddings::generator::error::invalid_request);
  embedding_detail::write_embed_error_out(runtime_ev);
  embedding_detail::emit_embed_done(runtime_ev);
  embedding_detail::emit_embed_error(runtime_ev);

  CHECK(output_dimension == 321);
  CHECK(embed_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(probe.done_called);
  CHECK(probe.error_called);
  CHECK(probe.request == &request);
}

TEST_CASE("embeddings vision pointwise direct path matches scalar pointwise reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t output_channels = 80;
  constexpr int32_t input_channels = 48;
  constexpr int32_t pixel_count = 9;
  constexpr size_t weight_count =
      static_cast<size_t>(output_channels) * static_cast<size_t>(input_channels);

  std::array<uint16_t, weight_count> weight_storage = {};
  for (size_t idx = 0; idx < weight_storage.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 7u) % 41u) - 20;
    const float value = static_cast<float>(centered) * 0.03125f;
    weight_storage[idx] = emel::kernel::detail::quant::fp32_to_fp16(value);
  }

  emel::model::data::tensor_record weight_tensor = {};
  weight_tensor.data = weight_storage.data();
  weight_tensor.data_size = static_cast<uint64_t>(weight_storage.size() * sizeof(uint16_t));
  weight_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  weight_tensor.n_dims = 4;
  weight_tensor.dims[0] = 1u;
  weight_tensor.dims[1] = 1u;
  weight_tensor.dims[2] = static_cast<uint64_t>(input_channels);
  weight_tensor.dims[3] = static_cast<uint64_t>(output_channels);

  emel::embeddings::generator::action::matrix_view matrix = {};
  REQUIRE(embedding_detail::bind_pointwise_f16(
      weight_tensor, output_channels, input_channels, matrix));
  REQUIRE(matrix.packed_rhs_f32 != nullptr);

  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(input_channels)> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 5u) % 29u) - 14;
    input[idx] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      output_direct = {};
  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      output_scalar = {};

  REQUIRE(embedding_detail::pointwise_conv_hwc_direct_f32(
      matrix, input.data(), pixel_count, output_direct.data()));
  REQUIRE(embedding_detail::pointwise_conv_hwc(
      matrix, input.data(), pixel_count, output_scalar.data()));

  for (size_t idx = 0; idx < output_direct.size(); ++idx) {
    CHECK(output_direct[idx] == doctest::Approx(output_scalar[idx]).epsilon(1.0e-5f));
  }
#endif
}

TEST_CASE("embeddings vision pointwise direct path matches scalar reference across output tails") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t output_channels = 77;
  constexpr int32_t input_channels = 48;
  constexpr int32_t pixel_count = 9;
  constexpr size_t weight_count =
      static_cast<size_t>(output_channels) * static_cast<size_t>(input_channels);

  std::array<uint16_t, weight_count> weight_storage = {};
  for (size_t idx = 0; idx < weight_storage.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 9u) % 53u) - 26;
    const float value = static_cast<float>(centered) * 0.015625f;
    weight_storage[idx] = emel::kernel::detail::quant::fp32_to_fp16(value);
  }

  emel::model::data::tensor_record weight_tensor = {};
  weight_tensor.data = weight_storage.data();
  weight_tensor.data_size = static_cast<uint64_t>(weight_storage.size() * sizeof(uint16_t));
  weight_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  weight_tensor.n_dims = 4;
  weight_tensor.dims[0] = 1u;
  weight_tensor.dims[1] = 1u;
  weight_tensor.dims[2] = static_cast<uint64_t>(input_channels);
  weight_tensor.dims[3] = static_cast<uint64_t>(output_channels);

  emel::embeddings::generator::action::matrix_view matrix = {};
  REQUIRE(embedding_detail::bind_pointwise_f16(
      weight_tensor, output_channels, input_channels, matrix));
  REQUIRE(matrix.packed_rhs_f32 != nullptr);

  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(input_channels)> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 3u) % 31u) - 15;
    input[idx] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      output_direct = {};
  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      output_scalar = {};

  REQUIRE(embedding_detail::pointwise_conv_hwc_direct_f32(
      matrix, input.data(), pixel_count, output_direct.data()));
  REQUIRE(embedding_detail::pointwise_conv_hwc(
      matrix, input.data(), pixel_count, output_scalar.data()));

  for (size_t idx = 0; idx < output_direct.size(); ++idx) {
    CHECK(output_direct[idx] == doctest::Approx(output_scalar[idx]).epsilon(1.0e-5f));
  }
#endif
}

TEST_CASE("embeddings vision batch norm precomputed affine matches scalar reference") {
  constexpr int32_t channels = 29;
  constexpr int32_t spatial = 3;
  constexpr float epsilon = 1.0e-5f;

  std::array<float, channels> weight = {};
  std::array<float, channels> bias = {};
  std::array<float, channels> running_mean = {};
  std::array<float, channels> running_var = {};
  for (int32_t channel = 0; channel < channels; ++channel) {
    weight[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 7) - 3) * 0.125f + 1.0f;
    bias[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 9) - 4) * 0.0625f;
    running_mean[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 11) - 5) * 0.03125f;
    running_var[static_cast<size_t>(channel)] =
        0.5f + static_cast<float>((channel % 5) + 1) * 0.125f;
  }

  const auto make_vector_tensor = [](float * data, const int32_t size) noexcept {
    emel::model::data::tensor_record tensor = {};
    tensor.data = data;
    tensor.data_size = static_cast<uint64_t>(size) * sizeof(float);
    tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<uint64_t>(size);
    return tensor;
  };

  emel::embeddings::generator::action::batch_norm_view batch_norm = {};
  REQUIRE(embedding_detail::bind_batch_norm(make_vector_tensor(weight.data(), channels),
                                            make_vector_tensor(bias.data(), channels),
                                            make_vector_tensor(running_mean.data(), channels),
                                            make_vector_tensor(running_var.data(), channels),
                                            channels,
                                            batch_norm));
  REQUIRE(batch_norm.scale != nullptr);
  REQUIRE(batch_norm.shift != nullptr);

  std::array<float, static_cast<size_t>(spatial) * static_cast<size_t>(spatial) *
                        static_cast<size_t>(channels)>
      values = {};
  std::array<float, values.size()> expected = {};
  for (size_t idx = 0; idx < values.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 7u) % 41u) - 20;
    values[idx] = static_cast<float>(centered) * 0.0625f;
    expected[idx] = values[idx];
  }

  for (int32_t pixel_index = 0; pixel_index < spatial * spatial; ++pixel_index) {
    for (int32_t channel = 0; channel < channels; ++channel) {
      const size_t idx = static_cast<size_t>(pixel_index) * static_cast<size_t>(channels) +
          static_cast<size_t>(channel);
      const float scale = weight[static_cast<size_t>(channel)] /
          std::sqrt(running_var[static_cast<size_t>(channel)] + epsilon);
      const float shifted =
          (expected[idx] - running_mean[static_cast<size_t>(channel)]) * scale +
          bias[static_cast<size_t>(channel)];
      expected[idx] = std::max(shifted, 0.0f);
    }
  }

  embedding_detail::apply_batch_norm_hwc<true>(values.data(), spatial, batch_norm, epsilon);

  for (size_t idx = 0; idx < values.size(); ++idx) {
    CHECK(values[idx] == doctest::Approx(expected[idx]).epsilon(1.0e-5f));
  }
}

TEST_CASE("embeddings vision fused pointwise batch norm matches unfused reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t output_channels = 77;
  constexpr int32_t input_channels = 48;
  constexpr int32_t pixel_count = 9;
  constexpr size_t weight_count =
      static_cast<size_t>(output_channels) * static_cast<size_t>(input_channels);

  std::array<uint16_t, weight_count> weight_storage = {};
  for (size_t idx = 0; idx < weight_storage.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 13u) % 67u) - 33;
    const float value = static_cast<float>(centered) * 0.015625f;
    weight_storage[idx] = emel::kernel::detail::quant::fp32_to_fp16(value);
  }

  emel::model::data::tensor_record weight_tensor = {};
  weight_tensor.data = weight_storage.data();
  weight_tensor.data_size = static_cast<uint64_t>(weight_storage.size() * sizeof(uint16_t));
  weight_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  weight_tensor.n_dims = 4;
  weight_tensor.dims[0] = 1u;
  weight_tensor.dims[1] = 1u;
  weight_tensor.dims[2] = static_cast<uint64_t>(input_channels);
  weight_tensor.dims[3] = static_cast<uint64_t>(output_channels);

  emel::embeddings::generator::action::matrix_view matrix = {};
  REQUIRE(embedding_detail::bind_pointwise_f16(
      weight_tensor, output_channels, input_channels, matrix));

  std::array<float, output_channels> bn_weight = {};
  std::array<float, output_channels> bn_bias = {};
  std::array<float, output_channels> bn_mean = {};
  std::array<float, output_channels> bn_var = {};
  for (int32_t channel = 0; channel < output_channels; ++channel) {
    bn_weight[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 5) - 2) * 0.125f + 1.0f;
    bn_bias[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 7) - 3) * 0.0625f;
    bn_mean[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 9) - 4) * 0.03125f;
    bn_var[static_cast<size_t>(channel)] =
        0.5f + static_cast<float>((channel % 6) + 1) * 0.125f;
  }

  const auto make_vector_tensor = [](float * data, const int32_t size) noexcept {
    emel::model::data::tensor_record tensor = {};
    tensor.data = data;
    tensor.data_size = static_cast<uint64_t>(size) * sizeof(float);
    tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<uint64_t>(size);
    return tensor;
  };

  emel::embeddings::generator::action::batch_norm_view batch_norm = {};
  REQUIRE(embedding_detail::bind_batch_norm(make_vector_tensor(bn_weight.data(), output_channels),
                                            make_vector_tensor(bn_bias.data(), output_channels),
                                            make_vector_tensor(bn_mean.data(), output_channels),
                                            make_vector_tensor(bn_var.data(), output_channels),
                                            output_channels,
                                            batch_norm));

  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(input_channels)> input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 5u) % 29u) - 14;
    input[idx] = static_cast<float>(centered) * 0.0625f;
  }

  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      fused_output = {};
  std::array<float, static_cast<size_t>(pixel_count) * static_cast<size_t>(output_channels)>
      unfused_output = {};

  REQUIRE(embedding_detail::pointwise_conv_hwc_direct_f32_bn<true>(
      matrix, input.data(), pixel_count, batch_norm, fused_output.data()));
  REQUIRE(embedding_detail::pointwise_conv_hwc_direct_f32(
      matrix, input.data(), pixel_count, unfused_output.data()));
  embedding_detail::apply_batch_norm_hwc<true>(unfused_output.data(), 3, batch_norm, 1.0e-5f);

  for (size_t idx = 0; idx < fused_output.size(); ++idx) {
    CHECK(fused_output[idx] == doctest::Approx(unfused_output[idx]).epsilon(1.0e-5f));
  }
#endif
}

TEST_CASE("embeddings vision fused standard conv batch norm matches unfused reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t kernel_h = 3;
  constexpr int32_t kernel_w = 3;
  constexpr int32_t input_channels = 48;
  constexpr int32_t output_channels = 29;
  constexpr int32_t input_spatial = 8;
  constexpr int32_t stride = 2;
  constexpr size_t weight_count = static_cast<size_t>(kernel_h) * static_cast<size_t>(kernel_w) *
      static_cast<size_t>(input_channels) * static_cast<size_t>(output_channels);

  std::array<uint16_t, weight_count> weight_storage = {};
  for (size_t idx = 0; idx < weight_storage.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 11u) % 37u) - 18;
    const float value = static_cast<float>(centered) * 0.015625f;
    weight_storage[idx] = emel::kernel::detail::quant::fp32_to_fp16(value);
  }

  emel::model::data::tensor_record weight_tensor = {};
  weight_tensor.data = weight_storage.data();
  weight_tensor.data_size = static_cast<uint64_t>(weight_storage.size() * sizeof(uint16_t));
  weight_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  weight_tensor.n_dims = 4;
  weight_tensor.dims[0] = static_cast<uint64_t>(kernel_w);
  weight_tensor.dims[1] = static_cast<uint64_t>(kernel_h);
  weight_tensor.dims[2] = static_cast<uint64_t>(input_channels);
  weight_tensor.dims[3] = static_cast<uint64_t>(output_channels);

  emel::embeddings::generator::action::conv2d_view conv = {};
  REQUIRE(embedding_detail::bind_conv_f16_hwio(
      weight_tensor, kernel_h, kernel_w, input_channels, output_channels, conv));

  std::array<float, output_channels> bn_weight = {};
  std::array<float, output_channels> bn_bias = {};
  std::array<float, output_channels> bn_mean = {};
  std::array<float, output_channels> bn_var = {};
  for (int32_t channel = 0; channel < output_channels; ++channel) {
    bn_weight[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 5) - 2) * 0.125f + 1.0f;
    bn_bias[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 7) - 3) * 0.0625f;
    bn_mean[static_cast<size_t>(channel)] =
        static_cast<float>((channel % 9) - 4) * 0.03125f;
    bn_var[static_cast<size_t>(channel)] =
        0.5f + static_cast<float>((channel % 6) + 1) * 0.125f;
  }

  const auto make_vector_tensor = [](float * data, const int32_t size) noexcept {
    emel::model::data::tensor_record tensor = {};
    tensor.data = data;
    tensor.data_size = static_cast<uint64_t>(size) * sizeof(float);
    tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<uint64_t>(size);
    return tensor;
  };

  emel::embeddings::generator::action::batch_norm_view batch_norm = {};
  REQUIRE(embedding_detail::bind_batch_norm(make_vector_tensor(bn_weight.data(), output_channels),
                                            make_vector_tensor(bn_bias.data(), output_channels),
                                            make_vector_tensor(bn_mean.data(), output_channels),
                                            make_vector_tensor(bn_var.data(), output_channels),
                                            output_channels,
                                            batch_norm));

  std::array<float, static_cast<size_t>(input_spatial) * static_cast<size_t>(input_spatial) *
                        static_cast<size_t>(input_channels)>
      input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 3u) % 19u) - 9;
    input[idx] = static_cast<float>(centered) * 0.0625f;
  }

  constexpr int32_t output_spatial = 4;
  std::array<float, static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
                        static_cast<size_t>(output_channels)>
      fused_output = {};
  std::array<float, fused_output.size()> unfused_output = {};
  std::array<float, static_cast<size_t>(kernel_h) * static_cast<size_t>(kernel_w) *
                        static_cast<size_t>(input_channels)>
      patch_buffer = {};
  int32_t fused_spatial = 0;
  int32_t unfused_spatial = 0;

  REQUIRE(embedding_detail::standard_conv_hwc_bn<true>(conv,
                                                       input.data(),
                                                       input_spatial,
                                                       stride,
                                                       patch_buffer.data(),
                                                       patch_buffer.size(),
                                                       batch_norm,
                                                       fused_output.data(),
                                                       fused_spatial));
  REQUIRE(embedding_detail::standard_conv_hwc(conv,
                                              input.data(),
                                              input_spatial,
                                              stride,
                                              patch_buffer.data(),
                                              patch_buffer.size(),
                                              unfused_output.data(),
                                              unfused_spatial));
  REQUIRE(fused_spatial == unfused_spatial);
  embedding_detail::apply_batch_norm_hwc<true>(
      unfused_output.data(), unfused_spatial, batch_norm, 1.0e-5f);

  for (size_t idx = 0; idx < fused_output.size(); ++idx) {
    CHECK(fused_output[idx] == doctest::Approx(unfused_output[idx]).epsilon(1.0e-5f));
  }
#endif
}

TEST_CASE("embeddings vision standard conv direct path matches patch reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t kernel_h = 3;
  constexpr int32_t kernel_w = 3;
  constexpr int32_t input_channels = 48;
  constexpr int32_t output_channels = 32;
  constexpr int32_t input_spatial = 8;
  constexpr int32_t stride = 2;
  constexpr size_t weight_count = static_cast<size_t>(kernel_h) * static_cast<size_t>(kernel_w) *
      static_cast<size_t>(input_channels) * static_cast<size_t>(output_channels);

  std::array<uint16_t, weight_count> weight_storage = {};
  for (size_t idx = 0; idx < weight_storage.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 11u) % 37u) - 18;
    const float value = static_cast<float>(centered) * 0.015625f;
    weight_storage[idx] = emel::kernel::detail::quant::fp32_to_fp16(value);
  }

  emel::model::data::tensor_record weight_tensor = {};
  weight_tensor.data = weight_storage.data();
  weight_tensor.data_size = static_cast<uint64_t>(weight_storage.size() * sizeof(uint16_t));
  weight_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  weight_tensor.n_dims = 4;
  weight_tensor.dims[0] = static_cast<uint64_t>(kernel_w);
  weight_tensor.dims[1] = static_cast<uint64_t>(kernel_h);
  weight_tensor.dims[2] = static_cast<uint64_t>(input_channels);
  weight_tensor.dims[3] = static_cast<uint64_t>(output_channels);

  emel::embeddings::generator::action::conv2d_view conv = {};
  REQUIRE(embedding_detail::bind_conv_f16_hwio(
      weight_tensor, kernel_h, kernel_w, input_channels, output_channels, conv));
  REQUIRE(conv.kernel_major_f32 != nullptr);

  std::array<float, static_cast<size_t>(input_spatial) * static_cast<size_t>(input_spatial) *
                        static_cast<size_t>(input_channels)>
      input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 3u) % 19u) - 9;
    input[idx] = static_cast<float>(centered) * 0.0625f;
  }

  const int32_t output_spatial =
      embedding_detail::output_dim_same(input_spatial, kernel_h, stride);
  std::vector<float> output_direct(
      static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
          static_cast<size_t>(output_channels),
      0.0f);
  std::vector<float> output_reference(output_direct.size(), 0.0f);
  std::vector<float> patch_buffer(
      static_cast<size_t>(kernel_h) * static_cast<size_t>(kernel_w) *
          static_cast<size_t>(input_channels),
      0.0f);
  int32_t output_direct_spatial = 0;
  int32_t output_reference_height = 0;
  int32_t output_reference_width = 0;

  REQUIRE(embedding_detail::standard_conv_hwc(conv,
                                              input.data(),
                                              input_spatial,
                                              stride,
                                              patch_buffer.data(),
                                              patch_buffer.size(),
                                              output_direct.data(),
                                              output_direct_spatial));
  REQUIRE(embedding_detail::standard_conv_hwc_rect(conv,
                                                   input.data(),
                                                   input_spatial,
                                                   input_spatial,
                                                   stride,
                                                   stride,
                                                   patch_buffer.data(),
                                                   patch_buffer.size(),
                                                   output_reference.data(),
                                                   output_reference_height,
                                                   output_reference_width));
  REQUIRE(output_direct_spatial == output_reference_height);
  REQUIRE(output_reference_height == output_reference_width);

  for (size_t idx = 0; idx < output_direct.size(); ++idx) {
    CHECK(output_direct[idx] == doctest::Approx(output_reference[idx]).epsilon(1.0e-5f));
  }
#endif
}

TEST_CASE("embeddings vision standard conv direct path matches patch reference across output tails") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t kernel_h = 3;
  constexpr int32_t kernel_w = 3;
  constexpr int32_t input_channels = 32;
  constexpr int32_t output_channels = 29;
  constexpr int32_t input_spatial = 8;
  constexpr int32_t stride = 2;
  constexpr size_t weight_count = static_cast<size_t>(kernel_h) * static_cast<size_t>(kernel_w) *
      static_cast<size_t>(input_channels) * static_cast<size_t>(output_channels);

  std::array<uint16_t, weight_count> weight_storage = {};
  for (size_t idx = 0; idx < weight_storage.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 13u) % 43u) - 21;
    const float value = static_cast<float>(centered) * 0.015625f;
    weight_storage[idx] = emel::kernel::detail::quant::fp32_to_fp16(value);
  }

  emel::model::data::tensor_record weight_tensor = {};
  weight_tensor.data = weight_storage.data();
  weight_tensor.data_size = static_cast<uint64_t>(weight_storage.size() * sizeof(uint16_t));
  weight_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  weight_tensor.n_dims = 4;
  weight_tensor.dims[0] = static_cast<uint64_t>(kernel_w);
  weight_tensor.dims[1] = static_cast<uint64_t>(kernel_h);
  weight_tensor.dims[2] = static_cast<uint64_t>(input_channels);
  weight_tensor.dims[3] = static_cast<uint64_t>(output_channels);

  emel::embeddings::generator::action::conv2d_view conv = {};
  REQUIRE(embedding_detail::bind_conv_f16_hwio(
      weight_tensor, kernel_h, kernel_w, input_channels, output_channels, conv));
  REQUIRE(conv.kernel_major_f32 != nullptr);

  std::array<float, static_cast<size_t>(input_spatial) * static_cast<size_t>(input_spatial) *
                        static_cast<size_t>(input_channels)>
      input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 7u) % 47u) - 23;
    input[idx] = static_cast<float>(centered) * 0.03125f;
  }

  constexpr int32_t output_spatial = 4;
  std::array<float, static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
                        static_cast<size_t>(output_channels)>
      output_direct = {};
  std::array<float, static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
                        static_cast<size_t>(output_channels)>
      output_reference = {};
  std::array<float, static_cast<size_t>(input_channels) * static_cast<size_t>(kernel_h) *
                        static_cast<size_t>(kernel_w)>
      patch_buffer = {};
  int32_t direct_spatial = 0;
  int32_t reference_height = 0;
  int32_t reference_width = 0;

  REQUIRE(embedding_detail::standard_conv_hwc(conv,
                                              input.data(),
                                              input_spatial,
                                              stride,
                                              patch_buffer.data(),
                                              patch_buffer.size(),
                                              output_direct.data(),
                                              direct_spatial));
  REQUIRE(embedding_detail::standard_conv_hwc_rect(conv,
                                                   input.data(),
                                                   input_spatial,
                                                   input_spatial,
                                                   stride,
                                                   stride,
                                                   patch_buffer.data(),
                                                   patch_buffer.size(),
                                                   output_reference.data(),
                                                   reference_height,
                                                   reference_width));
  REQUIRE(direct_spatial == reference_height);
  REQUIRE(reference_height == reference_width);

  for (size_t idx = 0; idx < output_direct.size(); ++idx) {
    CHECK(output_direct[idx] == doctest::Approx(output_reference[idx]).epsilon(1.0e-5f));
  }
#endif
}

TEST_CASE("embeddings vision depthwise path matches rect reference") {
#if !(defined(__aarch64__) || defined(__ARM_NEON))
  return;
#else
  constexpr int32_t kernel_h = 3;
  constexpr int32_t kernel_w = 3;
  constexpr int32_t channels = 160;
  constexpr int32_t input_spatial = 8;
  constexpr int32_t stride = 2;
  constexpr size_t weight_count =
      static_cast<size_t>(kernel_h) * static_cast<size_t>(kernel_w) * static_cast<size_t>(channels);

  std::array<uint16_t, weight_count> weight_storage = {};
  for (size_t idx = 0; idx < weight_storage.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 13u) % 43u) - 21;
    const float value = static_cast<float>(centered) * 0.015625f;
    weight_storage[idx] = emel::kernel::detail::quant::fp32_to_fp16(value);
  }

  emel::model::data::tensor_record weight_tensor = {};
  weight_tensor.data = weight_storage.data();
  weight_tensor.data_size = static_cast<uint64_t>(weight_storage.size() * sizeof(uint16_t));
  weight_tensor.type = static_cast<int32_t>(emel::kernel::detail::dtype_f16);
  weight_tensor.n_dims = 4;
  weight_tensor.dims[0] = static_cast<uint64_t>(kernel_w);
  weight_tensor.dims[1] = static_cast<uint64_t>(kernel_h);
  weight_tensor.dims[2] = 1u;
  weight_tensor.dims[3] = static_cast<uint64_t>(channels);

  emel::embeddings::generator::action::conv2d_view conv = {};
  REQUIRE(embedding_detail::bind_conv_f16_hwio(
      weight_tensor, kernel_h, kernel_w, 1, channels, conv));
  REQUIRE(conv.depthwise_kernel_major_f32 != nullptr);

  std::array<float, static_cast<size_t>(input_spatial) * static_cast<size_t>(input_spatial) *
                        static_cast<size_t>(channels)>
      input = {};
  for (size_t idx = 0; idx < input.size(); ++idx) {
    const int32_t centered = static_cast<int32_t>((idx * 3u) % 23u) - 11;
    input[idx] = static_cast<float>(centered) * 0.03125f;
  }

  const int32_t output_spatial =
      embedding_detail::output_dim_same(input_spatial, kernel_h, stride);
  std::vector<float> output_direct(
      static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
          static_cast<size_t>(channels),
      0.0f);
  std::vector<float> output_reference(output_direct.size(), 0.0f);
  int32_t output_direct_spatial = 0;
  int32_t output_reference_height = 0;
  int32_t output_reference_width = 0;

  REQUIRE(embedding_detail::depthwise_conv_hwc(
      conv, input.data(), input_spatial, stride, output_direct.data(), output_direct_spatial));
  REQUIRE(embedding_detail::depthwise_conv_hwc_rect(conv,
                                                    input.data(),
                                                    input_spatial,
                                                    input_spatial,
                                                    stride,
                                                    stride,
                                                    output_reference.data(),
                                                    output_reference_height,
                                                    output_reference_width));
  REQUIRE(output_direct_spatial == output_reference_height);
  REQUIRE(output_reference_height == output_reference_width);

  for (size_t idx = 0; idx < output_direct.size(); ++idx) {
    CHECK(output_direct[idx] == doctest::Approx(output_reference[idx]).epsilon(1.0e-5f));
  }
#endif
}
