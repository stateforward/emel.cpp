#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "emel/diarization/sortformer/detail.hpp"
#include "emel/diarization/sortformer/encoder/detail.hpp"
#include "emel/diarization/sortformer/executor/detail.hpp"
#include "emel/diarization/sortformer/executor/sm.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/transformer/detail.hpp"
#include "emel/kernel/aarch64/actions.hpp"
#include "emel/model/data.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace {

namespace encoder_detail = emel::diarization::sortformer::encoder::detail;
namespace executor = emel::diarization::sortformer::executor;
namespace executor_detail = emel::diarization::sortformer::executor::detail;
namespace modules_detail = emel::diarization::sortformer::modules::detail;
namespace sortformer_detail = emel::diarization::sortformer::detail;
namespace transformer_detail = emel::diarization::sortformer::transformer::detail;

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, encoder_detail::k_pre_tensor_count> k_pre_specs{{
    {"enc.pre.conv.0.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.0.w", 4, {3, 3, 1, 256}},
    {"enc.pre.conv.2.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.2.w", 4, {3, 3, 1, 256}},
    {"enc.pre.conv.3.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.3.w", 4, {1, 1, 256, 256}},
    {"enc.pre.conv.5.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.5.w", 4, {3, 3, 1, 256}},
    {"enc.pre.conv.6.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.6.w", 4, {1, 1, 256, 256}},
    {"enc.pre.out.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"enc.pre.out.w", 2, {4096, encoder_detail::k_model_dim, 0, 0}},
}};

constexpr std::array<tensor_spec, encoder_detail::k_layer_tensor_count> k_layer_specs{{
    {"conv.bn.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.rm", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.rv", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.sc", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.sh", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.bn.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.dw.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.dw.w", 3, {encoder_detail::k_depthwise_kernel, 1, encoder_detail::k_model_dim, 0}},
    {"conv.pw1.b", 1, {1024, 0, 0, 0}},
    {"conv.pw1.w", 3, {1, encoder_detail::k_model_dim, 1024, 0}},
    {"conv.pw2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"conv.pw2.w", 3, {1, encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0}},
    {"ff1.l1.b", 1, {encoder_detail::k_feed_forward_dim, 0, 0, 0}},
    {"ff1.l1.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_feed_forward_dim, 0, 0}},
    {"ff1.l2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"ff1.l2.w", 2, {encoder_detail::k_feed_forward_dim, encoder_detail::k_model_dim, 0, 0}},
    {"ff2.l1.b", 1, {encoder_detail::k_feed_forward_dim, 0, 0, 0}},
    {"ff2.l1.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_feed_forward_dim, 0, 0}},
    {"ff2.l2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"ff2.l2.w", 2, {encoder_detail::k_feed_forward_dim, encoder_detail::k_model_dim, 0, 0}},
    {"nc.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nc.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff1.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff1.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nff2.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"no.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"no.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nsa.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"nsa.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.k.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.k.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.o.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.o.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.p.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.q.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.q.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.v.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
    {"att.v.w", 2, {encoder_detail::k_model_dim, encoder_detail::k_model_dim, 0, 0}},
    {"att.pbu", 2, {encoder_detail::k_attention_head_dim, encoder_detail::k_attention_head_count, 0, 0}},
    {"att.pbv", 2, {encoder_detail::k_attention_head_dim, encoder_detail::k_attention_head_count, 0, 0}},
}};

void append_name(emel::model::data & model,
                 emel::model::data::tensor_record & tensor,
                 const std::string_view name) {
  const auto offset = model.name_bytes_used;
  std::memcpy(model.name_storage.data() + offset, name.data(), name.size());
  tensor.name_offset = offset;
  tensor.name_length = static_cast<uint32_t>(name.size());
  model.name_bytes_used += static_cast<uint32_t>(name.size());
}

void append_tensor(emel::model::data & model, const tensor_spec & spec) {
  static constexpr float k_dummy = 1.0f;
  auto & tensor = model.tensors[model.n_tensors];
  append_name(model, tensor, spec.name);
  tensor.n_dims = spec.n_dims;
  tensor.dims = spec.dims;
  tensor.data = &k_dummy;
  tensor.data_size = sizeof(k_dummy);
  ++model.n_tensors;
}

void append_tensor_with_data(emel::model::data & model,
                             const tensor_spec & spec,
                             std::span<const float> data) {
  auto & tensor = model.tensors[model.n_tensors];
  append_name(model, tensor, spec.name);
  tensor.n_dims = spec.n_dims;
  tensor.dims = spec.dims;
  tensor.data = data.data();
  tensor.data_size = data.size_bytes();
  ++model.n_tensors;
}

void append_layer_tensor(emel::model::data & model,
                         const int32_t layer,
                         const tensor_spec & spec) {
  std::array<char, 64> name = {};
  const int written = std::snprintf(name.data(),
                                    name.size(),
                                    "enc.l%d.%.*s",
                                    layer,
                                    static_cast<int>(spec.name.size()),
                                    spec.name.data());
  REQUIRE(written > 0);
  tensor_spec named_spec = spec;
  named_spec.name = std::string_view{name.data(), static_cast<size_t>(written)};
  append_tensor(model, named_spec);
}

size_t tensor_value_count(const tensor_spec & spec) noexcept {
  size_t count = 1u;
  for (int32_t dim = 0; dim < spec.n_dims; ++dim) {
    count *= static_cast<size_t>(spec.dims[static_cast<size_t>(dim)]);
  }
  return count;
}

void build_encoder_model(emel::model::data & model,
                         const bool include_all_tensors,
                         const bool valid_shapes) {
  std::memset(&model, 0, sizeof(model));

  for (const auto & spec : k_pre_specs) {
    append_tensor(model, spec);
  }

  for (int32_t layer = 0; layer < encoder_detail::k_layer_count; ++layer) {
    for (size_t index = 0u; index < k_layer_specs.size(); ++index) {
      if (!include_all_tensors && layer == 16 && index == k_layer_specs.size() - 1u) {
        continue;
      }

      tensor_spec spec = k_layer_specs[index];
      if (!valid_shapes && layer == 0 && spec.name == "att.q.w") {
        spec.dims = {encoder_detail::k_model_dim - 1, encoder_detail::k_model_dim, 0, 0};
      }
      append_layer_tensor(model, layer, spec);
    }
  }
}

std::unique_ptr<emel::model::data> make_encoder_model(const bool include_all_tensors,
                                                      const bool valid_shapes) {
  auto model = std::make_unique<emel::model::data>();
  build_encoder_model(*model, include_all_tensors, valid_shapes);
  return model;
}

struct pre_encoder_fixture {
  emel::model::data model = {};
  std::vector<float> channel_bias =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count), 0.0f);
  std::vector<float> feature_depthwise =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count * 9), 0.0f);
  std::vector<float> channel_depthwise =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count * 9), 0.0f);
  std::vector<float> channel_pointwise =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count *
                                             encoder_detail::k_pre_channel_count),
                         0.0f);
  std::vector<float> output_bias =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim), 0.0f);
  std::vector<float> output_weight =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_model_dim *
                                             encoder_detail::k_pre_expanded_dim),
                         0.0f);
  std::vector<std::vector<float>> layer_storage = {};

  pre_encoder_fixture() {
    std::memset(&model, 0, sizeof(model));
    layer_storage.reserve(static_cast<size_t>(encoder_detail::k_layer_count *
                                              encoder_detail::k_layer_tensor_count));

    for (int32_t channel = 0; channel < encoder_detail::k_pre_channel_count; ++channel) {
      feature_depthwise[(static_cast<size_t>(channel) * 9u) + 4u] = 1.0f;
      channel_depthwise[(static_cast<size_t>(channel) * 9u) + 4u] = 1.0f;
      const size_t pointwise_index =
          (static_cast<size_t>(channel) * static_cast<size_t>(encoder_detail::k_pre_channel_count)) +
          static_cast<size_t>(channel);
      channel_pointwise[pointwise_index] = 1.0f;
    }

    for (int32_t dim = 0; dim < encoder_detail::k_model_dim; ++dim) {
      const int32_t channel = dim % encoder_detail::k_pre_channel_count;
      const size_t weight_index =
          (static_cast<size_t>(dim) * static_cast<size_t>(encoder_detail::k_pre_expanded_dim)) +
          (static_cast<size_t>(channel) * static_cast<size_t>(encoder_detail::k_pre_expand_lanes));
      output_weight[weight_index] = 1.0f;
    }

    append_tensor_with_data(model, k_pre_specs[0], channel_bias);
    append_tensor_with_data(model, k_pre_specs[1], feature_depthwise);
    append_tensor_with_data(model, k_pre_specs[2], channel_bias);
    append_tensor_with_data(model, k_pre_specs[3], channel_depthwise);
    append_tensor_with_data(model, k_pre_specs[4], channel_bias);
    append_tensor_with_data(model, k_pre_specs[5], channel_pointwise);
    append_tensor_with_data(model, k_pre_specs[6], channel_bias);
    append_tensor_with_data(model, k_pre_specs[7], channel_depthwise);
    append_tensor_with_data(model, k_pre_specs[8], channel_bias);
    append_tensor_with_data(model, k_pre_specs[9], channel_pointwise);
    append_tensor_with_data(model, k_pre_specs[10], output_bias);
    append_tensor_with_data(model, k_pre_specs[11], output_weight);

    for (int32_t layer = 0; layer < encoder_detail::k_layer_count; ++layer) {
      for (const auto & spec : k_layer_specs) {
        layer_storage.push_back(make_layer_values(spec));
        append_layer_tensor_with_data(model, layer, spec, layer_storage.back());
      }
    }
  }

  std::vector<float> make_layer_values(const tensor_spec & spec) const {
    std::vector<float> values(tensor_value_count(spec), 0.0f);
    if ((spec.name == "nff1.w") || (spec.name == "nff2.w") ||
        (spec.name == "nsa.w") || (spec.name == "nc.w") ||
        (spec.name == "no.w") || (spec.name == "conv.bn.sc") ||
        (spec.name == "conv.bn.rv")) {
      std::fill(values.begin(), values.end(), 1.0f);
    }
    return values;
  }

  void append_layer_tensor_with_data(emel::model::data & model,
                                     const int32_t layer,
                                     const tensor_spec & spec,
                                     std::span<const float> data) {
    std::array<char, 64> name = {};
    const int written = std::snprintf(name.data(),
                                      name.size(),
                                      "enc.l%d.%.*s",
                                      layer,
                                      static_cast<int>(spec.name.size()),
                                      spec.name.data());
    REQUIRE(written > 0);
    tensor_spec named_spec = spec;
    named_spec.name = std::string_view{name.data(), static_cast<size_t>(written)};
    append_tensor_with_data(model, named_spec, data);
  }
};

}  // namespace

TEST_CASE("sortformer encoder binds maintained tensor contract") {
  auto model = make_encoder_model(true, true);

  encoder_detail::contract contract = {};
  REQUIRE(encoder_detail::bind_contract(*model, contract));
  CHECK(contract.tensor_count ==
        static_cast<uint32_t>(encoder_detail::k_pre_tensor_count +
                              (encoder_detail::k_layer_count *
                               encoder_detail::k_layer_tensor_count)));
  CHECK(contract.pre.front().name == "enc.pre.conv.0.b");
  CHECK(contract.layers[0][0].name == "enc.l0.conv.bn.b");
  CHECK(contract.layers[16][encoder_detail::k_layer_tensor_count - 1u].name ==
        "enc.l16.att.pbv");
}

TEST_CASE("sortformer encoder rejects missing maintained tensor") {
  auto model = make_encoder_model(false, true);

  encoder_detail::contract contract = {};
  CHECK_FALSE(encoder_detail::bind_contract(*model, contract));
}

TEST_CASE("sortformer encoder rejects maintained tensor shape drift") {
  auto model = make_encoder_model(true, false);

  encoder_detail::contract contract = {};
  CHECK_FALSE(encoder_detail::bind_contract(*model, contract));
}

TEST_CASE("sortformer encoder kernels are deterministic") {
  std::array<float, encoder_detail::k_model_dim> input = {};
  std::array<float, encoder_detail::k_model_dim> scale = {};
  std::array<float, encoder_detail::k_model_dim> bias = {};
  std::array<float, encoder_detail::k_model_dim> output = {};

  for (size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<float>(index % 11u) * 0.25f;
    scale[index] = 1.0f + (static_cast<float>(index % 5u) * 0.125f);
    bias[index] = static_cast<float>(index % 7u) * -0.03125f;
  }

  REQUIRE(encoder_detail::compute_affine_512(input, scale, bias, output));
  CHECK(output[0] == doctest::Approx(0.0f));
  CHECK(output[10] == doctest::Approx((2.5f * 1.0f) + (-0.09375f)));

  const std::array<float, 3> dense_input{1.0f, -2.0f, 0.5f};
  const std::array<float, 6> weights{1.0f, 2.0f, 3.0f, -1.0f, 0.25f, 0.5f};
  const std::array<float, 2> dense_bias{0.25f, -0.75f};
  std::array<float, 2> dense_output{};

  REQUIRE(sortformer_detail::compute_dense(dense_input, weights, dense_bias, dense_output));
  CHECK(dense_output[0] == doctest::Approx(-1.25f));
  CHECK(dense_output[1] == doctest::Approx(-2.0f));
}

TEST_CASE("sortformer encoder dense kernel rejects invalid shapes") {
  const std::array<float, 2> input{1.0f, 2.0f};
  const std::array<float, 3> weights{1.0f, 2.0f, 3.0f};
  const std::array<float, 2> bias{0.0f, 0.0f};
  std::array<float, 2> output{};

  CHECK_FALSE(sortformer_detail::compute_dense(input, weights, bias, output));
}

TEST_CASE("sortformer dense batch helpers cover transposed prepared residual paths") {
  constexpr size_t row_count = 3u;
  constexpr size_t input_dim = 4u;
  constexpr size_t output_dim = 5u;

  std::array<float, row_count * input_dim> input_rows{};
  std::array<float, input_dim * output_dim> weights{};
  std::array<float, output_dim> bias{};
  std::array<float, row_count * output_dim> residual{};
  std::array<float, input_dim * row_count> transposed_input{};
  std::array<float, output_dim * row_count> transposed_output{};
  std::array<float, row_count * output_dim> prepared_output{};
  std::array<float, row_count * output_dim> fused_output{};
  sortformer_detail::dense_weight_cache cache{};
  cache.lhs_4row.resize(static_cast<size_t>(
      emel::kernel::aarch64::detail::prepared_f32_lhs_4row_value_count(input_dim, output_dim)));

  for (size_t index = 0u; index < input_rows.size(); ++index) {
    input_rows[index] = static_cast<float>(static_cast<int>(index % 9u) - 4) * 0.125f;
  }
  for (size_t index = 0u; index < weights.size(); ++index) {
    weights[index] = static_cast<float>(static_cast<int>((index * 3u) % 11u) - 5) * 0.0625f;
  }
  for (size_t index = 0u; index < bias.size(); ++index) {
    bias[index] = static_cast<float>(static_cast<int>(index) - 2) * 0.03125f;
  }
  for (size_t index = 0u; index < residual.size(); ++index) {
    residual[index] = static_cast<float>(static_cast<int>((index * 5u) % 13u) - 6) * 0.015625f;
  }

  REQUIRE(sortformer_detail::prepare_dense_weight_cache(weights, input_dim, output_dim, cache));
  CHECK(sortformer_detail::prepare_dense_weight_cache(weights, input_dim, output_dim, cache));
  REQUIRE(sortformer_detail::compute_dense_batch_prepared(input_rows,
                                                          row_count,
                                                          input_dim,
                                                          weights,
                                                          cache,
                                                          bias,
                                                          output_dim,
                                                          transposed_input,
                                                          transposed_output,
                                                          prepared_output));
  REQUIRE(sortformer_detail::compute_dense_batch_residual_prepared(input_rows,
                                                                   row_count,
                                                                   input_dim,
                                                                   weights,
                                                                   cache,
                                                                   bias,
                                                                   output_dim,
                                                                   residual,
                                                                   transposed_input,
                                                                   transposed_output,
                                                                   fused_output));
  for (size_t index = 0u; index < fused_output.size(); ++index) {
    CHECK(fused_output[index] == doctest::Approx(prepared_output[index] + residual[index]));
  }

  REQUIRE(sortformer_detail::transpose_dense_input(input_rows,
                                                   row_count,
                                                   input_dim,
                                                   transposed_input));
  REQUIRE(sortformer_detail::compute_dense_batch_from_transposed_scaled_residual_prepared(
      transposed_input,
      row_count,
      input_dim,
      weights,
      cache,
      bias,
      output_dim,
      0.5f,
      residual,
      transposed_output,
      fused_output));
  for (size_t index = 0u; index < fused_output.size(); ++index) {
    CHECK(fused_output[index] == doctest::Approx(residual[index] + (prepared_output[index] * 0.5f)));
  }

  CHECK_FALSE(sortformer_detail::prepare_dense_weight_cache(
      weights,
      0u,
      output_dim,
      cache));
  CHECK_FALSE(sortformer_detail::compute_dense_batch_from_transposed_scaled_residual_prepared(
      transposed_input,
      row_count,
      input_dim,
      weights,
      cache,
      bias,
      output_dim,
      0.5f,
      std::span<const float>{residual.data(), residual.size() - 1u},
      transposed_output,
      fused_output));
}

TEST_CASE("sortformer dense batch helpers cover unprepared transposed variants") {
  constexpr size_t row_count = 4u;
  constexpr size_t input_dim = 3u;
  constexpr size_t output_dim = 4u;

  std::array<float, row_count * input_dim> input_rows{};
  std::array<float, input_dim * output_dim> weights{};
  std::array<float, output_dim> bias{};
  std::array<float, input_dim * row_count> transposed_input{};
  std::array<float, output_dim * row_count> transposed_output{};
  std::array<float, output_dim * row_count> prepared_transposed_output{};
  std::array<float, output_dim * row_count> from_transposed_scratch{};
  std::array<float, output_dim * row_count> prepared_from_transposed_scratch{};
  std::array<float, output_dim * row_count> without_bias_scratch{};
  std::array<float, output_dim * row_count> without_bias_prepared_scratch{};
  std::array<float, row_count * output_dim> batch_output{};
  std::array<float, row_count * output_dim> from_transposed_output{};
  std::array<float, row_count * output_dim> prepared_from_transposed_output{};
  std::array<float, row_count * output_dim> without_bias_output{};
  std::array<float, row_count * output_dim> without_bias_prepared_output{};
  std::array<float, output_dim> first_row_without_bias{};
  sortformer_detail::dense_weight_cache cache{};
  cache.lhs_4row.resize(static_cast<size_t>(
      emel::kernel::aarch64::detail::prepared_f32_lhs_4row_value_count(input_dim, output_dim)));

  for (size_t index = 0u; index < input_rows.size(); ++index) {
    input_rows[index] = static_cast<float>(static_cast<int>((index * 5u) % 17u) - 8) * 0.125f;
  }
  for (size_t index = 0u; index < weights.size(); ++index) {
    weights[index] = static_cast<float>(static_cast<int>((index * 7u) % 19u) - 9) * 0.0625f;
  }
  for (size_t index = 0u; index < bias.size(); ++index) {
    bias[index] = static_cast<float>(static_cast<int>(index) - 1) * 0.25f;
  }

  REQUIRE(sortformer_detail::prepare_dense_weight_cache(weights, input_dim, output_dim, cache));
  REQUIRE(sortformer_detail::compute_dense_batch(input_rows,
                                                 row_count,
                                                 input_dim,
                                                 weights,
                                                 bias,
                                                 output_dim,
                                                 transposed_input,
                                                 transposed_output,
                                                 batch_output));
  REQUIRE(sortformer_detail::transpose_dense_input(input_rows,
                                                   row_count,
                                                   input_dim,
                                                   transposed_input));
  REQUIRE(sortformer_detail::compute_dense_batch_to_transposed(input_rows,
                                                               row_count,
                                                               input_dim,
                                                               weights,
                                                               bias,
                                                               output_dim,
                                                               transposed_input,
                                                               transposed_output));
  REQUIRE(sortformer_detail::compute_dense_batch_to_transposed_prepared(input_rows,
                                                                        row_count,
                                                                        input_dim,
                                                                        weights,
                                                                        cache,
                                                                        bias,
                                                                        output_dim,
                                                                        transposed_input,
                                                                        prepared_transposed_output));
  REQUIRE(sortformer_detail::compute_dense_batch_from_transposed(transposed_input,
                                                                 row_count,
                                                                 input_dim,
                                                                 weights,
                                                                 bias,
                                                                 output_dim,
                                                                 from_transposed_scratch,
                                                                 from_transposed_output));
  REQUIRE(sortformer_detail::compute_dense_batch_from_transposed_prepared(
      transposed_input,
      row_count,
      input_dim,
      weights,
      cache,
      bias,
      output_dim,
      prepared_from_transposed_scratch,
      prepared_from_transposed_output));
  REQUIRE(sortformer_detail::compute_dense_batch_without_bias(input_rows,
                                                             row_count,
                                                             input_dim,
                                                             weights,
                                                             output_dim,
                                                             transposed_input,
                                                             without_bias_scratch,
                                                             without_bias_output));
  REQUIRE(sortformer_detail::compute_dense_batch_without_bias_prepared(input_rows,
                                                                      row_count,
                                                                      input_dim,
                                                                      weights,
                                                                      cache,
                                                                      output_dim,
                                                                      transposed_input,
                                                                      without_bias_prepared_scratch,
                                                                      without_bias_prepared_output));
  REQUIRE(sortformer_detail::compute_dense_without_bias(
      std::span<const float>{input_rows.data(), input_dim},
      weights,
      first_row_without_bias));

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      const size_t row_major = (row * output_dim) + output_index;
      const size_t transposed = (output_index * row_count) + row;
      CHECK(transposed_output[transposed] == doctest::Approx(batch_output[row_major]));
      CHECK(prepared_transposed_output[transposed] == doctest::Approx(transposed_output[transposed]));
      CHECK(from_transposed_output[row_major] == doctest::Approx(batch_output[row_major]));
      CHECK(prepared_from_transposed_output[row_major] == doctest::Approx(batch_output[row_major]));
      CHECK(without_bias_prepared_output[row_major] ==
            doctest::Approx(without_bias_output[row_major]));
    }
  }
  for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
    CHECK(first_row_without_bias[output_index] ==
          doctest::Approx(without_bias_output[output_index]));
  }

  CHECK_FALSE(sortformer_detail::transpose_dense_input(input_rows,
                                                       0u,
                                                       input_dim,
                                                       transposed_input));
  CHECK_FALSE(sortformer_detail::compute_dense_without_bias(
      std::span<const float>{},
      weights,
      first_row_without_bias));
  CHECK_FALSE(sortformer_detail::compute_dense_batch_without_bias(input_rows,
                                                                 row_count,
                                                                 input_dim,
                                                                 weights,
                                                                 output_dim,
                                                                 transposed_input,
                                                                 transposed_output,
                                                                 std::span<float>{
                                                                     without_bias_output.data(),
                                                                     without_bias_output.size() - 1u}));
  CHECK_FALSE(sortformer_detail::compute_dense_batch_to_transposed(input_rows,
                                                                  row_count,
                                                                  input_dim,
                                                                  weights,
                                                                  bias,
                                                                  output_dim,
                                                                  transposed_input,
                                                                  std::span<float>{
                                                                      transposed_output.data(),
                                                                      transposed_output.size() - 1u}));
  CHECK_FALSE(sortformer_detail::compute_dense_batch_from_transposed(transposed_input,
                                                                    row_count,
                                                                    input_dim,
                                                                    weights,
                                                                    bias,
                                                                    output_dim,
                                                                    transposed_output,
                                                                    std::span<float>{
                                                                        batch_output.data(),
                                                                        batch_output.size() - 1u}));
}

TEST_CASE("sortformer encoder derives maintained frames from acoustic features") {
  auto fixture = std::make_unique<pre_encoder_fixture>();
  encoder_detail::contract contract = {};
  REQUIRE(encoder_detail::bind_contract(fixture->model, contract));

  std::vector<float> features(static_cast<size_t>(
      encoder_detail::k_required_feature_value_count));
  for (size_t index = 0u; index < features.size(); ++index) {
    features[index] = static_cast<float>(index % 37u) * 0.015625f;
  }

  encoder_detail::pre_encoder_workspace workspace = {};
  std::vector<float> encoder_frames(static_cast<size_t>(
      encoder_detail::k_required_encoder_value_count));

  REQUIRE(encoder_detail::compute_encoder_frames_from_features(features,
                                                               contract,
                                                               workspace,
                                                               encoder_frames));
  CHECK(std::all_of(encoder_frames.begin(), encoder_frames.end(), [](const float value) {
    return std::isfinite(value);
  }));
}

TEST_CASE("sortformer encoder rejects invalid feature-to-frame shapes") {
  auto fixture = std::make_unique<pre_encoder_fixture>();
  encoder_detail::contract contract = {};
  REQUIRE(encoder_detail::bind_contract(fixture->model, contract));

  std::vector<float> features(static_cast<size_t>(
      encoder_detail::k_required_feature_value_count - 1));
  std::vector<float> encoder_frames(static_cast<size_t>(
      encoder_detail::k_required_encoder_value_count));
  encoder_detail::pre_encoder_workspace workspace = {};

  CHECK_FALSE(encoder_detail::compute_encoder_frames_from_features(features,
                                                                   contract,
                                                                   workspace,
                                                                   encoder_frames));

  features.resize(static_cast<size_t>(encoder_detail::k_required_feature_value_count));
  encoder_frames.resize(static_cast<size_t>(encoder_detail::k_required_encoder_value_count - 1));
  CHECK_FALSE(encoder_detail::compute_encoder_frames_from_features(features,
                                                                   contract,
                                                                   workspace,
                                                                   encoder_frames));
}

TEST_CASE("sortformer encoder rejects unbound feature-to-frame contract") {
  std::vector<float> features(static_cast<size_t>(
      encoder_detail::k_required_feature_value_count), 0.25f);
  std::vector<float> encoder_frames(static_cast<size_t>(
      encoder_detail::k_required_encoder_value_count));
  encoder_detail::pre_encoder_workspace workspace = {};
  encoder_detail::contract contract = {};

  CHECK_FALSE(encoder_detail::compute_encoder_frames_from_features(features,
                                                                   contract,
                                                                   workspace,
                                                                   encoder_frames));
}
