#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <span>
#include <string_view>

#include "doctest/doctest.h"

#include "emel/diarization/sortformer/transformer/detail.hpp"
#include "emel/model/data.hpp"

namespace {

namespace transformer_detail = emel::diarization::sortformer::transformer::detail;

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, transformer_detail::k_layer_tensor_count> k_layer_specs{{
    {"sa.k.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.k.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"sa.o.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.o.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"sa.q.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.q.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"sa.v.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"sa.v.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_hidden_dim, 0, 0}},
    {"ln1.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ln1.w", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ln2.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ln2.w", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ff.di.b", 1, {transformer_detail::k_inner_dim, 0, 0, 0}},
    {"ff.di.w", 2, {transformer_detail::k_hidden_dim, transformer_detail::k_inner_dim, 0, 0}},
    {"ff.do.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
    {"ff.do.w", 2, {transformer_detail::k_inner_dim, transformer_detail::k_hidden_dim, 0, 0}},
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

void append_layer_tensor(emel::model::data & model,
                         const int32_t layer,
                         const tensor_spec & spec) {
  std::array<char, 64> name = {};
  const int written = std::snprintf(name.data(),
                                    name.size(),
                                    "te.l%d.%.*s",
                                    layer,
                                    static_cast<int>(spec.name.size()),
                                    spec.name.data());
  REQUIRE(written > 0);
  tensor_spec named_spec = spec;
  named_spec.name = std::string_view{name.data(), static_cast<size_t>(written)};
  append_tensor(model, named_spec);
}

void build_transformer_model(emel::model::data & model,
                             const bool include_all_tensors,
                             const bool valid_shapes) {
  std::memset(&model, 0, sizeof(model));

  for (int32_t layer = 0; layer < transformer_detail::k_layer_count; ++layer) {
    for (size_t index = 0u; index < k_layer_specs.size(); ++index) {
      if (!include_all_tensors && layer == 17 && index == k_layer_specs.size() - 1u) {
        continue;
      }

      tensor_spec spec = k_layer_specs[index];
      if (!valid_shapes && layer == 0 && spec.name == "ff.di.w") {
        spec.dims = {transformer_detail::k_hidden_dim,
                     transformer_detail::k_inner_dim - 1,
                     0,
                     0};
      }
      append_layer_tensor(model, layer, spec);
    }
  }
}

void fill_identity(std::span<float, transformer_detail::k_hidden_dim *
                                       transformer_detail::k_hidden_dim> weights) {
  std::fill(weights.begin(), weights.end(), 0.0f);
  for (size_t index = 0u; index < static_cast<size_t>(transformer_detail::k_hidden_dim);
       ++index) {
    weights[(index * static_cast<size_t>(transformer_detail::k_hidden_dim)) + index] = 1.0f;
  }
}

void prepare_test_weight_caches(
    std::span<const float, transformer_detail::k_hidden_dim *
                               transformer_detail::k_hidden_dim> projection,
    std::span<const float, transformer_detail::k_inner_dim *
                               transformer_detail::k_hidden_dim> feed_forward_in_weight,
    std::span<const float, transformer_detail::k_hidden_dim *
                               transformer_detail::k_inner_dim> feed_forward_out_weight,
    transformer_detail::layer_workspace & workspace) {
  namespace sortformer_detail = emel::diarization::sortformer::detail;
  REQUIRE(sortformer_detail::prepare_dense_weight_cache(
      projection,
      static_cast<size_t>(transformer_detail::k_hidden_dim),
      static_cast<size_t>(transformer_detail::k_hidden_dim),
      workspace.attention_weight_caches[0]));
  REQUIRE(sortformer_detail::prepare_dense_weight_cache(
      feed_forward_in_weight,
      static_cast<size_t>(transformer_detail::k_hidden_dim),
      static_cast<size_t>(transformer_detail::k_inner_dim),
      workspace.feed_forward_weight_caches[0]));
  REQUIRE(sortformer_detail::prepare_dense_weight_cache(
      feed_forward_out_weight,
      static_cast<size_t>(transformer_detail::k_inner_dim),
      static_cast<size_t>(transformer_detail::k_hidden_dim),
      workspace.feed_forward_weight_caches[1]));
}

}  // namespace

TEST_CASE("sortformer transformer binds maintained tensor contract") {
  emel::model::data model = {};
  build_transformer_model(model, true, true);

  transformer_detail::contract contract = {};
  REQUIRE(transformer_detail::bind_contract(model, contract));
  CHECK(contract.tensor_count == static_cast<uint32_t>(
                                     transformer_detail::k_layer_count *
                                     transformer_detail::k_layer_tensor_count));
  CHECK(contract.layers[0].key_bias.name == "te.l0.sa.k.b");
  CHECK(contract.layers[17].feed_forward_out_weight.name == "te.l17.ff.do.w");
}

TEST_CASE("sortformer transformer rejects missing maintained tensor") {
  emel::model::data model = {};
  build_transformer_model(model, false, true);

  transformer_detail::contract contract = {};
  CHECK_FALSE(transformer_detail::bind_contract(model, contract));
}

TEST_CASE("sortformer transformer rejects maintained shape drift") {
  emel::model::data model = {};
  build_transformer_model(model, true, false);

  transformer_detail::contract contract = {};
  CHECK_FALSE(transformer_detail::bind_contract(model, contract));
}

TEST_CASE("sortformer transformer layer norm is deterministic") {
  std::array<float, transformer_detail::k_hidden_dim> input = {};
  std::array<float, transformer_detail::k_hidden_dim> scale = {};
  std::array<float, transformer_detail::k_hidden_dim> bias = {};
  std::array<float, transformer_detail::k_hidden_dim> output = {};

  for (size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<float>(index % 9u) - 4.0f;
    scale[index] = 1.0f;
    bias[index] = 0.0f;
  }

  REQUIRE(transformer_detail::compute_layer_norm(input, scale, bias, output));

  float mean = 0.0f;
  for (const float value : output) {
    mean += value;
  }
  mean /= static_cast<float>(output.size());
  CHECK(mean == doctest::Approx(0.0f).epsilon(0.0001f));
}

TEST_CASE("sortformer transformer native layer kernel is deterministic") {
  constexpr uint32_t k_frame_count = 2;
  std::array<float, k_frame_count * transformer_detail::k_hidden_dim> input = {};
  std::array<float, k_frame_count * transformer_detail::k_hidden_dim> output_a = {};
  std::array<float, k_frame_count * transformer_detail::k_hidden_dim> output_b = {};
  std::array<float, transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>
      projection = {};
  std::array<float, transformer_detail::k_hidden_dim> hidden_bias = {};
  std::array<float, transformer_detail::k_hidden_dim> norm_weight = {};
  std::array<float, transformer_detail::k_hidden_dim> norm_bias = {};
  std::array<float, transformer_detail::k_inner_dim * transformer_detail::k_hidden_dim>
      feed_forward_in_weight = {};
  std::array<float, transformer_detail::k_inner_dim> feed_forward_in_bias = {};
  std::array<float, transformer_detail::k_hidden_dim * transformer_detail::k_inner_dim>
      feed_forward_out_weight = {};
  std::array<float, transformer_detail::k_hidden_dim> feed_forward_out_bias = {};
  transformer_detail::layer_workspace workspace_a = {};
  transformer_detail::layer_workspace workspace_b = {};

  fill_identity(projection);
  prepare_test_weight_caches(
      projection, feed_forward_in_weight, feed_forward_out_weight, workspace_a);
  prepare_test_weight_caches(
      projection, feed_forward_in_weight, feed_forward_out_weight, workspace_b);
  norm_weight.fill(1.0f);
  for (size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<float>(index % 17u) * 0.0625f;
  }

  REQUIRE(transformer_detail::compute_transformer_layer(input,
                                                        k_frame_count,
                                                        projection,
                                                        hidden_bias,
                                                        projection,
                                                        hidden_bias,
                                                        projection,
                                                        hidden_bias,
                                                        projection,
                                                        hidden_bias,
                                                        norm_weight,
                                                        norm_bias,
                                                        feed_forward_in_weight,
                                                        feed_forward_in_bias,
                                                        feed_forward_out_weight,
                                                        feed_forward_out_bias,
                                                        workspace_a.attention_weight_caches[0],
                                                        workspace_a.attention_weight_caches[0],
                                                        workspace_a.attention_weight_caches[0],
                                                        workspace_a.attention_weight_caches[0],
                                                        workspace_a.feed_forward_weight_caches[0],
                                                        workspace_a.feed_forward_weight_caches[1],
                                                        norm_weight,
                                                        norm_bias,
                                                        workspace_a,
                                                        output_a));
  REQUIRE(transformer_detail::compute_transformer_layer(input,
                                                        k_frame_count,
                                                        projection,
                                                        hidden_bias,
                                                        projection,
                                                        hidden_bias,
                                                        projection,
                                                        hidden_bias,
                                                        projection,
                                                        hidden_bias,
                                                        norm_weight,
                                                        norm_bias,
                                                        feed_forward_in_weight,
                                                        feed_forward_in_bias,
                                                        feed_forward_out_weight,
                                                        feed_forward_out_bias,
                                                        workspace_b.attention_weight_caches[0],
                                                        workspace_b.attention_weight_caches[0],
                                                        workspace_b.attention_weight_caches[0],
                                                        workspace_b.attention_weight_caches[0],
                                                        workspace_b.feed_forward_weight_caches[0],
                                                        workspace_b.feed_forward_weight_caches[1],
                                                        norm_weight,
                                                        norm_bias,
                                                        workspace_b,
                                                        output_b));

  CHECK(output_a == output_b);
  CHECK(output_a[0] != doctest::Approx(0.0f));
}

TEST_CASE("sortformer transformer native layer rejects invalid frame shapes") {
  std::array<float, transformer_detail::k_hidden_dim> input = {};
  std::array<float, transformer_detail::k_hidden_dim> output = {};
  std::array<float, transformer_detail::k_hidden_dim * transformer_detail::k_hidden_dim>
      projection = {};
  std::array<float, transformer_detail::k_hidden_dim> hidden_bias = {};
  std::array<float, transformer_detail::k_hidden_dim> norm_weight = {};
  std::array<float, transformer_detail::k_hidden_dim> norm_bias = {};
  std::array<float, transformer_detail::k_inner_dim * transformer_detail::k_hidden_dim>
      feed_forward_in_weight = {};
  std::array<float, transformer_detail::k_inner_dim> feed_forward_in_bias = {};
  std::array<float, transformer_detail::k_hidden_dim * transformer_detail::k_inner_dim>
      feed_forward_out_weight = {};
  std::array<float, transformer_detail::k_hidden_dim> feed_forward_out_bias = {};
  transformer_detail::layer_workspace workspace = {};

  CHECK_FALSE(transformer_detail::compute_transformer_layer(input,
                                                            0,
                                                            projection,
                                                            hidden_bias,
                                                            projection,
                                                            hidden_bias,
                                                            projection,
                                                            hidden_bias,
                                                            projection,
                                                            hidden_bias,
                                                            norm_weight,
                                                            norm_bias,
                                                            feed_forward_in_weight,
                                                            feed_forward_in_bias,
                                                            feed_forward_out_weight,
                                                            feed_forward_out_bias,
                                                            workspace.attention_weight_caches[0],
                                                            workspace.attention_weight_caches[0],
                                                            workspace.attention_weight_caches[0],
                                                            workspace.attention_weight_caches[0],
                                                            workspace.feed_forward_weight_caches[0],
                                                            workspace.feed_forward_weight_caches[1],
                                                            norm_weight,
                                                            norm_bias,
                                                            workspace,
                                                            output));
}
