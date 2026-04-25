#include <array>
#include <cstring>
#include <span>
#include <string_view>

#include "doctest/doctest.h"

#include "emel/diarization/sortformer/cache/detail.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/model/data.hpp"

namespace {

namespace cache_detail = emel::diarization::sortformer::cache::detail;
namespace modules_detail = emel::diarization::sortformer::modules::detail;

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, modules_detail::k_tensor_count> k_specs{{
    {"mods.ep.b", 1, {modules_detail::k_hidden_dim, 0, 0, 0}},
    {"mods.ep.w", 2, {modules_detail::k_encoder_dim, modules_detail::k_hidden_dim, 0, 0}},
    {"mods.fh2h.b", 1, {modules_detail::k_hidden_dim, 0, 0, 0}},
    {"mods.fh2h.w", 2, {modules_detail::k_hidden_dim, modules_detail::k_hidden_dim, 0, 0}},
    {"mods.h2s.b", 1, {modules_detail::k_speaker_count, 0, 0, 0}},
    {"mods.h2s.w", 2, {modules_detail::k_pair_hidden_dim, modules_detail::k_speaker_count, 0, 0}},
    {"mods.sh2s.b", 1, {modules_detail::k_speaker_count, 0, 0, 0}},
    {"mods.sh2s.w", 2, {modules_detail::k_hidden_dim, modules_detail::k_speaker_count, 0, 0}},
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

void build_modules_model(emel::model::data & model,
                         const bool include_all_tensors,
                         const bool valid_shapes) {
  std::memset(&model, 0, sizeof(model));
  for (size_t index = 0u; index < k_specs.size(); ++index) {
    if (!include_all_tensors && index == k_specs.size() - 1u) {
      continue;
    }

    tensor_spec spec = k_specs[index];
    if (!valid_shapes && spec.name == "mods.ep.w") {
      spec.dims = {modules_detail::k_encoder_dim, modules_detail::k_hidden_dim - 1, 0, 0};
    }
    append_tensor(model, spec);
  }
}

}  // namespace

TEST_CASE("sortformer modules bind maintained tensor contract") {
  emel::model::data model = {};
  build_modules_model(model, true, true);

  modules_detail::contract contract = {};
  REQUIRE(modules_detail::bind_contract(model, contract));
  CHECK(contract.tensor_count == static_cast<uint32_t>(modules_detail::k_tensor_count));
  CHECK(contract.encoder_projection_weight.name == "mods.ep.w");
  CHECK(contract.hidden_to_speaker_weight.name == "mods.h2s.w");
  CHECK(contract.speaker_hidden_to_speaker_weight.name == "mods.sh2s.w");
}

TEST_CASE("sortformer modules reject missing maintained tensor") {
  emel::model::data model = {};
  build_modules_model(model, false, true);

  modules_detail::contract contract = {};
  CHECK_FALSE(modules_detail::bind_contract(model, contract));
}

TEST_CASE("sortformer modules reject maintained shape drift") {
  emel::model::data model = {};
  build_modules_model(model, true, false);

  modules_detail::contract contract = {};
  CHECK_FALSE(modules_detail::bind_contract(model, contract));
}

TEST_CASE("sortformer speaker cache reset write and read are deterministic") {
  cache_detail::state cache = {};
  std::array<float, cache_detail::k_hidden_dim> hidden = {};
  std::array<float, cache_detail::k_hidden_dim> readback = {};

  for (size_t index = 0u; index < hidden.size(); ++index) {
    hidden[index] = static_cast<float>(index % 13u) * 0.125f;
  }

  cache_detail::reset(cache);
  CHECK(cache.frame_count == 0);
  CHECK_FALSE(cache_detail::read_frame(cache, 0, readback));
  REQUIRE(cache_detail::write_frame(cache, 3, hidden));
  CHECK(cache.frame_count == 4);
  REQUIRE(cache_detail::read_frame(cache, 3, readback));
  CHECK(readback == hidden);
  CHECK_FALSE(cache_detail::write_frame(cache, cache_detail::k_cache_len, hidden));
}

TEST_CASE("sortformer modules compute projection and speaker logits") {
  std::array<float, modules_detail::k_encoder_dim> encoder_frame = {};
  std::array<float, modules_detail::k_hidden_dim * modules_detail::k_encoder_dim> weights = {};
  std::array<float, modules_detail::k_hidden_dim> bias = {};
  std::array<float, modules_detail::k_hidden_dim> hidden = {};

  encoder_frame[0] = 2.0f;
  encoder_frame[1] = -1.0f;
  weights[0] = 0.5f;
  weights[1] = 0.25f;
  bias[0] = 0.125f;

  REQUIRE(modules_detail::compute_encoder_projection(encoder_frame, weights, bias, hidden));
  CHECK(hidden[0] == doctest::Approx(0.875f));

  std::array<float, modules_detail::k_hidden_dim> cached_hidden = {};
  std::array<float, modules_detail::k_speaker_count * modules_detail::k_pair_hidden_dim>
      speaker_weights = {};
  std::array<float, modules_detail::k_speaker_count> speaker_bias = {};
  std::array<float, modules_detail::k_speaker_count> logits = {};

  cached_hidden[0] = 0.5f;
  speaker_weights[0] = 2.0f;
  speaker_weights[modules_detail::k_hidden_dim] = -1.0f;
  speaker_bias[0] = 0.25f;

  REQUIRE(modules_detail::compute_speaker_logits(hidden,
                                                 cached_hidden,
                                                 speaker_weights,
                                                 speaker_bias,
                                                 logits));
  CHECK(logits[0] == doctest::Approx(1.5f));
}
