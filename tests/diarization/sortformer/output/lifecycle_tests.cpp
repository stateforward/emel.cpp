#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/output/detail.hpp"
#include "emel/model/data.hpp"

namespace {

namespace modules_detail = emel::diarization::sortformer::modules::detail;
namespace output_detail = emel::diarization::sortformer::output::detail;

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

struct modules_fixture {
  emel::model::data model = {};
  std::vector<float> hidden_zero =
      std::vector<float>(static_cast<size_t>(modules_detail::k_hidden_dim), 0.0f);
  std::vector<float> encoder_projection =
      std::vector<float>(static_cast<size_t>(modules_detail::k_hidden_dim *
                                             modules_detail::k_encoder_dim),
                         0.0f);
  std::vector<float> frame_hidden =
      std::vector<float>(static_cast<size_t>(modules_detail::k_hidden_dim *
                                             modules_detail::k_hidden_dim),
                         0.0f);
  std::vector<float> speaker_bias =
      std::vector<float>(static_cast<size_t>(modules_detail::k_speaker_count), 0.0f);
  std::vector<float> speaker_pair =
      std::vector<float>(static_cast<size_t>(modules_detail::k_speaker_count *
                                             modules_detail::k_pair_hidden_dim),
                         0.0f);
  std::vector<float> speaker_hidden =
      std::vector<float>(static_cast<size_t>(modules_detail::k_speaker_count *
                                             modules_detail::k_hidden_dim),
                         0.0f);

  modules_fixture() {
    std::memset(&model, 0, sizeof(model));
    for (int32_t index = 0; index < modules_detail::k_hidden_dim; ++index) {
      frame_hidden[(static_cast<size_t>(index) * static_cast<size_t>(modules_detail::k_hidden_dim)) +
                   static_cast<size_t>(index)] = 1.0f;
    }
    speaker_hidden[modules_detail::k_hidden_dim + 1u] = 1.0f;
    speaker_hidden[(2u * static_cast<size_t>(modules_detail::k_hidden_dim)) + 2u] = -1.0f;
    speaker_hidden[(3u * static_cast<size_t>(modules_detail::k_hidden_dim)) + 3u] = 2.0f;

    for (const auto & spec : k_specs) {
      append_tensor(spec);
    }
  }

  void append_name(emel::model::data::tensor_record & tensor, const std::string_view name) {
    const auto offset = model.name_bytes_used;
    std::memcpy(model.name_storage.data() + offset, name.data(), name.size());
    tensor.name_offset = offset;
    tensor.name_length = static_cast<uint32_t>(name.size());
    model.name_bytes_used += static_cast<uint32_t>(name.size());
  }

  std::span<const float> data_for(const std::string_view name) const noexcept {
    if (name == "mods.ep.w") {
      return encoder_projection;
    }
    if (name == "mods.fh2h.w") {
      return frame_hidden;
    }
    if (name == "mods.h2s.w") {
      return speaker_pair;
    }
    if (name == "mods.h2s.b" || name == "mods.sh2s.b") {
      return speaker_bias;
    }
    if (name == "mods.sh2s.w") {
      return speaker_hidden;
    }
    return hidden_zero;
  }

  void append_tensor(const tensor_spec & spec) {
    auto & tensor = model.tensors[model.n_tensors];
    append_name(tensor, spec.name);
    tensor.n_dims = spec.n_dims;
    tensor.dims = spec.dims;
    const auto values = data_for(spec.name);
    tensor.data = values.data();
    tensor.data_size = values.size_bytes();
    ++model.n_tensors;
  }
};

}  // namespace

TEST_CASE("sortformer output computes deterministic speaker probabilities") {
  auto fixture = std::make_unique<modules_fixture>();
  modules_detail::contract contract = {};
  REQUIRE(modules_detail::bind_contract(fixture->model, contract));

  std::vector<float> hidden_frames(static_cast<size_t>(
      output_detail::k_required_hidden_value_count), 0.0f);
  hidden_frames[1] = 1.0f;
  hidden_frames[2] = 1.0f;
  hidden_frames[3] = 1.0f;
  std::vector<float> probabilities(static_cast<size_t>(
      output_detail::k_required_probability_value_count));

  REQUIRE(output_detail::compute_speaker_probabilities(hidden_frames,
                                                       contract,
                                                       probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.5f));
  CHECK(probabilities[1] == doctest::Approx(1.0f / (1.0f + std::exp(-1.0f))));
  CHECK(probabilities[2] == doctest::Approx(1.0f / (1.0f + std::exp(1.0f))));
  CHECK(probabilities[3] == doctest::Approx(1.0f / (1.0f + std::exp(-2.0f))));

  std::vector<float> second(probabilities.size());
  REQUIRE(output_detail::compute_speaker_probabilities(hidden_frames, contract, second));
  CHECK(second == probabilities);
}

TEST_CASE("sortformer output uses non-aliased frame-hidden projection") {
  auto fixture = std::make_unique<modules_fixture>();
  std::fill(fixture->frame_hidden.begin(), fixture->frame_hidden.end(), 0.0f);
  std::fill(fixture->speaker_hidden.begin(), fixture->speaker_hidden.end(), 0.0f);
  std::fill(fixture->speaker_bias.begin(), fixture->speaker_bias.end(), 0.0f);

  fixture->frame_hidden[0] = 1.0f;
  fixture->frame_hidden[1] = 1.0f;
  fixture->frame_hidden[static_cast<size_t>(modules_detail::k_hidden_dim)] = 1.0f;
  fixture->frame_hidden[static_cast<size_t>(modules_detail::k_hidden_dim) + 1u] = -1.0f;
  fixture->speaker_hidden[0] = 1.0f;
  fixture->speaker_hidden[static_cast<size_t>(modules_detail::k_hidden_dim) + 1u] = 1.0f;

  modules_detail::contract contract = {};
  REQUIRE(modules_detail::bind_contract(fixture->model, contract));

  std::vector<float> hidden_frames(static_cast<size_t>(
      output_detail::k_required_hidden_value_count), 0.0f);
  hidden_frames[0] = 0.25f;
  hidden_frames[1] = 0.5f;
  std::vector<float> probabilities(static_cast<size_t>(
      output_detail::k_required_probability_value_count), 0.0f);

  REQUIRE(output_detail::compute_speaker_probabilities(hidden_frames,
                                                       contract,
                                                       probabilities));
  CHECK(probabilities[0] == doctest::Approx(1.0f / (1.0f + std::exp(-0.75f))));
  CHECK(probabilities[1] == doctest::Approx(0.5f));
}

TEST_CASE("sortformer output rejects invalid probability inputs") {
  auto fixture = std::make_unique<modules_fixture>();
  modules_detail::contract contract = {};
  REQUIRE(modules_detail::bind_contract(fixture->model, contract));

  std::vector<float> hidden_frames(static_cast<size_t>(
      output_detail::k_required_hidden_value_count - 1), 0.0f);
  std::vector<float> probabilities(static_cast<size_t>(
      output_detail::k_required_probability_value_count));
  CHECK_FALSE(output_detail::compute_speaker_probabilities(hidden_frames,
                                                          contract,
                                                          probabilities));

  hidden_frames.resize(static_cast<size_t>(output_detail::k_required_hidden_value_count));
  probabilities.resize(static_cast<size_t>(output_detail::k_required_probability_value_count - 1));
  CHECK_FALSE(output_detail::compute_speaker_probabilities(hidden_frames,
                                                          contract,
                                                          probabilities));

  probabilities.resize(static_cast<size_t>(output_detail::k_required_probability_value_count));
  modules_detail::contract empty_contract = {};
  CHECK_FALSE(output_detail::compute_speaker_probabilities(hidden_frames,
                                                          empty_contract,
                                                          probabilities));
}

TEST_CASE("sortformer output decodes overlapping bounded segments") {
  std::vector<float> probabilities(static_cast<size_t>(
      output_detail::k_required_probability_value_count), 0.0f);
  for (int32_t frame = 1; frame < 4; ++frame) {
    probabilities[(static_cast<size_t>(frame) * output_detail::k_speaker_count) + 0u] = 0.75f;
  }
  for (int32_t frame = 2; frame < 6; ++frame) {
    probabilities[(static_cast<size_t>(frame) * output_detail::k_speaker_count) + 1u] = 0.875f;
  }

  std::array<output_detail::segment_record, 4> segments = {};
  int32_t segment_count = -1;
  REQUIRE(output_detail::decode_segments(probabilities,
                                         output_detail::k_default_activity_threshold,
                                         segments,
                                         segment_count));

  REQUIRE(segment_count == 2);
  CHECK(segments[0].speaker == 0);
  CHECK(segments[0].start_frame == 1);
  CHECK(segments[0].end_frame == 4);
  CHECK(segments[0].start_seconds == doctest::Approx(0.08f));
  CHECK(segments[0].end_seconds == doctest::Approx(0.32f));
  CHECK(segments[1].speaker == 1);
  CHECK(output_detail::speaker_label(segments[0].speaker) == "speaker_0");
  CHECK(output_detail::speaker_label(segments[1].speaker) == "speaker_1");
  CHECK(segments[1].start_frame == 2);
  CHECK(segments[1].end_frame == 6);
  CHECK(segments[1].start_seconds == doctest::Approx(0.16f));
  CHECK(segments[1].end_seconds == doctest::Approx(0.48f));
}

TEST_CASE("sortformer output rejects invalid segment inputs") {
  std::vector<float> probabilities(static_cast<size_t>(
      output_detail::k_required_probability_value_count), 0.0f);
  probabilities[0] = 0.75f;
  probabilities[output_detail::k_speaker_count] = 0.75f;
  probabilities[(2u * output_detail::k_speaker_count) + 1u] = 0.75f;

  std::array<output_detail::segment_record, 1> segments = {};
  int32_t segment_count = -1;
  CHECK_FALSE(output_detail::decode_segments(probabilities,
                                             output_detail::k_default_activity_threshold,
                                             segments,
                                             segment_count));
  CHECK(segment_count == 1);

  probabilities.resize(static_cast<size_t>(output_detail::k_required_probability_value_count - 1));
  CHECK_FALSE(output_detail::decode_segments(probabilities,
                                             output_detail::k_default_activity_threshold,
                                             segments,
                                             segment_count));

  probabilities.resize(static_cast<size_t>(output_detail::k_required_probability_value_count));
  CHECK_FALSE(output_detail::decode_segments(probabilities, -0.125f, segments, segment_count));
  CHECK_FALSE(output_detail::decode_segments(probabilities, 1.125f, segments, segment_count));
  CHECK(output_detail::speaker_label(-1).empty());
  CHECK(output_detail::speaker_label(output_detail::k_speaker_count).empty());
}
