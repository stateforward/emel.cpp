#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "emel/diarization/request/detail.hpp"
#include "emel/diarization/request/errors.hpp"
#include "emel/diarization/request/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace {

namespace diarization_request = emel::diarization::request;
namespace diarization_detail = emel::diarization::request::detail;
namespace feature_extractor_detail =
    emel::diarization::sortformer::encoder::feature_extractor::detail;

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

struct callback_probe {
  bool done_called = false;
  bool error_called = false;
  const diarization_request::event::prepare * request = nullptr;
  int32_t frame_count = 0;
  int32_t feature_bin_count = 0;
  emel::error::type err = emel::error::cast(diarization_request::error::none);

  void on_done(const diarization_request::events::prepare_done & ev) noexcept {
    done_called = true;
    request = ev.request;
    frame_count = ev.frame_count;
    feature_bin_count = ev.feature_bin_count;
  }

  void on_error(const diarization_request::events::prepare_error & ev) noexcept {
    error_called = true;
    request = ev.request;
    err = ev.err;
  }
};

constexpr std::array<tensor_spec, 2> k_feature_specs{{
    {"prep.feat.fb", 3, {feature_extractor_detail::k_fft_bin_count,
                         feature_extractor_detail::k_feature_bin_count,
                         1,
                         0}},
    {"prep.feat.win", 1, {feature_extractor_detail::k_window_length, 0, 0, 0}},
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

struct request_model_fixture {
  emel::model::data model = {};
  std::vector<float> filter_bank = std::vector<float>(
      static_cast<size_t>(feature_extractor_detail::k_fft_bin_count *
                          feature_extractor_detail::k_feature_bin_count),
      0.0f);
  std::vector<float> window =
      std::vector<float>(static_cast<size_t>(feature_extractor_detail::k_window_length), 1.0f);

  request_model_fixture() {
    std::memset(&model, 0, sizeof(model));

    for (int32_t mel = 0; mel < feature_extractor_detail::k_feature_bin_count; ++mel) {
      for (int32_t bin = 0; bin < feature_extractor_detail::k_fft_bin_count; ++bin) {
        const size_t offset =
            (static_cast<size_t>(mel) * static_cast<size_t>(feature_extractor_detail::k_fft_bin_count)) +
            static_cast<size_t>(bin);
        filter_bank[offset] =
            static_cast<float>(((mel + 3) * ((bin % 11) + 1))) * 1.0e-4f;
      }
    }

    append_tensor_with_data(model, k_feature_specs[0], filter_bank);
    append_tensor_with_data(model, k_feature_specs[1], window);
  }
};

emel::model::sortformer::detail::execution_contract make_contract(
    const emel::model::data & model) noexcept {
  emel::model::sortformer::detail::execution_contract contract = {};
  contract.model = &model;
  contract.sample_rate = diarization_detail::k_sample_rate;
  contract.speaker_count = diarization_detail::k_speaker_count;
  contract.frame_shift_ms = diarization_detail::k_frame_shift_ms;
  contract.chunk_len = diarization_detail::k_chunk_len;
  contract.chunk_right_context = diarization_detail::k_chunk_right_context;
  contract.feature_extractor.tensor_count = 2u;
  contract.encoder.tensor_count = 1u;
  contract.modules.tensor_count = 1u;
  contract.transformer_encoder.tensor_count = 1u;
  return contract;
}

std::vector<float> make_pcm() {
  std::vector<float> pcm(static_cast<size_t>(diarization_detail::k_required_sample_count));
  for (size_t index = 0u; index < pcm.size(); ++index) {
    const float low = static_cast<float>(index % 97u) / 96.0f;
    const float high = static_cast<float>((index / 97u) % 23u) / 22.0f;
    pcm[index] = (low * 0.75f) - (high * 0.25f);
  }
  return pcm;
}

diarization_request::event::prepare make_request(
    const emel::model::sortformer::detail::execution_contract & contract,
    std::span<const float> pcm,
    const int32_t sample_rate,
    const int32_t channel_count,
    std::span<float> features,
    int32_t & frame_count,
    int32_t & feature_bin_count,
    emel::error::type & err) noexcept {
  diarization_request::event::prepare request{
    contract,
    pcm,
    sample_rate,
    channel_count,
    features,
    frame_count,
    feature_bin_count,
  };
  request.error_out = &err;
  return request;
}

}  // namespace

TEST_CASE("diarization request extracts deterministic Sortformer features") {
  auto model = std::make_unique<request_model_fixture>();
  const auto contract = make_contract(model->model);
  auto pcm = make_pcm();
  std::vector<float> features(
      static_cast<size_t>(diarization_detail::k_required_feature_count));
  int32_t frame_count = -1;
  int32_t feature_bin_count = -1;
  emel::error::type err = emel::error::cast(diarization_request::error::feature_extractor);
  callback_probe probe = {};

  auto request = make_request(contract,
                              pcm,
                              diarization_detail::k_sample_rate,
                              diarization_detail::k_channel_count,
                              features,
                              frame_count,
                              feature_bin_count,
                              err);
  request.on_done =
      emel::callback<void(const diarization_request::events::prepare_done &)>::from<
          callback_probe,
          &callback_probe::on_done>(&probe);

  diarization_request::sm machine{};
  REQUIRE(machine.process_event(request));
  CHECK(machine.is(boost::sml::state<diarization_request::state_ready>));
  CHECK(err == emel::error::cast(diarization_request::error::none));
  CHECK(frame_count == feature_extractor_detail::k_feature_frame_count);
  CHECK(feature_bin_count == diarization_detail::k_feature_bin_count);
  CHECK(probe.done_called);
  CHECK_FALSE(probe.error_called);
  CHECK(probe.request == &request);
  CHECK(probe.frame_count == feature_extractor_detail::k_feature_frame_count);
  CHECK(probe.feature_bin_count == diarization_detail::k_feature_bin_count);

  CHECK(std::all_of(features.begin(), features.end(), [](const float value) {
    return std::isfinite(value);
  }));
  CHECK(features.front() != features.back());

  std::vector<float> second_features(features.size());
  int32_t second_frame_count = -1;
  int32_t second_feature_bin_count = -1;
  emel::error::type second_err =
      emel::error::cast(diarization_request::error::feature_extractor);
  auto second_request = make_request(contract,
                                     pcm,
                                     diarization_detail::k_sample_rate,
                                     diarization_detail::k_channel_count,
                                     second_features,
                                     second_frame_count,
                                     second_feature_bin_count,
                                     second_err);
  REQUIRE(machine.process_event(second_request));
  CHECK(second_err == emel::error::cast(diarization_request::error::none));
  CHECK(second_features == features);
}

TEST_CASE("diarization request rejects invalid sample rate") {
  auto model = std::make_unique<request_model_fixture>();
  const auto contract = make_contract(model->model);
  auto pcm = make_pcm();
  std::vector<float> features(
      static_cast<size_t>(diarization_detail::k_required_feature_count));
  int32_t frame_count = -1;
  int32_t feature_bin_count = -1;
  emel::error::type err = emel::error::cast(diarization_request::error::none);

  auto request = make_request(contract, pcm, 8000, 1, features, frame_count, feature_bin_count, err);
  diarization_request::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(diarization_request::error::sample_rate));
  CHECK(frame_count == 0);
  CHECK(feature_bin_count == 0);
}

TEST_CASE("diarization request rejects invalid channel count") {
  auto model = std::make_unique<request_model_fixture>();
  const auto contract = make_contract(model->model);
  auto pcm = make_pcm();
  std::vector<float> features(
      static_cast<size_t>(diarization_detail::k_required_feature_count));
  int32_t frame_count = -1;
  int32_t feature_bin_count = -1;
  emel::error::type err = emel::error::cast(diarization_request::error::none);

  auto request = make_request(contract, pcm, 16000, 2, features, frame_count, feature_bin_count, err);
  diarization_request::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(diarization_request::error::channel_count));
  CHECK(frame_count == 0);
  CHECK(feature_bin_count == 0);
}

TEST_CASE("diarization request rejects wrong pcm shape") {
  auto model = std::make_unique<request_model_fixture>();
  const auto contract = make_contract(model->model);
  std::vector<float> pcm(128u, 0.25f);
  std::vector<float> features(
      static_cast<size_t>(diarization_detail::k_required_feature_count));
  int32_t frame_count = -1;
  int32_t feature_bin_count = -1;
  emel::error::type err = emel::error::cast(diarization_request::error::none);

  auto request = make_request(contract, pcm, 16000, 1, features, frame_count, feature_bin_count, err);
  diarization_request::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(diarization_request::error::pcm_shape));
  CHECK(frame_count == 0);
  CHECK(feature_bin_count == 0);
}

TEST_CASE("diarization request rejects insufficient feature output capacity") {
  auto model = std::make_unique<request_model_fixture>();
  const auto contract = make_contract(model->model);
  auto pcm = make_pcm();
  std::vector<float> features(
      static_cast<size_t>(diarization_detail::k_required_feature_count - 1));
  int32_t frame_count = -1;
  int32_t feature_bin_count = -1;
  emel::error::type err = emel::error::cast(diarization_request::error::none);

  auto request = make_request(contract, pcm, 16000, 1, features, frame_count, feature_bin_count, err);
  diarization_request::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(diarization_request::error::capacity));
  CHECK(frame_count == 0);
  CHECK(feature_bin_count == 0);
}

TEST_CASE("diarization request rejects invalid Sortformer profile and emits error callback") {
  auto model = std::make_unique<request_model_fixture>();
  auto contract = make_contract(model->model);
  contract.chunk_len = diarization_detail::k_chunk_len - 1;
  auto pcm = make_pcm();
  std::vector<float> features(
      static_cast<size_t>(diarization_detail::k_required_feature_count));
  int32_t frame_count = -1;
  int32_t feature_bin_count = -1;
  emel::error::type err = emel::error::cast(diarization_request::error::none);
  callback_probe probe = {};

  auto request = make_request(contract, pcm, 16000, 1, features, frame_count, feature_bin_count, err);
  request.on_error =
      emel::callback<void(const diarization_request::events::prepare_error &)>::from<
          callback_probe,
          &callback_probe::on_error>(&probe);

  diarization_request::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(err == emel::error::cast(diarization_request::error::model_invalid));
  CHECK_FALSE(probe.done_called);
  CHECK(probe.error_called);
  CHECK(probe.request == &request);
  CHECK(probe.err == emel::error::cast(diarization_request::error::model_invalid));
  CHECK(frame_count == 0);
  CHECK(feature_bin_count == 0);
}
