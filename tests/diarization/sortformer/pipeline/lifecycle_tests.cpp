#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <stateforward/sml.hpp>
#include "doctest/doctest.h"

#include "diarization/sortformer_fixture.hpp"
#include "emel/diarization/sortformer/encoder/detail.hpp"
#include "emel/diarization/sortformer/encoder/feature_extractor/detail.hpp"
#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/transformer/detail.hpp"
#include "emel/model/data.hpp"
#include "emel/model/sortformer/detail.hpp"

namespace {

namespace fixture = emel::bench::diarization::sortformer_fixture;
namespace encoder_detail = emel::diarization::sortformer::encoder::detail;
namespace feature_detail =
    emel::diarization::sortformer::encoder::feature_extractor::detail;
namespace modules_detail = emel::diarization::sortformer::modules::detail;
namespace pipeline = emel::diarization::sortformer::pipeline;
namespace pipeline_detail = emel::diarization::sortformer::pipeline::detail;
namespace request_detail = emel::diarization::sortformer::request::detail;
namespace transformer_detail = emel::diarization::sortformer::transformer::detail;

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, 2> k_feature_specs{{
    {"prep.feat.fb", 3, {feature_detail::k_fft_bin_count,
                         feature_detail::k_feature_bin_count, 1, 0}},
    {"prep.feat.win", 1, {feature_detail::k_window_length, 0, 0, 0}},
}};

constexpr std::array<tensor_spec, encoder_detail::k_pre_tensor_count>
    k_encoder_pre_specs{{
        {"enc.pre.conv.0.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
        {"enc.pre.conv.0.w", 4, {3, 3, 1, encoder_detail::k_pre_channel_count}},
        {"enc.pre.conv.2.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
        {"enc.pre.conv.2.w", 4, {3, 3, 1, encoder_detail::k_pre_channel_count}},
        {"enc.pre.conv.3.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
        {"enc.pre.conv.3.w", 4, {1, 1, encoder_detail::k_pre_channel_count,
                                  encoder_detail::k_pre_channel_count}},
        {"enc.pre.conv.5.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
        {"enc.pre.conv.5.w", 4, {3, 3, 1, encoder_detail::k_pre_channel_count}},
        {"enc.pre.conv.6.b", 1, {encoder_detail::k_pre_channel_count, 0, 0, 0}},
        {"enc.pre.conv.6.w", 4, {1, 1, encoder_detail::k_pre_channel_count,
                                  encoder_detail::k_pre_channel_count}},
        {"enc.pre.out.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"enc.pre.out.w", 2, {encoder_detail::k_pre_expanded_dim,
                               encoder_detail::k_model_dim, 0, 0}},
    }};

constexpr std::array<tensor_spec, encoder_detail::k_layer_tensor_count>
    k_encoder_layer_specs{{
        {"conv.bn.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.bn.rm", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.bn.rv", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.bn.sc", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.bn.sh", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.bn.w", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.dw.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.dw.w", 3, {encoder_detail::k_depthwise_kernel, 1,
                           encoder_detail::k_model_dim, 0}},
        {"conv.pw1.b", 1, {1024, 0, 0, 0}},
        {"conv.pw1.w", 3, {1, encoder_detail::k_model_dim, 1024, 0}},
        {"conv.pw2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"conv.pw2.w", 3, {1, encoder_detail::k_model_dim,
                            encoder_detail::k_model_dim, 0}},
        {"ff1.l1.b", 1, {encoder_detail::k_feed_forward_dim, 0, 0, 0}},
        {"ff1.l1.w", 2, {encoder_detail::k_model_dim,
                          encoder_detail::k_feed_forward_dim, 0, 0}},
        {"ff1.l2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"ff1.l2.w", 2, {encoder_detail::k_feed_forward_dim,
                          encoder_detail::k_model_dim, 0, 0}},
        {"ff2.l1.b", 1, {encoder_detail::k_feed_forward_dim, 0, 0, 0}},
        {"ff2.l1.w", 2, {encoder_detail::k_model_dim,
                          encoder_detail::k_feed_forward_dim, 0, 0}},
        {"ff2.l2.b", 1, {encoder_detail::k_model_dim, 0, 0, 0}},
        {"ff2.l2.w", 2, {encoder_detail::k_feed_forward_dim,
                          encoder_detail::k_model_dim, 0, 0}},
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
        {"att.pbu", 2, {encoder_detail::k_attention_head_dim,
                         encoder_detail::k_attention_head_count, 0, 0}},
        {"att.pbv", 2, {encoder_detail::k_attention_head_dim,
                         encoder_detail::k_attention_head_count, 0, 0}},
    }};

constexpr std::array<tensor_spec, modules_detail::k_tensor_count> k_module_specs{{
    {"mods.ep.b", 1, {modules_detail::k_hidden_dim, 0, 0, 0}},
    {"mods.ep.w", 2, {modules_detail::k_encoder_dim, modules_detail::k_hidden_dim, 0, 0}},
    {"mods.fh2h.b", 1, {modules_detail::k_hidden_dim, 0, 0, 0}},
    {"mods.fh2h.w", 2, {modules_detail::k_hidden_dim, modules_detail::k_hidden_dim, 0, 0}},
    {"mods.h2s.b", 1, {modules_detail::k_speaker_count, 0, 0, 0}},
    {"mods.h2s.w", 2, {modules_detail::k_pair_hidden_dim,
                        modules_detail::k_speaker_count, 0, 0}},
    {"mods.sh2s.b", 1, {modules_detail::k_speaker_count, 0, 0, 0}},
    {"mods.sh2s.w", 2, {modules_detail::k_hidden_dim,
                         modules_detail::k_speaker_count, 0, 0}},
}};

constexpr std::array<tensor_spec, transformer_detail::k_layer_tensor_count>
    k_transformer_specs{{
        {"sa.k.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"sa.k.w", 2, {transformer_detail::k_hidden_dim,
                        transformer_detail::k_hidden_dim, 0, 0}},
        {"sa.o.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"sa.o.w", 2, {transformer_detail::k_hidden_dim,
                        transformer_detail::k_hidden_dim, 0, 0}},
        {"sa.q.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"sa.q.w", 2, {transformer_detail::k_hidden_dim,
                        transformer_detail::k_hidden_dim, 0, 0}},
        {"sa.v.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"sa.v.w", 2, {transformer_detail::k_hidden_dim,
                        transformer_detail::k_hidden_dim, 0, 0}},
        {"ln1.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"ln1.w", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"ln2.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"ln2.w", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"ff.di.b", 1, {transformer_detail::k_inner_dim, 0, 0, 0}},
        {"ff.di.w", 2, {transformer_detail::k_hidden_dim,
                         transformer_detail::k_inner_dim, 0, 0}},
        {"ff.do.b", 1, {transformer_detail::k_hidden_dim, 0, 0, 0}},
        {"ff.do.w", 2, {transformer_detail::k_inner_dim,
                         transformer_detail::k_hidden_dim, 0, 0}},
    }};

struct synthetic_model_fixture {
  emel::model::data model = {};
  std::vector<float> tensor_data =
      std::vector<float>(static_cast<size_t>(encoder_detail::k_pre_channel_count *
                                             encoder_detail::k_pre_channel_count),
                         0.0f);

  synthetic_model_fixture() {
    for (const auto & spec : k_feature_specs) {
      append_tensor(spec);
    }
    for (const auto & spec : k_encoder_pre_specs) {
      append_tensor(spec);
    }
    for (int32_t layer = 0; layer < encoder_detail::k_layer_count; ++layer) {
      for (const auto & spec : k_encoder_layer_specs) {
        append_prefixed_tensor("enc.l", layer, spec);
      }
    }
    for (const auto & spec : k_module_specs) {
      append_tensor(spec);
    }
    for (int32_t layer = 0; layer < transformer_detail::k_layer_count; ++layer) {
      for (const auto & spec : k_transformer_specs) {
        append_prefixed_tensor("te.l", layer, spec);
      }
    }
  }

  void append_name(emel::model::data::tensor_record & tensor,
                   const std::string_view name) {
    REQUIRE(model.name_bytes_used + name.size() <= model.name_storage.size());
    const auto offset = model.name_bytes_used;
    std::memcpy(model.name_storage.data() + offset, name.data(), name.size());
    tensor.name_offset = offset;
    tensor.name_length = static_cast<uint32_t>(name.size());
    model.name_bytes_used += static_cast<uint32_t>(name.size());
  }

  void append_tensor(const tensor_spec & spec) {
    REQUIRE(model.n_tensors < model.tensors.size());
    auto & tensor = model.tensors[model.n_tensors];
    append_name(tensor, spec.name);
    tensor.n_dims = spec.n_dims;
    tensor.dims = spec.dims;
    tensor.data = tensor_data.data();
    tensor.data_size = tensor_data.size() * sizeof(float);
    ++model.n_tensors;
  }

  void append_prefixed_tensor(const char * prefix,
                              const int32_t layer,
                              const tensor_spec & spec) {
    std::array<char, 64> name = {};
    const int written = std::snprintf(name.data(),
                                      name.size(),
                                      "%s%d.%.*s",
                                      prefix,
                                      layer,
                                      static_cast<int>(spec.name.size()),
                                      spec.name.data());
    REQUIRE(written > 0);
    REQUIRE(static_cast<size_t>(written) < name.size());
    tensor_spec named = spec;
    named.name = std::string_view{name.data(), static_cast<size_t>(written)};
    append_tensor(named);
  }
};

emel::model::sortformer::execution_contract make_synthetic_contract(
    const emel::model::data & model) noexcept {
  emel::model::sortformer::execution_contract contract = {};
  contract.model = &model;
  contract.sample_rate = pipeline_detail::k_sample_rate;
  contract.speaker_count = pipeline_detail::k_speaker_count;
  contract.frame_shift_ms = request_detail::k_frame_shift_ms;
  contract.chunk_len = pipeline_detail::k_frame_count;
  contract.chunk_right_context = request_detail::k_chunk_right_context;
  contract.feature_extractor.tensor_count = static_cast<uint32_t>(k_feature_specs.size());
  contract.encoder.tensor_count = static_cast<uint32_t>(
      k_encoder_pre_specs.size() +
      (encoder_detail::k_layer_count * k_encoder_layer_specs.size()));
  contract.modules.tensor_count = static_cast<uint32_t>(k_module_specs.size());
  contract.transformer_encoder.tensor_count = static_cast<uint32_t>(
      transformer_detail::k_layer_count * k_transformer_specs.size());
  return contract;
}

}  // namespace

TEST_CASE("sortformer pipeline runs maintained pcm to probabilities and segments") {
  auto model = std::make_unique<fixture::model_fixture>();
  fixture::pcm_fixture pcm{};
  fixture::run_result result{};

  REQUIRE(fixture::prepare(*model));
  REQUIRE(model->ready);
  REQUIRE(fixture::prepare(pcm));
  REQUIRE(pcm.ready);

  pipeline::sm machine{};
  REQUIRE(fixture::run_pipeline(machine, model->contract, pcm.pcm, pcm.sample_rate, result));
  CHECK(machine.is(stateforward::sml::state<pipeline::state_ready>));
  CHECK(result.err == pipeline_detail::to_error(pipeline::error::none));
  CHECK(result.frame_count == pipeline_detail::k_frame_count);
  CHECK(result.probability_count == pipeline_detail::k_required_probability_value_count);
  CHECK(result.segment_count > 0);
  CHECK(result.segment_count <= pipeline_detail::k_max_segment_count);
  CHECK(std::all_of(result.probabilities.begin(), result.probabilities.end(), [](const float value) {
    return std::isfinite(value);
  }));
}

TEST_CASE("sortformer pipeline rejects invalid sample rate") {
  auto model = std::make_unique<fixture::model_fixture>();
  fixture::pcm_fixture pcm{};
  fixture::run_result result{};

  REQUIRE(fixture::prepare(*model));
  REQUIRE(model->ready);
  REQUIRE(fixture::prepare(pcm));
  REQUIRE(pcm.ready);

  auto request = fixture::make_run_event(model->contract,
                                         pcm.pcm,
                                         8000,
                                         result.probabilities,
                                         result.segments,
                                         result.frame_count,
                                         result.probability_count,
                                         result.segment_count,
                                         result.err);
  pipeline::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(result.err == pipeline_detail::to_error(pipeline::error::sample_rate));
  CHECK(result.frame_count == 0);
  CHECK(result.probability_count == 0);
  CHECK(result.segment_count == 0);
}

TEST_CASE("sortformer pipeline rejects insufficient probability output capacity") {
  auto model = std::make_unique<fixture::model_fixture>();
  fixture::pcm_fixture pcm{};
  fixture::run_result result{};
  result.probabilities.resize(static_cast<size_t>(
      pipeline_detail::k_required_probability_value_count - 1));

  REQUIRE(fixture::prepare(*model));
  REQUIRE(model->ready);
  REQUIRE(fixture::prepare(pcm));
  REQUIRE(pcm.ready);

  auto request = fixture::make_run_event(model->contract,
                                         pcm.pcm,
                                         pcm.sample_rate,
                                         result.probabilities,
                                         result.segments,
                                         result.frame_count,
                                         result.probability_count,
                                         result.segment_count,
                                         result.err);
  pipeline::sm machine{};
  CHECK_FALSE(machine.process_event(request));
  CHECK(result.err == pipeline_detail::to_error(pipeline::error::probability_capacity));
  CHECK(result.frame_count == 0);
  CHECK(result.probability_count == 0);
  CHECK(result.segment_count == 0);
}

TEST_CASE("sortformer pipeline routes encoder kernel unavailability before compute") {
  namespace sml = stateforward::sml;

  auto model = std::make_unique<synthetic_model_fixture>();
  const auto contract = make_synthetic_contract(model->model);
  fixture::run_result result{};
  std::vector<float> pcm(static_cast<size_t>(pipeline_detail::k_required_sample_count), 0.0f);
  pipeline::action::context ctx = {};
  ctx.encoder_workspace.pre_encoder_rows.clear();

  auto request = fixture::make_run_event(contract,
                                         pcm,
                                         pipeline_detail::k_sample_rate,
                                         result.probabilities,
                                         result.segments,
                                         result.frame_count,
                                         result.probability_count,
                                         result.segment_count,
                                         result.err);
  pipeline::event::run_ctx runtime_ctx = {};
  pipeline::event::run_flow runtime_ev{request, runtime_ctx};
  stateforward::sml::sm<pipeline::model, stateforward::sml::testing> machine{ctx};

  REQUIRE(machine.is(sml::state<pipeline::state_ready>));
  REQUIRE(machine.process_event(runtime_ev));
  CHECK(machine.is(sml::state<pipeline::state_ready>));
  CHECK(runtime_ctx.err == pipeline_detail::to_error(pipeline::error::kernel));
  CHECK(result.err == pipeline_detail::to_error(pipeline::error::kernel));
  CHECK(result.frame_count == 0);
  CHECK(result.probability_count == 0);
  CHECK(result.segment_count == 0);
}

TEST_CASE("sortformer pipeline routes encoder compute failures after readiness") {
  namespace sml = stateforward::sml;

  auto model = std::make_unique<synthetic_model_fixture>();
  const auto contract = make_synthetic_contract(model->model);
  fixture::run_result result{};
  std::vector<float> pcm(static_cast<size_t>(pipeline_detail::k_required_sample_count), 0.0f);
  pipeline::action::context ctx = {};
  ctx.encoder_workspace.dense_transposed_input.clear();

  auto request = fixture::make_run_event(contract,
                                         pcm,
                                         pipeline_detail::k_sample_rate,
                                         result.probabilities,
                                         result.segments,
                                         result.frame_count,
                                         result.probability_count,
                                         result.segment_count,
                                         result.err);
  pipeline::event::run_ctx runtime_ctx = {};
  pipeline::event::run_flow runtime_ev{request, runtime_ctx};
  stateforward::sml::sm<pipeline::model, stateforward::sml::testing> machine{ctx};

  REQUIRE(machine.is(sml::state<pipeline::state_ready>));
  REQUIRE(machine.process_event(runtime_ev));
  CHECK(machine.is(sml::state<pipeline::state_ready>));
  CHECK(runtime_ctx.err == pipeline_detail::to_error(pipeline::error::kernel));
  CHECK(result.err == pipeline_detail::to_error(pipeline::error::kernel));
  CHECK(result.frame_count == 0);
  CHECK(result.probability_count == 0);
  CHECK(result.segment_count == 0);
}
