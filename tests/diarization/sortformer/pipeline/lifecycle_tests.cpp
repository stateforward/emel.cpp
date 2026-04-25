#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "diarization/sortformer_fixture.hpp"

namespace {

namespace fixture = emel::bench::diarization::sortformer_fixture;
namespace pipeline = emel::diarization::sortformer::pipeline;
namespace pipeline_detail = emel::diarization::sortformer::pipeline::detail;
namespace request_detail = emel::diarization::request::detail;

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
  CHECK(machine.is(boost::sml::state<pipeline::state_ready>));
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

TEST_CASE("sortformer pipeline reports encoder kernel failures before executor") {
  auto model = std::make_unique<fixture::model_fixture>();
  fixture::run_result result{};
  std::array<float, 1> pcm_stub = {};

  REQUIRE(fixture::prepare(*model));
  REQUIRE(model->ready);

  auto request = fixture::make_run_event(model->contract,
                                         pcm_stub,
                                         request_detail::k_sample_rate,
                                         result.probabilities,
                                         result.segments,
                                         result.frame_count,
                                         result.probability_count,
                                         result.segment_count,
                                         result.err);
  pipeline::event::run_ctx runtime_ctx = {};
  pipeline::event::run_flow runtime_ev{request, runtime_ctx};
  pipeline::action::context ctx = {};

  pipeline::action::effect_begin_run(runtime_ev, ctx);
  REQUIRE(runtime_ctx.err == pipeline_detail::to_error(pipeline::error::none));
  pipeline::action::effect_bind_encoder(runtime_ev, ctx);
  ctx.encoder.pre[0].tensor = nullptr;
  pipeline::action::effect_compute_encoder_frames(runtime_ev, ctx);

  CHECK(runtime_ctx.err == pipeline_detail::to_error(pipeline::error::kernel));
  CHECK(result.frame_count == 0);
  CHECK(result.probability_count == 0);
  CHECK(result.segment_count == 0);
}

TEST_CASE("sortformer pipeline reports probability kernel failures before decode") {
  auto model = std::make_unique<fixture::model_fixture>();
  fixture::run_result result{};
  std::array<float, 1> pcm_stub = {};
  std::fill(result.probabilities.begin(), result.probabilities.end(), -3.0f);

  REQUIRE(fixture::prepare(*model));
  REQUIRE(model->ready);

  auto request = fixture::make_run_event(model->contract,
                                         pcm_stub,
                                         request_detail::k_sample_rate,
                                         result.probabilities,
                                         result.segments,
                                         result.frame_count,
                                         result.probability_count,
                                         result.segment_count,
                                         result.err);
  pipeline::event::run_ctx runtime_ctx = {};
  pipeline::event::run_flow runtime_ev{request, runtime_ctx};
  pipeline::action::context ctx = {};

  pipeline::action::effect_begin_run(runtime_ev, ctx);
  pipeline::action::effect_bind_modules(runtime_ev, ctx);
  ctx.modules.speaker_hidden_to_speaker_weight.tensor = nullptr;
  pipeline::action::effect_compute_probabilities(runtime_ev, ctx);

  CHECK(runtime_ctx.err == pipeline_detail::to_error(pipeline::error::kernel));
  CHECK(result.frame_count == 0);
  CHECK(result.probability_count == 0);
  CHECK(result.segment_count == 0);
  CHECK(std::all_of(result.probabilities.begin(),
                    result.probabilities.end(),
                    [](const float value) {
                      return value == -3.0f;
                    }));
}
