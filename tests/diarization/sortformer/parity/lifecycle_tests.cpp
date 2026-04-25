#include "doctest/doctest.h"

#include <memory>

#include "diarization/sortformer_fixture.hpp"

namespace {

namespace fixture = emel::bench::diarization::sortformer_fixture;
namespace pipeline = emel::diarization::sortformer::pipeline;
namespace pipeline_detail = emel::diarization::sortformer::pipeline::detail;

}  // namespace

TEST_CASE("sortformer parity matches maintained real model and audio checksum baseline") {
  auto model = std::make_unique<fixture::model_fixture>();
  fixture::pcm_fixture pcm{};
  fixture::expected_output_baseline baseline{};
  fixture::run_result result{};

  REQUIRE(fixture::prepare(*model));
  REQUIRE(model->ready);
  REQUIRE(fixture::prepare(pcm));
  REQUIRE(pcm.ready);
  REQUIRE(fixture::prepare(baseline));
  REQUIRE(baseline.ready);

  pipeline::sm machine{};
  REQUIRE(fixture::run_pipeline(machine, model->contract, pcm.pcm, pcm.sample_rate, result));
  INFO("fixture_id=", fixture::k_fixture_id);
  INFO("model_path=", fixture::k_model_rel_path.data());
  INFO("audio_path=", fixture::k_audio_rel_path.data());
  CHECK(result.err == pipeline_detail::to_error(pipeline::error::none));
  CHECK(model->contract.model == &model->model);
  CHECK(pcm.sample_rate == 16000);
  REQUIRE(result.segment_count == baseline.segment_count);
  CHECK(fixture::compute_checksum(result.segments, result.segment_count) ==
        baseline.output_checksum);
}
