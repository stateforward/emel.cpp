#include "doctest/doctest.h"

#include "emel/emel.h"
#include "emel/sampler/candidate_builder/actions.hpp"
#include "emel/sampler/candidate_builder/sm.hpp"

namespace {

struct build_buffers {
  float logits[4] = {1.0f, 4.0f, 2.0f, -1.0f};
  int32_t ids[4] = {};
  float scores[4] = {};
  int32_t count = 0;
  int32_t err = EMEL_OK;
};

}  // namespace

TEST_CASE("candidate builder builds and normalizes scores") {
  emel::sampler::candidate_builder::sm machine{};
  build_buffers buffers{};

  emel::sampler::candidate_builder::event::build request{};
  request.logits = buffers.logits;
  request.vocab_size = 4;
  request.candidate_ids_out = buffers.ids;
  request.candidate_scores_out = buffers.scores;
  request.candidate_capacity = 4;
  request.candidate_count_out = &buffers.count;
  request.error_out = &buffers.err;

  CHECK(machine.process_event(request));
  CHECK(buffers.err == EMEL_OK);
  CHECK(buffers.count == 4);
  CHECK(buffers.ids[0] == 0);
  CHECK(buffers.ids[3] == 3);
  CHECK(buffers.scores[1] == doctest::Approx(0.0f));
  CHECK(buffers.scores[0] == doctest::Approx(-3.0f));
  CHECK(buffers.scores[2] == doctest::Approx(-2.0f));
}

TEST_CASE("candidate builder reports invalid arguments") {
  emel::sampler::candidate_builder::sm machine{};
  build_buffers buffers{};

  emel::sampler::candidate_builder::event::build request{};
  request.logits = buffers.logits;
  request.vocab_size = 4;
  request.candidate_ids_out = buffers.ids;
  request.candidate_scores_out = buffers.scores;
  request.candidate_capacity = 2;
  request.candidate_count_out = &buffers.count;
  request.error_out = &buffers.err;

  CHECK(!machine.process_event(request));
  CHECK(buffers.err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(buffers.count == 0);
}

TEST_CASE("candidate builder action error paths") {
  emel::sampler::candidate_builder::action::context ctx{};
  emel::sampler::candidate_builder::event::build request{};
  int32_t count = 0;
  int32_t err = EMEL_OK;

  request.candidate_count_out = &count;
  request.error_out = &err;
  emel::sampler::candidate_builder::action::begin_build(request, ctx);
  emel::sampler::candidate_builder::action::run_validate(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  request.logits = nullptr;
  request.vocab_size = 1;
  request.candidate_ids_out = nullptr;
  request.candidate_scores_out = nullptr;
  ctx.request = &request;
  emel::sampler::candidate_builder::action::run_build_candidates(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  request.candidate_scores_out = nullptr;
  ctx.candidate_count = 1;
  ctx.request = &request;
  emel::sampler::candidate_builder::action::run_normalize_scores(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  float scores[1] = {0.0f};
  request.candidate_scores_out = scores;
  ctx.candidate_count = 0;
  ctx.request = &request;
  emel::sampler::candidate_builder::action::run_normalize_scores(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ctx.request = &request;
  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.last_error = EMEL_OK;
  emel::sampler::candidate_builder::action::publish_error(ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(count == 0);
  CHECK(ctx.request == nullptr);
}
