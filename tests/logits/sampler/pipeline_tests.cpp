#include "doctest/doctest.h"

#include <limits>

#include "emel/emel.h"
#include "emel/logits/sampler/actions.hpp"
#include "emel/logits/sampler/sm.hpp"

namespace {

bool sampler_force_token(int32_t * candidate_ids,
                         float * candidate_scores,
                         int32_t candidate_count,
                         void *,
                         int32_t * error_out) {
  if (candidate_ids == nullptr || candidate_scores == nullptr || candidate_count <= 0) {
    if (error_out != nullptr) {
      *error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  for (int32_t i = 0; i < candidate_count; ++i) {
    candidate_scores[i] = -1.0f;
  }
  candidate_scores[1] = 5.0f;
  if (error_out != nullptr) {
    *error_out = EMEL_OK;
  }
  return true;
}

bool sampler_error(int32_t *, float *, int32_t, void *, int32_t * error_out) {
  if (error_out != nullptr) {
    *error_out = EMEL_ERR_BACKEND;
  }
  return false;
}

}  // namespace

TEST_CASE("sampler pipeline selects token after sampler") {
  emel::logits::sampler::sm machine{};

  float logits[3] = {0.0f, 0.1f, 0.2f};
  int32_t ids[3] = {};
  float scores[3] = {};
  int32_t selected = -1;
  int32_t err = EMEL_OK;
  emel::logits::sampler::event::sampler_fn samplers[1] = {sampler_force_token};

  emel::logits::sampler::event::sample request{};
  request.logits = logits;
  request.vocab_size = 3;
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_capacity = 3;
  request.sampler_fns = samplers;
  request.sampler_count = 1;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(selected == 1);
}

TEST_CASE("sampler pipeline handles zero samplers") {
  emel::logits::sampler::sm machine{};

  float logits[2] = {1.0f, 2.0f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::logits::sampler::event::sample request{};
  request.logits = logits;
  request.vocab_size = 2;
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_capacity = 2;
  request.sampler_fns = nullptr;
  request.sampler_count = 0;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(selected == 1);
}

TEST_CASE("sampler pipeline reports invalid arguments") {
  emel::logits::sampler::sm machine{};

  float logits[2] = {1.0f, 2.0f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::logits::sampler::event::sample request{};
  request.logits = logits;
  request.vocab_size = 2;
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_capacity = 1;
  request.sampler_fns = nullptr;
  request.sampler_count = 0;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(!machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline reports sampler errors") {
  emel::logits::sampler::sm machine{};

  float logits[2] = {0.0f, 0.1f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  int32_t err = EMEL_OK;
  emel::logits::sampler::event::sampler_fn samplers[1] = {sampler_error};

  emel::logits::sampler::event::sample request{};
  request.logits = logits;
  request.vocab_size = 2;
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_capacity = 2;
  request.sampler_fns = samplers;
  request.sampler_count = 1;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(!machine.process_event(request));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline categorical selection uses random draw") {
  emel::logits::sampler::sm machine{};

  float logits[2] = {0.0f, 0.0f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::logits::sampler::event::sample request{};
  request.logits = logits;
  request.vocab_size = 2;
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_capacity = 2;
  request.sampler_fns = nullptr;
  request.sampler_count = 0;
  request.policy = emel::logits::sampler::event::selection_policy::categorical;
  request.random_01 = 0.99f;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(selected == 1);
}

TEST_CASE("sampler pipeline categorical validates random range") {
  emel::logits::sampler::sm machine{};

  float logits[2] = {0.0f, 0.0f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::logits::sampler::event::sample request{};
  request.logits = logits;
  request.vocab_size = 2;
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_capacity = 2;
  request.sampler_fns = nullptr;
  request.sampler_count = 0;
  request.policy = emel::logits::sampler::event::selection_policy::categorical;
  request.random_01 = 1.0f;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(!machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline action error paths") {
  emel::logits::sampler::action::context ctx{};
  emel::logits::sampler::event::sample request{};
  int32_t selected = -1;
  int32_t err = EMEL_OK;
  request.selected_token_out = &selected;
  request.error_out = &err;

  emel::logits::sampler::action::begin_sample(request, ctx);
  emel::logits::sampler::action::run_prepare_candidates(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  float logits[1] = {0.0f};
  int32_t ids[1] = {};
  float scores[1] = {};
  request.logits = logits;
  request.vocab_size = 1;
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_capacity = 1;
  request.sampler_count = 1;
  ctx.request = &request;
  ctx.sampler_count = request.sampler_count;
  emel::logits::sampler::action::run_prepare_candidates(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::logits::sampler::action::run_apply_sampling(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::logits::sampler::event::sampler_fn samplers[1] = {sampler_error};
  request.sampler_fns = samplers;
  ctx.request = &request;
  ctx.candidate_count = 1;
  ctx.sampler_index = 0;
  ctx.sampler_count = 1;
  emel::logits::sampler::action::run_apply_sampling(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ctx.candidate_count = 0;
  emel::logits::sampler::action::run_select_token(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ids[0] = 1;
  scores[0] = std::numeric_limits<float>::quiet_NaN();
  request.policy = emel::logits::sampler::event::selection_policy::categorical;
  request.random_01 = 0.5f;
  ctx.request = &request;
  ctx.candidate_count = 1;
  emel::logits::sampler::action::run_select_token(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ctx.request = &request;
  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.last_error = EMEL_OK;
  emel::logits::sampler::action::publish_error(ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.request == nullptr);
}
