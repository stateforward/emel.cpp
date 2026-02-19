#include <limits>

#include "doctest/doctest.h"

#include "emel/emel.h"
#include "emel/sampler/token_selector/actions.hpp"
#include "emel/sampler/token_selector/sm.hpp"

TEST_CASE("token selector argmax picks highest score") {
  emel::sampler::token_selector::sm machine{};

  int32_t ids[3] = {10, 11, 12};
  float scores[3] = {0.1f, 0.7f, 0.2f};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::sampler::token_selector::event::select_token request{};
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_count = 3;
  request.policy = emel::sampler::token_selector::event::selection_policy::argmax;
  request.random_01 = 0.0f;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(selected == 11);
}

TEST_CASE("token selector categorical uses random draw") {
  emel::sampler::token_selector::sm machine{};

  int32_t ids[2] = {5, 7};
  float scores[2] = {0.0f, 0.0f};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::sampler::token_selector::event::select_token request{};
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_count = 2;
  request.policy = emel::sampler::token_selector::event::selection_policy::categorical;
  request.random_01 = 0.99f;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(selected == 7);
}

TEST_CASE("token selector reports invalid arguments") {
  emel::sampler::token_selector::sm machine{};

  int32_t ids[1] = {1};
  float scores[1] = {0.0f};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::sampler::token_selector::event::select_token request{};
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_count = 0;
  request.policy = emel::sampler::token_selector::event::selection_policy::argmax;
  request.random_01 = 0.0f;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(!machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token selector validates random range") {
  emel::sampler::token_selector::sm machine{};

  int32_t ids[1] = {1};
  float scores[1] = {0.0f};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  emel::sampler::token_selector::event::select_token request{};
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_count = 1;
  request.policy = emel::sampler::token_selector::event::selection_policy::categorical;
  request.random_01 = 1.0f;
  request.selected_token_out = &selected;
  request.error_out = &err;

  CHECK(!machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("token selector action error paths") {
  emel::sampler::token_selector::action::context ctx{};
  emel::sampler::token_selector::event::select_token request{};
  int32_t selected = -1;
  int32_t err = EMEL_OK;

  request.selected_token_out = &selected;
  request.error_out = &err;
  emel::sampler::token_selector::action::begin_select_token(request, ctx);
  emel::sampler::token_selector::action::run_validate(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  int32_t ids[1] = {1};
  float scores[1] = {std::numeric_limits<float>::quiet_NaN()};
  request.candidate_ids = ids;
  request.candidate_scores = scores;
  request.candidate_count = 1;
  request.policy = emel::sampler::token_selector::event::selection_policy::categorical;
  request.random_01 = 0.5f;
  ctx.request = &request;
  emel::sampler::token_selector::action::run_select(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ctx.request = &request;
  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.last_error = EMEL_OK;
  emel::sampler::token_selector::action::publish_error(ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.request == nullptr);
}
