#include "doctest/doctest.h"

#include "emel/emel.h"
#include "emel/error/error.hpp"
#include "emel/logits/sampler/sm.hpp"

namespace {

emel::error::type sampler_shift_scores(int32_t &,
                                       float & candidate_scores,
                                       int32_t & candidate_count,
                                       int32_t &) {
  for (int32_t i = 0; i < candidate_count; ++i) {
    (&candidate_scores)[i] = -1.0f;
  }
  (&candidate_scores)[1] = 5.0f;
  return emel::error::cast(emel::logits::sampler::error::none);
}

emel::error::type sampler_select_argmax(int32_t & candidate_ids,
                                        float & candidate_scores,
                                        int32_t & candidate_count,
                                        int32_t & selected_token_out) {
  int32_t best_idx = 0;
  float best_score = (&candidate_scores)[0];
  for (int32_t i = 1; i < candidate_count; ++i) {
    if ((&candidate_scores)[i] > best_score) {
      best_score = (&candidate_scores)[i];
      best_idx = i;
    }
  }
  selected_token_out = (&candidate_ids)[best_idx];
  return emel::error::cast(emel::logits::sampler::error::none);
}

emel::error::type sampler_select_fixed(int32_t &,
                                       float &,
                                       int32_t &,
                                       int32_t & selected_token_out) {
  selected_token_out = 7;
  return emel::error::cast(emel::logits::sampler::error::none);
}

emel::error::type sampler_error(int32_t &,
                                float &,
                                int32_t &,
                                int32_t &) {
  return emel::error::cast(emel::logits::sampler::error::backend_error);
}

emel::error::type sampler_set_invalid_candidate_count(int32_t &,
                                                      float &,
                                                      int32_t & candidate_count,
                                                      int32_t &) {
  candidate_count = 0;
  return emel::error::cast(emel::logits::sampler::error::none);
}

}  // namespace

TEST_CASE("sampler pipeline selects token via selector sampler") {
  float logits[3] = {0.0f, 0.1f, 0.2f};
  int32_t ids[3] = {};
  float scores[3] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sampler_fn samplers[2] = {
      sampler_shift_scores,
      sampler_select_argmax,
  };
  emel::logits::sampler::sm machine{samplers, 2};

  emel::logits::sampler::event::sample_logits request{
      logits[0], 3, ids[0], scores[0], 3, selected, err};

  CHECK(machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::none));
  CHECK(selected == 1);
}

TEST_CASE("sampler pipeline reports invalid request when no samplers are configured") {
  emel::logits::sampler::sm machine{};

  float logits[2] = {1.0f, 2.0f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sample_logits request{
      logits[0], 2, ids[0], scores[0], 2, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline accepts valid preselected token without sampler chain") {
  emel::logits::sampler::sm machine{};

  int32_t selected = 2;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::backend_error);

  emel::logits::sampler::event::sample_preselected request{3, selected, err};

  CHECK(machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::none));
  CHECK(selected == 2);
}

TEST_CASE("sampler pipeline rejects invalid preselected token") {
  emel::logits::sampler::sm machine{};

  int32_t selected = 7;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sample_preselected request{3, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == 7);
}

TEST_CASE("sampler pipeline reports invalid arguments") {
  emel::logits::sampler::sm machine{};

  float logits[2] = {1.0f, 2.0f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sample_logits request{
      logits[0], 2, ids[0], scores[0], 1, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline reports invalid context sampler table") {
  emel::logits::sampler::sm machine{nullptr, 1};

  float logits[1] = {0.0f};
  int32_t ids[1] = {};
  float scores[1] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sample_logits request{
      logits[0], 1, ids[0], scores[0], 1, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline reports missing sampler function") {
  float logits[1] = {0.0f};
  int32_t ids[1] = {};
  float scores[1] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sampler_fn samplers[1] = {nullptr};
  emel::logits::sampler::sm machine{samplers, 1};

  emel::logits::sampler::event::sample_logits request{
      logits[0], 1, ids[0], scores[0], 1, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline reports sampler errors") {
  float logits[2] = {0.0f, 0.1f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sampler_fn samplers[1] = {sampler_error};
  emel::logits::sampler::sm machine{samplers, 1};

  emel::logits::sampler::event::sample_logits request{
      logits[0], 2, ids[0], scores[0], 2, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::backend_error));
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline reports invalid request when no selector sampler sets a token") {
  float logits[2] = {0.0f, 0.1f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sampler_fn samplers[1] = {sampler_shift_scores};
  emel::logits::sampler::sm machine{samplers, 1};

  emel::logits::sampler::event::sample_logits request{
      logits[0], 2, ids[0], scores[0], 2, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline reports invalid request when sampler sets invalid candidate_count") {
  float logits[3] = {0.0f, 0.1f, 0.2f};
  int32_t ids[3] = {};
  float scores[3] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sampler_fn samplers[1] = {sampler_set_invalid_candidate_count};
  emel::logits::sampler::sm machine{samplers, 1};

  emel::logits::sampler::event::sample_logits request{
      logits[0], 3, ids[0], scores[0], 3, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == -1);
}

TEST_CASE("sampler pipeline reports invalid request when selector picks out-of-range token") {
  float logits[3] = {0.0f, 0.1f, 0.2f};
  int32_t ids[3] = {};
  float scores[3] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::event::sampler_fn samplers[1] = {sampler_select_fixed};
  emel::logits::sampler::sm machine{samplers, 1};

  emel::logits::sampler::event::sample_logits request{
      logits[0], 3, ids[0], scores[0], 3, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == 7);
}
