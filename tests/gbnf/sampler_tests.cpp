#include "doctest/doctest.h"

#include "emel/error/error.hpp"
#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/sampler/sm.hpp"

TEST_CASE("gbnf sampler filters candidates accepted by grammar") {
  emel::gbnf::grammar grammar{};
  grammar.rule_count = 3;

  int32_t candidate_ids[4] = {2, 4, 1, 9};
  float candidate_scores[4] = {0.2f, 0.4f, 0.1f, 0.9f};
  int32_t candidate_count = 4;
  int32_t selected_token = -1;
  emel::error::type err = emel::error::cast(emel::gbnf::sampler::error::none);

  emel::gbnf::sampler::sm machine{grammar, 0};
  emel::gbnf::sampler::event::sample request{
    candidate_ids[0],
    candidate_scores[0],
    candidate_count,
    selected_token,
    err,
  };

  CHECK(machine.process_event(request));
  CHECK(err == emel::error::cast(emel::gbnf::sampler::error::none));
  CHECK(candidate_count == 2);
  CHECK(candidate_ids[0] == 2);
  CHECK(candidate_ids[1] == 1);
  CHECK(candidate_scores[0] == doctest::Approx(0.2f));
  CHECK(candidate_scores[1] == doctest::Approx(0.1f));
  CHECK(selected_token == -1);
}

TEST_CASE("gbnf sampler reports invalid request for zero candidate_count") {
  emel::gbnf::grammar grammar{};
  grammar.rule_count = 2;

  int32_t candidate_ids[1] = {0};
  float candidate_scores[1] = {0.0f};
  int32_t candidate_count = 0;
  int32_t selected_token = -1;
  emel::error::type err = emel::error::cast(emel::gbnf::sampler::error::none);

  emel::gbnf::sampler::sm machine{grammar, 0};
  emel::gbnf::sampler::event::sample request{
    candidate_ids[0],
    candidate_scores[0],
    candidate_count,
    selected_token,
    err,
  };

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::gbnf::sampler::error::invalid_request));
  CHECK(candidate_count == 0);
}

TEST_CASE("gbnf sampler reports parse_failed when grammar rejects all candidates") {
  emel::gbnf::grammar grammar{};
  grammar.rule_count = 2;

  int32_t candidate_ids[2] = {8, 5};
  float candidate_scores[2] = {0.8f, 0.5f};
  int32_t candidate_count = 2;
  int32_t selected_token = -1;
  emel::error::type err = emel::error::cast(emel::gbnf::sampler::error::none);

  emel::gbnf::sampler::sm machine{grammar, 0};
  emel::gbnf::sampler::event::sample request{
    candidate_ids[0],
    candidate_scores[0],
    candidate_count,
    selected_token,
    err,
  };

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::gbnf::sampler::error::parse_failed));
  CHECK(candidate_count == 0);
}

TEST_CASE("gbnf sampler reports invalid request when grammar is not configured") {
  int32_t candidate_ids[1] = {0};
  float candidate_scores[1] = {0.1f};
  int32_t candidate_count = 1;
  int32_t selected_token = -1;
  emel::error::type err = emel::error::cast(emel::gbnf::sampler::error::none);

  emel::gbnf::sampler::sm machine{};
  emel::gbnf::sampler::event::sample request{
    candidate_ids[0],
    candidate_scores[0],
    candidate_count,
    selected_token,
    err,
  };

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::gbnf::sampler::error::invalid_request));
  CHECK(candidate_count == 0);
}

TEST_CASE("gbnf sampler adapter method matches sampler_fn return contract") {
  emel::gbnf::grammar grammar{};
  grammar.rule_count = 2;

  int32_t candidate_ids[3] = {0, 7, 1};
  float candidate_scores[3] = {0.5f, 0.2f, 0.9f};
  int32_t candidate_count = 3;
  int32_t selected_token = -1;

  emel::gbnf::sampler::sm machine{grammar, 0};
  const emel::error::type err = machine.sample(
      candidate_ids[0], candidate_scores[0], candidate_count, selected_token);

  CHECK(err == emel::error::cast(emel::gbnf::sampler::error::none));
  CHECK(candidate_count == 2);
  CHECK(candidate_ids[0] == 0);
  CHECK(candidate_ids[1] == 1);
}
