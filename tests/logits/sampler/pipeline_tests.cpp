#include "doctest/doctest.h"

#include "emel/emel.h"
#include "emel/error/error.hpp"
#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/sampler/sm.hpp"
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

emel::error::type sampler_select_first(int32_t & candidate_ids,
                                       float &,
                                       int32_t &,
                                       int32_t & selected_token_out) {
  selected_token_out = (&candidate_ids)[0];
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

bool configure_sampler_chain(emel::logits::sampler::sm & machine,
                             emel::logits::sampler::fn & sampler_fns,
                             int32_t sampler_count,
                             emel::error::type & err) {
  emel::logits::sampler::event::configure request{sampler_fns, sampler_count, err};
  return machine.process_event(request);
}

}  // namespace

TEST_CASE("sampler pipeline selects token via selector sampler") {
  float logits[3] = {0.0f, 0.1f, 0.2f};
  int32_t ids[3] = {};
  float scores[3] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::fn samplers[2] = {
      emel::logits::sampler::fn::from<sampler_shift_scores>(),
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 2, err));

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

TEST_CASE("sampler configure rejects invalid sampler count") {
  emel::logits::sampler::sm machine{};
  emel::logits::sampler::fn sampler = emel::logits::sampler::fn::from<sampler_select_first>();
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  CHECK_FALSE(configure_sampler_chain(machine, sampler, 0, err));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
}

TEST_CASE("sampler configure can replace the sampler chain") {
  float logits[3] = {0.0f, 0.1f, 0.2f};
  int32_t ids[3] = {};
  float scores[3] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::sm machine{};
  emel::logits::sampler::fn first_chain[2] = {
      emel::logits::sampler::fn::from<sampler_shift_scores>(),
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  REQUIRE(configure_sampler_chain(machine, first_chain[0], 2, err));

  emel::logits::sampler::event::sample_logits first_request{
      logits[0], 3, ids[0], scores[0], 3, selected, err};
  REQUIRE(machine.process_event(first_request));
  CHECK(selected == 1);

  emel::logits::sampler::fn second_chain[1] = {
      emel::logits::sampler::fn::from<sampler_select_first>(),
  };
  REQUIRE(configure_sampler_chain(machine, second_chain[0], 1, err));

  selected = -1;
  emel::logits::sampler::event::sample_logits second_request{
      logits[0], 3, ids[0], scores[0], 3, selected, err};
  CHECK(machine.process_event(second_request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::none));
  CHECK(selected == 0);
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
  emel::logits::sampler::sm machine{};
  emel::logits::sampler::fn sampler = emel::logits::sampler::fn::from<sampler_select_first>();
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  CHECK_FALSE(configure_sampler_chain(machine, sampler, 0, err));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
}

TEST_CASE("sampler pipeline reports missing sampler function") {
  float logits[1] = {0.0f};
  int32_t ids[1] = {};
  float scores[1] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::logits::sampler::fn samplers[1] = {};
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 1, err));

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

  emel::logits::sampler::fn samplers[1] = {
      emel::logits::sampler::fn::from<sampler_error>(),
  };
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 1, err));

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

  emel::logits::sampler::fn samplers[1] = {
      emel::logits::sampler::fn::from<sampler_shift_scores>(),
  };
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 1, err));

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

  emel::logits::sampler::fn samplers[1] = {
      emel::logits::sampler::fn::from<sampler_set_invalid_candidate_count>(),
  };
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 1, err));

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

  emel::logits::sampler::fn samplers[1] = {
      emel::logits::sampler::fn::from<sampler_select_fixed>(),
  };
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 1, err));

  emel::logits::sampler::event::sample_logits request{
      logits[0], 3, ids[0], scores[0], 3, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(selected == 7);
}

TEST_CASE("sampler pipeline supports gbnf_filter then argmax sampler") {
  float logits[4] = {0.5f, 2.0f, -1.0f, 8.0f};
  int32_t ids[4] = {};
  float scores[4] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::gbnf::grammar grammar{};
  grammar.rule_count = 3;
  emel::gbnf::sampler::sm gbnf_sampler{grammar, 0};

  emel::logits::sampler::fn samplers[2] = {
      emel::gbnf::sampler::make_logits_sampler_fn(gbnf_sampler),
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 2, err));

  emel::logits::sampler::event::sample_logits request{
      logits[0], 4, ids[0], scores[0], 4, selected, err};

  CHECK(machine.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::none));
  CHECK(selected == 1);
}

TEST_CASE("sampler pipeline propagates gbnf sampler errors") {
  float logits[2] = {0.9f, 0.7f};
  int32_t ids[2] = {};
  float scores[2] = {};
  int32_t selected = -1;
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);

  emel::gbnf::grammar invalid_grammar{};
  invalid_grammar.rule_count = 0;
  emel::gbnf::sampler::sm gbnf_sampler{invalid_grammar, 0};

  emel::logits::sampler::fn samplers[2] = {
      emel::gbnf::sampler::make_logits_sampler_fn(gbnf_sampler),
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  emel::logits::sampler::sm machine{};
  REQUIRE(configure_sampler_chain(machine, samplers[0], 2, err));

  emel::logits::sampler::event::sample_logits request{
      logits[0], 2, ids[0], scores[0], 2, selected, err};

  CHECK(!machine.process_event(request));
  CHECK(err == emel::error::cast(emel::gbnf::sampler::error::invalid_request));
  CHECK(selected == -1);
}
