#include "doctest/doctest.h"

#include "emel/model/weight_loader/sm.hpp"

TEST_CASE("weight_loader bind plan apply lifecycle") {
  emel::model::weight_loader::sm machine{};
  emel::model::data::tensor_record tensors[1] = {};

  emel::model::weight_loader::event::bind_storage bind{
    .tensors = tensors,
    .tensor_count = 1,
  };
  CHECK(machine.process_event(bind));

  emel::model::weight_loader::effect_request effects[1] = {};
  uint32_t effect_count = 0;
  emel::model::weight_loader::event::plan_load plan{
    .effects_out = effects,
    .effect_capacity = 1,
    .effect_count_out = &effect_count,
  };
  CHECK(machine.process_event(plan));
  CHECK(effect_count == 0);

  emel::model::weight_loader::event::apply_effect_results apply{
    .results = nullptr,
    .result_count = 0,
  };
  CHECK(machine.process_event(apply));
}

TEST_CASE("weight_loader bind rejects invalid inputs") {
  emel::model::weight_loader::sm machine{};
  emel::model::weight_loader::event::bind_storage bind{};
  CHECK_FALSE(machine.process_event(bind));
}
