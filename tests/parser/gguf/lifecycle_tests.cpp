#include "doctest/doctest.h"

#include "emel/model/data.hpp"
#include "emel/parser/gguf/sm.hpp"

TEST_CASE("gguf probe bind parse lifecycle") {
  emel::parser::gguf::sm machine{};

  uint8_t file_bytes[4] = {0};
  emel::parser::gguf::requirements req{};
  emel::parser::gguf::event::probe probe{
    .file_image = file_bytes,
    .size = sizeof(file_bytes),
    .requirements_out = &req,
  };
  CHECK(machine.process_event(probe));

  uint8_t kv_arena[8] = {};
  emel::model::data::tensor_record tensors[1] = {};
  emel::parser::gguf::event::bind_storage bind{
    .kv_arena = kv_arena,
    .kv_arena_size = sizeof(kv_arena),
    .tensors = tensors,
    .tensor_capacity = 1,
  };
  CHECK(machine.process_event(bind));

  emel::parser::gguf::event::parse parse{
    .file_image = file_bytes,
    .size = sizeof(file_bytes),
  };
  CHECK(machine.process_event(parse));
}

TEST_CASE("gguf probe rejects invalid inputs") {
  emel::parser::gguf::sm machine{};
  emel::parser::gguf::event::probe probe{};
  CHECK_FALSE(machine.process_event(probe));
}
