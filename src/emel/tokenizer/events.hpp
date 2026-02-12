#pragma once

#include <cstdint>

namespace emel::tokenizer::event {

struct tokenize {
  const char * text = nullptr;
};

struct tokenizing_done {};
struct tokenizing_error {};

}  // namespace emel::tokenizer::event

namespace emel::tokenizer::events {

struct tokenizer_done {};
struct tokenizer_error {};

}  // namespace emel::tokenizer::events
