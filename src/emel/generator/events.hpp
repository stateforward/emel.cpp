#pragma once

#include <cstdint>

namespace emel::generator::event {

struct generate {
  const char * prompt = nullptr;
  int32_t max_tokens = 0;
};

struct prompt_tokenized_done {};
struct prompt_tokenized_error {};
struct prefill_done {};
struct prefill_error {};
struct decode_step_done {};
struct decode_step_error {};
struct stop_condition_met {};

}  // namespace emel::generator::event

namespace emel::generator::events {

struct generation_done {};
struct generation_error {};

}  // namespace emel::generator::events
