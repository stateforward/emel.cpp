#pragma once

#include <cstdint>
#include <string_view>

namespace emel::generator::event {

struct generate {
  std::string_view prompt = {};
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
