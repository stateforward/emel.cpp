#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/emel.h"

namespace emel::generator::events {

struct generation_done;
struct generation_error;

}  // namespace emel::generator::events

namespace emel::generator::event {

struct generate {
  std::string_view prompt = {};
  int32_t max_tokens = 0;
  int32_t * error_out = nullptr;
  emel::callback<bool(const events::generation_done &)> dispatch_done = {};
  emel::callback<bool(const events::generation_error &)> dispatch_error = {};
};

}  // namespace emel::generator::event

namespace emel::generator::events {

struct generation_done {
  const event::generate * request = nullptr;
  int32_t tokens_generated = 0;
};

struct generation_error {
  const event::generate * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
  int32_t tokens_generated = 0;
};

}  // namespace emel::generator::events
