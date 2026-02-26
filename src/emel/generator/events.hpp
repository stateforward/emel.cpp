#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/generator/errors.hpp"

namespace emel::generator::events {

struct generation_done;
struct generation_error;

}  // namespace emel::generator::events

namespace emel::generator::event {

struct generate {
  std::string_view prompt = {};
  int32_t max_tokens = 0;
  emel::error::type * error_out = nullptr;
  const emel::callback<void(const events::generation_done &)> & on_done;
  const emel::callback<void(const events::generation_error &)> & on_error;
};

struct generate_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t tokens_generated = 0;
  int32_t target_tokens = 0;
};

// Internal event used by generator::sm wrapper; not part of public API.
struct generate_run {
  const generate & request;
  generate_ctx & ctx;
};

}  // namespace emel::generator::event

namespace emel::generator::events {

struct generation_done {
  const event::generate * request = nullptr;
  int32_t tokens_generated = 0;
};

struct generation_error {
  const event::generate * request = nullptr;
  emel::error::type err = emel::error::type{};
  int32_t tokens_generated = 0;
};

}  // namespace emel::generator::events
