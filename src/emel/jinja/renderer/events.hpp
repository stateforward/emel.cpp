#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/jinja/ast.hpp"
#include "emel/jinja/value.hpp"

namespace emel::jinja::events {

struct rendering_done;
struct rendering_error;

}  // namespace emel::jinja::events

namespace emel::jinja::event {

struct render {
  const emel::jinja::program * program = nullptr;
  const emel::jinja::object_value * globals = nullptr;
  std::string_view source_text = {};
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t * output_length = nullptr;
  bool * output_truncated = nullptr;
  int32_t * error_out = nullptr;
  size_t * error_pos_out = nullptr;
  void * owner_sm = nullptr;
  ::emel::callback<bool(const ::emel::jinja::events::rendering_done &)> dispatch_done = {};
  ::emel::callback<bool(const ::emel::jinja::events::rendering_error &)> dispatch_error = {};
};

}  // namespace emel::jinja::event

namespace emel::jinja::events {

struct rendering_done {
  const event::render * request = nullptr;
  size_t output_length = 0;
  bool output_truncated = false;
};

struct rendering_error {
  const event::render * request = nullptr;
  int32_t err = 0;
  size_t error_pos = 0;
};

}  // namespace emel::jinja::events
