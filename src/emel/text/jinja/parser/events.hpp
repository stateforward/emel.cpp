#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/text/jinja/types.hpp"

namespace emel::text::jinja::events {

struct parsing_done;
struct parsing_error;

} // namespace emel::text::jinja::events

namespace emel::text::jinja::event {

struct parse {
  std::string_view template_text = {};
  emel::text::jinja::program * program_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  ::emel::callback<bool(const ::emel::text::jinja::events::parsing_done &)>
      dispatch_done = {};
  ::emel::callback<bool(const ::emel::text::jinja::events::parsing_error &)>
      dispatch_error = {};
};

} // namespace emel::text::jinja::event

namespace emel::text::jinja::events {

struct parsing_done {
  const event::parse * request = nullptr;
};

struct parsing_error {
  const event::parse * request = nullptr;
  int32_t err = 0;
};

} // namespace emel::text::jinja::events
