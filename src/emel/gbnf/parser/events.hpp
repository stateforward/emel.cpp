#pragma once

#include "emel/callback.hpp"
#include "emel/gbnf/types.hpp"
#include <cstdint>
#include <string_view>

namespace emel::gbnf::events {

struct parsing_done;
struct parsing_error;

} // namespace emel::gbnf::events

namespace emel::gbnf::event {

struct parse {
  std::string_view grammar_text = {};
  emel::gbnf::grammar *grammar_out = nullptr;
  int32_t *error_out = nullptr;
  void *owner_sm = nullptr;
  ::emel::callback<bool(const ::emel::gbnf::events::parsing_done &)>
      dispatch_done = {};
  ::emel::callback<bool(const ::emel::gbnf::events::parsing_error &)>
      dispatch_error = {};
};

} // namespace emel::gbnf::event

namespace emel::gbnf::events {

struct parsing_done {
  const event::parse *request = nullptr;
};

struct parsing_error {
  const event::parse *request = nullptr;
  int32_t err = 0;
};

} // namespace emel::gbnf::events
