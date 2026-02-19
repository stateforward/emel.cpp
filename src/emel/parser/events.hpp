#pragma once

#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::model::loader::event {
struct load;
}  // namespace emel::model::loader::event

namespace emel::model::loader::events {
struct parsing_done;
struct parsing_error;
}  // namespace emel::model::loader::events

namespace emel::parser::event {
struct parse_model;
}  // namespace emel::parser::event

namespace emel::parser::events {

struct parsing_done {
  const event::parse_model * request = nullptr;
};

struct parsing_error {
  const event::parse_model * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

}  // namespace emel::parser::events

namespace emel::parser::event {

struct parse_model {
  emel::model::data * model = nullptr;
  std::string_view model_path = {};
  const void * architectures = nullptr;
  int32_t n_architectures = 0;
  void * file_handle = nullptr;
  void * format_ctx = nullptr;
  bool map_tensors = true;

  const emel::model::loader::event::load * loader_request = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const emel::model::loader::events::parsing_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const emel::model::loader::events::parsing_error &) = nullptr;
};

}  // namespace emel::parser::event
