#pragma once

#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/model/loader/events.hpp"

namespace emel::model::parser::event {
struct parse_model;
}  // namespace emel::model::parser::event

namespace emel::model::parser::events {

struct parse_architecture_done {
  const event::parse_model * request = nullptr;
};

struct parse_architecture_error {
  const event::parse_model * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct map_architecture_done {
  const event::parse_model * request = nullptr;
};

struct map_architecture_error {
  const event::parse_model * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct parse_hparams_done {
  const event::parse_model * request = nullptr;
};

struct parse_hparams_error {
  const event::parse_model * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct parse_vocab_done {
  const event::parse_model * request = nullptr;
};

struct parse_vocab_error {
  const event::parse_model * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct map_tensors_done {
  const event::parse_model * request = nullptr;
};

struct map_tensors_error {
  const event::parse_model * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct parsing_done {
  const event::parse_model * request = nullptr;
};

struct parsing_error {
  const event::parse_model * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

}  // namespace emel::model::parser::events

namespace emel::model::parser::event {

struct parse_model;

using parse_architecture_fn = bool (*)(const parse_model &, int32_t * err_out);
using map_architecture_fn = bool (*)(const parse_model &, int32_t * err_out);
using parse_hparams_fn = bool (*)(const parse_model &, int32_t * err_out);
using parse_vocab_fn = bool (*)(const parse_model &, int32_t * err_out);
using map_tensors_fn = bool (*)(const parse_model &, int32_t * err_out);

struct parse_model {
  emel::model::data * model = nullptr;
  std::string_view model_path = {};
  const void * architectures = nullptr;
  int32_t n_architectures = 0;
  void * file_handle = nullptr;
  void * format_ctx = nullptr;
  bool map_tensors = true;

  parse_architecture_fn parse_architecture = nullptr;
  map_architecture_fn map_architecture = nullptr;
  parse_hparams_fn parse_hparams = nullptr;
  parse_vocab_fn parse_vocab = nullptr;
  map_tensors_fn map_tensors_impl = nullptr;

  const emel::model::loader::event::load * loader_request = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const emel::model::loader::events::parsing_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const emel::model::loader::events::parsing_error &) = nullptr;
};

}  // namespace emel::model::parser::event
