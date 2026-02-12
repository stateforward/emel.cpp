#pragma once

#include <cstdint>

#include "emel/model/data.hpp"

namespace emel::model::loader {
struct sm;
}  // namespace emel::model::loader

namespace emel::model::parser::event {

struct parse_model {
  emel::model::data * model = nullptr;
  const void * architectures = nullptr;
  int32_t n_architectures = 0;
  void * file_handle = nullptr;
  void * format_ctx = nullptr;
  emel::model::loader::sm * model_loader_sm = nullptr;
};

struct parse_architecture_done {};
struct parse_architecture_error {};
struct map_architecture_done {};
struct map_architecture_error {};
struct parse_hparams_done {};
struct parse_hparams_error {};
struct parse_vocab_done {};
struct parse_vocab_error {};
struct map_tensors_done {};
struct map_tensors_error {};

}  // namespace emel::model::parser::event

namespace emel::model::parser::events {

struct parsing_done {};
struct parsing_error {};

}  // namespace emel::model::parser::events
