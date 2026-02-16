#pragma once

#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::model::parser::event {
struct parse_model;
}  // namespace emel::model::parser::event

namespace emel::model::weight_loader::event {
struct load_weights;
}  // namespace emel::model::weight_loader::event

namespace emel::model::loader::event {

struct load;

}  // namespace emel::model::loader::event

namespace emel::model::loader::events {

struct mapping_parser_done {
  const event::load * request = nullptr;
};

struct mapping_parser_error {
  const event::load * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct parsing_done {
  const event::load * request = nullptr;
};

struct parsing_error {
  const event::load * request = nullptr;
  int32_t err = EMEL_ERR_PARSE_FAILED;
};

struct loading_done {
  const event::load * request = nullptr;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

struct loading_error {
  const event::load * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct layers_mapped {
  const event::load * request = nullptr;
};

struct layers_map_error {
  const event::load * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

struct structure_validated {
  const event::load * request = nullptr;
};

struct structure_error {
  const event::load * request = nullptr;
  int32_t err = EMEL_ERR_MODEL_INVALID;
};

struct architecture_validated {
  const event::load * request = nullptr;
};

struct architecture_error {
  const event::load * request = nullptr;
  int32_t err = EMEL_ERR_MODEL_INVALID;
};

struct load_done {
  const event::load * request = nullptr;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

struct load_error {
  const event::load * request = nullptr;
  int32_t err = EMEL_ERR_BACKEND;
};

}  // namespace emel::model::loader::events

namespace emel::model::loader::event {

using map_parser_fn = bool (*)(const load &, int32_t * err_out);
using map_layers_fn = bool (*)(const load &, int32_t * err_out);
using validate_structure_fn = bool (*)(const load &, int32_t * err_out);
using validate_architecture_fn = bool (*)(const load &, int32_t * err_out);

struct load {
  emel::model::data & model_data;
  std::string_view model_path = {};
  bool request_mmap = true;
  bool request_direct_io = false;
  bool check_tensors = true;
  bool vocab_only = false;
  bool no_alloc = false;
  bool validate_architecture = true;
  bool mmap_supported = true;
  bool direct_io_supported = false;
  const void * architectures = nullptr;
  int32_t n_architectures = 0;
  void * weights_buffer = nullptr;
  uint64_t weights_buffer_size = 0;
  void * file_handle = nullptr;
  void * format_ctx = nullptr;
  bool (*progress_callback)(float progress, void * user_data) = nullptr;
  void * progress_user_data = nullptr;

  map_parser_fn map_parser = nullptr;
  bool (*parse_architecture)(const emel::model::parser::event::parse_model &, int32_t * err_out) = nullptr;
  bool (*map_architecture)(const emel::model::parser::event::parse_model &, int32_t * err_out) = nullptr;
  bool (*parse_hparams)(const emel::model::parser::event::parse_model &, int32_t * err_out) = nullptr;
  bool (*parse_vocab)(const emel::model::parser::event::parse_model &, int32_t * err_out) = nullptr;
  bool (*map_tensors)(const emel::model::parser::event::parse_model &, int32_t * err_out) = nullptr;
  bool (*map_mmap)(const emel::model::weight_loader::event::load_weights &,
                   uint64_t * bytes_done,
                   uint64_t * bytes_total,
                   int32_t * err_out) = nullptr;
  bool (*load_streamed)(const emel::model::weight_loader::event::load_weights &,
                        uint64_t * bytes_done,
                        uint64_t * bytes_total,
                        int32_t * err_out) = nullptr;
  map_layers_fn map_layers = nullptr;
  validate_structure_fn validate_structure = nullptr;
  validate_architecture_fn validate_architecture_impl = nullptr;

  void * parser_sm = nullptr;
  bool (*dispatch_parse_model)(void * parser_sm,
                               const emel::model::parser::event::parse_model &) = nullptr;
  void * weight_loader_sm = nullptr;
  bool (*dispatch_load_weights)(void * weight_loader_sm,
                                const emel::model::weight_loader::event::load_weights &) = nullptr;
  void * buffer_allocator_sm = nullptr;
  void * loader_sm = nullptr;
  bool (*dispatch_parsing_done)(void * loader_sm,
                                const emel::model::loader::events::parsing_done &) = nullptr;
  bool (*dispatch_parsing_error)(void * loader_sm,
                                 const emel::model::loader::events::parsing_error &) = nullptr;
  bool (*dispatch_loading_done)(void * loader_sm,
                                const emel::model::loader::events::loading_done &) = nullptr;
  bool (*dispatch_loading_error)(void * loader_sm,
                                 const emel::model::loader::events::loading_error &) = nullptr;

  int32_t * error_out = nullptr;

  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::load_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::load_error &) = nullptr;
};

}  // namespace emel::model::loader::event
