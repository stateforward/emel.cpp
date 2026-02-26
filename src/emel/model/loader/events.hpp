#pragma once

#include <cstdint>
#include <string_view>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/parser/gguf/events.hpp"

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

using map_layers_fn = bool (*)(const load &, int32_t * err_out);
using validate_structure_fn = bool (*)(const load &, int32_t * err_out);
using validate_architecture_fn = bool (*)(const load &, int32_t * err_out);

struct load {
  emel::model::data & model_data;
  std::string_view model_path = {};
  const void * file_image = nullptr;
  uint64_t file_size = 0;
  bool check_tensors = true;
  bool vocab_only = false;
  bool validate_architecture = true;

  void * parser_sm = nullptr;
  bool (*dispatch_probe)(void * parser_sm, const emel::parser::gguf::event::probe &) = nullptr;
  bool (*dispatch_bind_storage)(void * parser_sm,
                                const emel::parser::gguf::event::bind_storage &) = nullptr;
  bool (*dispatch_parse)(void * parser_sm, const emel::parser::gguf::event::parse &) = nullptr;
  void * parser_kv_arena = nullptr;
  uint64_t parser_kv_arena_size = 0;
  void * parser_kv_entries = nullptr;
  uint32_t parser_kv_entry_capacity = 0;
  emel::model::data::tensor_record * parser_tensors = nullptr;
  uint32_t parser_tensor_capacity = 0;

  void * weight_loader_sm = nullptr;
  bool (*dispatch_bind_weights)(void * weight_loader_sm,
                                const emel::model::weight_loader::event::bind_storage &) = nullptr;
  bool (*dispatch_plan_load)(void * weight_loader_sm,
                             const emel::model::weight_loader::event::plan_load &) = nullptr;
  bool (*dispatch_apply_results)(void * weight_loader_sm,
                                 const emel::model::weight_loader::event::apply_effect_results &) =
    nullptr;
  emel::model::weight_loader::effect_request * effect_requests = nullptr;
  uint32_t effect_capacity = 0;
  emel::model::weight_loader::effect_result * effect_results = nullptr;
  uint32_t effect_result_capacity = 0;

  map_layers_fn map_layers = nullptr;
  validate_structure_fn validate_structure = nullptr;
  validate_architecture_fn validate_architecture_impl = nullptr;

  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::load_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::load_error &) = nullptr;
};

}  // namespace emel::model::loader::event
