#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::loader::events {

struct load_done;
struct load_error;

}  // namespace emel::model::loader::events

namespace emel::model::loader::event {

struct load;

using parse_model_fn = emel::callback<emel::error::type(const load &)>;
using load_weights_fn = emel::callback<
  emel::error::type(const load &, uint64_t &, uint64_t &, bool &)>;
using map_layers_fn = emel::callback<emel::error::type(const load &)>;
using validate_structure_fn = emel::callback<emel::error::type(const load &)>;
using validate_architecture_fn = emel::callback<emel::error::type(const load &)>;

struct load {
  emel::model::data & model_data;
  parse_model_fn & parse_model;

  std::string_view model_path = {};
  const void * file_image = nullptr;
  uint64_t file_size = 0;
  bool check_tensors = true;
  bool vocab_only = false;
  bool validate_architecture = true;

  load_weights_fn load_weights = {};
  map_layers_fn map_layers = {};
  validate_structure_fn validate_structure = {};
  validate_architecture_fn validate_architecture_impl = {};

  emel::callback<void(const events::load_done &)> on_done = {};
  emel::callback<void(const events::load_error &)> on_error = {};

  load(emel::model::data & model_data_in, parse_model_fn & parse_model_in) noexcept
      : model_data(model_data_in), parse_model(parse_model_in) {}
};

struct load_ctx {
  emel::error::type err = emel::error::cast(error::none);
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

struct load_runtime {
  const load & request;
  load_ctx & ctx;
};

}  // namespace emel::model::loader::event

namespace emel::model::loader::events {

struct load_done {
  const event::load & request;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

struct load_error {
  const event::load & request;
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::model::loader::events
