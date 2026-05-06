#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/events.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"

namespace emel::io::loader {

struct sm;

} // namespace emel::io::loader

namespace emel::model::loader::events {

struct tensor_bind_done;
struct tensor_bind_error;
struct tensor_plan_done;
struct tensor_plan_error;
struct tensor_apply_done;
struct tensor_apply_error;
struct io_load_done;
struct io_load_error;
struct load_done;
struct load_error;

} // namespace emel::model::loader::events

namespace emel::model::loader::event {

struct load;

using parse_model_fn = emel::callback<emel::error::type(const load &)>;
using map_layers_fn = emel::callback<emel::error::type(const load &)>;
using validate_structure_fn = emel::callback<emel::error::type(const load &)>;
using validate_architecture_fn =
    emel::callback<emel::error::type(const load &)>;

struct load {
  emel::model::data &model_data;
  parse_model_fn &parse_model;

  std::string_view model_path = {};
  const void *file_image = nullptr;
  uint64_t file_size = 0;
  bool check_tensors = true;
  bool vocab_only = false;
  bool validate_architecture = true;

  emel::model::tensor::sm *tensor_loader = nullptr;
  emel::io::loader::sm *io_loader = nullptr;
  emel::io::loader::event::strategy_kind io_strategy =
      emel::io::loader::event::strategy_kind::none;
  std::span<emel::model::tensor::effect_request> effect_requests = {};
  std::span<emel::model::tensor::effect_result> effect_results = {};
  std::span<emel::io::event::tensor_load_span> io_load_spans = {};
  map_layers_fn map_layers = {};
  validate_structure_fn validate_structure = {};
  validate_architecture_fn validate_architecture_impl = {};

  emel::callback<void(const events::load_done &)> on_done = {};
  emel::callback<void(const events::load_error &)> on_error = {};

  load(emel::model::data &model_data_in,
       parse_model_fn &parse_model_in) noexcept
      : model_data(model_data_in), parse_model(parse_model_in) {}
};

struct load_ctx {
  emel::error::type err = emel::error::cast(error::none);
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  emel::io::loader::event::strategy_kind used_io_strategy =
      emel::io::loader::event::strategy_kind::none;
};

struct tensor_phase_events {
  events::tensor_bind_done &bind_done;
  events::tensor_bind_error &bind_error;
  events::tensor_plan_done &plan_done;
  events::tensor_plan_error &plan_error;
  events::tensor_apply_done &apply_done;
  events::tensor_apply_error &apply_error;
};

struct io_phase_events {
  events::io_load_done &load_done;
  events::io_load_error &load_error;
};

struct load_runtime {
  const load &request;
  load_ctx &ctx;
  mutable tensor_phase_events tensor_events;
  mutable io_phase_events *io_events = nullptr;
};

} // namespace emel::model::loader::event

namespace emel::model::loader::events {

struct tensor_bind_done {
  bool raised = false;
};

struct tensor_bind_error {
  bool raised = false;
  emel::error::type err = emel::error::cast(emel::model::tensor::error::none);
};

struct tensor_plan_done {
  bool raised = false;
  uint32_t effect_count = 0u;
};

struct tensor_plan_error {
  bool raised = false;
  emel::error::type err = emel::error::cast(emel::model::tensor::error::none);
};

struct tensor_apply_done {
  bool raised = false;
};

struct tensor_apply_error {
  bool raised = false;
  emel::error::type err = emel::error::cast(emel::model::tensor::error::none);
};

struct io_load_done {
  bool raised = false;
  uint32_t expected_count = 0u;
  uint32_t done_count = 0u;
  uint64_t bytes_done = 0u;
};

struct io_load_error {
  bool raised = false;
  emel::error::type err = emel::error::cast(emel::io::loader::error::none);
  emel::error::type strategy_err =
      emel::error::cast(emel::io::loader::error::none);
  uint32_t failed_index = 0u;
};

struct load_done {
  const event::load &request;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  emel::io::loader::event::strategy_kind used_io_strategy =
      emel::io::loader::event::strategy_kind::none;
};

struct load_error {
  const event::load &request;
  emel::error::type err = emel::error::cast(error::none);
  emel::io::loader::event::strategy_kind requested_io_strategy =
      emel::io::loader::event::strategy_kind::none;
  emel::io::loader::event::strategy_kind used_io_strategy =
      emel::io::loader::event::strategy_kind::none;
};

} // namespace emel::model::loader::events
