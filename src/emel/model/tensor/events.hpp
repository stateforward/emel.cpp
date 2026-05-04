#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/tensor/errors.hpp"

namespace emel::model::tensor::event {

enum class lifecycle : uint8_t {
  unbound = 0u,
  resident = 1u,
  evicted = 2u,
  internal_error = 3u,
};

struct tensor_state {
  lifecycle lifecycle_state = lifecycle::unbound;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  uint64_t file_offset = 0u;
  uint64_t data_size = 0u;
  uint16_t file_index = 0u;
  int32_t tensor_type = 0;
};

enum class effect_kind : uint8_t {
  k_none = 0,
  k_io_load = 1,
};

struct effect_request {
  effect_kind kind = effect_kind::k_none;
  emel::io::loader::event::strategy_kind strategy =
      emel::io::loader::event::strategy_kind::none;
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t offset = 0;
  uint64_t size = 0;
  void *target = nullptr;
};

struct effect_result {
  effect_kind kind = effect_kind::k_none;
  void *handle = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct bind_storage;
struct plan_load;
struct apply_effect_results;

struct bind_tensor {
  int32_t tensor_id = 0;
  const emel::model::data::tensor_record &tensor_record;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  int32_t *error_out = nullptr;

  bind_tensor(const int32_t tensor_id_in,
              const emel::model::data::tensor_record &tensor_record_in,
              const void *buffer_in, const uint64_t buffer_bytes_in) noexcept
      : tensor_id(tensor_id_in), tensor_record(tensor_record_in),
        buffer(buffer_in), buffer_bytes(buffer_bytes_in) {}
};

struct evict_tensor {
  int32_t tensor_id = 0;
  int32_t *error_out = nullptr;
};

struct capture_tensor_state {
  int32_t tensor_id = 0;
  tensor_state *state_out = nullptr;
  int32_t *error_out = nullptr;
};

} // namespace emel::model::tensor::event

namespace emel::model::tensor::events {

struct bind_storage_done;
struct bind_storage_error;
struct plan_load_done;
struct plan_load_error;
struct apply_effect_results_done;
struct apply_effect_results_error;

struct bind_tensor_done {
  const event::bind_tensor *request = nullptr;
};
struct bind_tensor_error {
  int32_t err = 0;
  const event::bind_tensor *request = nullptr;
};

struct evict_tensor_done {
  const event::evict_tensor *request = nullptr;
};
struct evict_tensor_error {
  int32_t err = 0;
  const event::evict_tensor *request = nullptr;
};

struct capture_tensor_state_done {
  const event::capture_tensor_state *request = nullptr;
};
struct capture_tensor_state_error {
  int32_t err = 0;
  const event::capture_tensor_state *request = nullptr;
};

} // namespace emel::model::tensor::events

namespace emel::model::tensor::event {

struct bind_storage {
  std::span<emel::model::data::tensor_record> tensors = {};
  emel::callback<void(const events::bind_storage_done &)> on_done = {};
  emel::callback<void(const events::bind_storage_error &)> on_error = {};

  explicit bind_storage(
      std::span<emel::model::data::tensor_record> tensors_in) noexcept
      : tensors(tensors_in) {}
};

struct plan_load {
  std::span<effect_request> effects = {};
  emel::io::loader::event::strategy_kind strategy =
      emel::io::loader::event::strategy_kind::none;
  emel::callback<void(const events::plan_load_done &)> on_done = {};
  emel::callback<void(const events::plan_load_error &)> on_error = {};

  explicit plan_load(std::span<effect_request> effects_in) noexcept
      : effects(effects_in) {}
};

struct apply_effect_results {
  std::span<const effect_result> results = {};
  std::span<emel::model::data::tensor_record> tensors = {};
  emel::callback<void(const events::apply_effect_results_done &)> on_done = {};
  emel::callback<void(const events::apply_effect_results_error &)> on_error =
      {};

  explicit apply_effect_results(
      std::span<const effect_result> results_in) noexcept
      : results(results_in) {}

  apply_effect_results(
      std::span<const effect_result> results_in,
      std::span<emel::model::data::tensor_record> tensors_in) noexcept
      : results(results_in), tensors(tensors_in) {}
};

} // namespace emel::model::tensor::event

namespace emel::model::tensor::events {

struct bind_storage_done {
  const event::bind_storage &request;
};

struct bind_storage_error {
  const event::bind_storage &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct plan_load_done {
  const event::plan_load &request;
  uint32_t effect_count = 0u;
};

struct plan_load_error {
  const event::plan_load &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct apply_effect_results_done {
  const event::apply_effect_results &request;
};

struct apply_effect_results_error {
  const event::apply_effect_results &request;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::model::tensor::events

namespace emel::model::tensor {

using effect_kind = event::effect_kind;
using effect_request = event::effect_request;
using effect_result = event::effect_result;

} // namespace emel::model::tensor

namespace emel::model::tensor::events {

using bind_done = bind_storage_done;
using bind_error = bind_storage_error;
using plan_done = plan_load_done;
using plan_error = plan_load_error;
using apply_done = apply_effect_results_done;
using apply_error = apply_effect_results_error;

} // namespace emel::model::tensor::events
