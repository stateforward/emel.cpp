#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::model::weight_loader {

enum class effect_kind : uint8_t {
  k_none = 0,
};

struct effect_request {
  effect_kind kind = effect_kind::k_none;
  uint64_t offset = 0;
  uint64_t size = 0;
  void * target = nullptr;
};

struct effect_result {
  effect_kind kind = effect_kind::k_none;
  void * handle = nullptr;
  int32_t err = EMEL_OK;
};

namespace events {
struct bind_done;
struct bind_error;
struct plan_done;
struct plan_error;
struct apply_done;
struct apply_error;
}  // namespace events

namespace event {

struct bind_storage {
  emel::model::data::tensor_record * tensors = nullptr;
  uint32_t tensor_count = 0;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::bind_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::bind_error &) = nullptr;
};

struct plan_load {
  effect_request * effects_out = nullptr;
  uint32_t effect_capacity = 0;
  uint32_t * effect_count_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::plan_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::plan_error &) = nullptr;
};

struct apply_effect_results {
  const effect_result * results = nullptr;
  uint32_t result_count = 0;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::apply_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::apply_error &) = nullptr;
};

}  // namespace event

namespace events {

struct bind_done {
  const event::bind_storage * request = nullptr;
};

struct bind_error {
  const event::bind_storage * request = nullptr;
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
};

struct plan_done {
  const event::plan_load * request = nullptr;
  uint32_t effect_count = 0;
};

struct plan_error {
  const event::plan_load * request = nullptr;
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
};

struct apply_done {
  const event::apply_effect_results * request = nullptr;
};

struct apply_error {
  const event::apply_effect_results * request = nullptr;
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
};

}  // namespace events

}  // namespace emel::model::weight_loader
