#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/weight_loader/errors.hpp"

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
  emel::error::type err = emel::error::cast(error::none);
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
  std::span<emel::model::data::tensor_record> tensors = {};
  emel::callback<void(const events::bind_done &)> on_done = {};
  emel::callback<void(const events::bind_error &)> on_error = {};

  explicit bind_storage(std::span<emel::model::data::tensor_record> tensors_in) noexcept
      : tensors(tensors_in) {}
};

struct plan_load {
  std::span<effect_request> effects = {};
  emel::callback<void(const events::plan_done &)> on_done = {};
  emel::callback<void(const events::plan_error &)> on_error = {};

  explicit plan_load(std::span<effect_request> effects_in) noexcept : effects(effects_in) {}
};

struct apply_effect_results {
  std::span<const effect_result> results = {};
  emel::callback<void(const events::apply_done &)> on_done = {};
  emel::callback<void(const events::apply_error &)> on_error = {};

  explicit apply_effect_results(std::span<const effect_result> results_in) noexcept
      : results(results_in) {}
};

struct bind_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct bind_runtime {
  const bind_storage & request;
  bind_ctx & ctx;
};

struct plan_ctx {
  emel::error::type err = emel::error::cast(error::none);
  uint32_t effect_count = 0u;
};

struct plan_runtime {
  const plan_load & request;
  plan_ctx & ctx;
};

struct apply_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct apply_runtime {
  const apply_effect_results & request;
  apply_ctx & ctx;
};

}  // namespace event

namespace events {

struct bind_done {
  const event::bind_storage & request;
};

struct bind_error {
  const event::bind_storage & request;
  emel::error::type err = emel::error::cast(error::none);
};

struct plan_done {
  const event::plan_load & request;
  uint32_t effect_count = 0u;
};

struct plan_error {
  const event::plan_load & request;
  emel::error::type err = emel::error::cast(error::none);
};

struct apply_done {
  const event::apply_effect_results & request;
};

struct apply_error {
  const event::apply_effect_results & request;
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace events

}  // namespace emel::model::weight_loader
