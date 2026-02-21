#pragma once

#include <cstdint>

namespace emel::memory::coordinator::event {

enum class memory_status : int32_t {
  success = 0,
  no_update = 1,
  failed_prepare = 2,
  failed_compute = 3,
};

struct prepare_update;
struct prepare_batch;
struct prepare_full;

using prepare_update_fn =
  memory_status (*)(const prepare_update & request, void * user_data, int32_t * err_out) noexcept;
using prepare_batch_fn =
  memory_status (*)(const prepare_batch & request, void * user_data, int32_t * err_out) noexcept;
using prepare_full_fn =
  memory_status (*)(const prepare_full & request, void * user_data, int32_t * err_out) noexcept;

struct prepare_update {
  bool optimize = false;
  memory_status * status_out = nullptr;
  int32_t * error_out = nullptr;
  prepare_update_fn prepare_fn = nullptr;
  void * prepare_ctx = nullptr;
};

struct prepare_batch {
  int32_t n_ubatch = 0;
  int32_t n_ubatches_total = 0;
  memory_status * status_out = nullptr;
  int32_t * error_out = nullptr;
  prepare_batch_fn prepare_fn = nullptr;
  void * prepare_ctx = nullptr;
};

struct prepare_full {
  memory_status * status_out = nullptr;
  int32_t * error_out = nullptr;
  prepare_full_fn prepare_fn = nullptr;
  void * prepare_ctx = nullptr;
};

struct validate_update {
  const prepare_update * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_batch {
  const prepare_batch * request = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_full {
  const prepare_full * request = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_update_step {
  const prepare_update * request = nullptr;
  memory_status * prepared_status_out = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_batch_step {
  const prepare_batch * request = nullptr;
  memory_status * prepared_status_out = nullptr;
  int32_t * error_out = nullptr;
};

struct prepare_full_step {
  const prepare_full * request = nullptr;
  memory_status * prepared_status_out = nullptr;
  int32_t * error_out = nullptr;
};

struct apply_update_step {
  const prepare_update * request = nullptr;
  memory_status prepared_status = memory_status::success;
  int32_t * error_out = nullptr;
};

struct publish_update {
  const prepare_update * request = nullptr;
  memory_status prepared_status = memory_status::success;
  int32_t * error_out = nullptr;
};

struct publish_batch {
  const prepare_batch * request = nullptr;
  memory_status prepared_status = memory_status::success;
  int32_t * error_out = nullptr;
};

struct publish_full {
  const prepare_full * request = nullptr;
  memory_status prepared_status = memory_status::success;
  int32_t * error_out = nullptr;
};

}  // namespace emel::memory::coordinator::event

namespace emel::memory::coordinator::events {

struct validate_done {
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};

struct prepare_done {
  event::memory_status prepared_status = event::memory_status::success;
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};
struct prepare_error {
  int32_t err = 0;
  event::memory_status prepared_status = event::memory_status::failed_prepare;
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};

struct apply_done {
  event::memory_status prepared_status = event::memory_status::success;
  const event::prepare_update * update_request = nullptr;
};
struct apply_error {
  int32_t err = 0;
  event::memory_status prepared_status = event::memory_status::failed_prepare;
  const event::prepare_update * update_request = nullptr;
};

struct publish_done {
  event::memory_status prepared_status = event::memory_status::success;
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};
struct publish_error {
  int32_t err = 0;
  event::memory_status prepared_status = event::memory_status::failed_prepare;
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};

struct memory_done {
  event::memory_status status = event::memory_status::success;
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};

struct memory_error {
  int32_t err = 0;
  event::memory_status status = event::memory_status::failed_prepare;
  const event::prepare_update * update_request = nullptr;
  const event::prepare_batch * batch_request = nullptr;
  const event::prepare_full * full_request = nullptr;
};

}  // namespace emel::memory::coordinator::events
