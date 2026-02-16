#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"

namespace emel::model::weight_loader::event {
struct load_weights;
}  // namespace emel::model::weight_loader::event

namespace emel::model::weight_loader::events {

struct weights_loaded {
  const event::load_weights * request = nullptr;
  int32_t err = EMEL_OK;
  bool used_mmap = false;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
};

}  // namespace emel::model::weight_loader::events

namespace emel::model::weight_loader::event {

struct load_weights;

using map_mmap_fn = bool (*)(const load_weights &,
                             uint64_t * bytes_done,
                             uint64_t * bytes_total,
                             int32_t * err_out);
using load_streamed_fn = bool (*)(const load_weights &,
                                  uint64_t * bytes_done,
                                  uint64_t * bytes_total,
                                  int32_t * err_out);

struct load_weights {
  bool request_mmap = true;
  bool request_direct_io = false;
  bool check_tensors = true;
  bool no_alloc = false;
  bool mmap_supported = true;
  bool direct_io_supported = false;

  void * buffer_allocator_sm = nullptr;

  map_mmap_fn map_mmap = nullptr;
  load_streamed_fn load_streamed = nullptr;
  bool (*progress_callback)(float progress, void * user_data) = nullptr;
  void * progress_user_data = nullptr;

  const emel::model::loader::event::load * loader_request = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const emel::model::loader::events::loading_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const emel::model::loader::events::loading_error &) = nullptr;
};

}  // namespace emel::model::weight_loader::event
