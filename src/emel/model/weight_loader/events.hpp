#pragma once

namespace emel::buffer_allocator {
struct sm;
}  // namespace emel::buffer_allocator

namespace emel::model::loader {
struct sm;
}  // namespace emel::model::loader

namespace emel::model::weight_loader::event {

struct load_weights {
  bool request_mmap = true;
  bool request_direct_io = false;
  bool check_tensors = true;
  bool no_alloc = false;
  bool mmap_supported = true;
  bool direct_io_supported = false;
  emel::buffer_allocator::sm * buffer_allocator_sm = nullptr;
  emel::model::loader::sm * model_loader_sm = nullptr;
};

struct transport_selected {};
struct weights_loaded {};

}  // namespace emel::model::weight_loader::event

namespace emel::model::weight_loader::events {

struct loading_done {};
struct loading_error {};

}  // namespace emel::model::weight_loader::events
