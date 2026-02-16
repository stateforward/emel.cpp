#include "emel/emel.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/sm.hpp"

namespace {

using emel_allocator_sm = emel::buffer::allocator::sm;
using emel_graph_view = emel::buffer::allocator::event::graph_view;
using emel_tensor_desc = emel::buffer::allocator::event::tensor_desc;
struct graph_storage {
  std::array<emel_tensor_desc, emel::buffer::allocator::action::k_max_graph_tensors> nodes = {};
  std::array<emel_tensor_desc, emel::buffer::allocator::action::k_max_graph_tensors> leafs = {};
};

emel_status normalize_status(const bool ok, const int32_t err) noexcept {
  if (ok && err == EMEL_OK) {
    return EMEL_OK;
  }
  if (err == EMEL_OK) {
    return EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
  }
  return static_cast<emel_status>(err);
}

bool build_graph_view(
    const emel_buffer_graph_view * input,
    graph_storage & storage,
    emel_graph_view & out,
    int32_t & err) noexcept {
  if (input == nullptr) {
    err = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }
  if (input->n_nodes < 0 || input->n_leafs < 0) {
    err = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }
  if (input->n_nodes > emel::buffer::allocator::action::k_max_graph_tensors ||
      input->n_leafs > emel::buffer::allocator::action::k_max_graph_tensors) {
    err = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }
  if ((input->n_nodes > 0 && input->nodes == nullptr) ||
      (input->n_leafs > 0 && input->leafs == nullptr)) {
    err = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }

  for (int32_t i = 0; i < input->n_nodes; ++i) {
    const emel_buffer_tensor_desc & in = input->nodes[i];
    emel_tensor_desc & dst = storage.nodes[i];
    dst.tensor_id = in.tensor_id;
    dst.alloc_size = in.alloc_size;
    for (int32_t j = 0; j < static_cast<int32_t>(emel::buffer::allocator::event::k_max_sources); ++j) {
      dst.src_ids[j] = in.src_ids[j];
    }
    dst.is_view = in.is_view != 0;
    dst.view_src_id = in.view_src_id;
    dst.is_input = in.is_input != 0;
    dst.is_output = in.is_output != 0;
    dst.can_inplace = in.can_inplace != 0;
    dst.has_external_data = in.has_external_data != 0;
  }

  for (int32_t i = 0; i < input->n_leafs; ++i) {
    const emel_buffer_tensor_desc & in = input->leafs[i];
    emel_tensor_desc & dst = storage.leafs[i];
    dst.tensor_id = in.tensor_id;
    dst.alloc_size = in.alloc_size;
    for (int32_t j = 0; j < static_cast<int32_t>(emel::buffer::allocator::event::k_max_sources); ++j) {
      dst.src_ids[j] = in.src_ids[j];
    }
    dst.is_view = in.is_view != 0;
    dst.view_src_id = in.view_src_id;
    dst.is_input = in.is_input != 0;
    dst.is_output = in.is_output != 0;
    dst.can_inplace = in.can_inplace != 0;
    dst.has_external_data = in.has_external_data != 0;
  }

  out = emel_graph_view{
    .nodes = input->n_nodes > 0 ? storage.nodes.data() : nullptr,
    .n_nodes = input->n_nodes,
    .leafs = input->n_leafs > 0 ? storage.leafs.data() : nullptr,
    .n_leafs = input->n_leafs,
  };
  err = EMEL_OK;
  return true;
}

}  // namespace

struct emel_buffer_allocator {
  emel_allocator_sm sm;
};

static_assert(EMEL_BUFFER_MAX_SOURCES == emel::buffer::allocator::event::k_max_sources);

extern "C" size_t emel_buffer_allocator_storage_size(void) {
  return sizeof(emel_buffer_allocator);
}

extern "C" size_t emel_buffer_allocator_storage_alignment(void) {
  return alignof(emel_buffer_allocator);
}

extern "C" emel_buffer_allocator * emel_buffer_allocator_init(
    void * storage, size_t storage_size) {
  if (storage == nullptr || storage_size < sizeof(emel_buffer_allocator)) {
    return nullptr;
  }
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(storage);
  if (addr % alignof(emel_buffer_allocator) != 0) {
    return nullptr;
  }
  return new (storage) emel_buffer_allocator{};
}

extern "C" void emel_buffer_allocator_destroy(emel_buffer_allocator * allocator) {
  if (allocator == nullptr) {
    return;
  }
  allocator->~emel_buffer_allocator();
}

extern "C" emel_status emel_buffer_allocator_initialize(
    emel_buffer_allocator * allocator,
    int32_t buffer_count,
    const int32_t * buffer_alignments,
    const int32_t * buffer_max_sizes) {
  if (allocator == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  int32_t err = EMEL_OK;
  const bool ok = allocator->sm.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = buffer_count,
    .buffer_alignments = buffer_alignments,
    .buffer_max_sizes = buffer_max_sizes,
    .chunk_allocator_sm = nullptr,
    .error_out = &err,
  });
  return normalize_status(ok, err);
}

extern "C" emel_status emel_buffer_allocator_reserve_n_size(
    emel_buffer_allocator * allocator,
    const emel_buffer_graph_view * graph,
    const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids,
    int32_t * sizes_out,
    int32_t sizes_out_count) {
  if (allocator == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  graph_storage storage{};
  emel_graph_view view{};
  int32_t err = EMEL_OK;
  if (!build_graph_view(graph, storage, view, err)) {
    return static_cast<emel_status>(err);
  }
  const bool ok = allocator->sm.process_event(emel::buffer::allocator::event::reserve_n_size{
    .graph = view,
    .node_buffer_ids = node_buffer_ids,
    .leaf_buffer_ids = leaf_buffer_ids,
    .sizes_out = sizes_out,
    .sizes_out_count = sizes_out_count,
    .buffer_planner_sm = nullptr,
    .chunk_allocator_sm = nullptr,
    .strategy = nullptr,
    .error_out = &err,
  });
  return normalize_status(ok, err);
}

extern "C" emel_status emel_buffer_allocator_reserve_n(
    emel_buffer_allocator * allocator,
    const emel_buffer_graph_view * graph,
    const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids) {
  if (allocator == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  graph_storage storage{};
  emel_graph_view view{};
  int32_t err = EMEL_OK;
  if (!build_graph_view(graph, storage, view, err)) {
    return static_cast<emel_status>(err);
  }
  const bool ok = allocator->sm.process_event(emel::buffer::allocator::event::reserve_n{
    .graph = view,
    .node_buffer_ids = node_buffer_ids,
    .leaf_buffer_ids = leaf_buffer_ids,
    .buffer_planner_sm = nullptr,
    .chunk_allocator_sm = nullptr,
    .strategy = nullptr,
    .error_out = &err,
  });
  return normalize_status(ok, err);
}

extern "C" emel_status emel_buffer_allocator_reserve(
    emel_buffer_allocator * allocator,
    const emel_buffer_graph_view * graph) {
  if (allocator == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  graph_storage storage{};
  emel_graph_view view{};
  int32_t err = EMEL_OK;
  if (!build_graph_view(graph, storage, view, err)) {
    return static_cast<emel_status>(err);
  }
  const bool ok = allocator->sm.process_event(emel::buffer::allocator::event::reserve{
    .graph = view,
    .buffer_planner_sm = nullptr,
    .chunk_allocator_sm = nullptr,
    .strategy = nullptr,
    .error_out = &err,
  });
  return normalize_status(ok, err);
}

extern "C" emel_status emel_buffer_allocator_alloc_graph(
    emel_buffer_allocator * allocator,
    const emel_buffer_graph_view * graph) {
  if (allocator == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  graph_storage storage{};
  emel_graph_view view{};
  int32_t err = EMEL_OK;
  if (!build_graph_view(graph, storage, view, err)) {
    return static_cast<emel_status>(err);
  }
  const bool ok = allocator->sm.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = view,
    .buffer_planner_sm = nullptr,
    .chunk_allocator_sm = nullptr,
    .buffer_realloc_analyzer_sm = nullptr,
    .strategy = nullptr,
    .error_out = &err,
  });
  return normalize_status(ok, err);
}

extern "C" emel_status emel_buffer_allocator_release(emel_buffer_allocator * allocator) {
  if (allocator == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  int32_t err = EMEL_OK;
  const bool ok = allocator->sm.process_event(emel::buffer::allocator::event::release{
    .chunk_allocator_sm = nullptr,
    .error_out = &err,
  });
  return normalize_status(ok, err);
}

extern "C" emel_status emel_buffer_allocator_alloc_tensors_from_buft_size(
    emel_buffer_allocator * allocator,
    const emel_buffer_graph_view * graph,
    const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids,
    int32_t * sizes_out,
    int32_t sizes_out_count) {
  return emel_buffer_allocator_reserve_n_size(
    allocator, graph, node_buffer_ids, leaf_buffer_ids, sizes_out, sizes_out_count);
}

extern "C" emel_status emel_buffer_allocator_alloc_tensors_from_buft(
    emel_buffer_allocator * allocator,
    const emel_buffer_graph_view * graph,
    const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids) {
  const emel_status reserve_status =
    emel_buffer_allocator_reserve_n(allocator, graph, node_buffer_ids, leaf_buffer_ids);
  if (reserve_status != EMEL_OK) {
    return reserve_status;
  }
  return emel_buffer_allocator_alloc_graph(allocator, graph);
}

extern "C" emel_status emel_buffer_allocator_alloc_tensors(
    emel_buffer_allocator * allocator,
    const emel_buffer_graph_view * graph) {
  const emel_status reserve_status = emel_buffer_allocator_reserve(allocator, graph);
  if (reserve_status != EMEL_OK) {
    return reserve_status;  // GCOVR_EXCL_LINE
  }
  return emel_buffer_allocator_alloc_graph(allocator, graph);
}

extern "C" int32_t emel_buffer_allocator_buffer_size(
    const emel_buffer_allocator * allocator, int32_t buffer_id) {
  if (allocator == nullptr) {
    return 0;
  }
  return allocator->sm.get_buffer_size(buffer_id);
}

extern "C" int32_t emel_buffer_allocator_buffer_chunk_id(
    const emel_buffer_allocator * allocator, int32_t buffer_id) {
  if (allocator == nullptr) {
    return -1;
  }
  return allocator->sm.get_buffer_chunk_id(buffer_id);
}

extern "C" uint64_t emel_buffer_allocator_buffer_chunk_offset(
    const emel_buffer_allocator * allocator, int32_t buffer_id) {
  if (allocator == nullptr) {
    return 0;
  }
  return allocator->sm.get_buffer_chunk_offset(buffer_id);
}

extern "C" uint64_t emel_buffer_allocator_buffer_alloc_size(
    const emel_buffer_allocator * allocator, int32_t buffer_id) {
  if (allocator == nullptr) {
    return 0;
  }
  return allocator->sm.get_buffer_alloc_size(buffer_id);
}
