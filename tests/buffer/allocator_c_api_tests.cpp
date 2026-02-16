#include <array>
#include <cstddef>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"

namespace {

emel_buffer_graph_view make_valid_graph(
    std::array<emel_buffer_tensor_desc, 2> & nodes,
    std::array<emel_buffer_tensor_desc, 1> & leafs) {
  leafs[0] = emel_buffer_tensor_desc{
    .tensor_id = 100,
    .alloc_size = 128,
    .src_ids = {-1, -1, -1, -1},
    .is_view = 0,
    ._pad0 = {0, 0, 0},
    .view_src_id = -1,
    .is_input = 1,
    .is_output = 0,
    .can_inplace = 1,
    .has_external_data = 0,
  };
  nodes[0] = emel_buffer_tensor_desc{
    .tensor_id = 200,
    .alloc_size = 256,
    .src_ids = {100, -1, -1, -1},
    .is_view = 0,
    ._pad0 = {0, 0, 0},
    .view_src_id = -1,
    .is_input = 0,
    .is_output = 0,
    .can_inplace = 1,
    .has_external_data = 0,
  };
  nodes[1] = emel_buffer_tensor_desc{
    .tensor_id = 201,
    .alloc_size = 512,
    .src_ids = {200, -1, -1, -1},
    .is_view = 0,
    ._pad0 = {0, 0, 0},
    .view_src_id = -1,
    .is_input = 0,
    .is_output = 1,
    .can_inplace = 0,
    .has_external_data = 0,
  };

  return emel_buffer_graph_view{
    .nodes = nodes.data(),
    .n_nodes = 2,
    .leafs = leafs.data(),
    .n_leafs = 1,
  };
}

}  // namespace

TEST_CASE("buffer_allocator_c_api_reports_invalid_arguments") {
  alignas(64) static std::array<std::byte, 524288> storage = {};
  CHECK(emel_buffer_allocator_storage_size() <= storage.size());
  const size_t alignment = emel_buffer_allocator_storage_alignment();
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(storage.data());
  CHECK((addr % alignment) == 0);

  emel_buffer_allocator * allocator =
    emel_buffer_allocator_init(storage.data(), storage.size());
  REQUIRE(allocator != nullptr);

  CHECK(emel_buffer_allocator_initialize(allocator, 0, nullptr, nullptr) ==
        EMEL_ERR_INVALID_ARGUMENT);

  CHECK(emel_buffer_allocator_reserve_n_size(
          allocator, nullptr, nullptr, nullptr, nullptr, 0) ==
        EMEL_ERR_INVALID_ARGUMENT);

  emel_buffer_allocator_destroy(allocator);
}

TEST_CASE("buffer_allocator_c_api_alloc_tensors_wrappers") {
  alignas(64) static std::array<std::byte, 524288> storage = {};
  emel_buffer_allocator * allocator =
    emel_buffer_allocator_init(storage.data(), storage.size());
  REQUIRE(allocator != nullptr);

  std::array<int32_t, 1> alignments = {{16}};
  std::array<int32_t, 1> max_sizes = {{0}};
  REQUIRE(emel_buffer_allocator_initialize(
            allocator,
            1,
            alignments.data(),
            max_sizes.data()) == EMEL_OK);

  std::array<emel_buffer_tensor_desc, 2> nodes = {};
  std::array<emel_buffer_tensor_desc, 1> leafs = {};
  emel_buffer_graph_view graph = make_valid_graph(nodes, leafs);

  std::array<int32_t, 1> sizes = {{0}};
  CHECK(emel_buffer_allocator_alloc_tensors_from_buft_size(
          allocator,
          &graph,
          nullptr,
          nullptr,
          sizes.data(),
          static_cast<int32_t>(sizes.size())) == EMEL_OK);
  CHECK(sizes[0] > 0);

  CHECK(emel_buffer_allocator_alloc_tensors_from_buft(
          allocator,
          &graph,
          nullptr,
          nullptr) == EMEL_OK);

  CHECK(emel_buffer_allocator_alloc_tensors(allocator, &graph) == EMEL_OK);

  emel_buffer_allocator_destroy(allocator);
}
