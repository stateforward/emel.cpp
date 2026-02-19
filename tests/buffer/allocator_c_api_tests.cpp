#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/buffer/allocator/actions.hpp"

namespace {

struct aligned_storage {
  std::unique_ptr<std::byte, decltype(&std::free)> data{nullptr, &std::free};
  size_t size = 0;
  size_t alignment = 0;
};

aligned_storage make_allocator_storage() {
  const size_t size = emel_buffer_allocator_storage_size();
  const size_t alignment = emel_buffer_allocator_storage_alignment();
  const size_t padded = (size + alignment - 1) / alignment * alignment;
  void * raw = std::aligned_alloc(alignment, padded);
  return aligned_storage{
    std::unique_ptr<std::byte, decltype(&std::free)>(static_cast<std::byte *>(raw), &std::free),
    padded,
    alignment,
  };
}

void fill_src_ids(emel_buffer_tensor_desc & tensor, std::initializer_list<int32_t> srcs) {
  for (size_t i = 0; i < EMEL_BUFFER_MAX_SOURCES; ++i) {
    tensor.src_ids[i] = -1;
  }
  size_t idx = 0;
  for (const int32_t src : srcs) {
    if (idx >= EMEL_BUFFER_MAX_SOURCES) {
      break;
    }
    tensor.src_ids[idx++] = src;
  }
}

emel_buffer_graph_view make_valid_graph(
    std::array<emel_buffer_tensor_desc, 2> & nodes,
    std::array<emel_buffer_tensor_desc, 1> & leafs) {
  leafs[0] = emel_buffer_tensor_desc{
    .tensor_id = 100,
    .alloc_size = 128,
    .src_ids = {},
    .is_view = 0,
    ._pad0 = {0, 0, 0},
    .view_src_id = -1,
    .is_input = 1,
    .is_output = 0,
    .can_inplace = 1,
    .has_external_data = 0,
  };
  fill_src_ids(leafs[0], {});
  nodes[0] = emel_buffer_tensor_desc{
    .tensor_id = 200,
    .alloc_size = 256,
    .src_ids = {},
    .is_view = 0,
    ._pad0 = {0, 0, 0},
    .view_src_id = -1,
    .is_input = 0,
    .is_output = 0,
    .can_inplace = 1,
    .has_external_data = 0,
  };
  fill_src_ids(nodes[0], {100});
  nodes[1] = emel_buffer_tensor_desc{
    .tensor_id = 201,
    .alloc_size = 512,
    .src_ids = {},
    .is_view = 0,
    ._pad0 = {0, 0, 0},
    .view_src_id = -1,
    .is_input = 0,
    .is_output = 1,
    .can_inplace = 0,
    .has_external_data = 0,
  };
  fill_src_ids(nodes[1], {200});

  return emel_buffer_graph_view{
    .nodes = nodes.data(),
    .n_nodes = 2,
    .leafs = leafs.data(),
    .n_leafs = 1,
  };
}

}  // namespace

TEST_CASE("buffer_allocator_c_api_reports_invalid_arguments") {
  auto storage = make_allocator_storage();
  REQUIRE(storage.data != nullptr);
  CHECK(emel_buffer_allocator_storage_size() <= storage.size);
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(storage.data.get());
  CHECK((addr % storage.alignment) == 0);

  emel_buffer_allocator * allocator =
    emel_buffer_allocator_init(storage.data.get(), storage.size);
  REQUIRE(allocator != nullptr);

  CHECK(emel_buffer_allocator_initialize(allocator, 0, nullptr, nullptr) ==
        EMEL_ERR_INVALID_ARGUMENT);

  CHECK(emel_buffer_allocator_reserve_n_size(
          allocator, nullptr, nullptr, nullptr, nullptr, 0) ==
        EMEL_ERR_INVALID_ARGUMENT);

  emel_buffer_allocator_destroy(allocator);
}

TEST_CASE("buffer_allocator_c_api_alloc_tensors_wrappers") {
  auto storage = make_allocator_storage();
  REQUIRE(storage.data != nullptr);
  emel_buffer_allocator * allocator =
    emel_buffer_allocator_init(storage.data.get(), storage.size);
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

TEST_CASE("buffer_allocator_c_api_init_rejects_bad_storage") {
  alignas(64) std::array<std::byte, 64> storage = {};
  CHECK(emel_buffer_allocator_init(nullptr, 0) == nullptr);
  CHECK(emel_buffer_allocator_init(storage.data(), 1) == nullptr);
  CHECK(emel_buffer_allocator_init(storage.data() + 1, storage.size() - 1) == nullptr);
}

TEST_CASE("buffer_allocator_c_api_init_rejects_misaligned_storage") {
  alignas(64) std::array<std::byte, 1024> storage = {};
  CHECK(emel_buffer_allocator_init(storage.data() + 1, storage.size() - 1) == nullptr);
}

TEST_CASE("buffer_allocator_c_api_graph_validation_paths") {
  auto storage = make_allocator_storage();
  REQUIRE(storage.data != nullptr);
  emel_buffer_allocator * allocator =
    emel_buffer_allocator_init(storage.data.get(), storage.size);
  REQUIRE(allocator != nullptr);

  emel_buffer_graph_view graph{};
  graph.n_nodes = -1;
  CHECK(emel_buffer_allocator_reserve_n_size(
          allocator, &graph, nullptr, nullptr, nullptr, 0) ==
        EMEL_ERR_INVALID_ARGUMENT);

  graph.n_nodes = 0;
  graph.n_leafs = -1;
  CHECK(emel_buffer_allocator_reserve_n(
          allocator, &graph, nullptr, nullptr) ==
        EMEL_ERR_INVALID_ARGUMENT);

  graph.n_nodes = emel::buffer::allocator::action::k_max_graph_tensors + 1;
  graph.n_leafs = 0;
  CHECK(emel_buffer_allocator_reserve(allocator, &graph) == EMEL_ERR_INVALID_ARGUMENT);

  graph.n_nodes = 1;
  graph.nodes = nullptr;
  CHECK(emel_buffer_allocator_alloc_graph(allocator, &graph) == EMEL_ERR_INVALID_ARGUMENT);

  emel_buffer_allocator_destroy(allocator);
}

TEST_CASE("buffer_allocator_c_api_alloc_tensors_error_paths") {
  auto storage = make_allocator_storage();
  REQUIRE(storage.data != nullptr);
  emel_buffer_allocator * allocator =
    emel_buffer_allocator_init(storage.data.get(), storage.size);
  REQUIRE(allocator != nullptr);

  emel_buffer_graph_view graph{};
  graph.n_nodes = 0;
  graph.n_leafs = 0;

  CHECK(emel_buffer_allocator_alloc_tensors_from_buft(
          allocator, &graph, nullptr, nullptr) ==
        EMEL_ERR_INVALID_ARGUMENT);

  CHECK(emel_buffer_allocator_alloc_tensors(allocator, &graph) ==
        EMEL_ERR_INVALID_ARGUMENT);

  emel_buffer_allocator_destroy(allocator);
}

TEST_CASE("buffer_allocator_c_api_release_and_query_paths") {
  auto storage = make_allocator_storage();
  REQUIRE(storage.data != nullptr);
  emel_buffer_allocator * allocator =
    emel_buffer_allocator_init(storage.data.get(), storage.size);
  REQUIRE(allocator != nullptr);

  std::array<int32_t, 1> alignments = {{16}};
  std::array<int32_t, 1> max_sizes = {{0}};
  REQUIRE(emel_buffer_allocator_initialize(
            allocator,
            1,
            alignments.data(),
            max_sizes.data()) == EMEL_OK);

  CHECK(emel_buffer_allocator_buffer_size(allocator, 0) == 0);
  CHECK(emel_buffer_allocator_buffer_chunk_id(allocator, 0) == -1);
  CHECK(emel_buffer_allocator_buffer_chunk_offset(allocator, 0) == 0);
  CHECK(emel_buffer_allocator_buffer_alloc_size(allocator, 0) == 0);
  CHECK(emel_buffer_allocator_release(allocator) == EMEL_OK);

  emel_buffer_allocator_destroy(allocator);
}

TEST_CASE("buffer_allocator_c_api_queries_handle_null") {
  CHECK(emel_buffer_allocator_buffer_size(nullptr, 0) == 0);
  CHECK(emel_buffer_allocator_buffer_chunk_id(nullptr, 0) == -1);
  CHECK(emel_buffer_allocator_buffer_chunk_offset(nullptr, 0) == 0);
  CHECK(emel_buffer_allocator_buffer_alloc_size(nullptr, 0) == 0);
  CHECK(emel_buffer_allocator_release(nullptr) == EMEL_ERR_INVALID_ARGUMENT);
  emel_buffer_allocator_destroy(nullptr);
}

TEST_CASE("buffer_allocator_c_api_null_allocator_operations") {
  emel_buffer_graph_view graph{};
  graph.n_nodes = 0;
  graph.n_leafs = 0;

  CHECK(emel_buffer_allocator_initialize(nullptr, 1, nullptr, nullptr) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel_buffer_allocator_reserve_n_size(nullptr, &graph, nullptr, nullptr, nullptr, 0) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel_buffer_allocator_reserve_n(nullptr, &graph, nullptr, nullptr) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel_buffer_allocator_reserve(nullptr, &graph) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel_buffer_allocator_alloc_graph(nullptr, &graph) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel_buffer_allocator_alloc_tensors_from_buft_size(
          nullptr, &graph, nullptr, nullptr, nullptr, 0) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel_buffer_allocator_alloc_tensors_from_buft(nullptr, &graph, nullptr, nullptr) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel_buffer_allocator_alloc_tensors(nullptr, &graph) == EMEL_ERR_INVALID_ARGUMENT);
}
