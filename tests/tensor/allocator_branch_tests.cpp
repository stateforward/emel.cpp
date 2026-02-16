#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/sm.hpp"

namespace {

using tensor_desc = emel::tensor::allocator::event::tensor_desc;

}  // namespace

TEST_CASE("tensor_allocator_allocates_multiple_chunks") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 10,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 11,
      .alloc_size = 32,
      .src_ids = {{10, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  std::array<int32_t, 2> chunk_sizes = {{0, 0}};
  int32_t chunk_count = 0;
  int32_t total = 0;
  int32_t error = EMEL_OK;
  emel_error_detail detail{};

  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 32,
    .no_alloc = true,
    .total_size_out = &total,
    .chunk_sizes_out = chunk_sizes.data(),
    .chunk_sizes_out_count = static_cast<int32_t>(chunk_sizes.size()),
    .chunk_count_out = &chunk_count,
    .error_out = &error,
    .detail_out = &detail,
  }));

  CHECK(error == EMEL_OK);
  CHECK(chunk_count == 2);
  CHECK(total >= 64);
}

TEST_CASE("tensor_allocator_reports_chunk_overflow_for_large_tensor") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 20,
      .alloc_size = 128,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total = 0;
  int32_t error = EMEL_OK;
  emel_error_detail detail{};

  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .total_size_out = &total,
    .error_out = &error,
    .detail_out = &detail,
  }));

  CHECK(error == EMEL_OK);
}

TEST_CASE("tensor_allocator_handles_view_and_external_data_inputs") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 3> tensors = {{
    tensor_desc{
      .tensor_id = 30,
      .alloc_size = 0,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = true,
    },
    tensor_desc{
      .tensor_id = 31,
      .alloc_size = 0,
      .src_ids = {{30, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 30,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 32,
      .alloc_size = 32,
      .src_ids = {{31, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total = 0;
  int32_t error = EMEL_OK;
  emel_error_detail detail{};

  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .total_size_out = &total,
    .error_out = &error,
    .detail_out = &detail,
  }));

  CHECK(error == EMEL_OK);
}
