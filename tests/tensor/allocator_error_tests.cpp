#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/sm.hpp"

namespace {

using tensor_desc = emel::tensor::allocator::event::tensor_desc;

}  // namespace

TEST_CASE("tensor_allocator_allocate_rejects_invalid_alignment") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 1,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total = 0;
  int32_t error = EMEL_OK;
  emel_error_detail detail{};

  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 3,
    .max_buffer_size = 64,
    .no_alloc = true,
    .total_size_out = &total,
    .error_out = &error,
    .detail_out = &detail,
  }));

  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.status == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_allocate_rejects_duplicate_tensor_ids") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 2,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 2,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total = 0;
  int32_t error = EMEL_OK;
  emel_error_detail detail{};

  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .total_size_out = &total,
    .error_out = &error,
    .detail_out = &detail,
  }));

  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.status == EMEL_ERR_INVALID_ARGUMENT);
}
