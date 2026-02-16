#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/sm.hpp"

namespace {

TEST_CASE("tensor_allocator_sm_allocate_success_and_release") {
  emel::tensor::allocator::sm machine{};
  std::array<emel::tensor::allocator::event::tensor_desc, 1> tensors = {{
    emel::tensor::allocator::event::tensor_desc{
      .tensor_id = 1,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  void * result_buffer = nullptr;
  int32_t total_size = 0;
  std::array<int32_t, 2> chunk_sizes = {{0, 0}};
  int32_t chunk_count = 0;
  int32_t err = EMEL_OK;
  emel_error_detail detail{};

  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .result_buffer_out = &result_buffer,
    .total_size_out = &total_size,
    .chunk_sizes_out = chunk_sizes.data(),
    .chunk_sizes_out_count = static_cast<int32_t>(chunk_sizes.size()),
    .chunk_count_out = &chunk_count,
    .error_out = &err,
    .detail_out = &detail,
  }));
  CHECK(err == EMEL_OK);
  CHECK(chunk_count >= 1);

  err = EMEL_OK;
  detail = {};
  CHECK(machine.process_event(emel::tensor::allocator::event::release{
    .error_out = &err,
    .detail_out = &detail,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("tensor_allocator_sm_validation_error_path") {
  emel::tensor::allocator::sm machine{};
  std::array<emel::tensor::allocator::event::tensor_desc, 1> tensors = {{
    emel::tensor::allocator::event::tensor_desc{
      .tensor_id = 1,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 0,
    .max_buffer_size = 64,
    .no_alloc = true,
    .error_out = &err,
    .detail_out = &detail,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
