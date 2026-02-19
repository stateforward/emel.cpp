#include <array>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/sm.hpp"

namespace {

using tensor_desc = emel::tensor::allocator::event::tensor_desc;

}  // namespace

TEST_CASE("tensor_allocator_sm_success_path_sets_outputs") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 0,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 1,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total_size = 0;
  int32_t chunk_count = 0;
  std::array<int32_t, 4> chunk_sizes = {{0, 0, 0, 0}};
  int32_t err = EMEL_OK;
  emel_error_detail detail{};

  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .total_size_out = &total_size,
    .chunk_sizes_out = chunk_sizes.data(),
    .chunk_sizes_out_count = static_cast<int32_t>(chunk_sizes.size()),
    .chunk_count_out = &chunk_count,
    .error_out = &err,
    .detail_out = &detail,
  }));

  CHECK(err == EMEL_OK);
  CHECK(chunk_count >= 1);
  CHECK(total_size >= 32);
}

TEST_CASE("tensor_allocator_sm_duplicate_ids_report_scan_error") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 2,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 2,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t err = EMEL_OK;
  emel_error_detail detail{};

  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .error_out = &err,
    .detail_out = &detail,
  }));

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.phase ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::scan_tensors));
  CHECK(detail.reason ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::duplicate_tensor_id));
}

TEST_CASE("tensor_allocator_sm_assemble_reports_chunk_size_capacity_error") {
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
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  std::array<int32_t, 1> chunk_sizes = {{0}};
  int32_t err = EMEL_OK;
  emel_error_detail detail{};

  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .chunk_sizes_out = chunk_sizes.data(),
    .chunk_sizes_out_count = 0,
    .error_out = &err,
    .detail_out = &detail,
  }));

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.phase ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::assemble));
}
