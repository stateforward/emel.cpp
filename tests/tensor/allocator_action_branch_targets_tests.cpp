#include <array>
#include <cstdint>
#include <cstdlib>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/actions.hpp"

namespace {

using tensor_desc = emel::tensor::allocator::event::tensor_desc;

}  // namespace

TEST_CASE("tensor_allocator_actions_allocate_ranges_rejects_zero_chunk_size") {
  emel::tensor::allocator::action::context c{};
  c.chunk_count = 1;
  c.chunk_sizes[0] = 0;
  c.no_alloc = false;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_allocate_ranges(
    emel::tensor::allocator::event::allocate_ranges{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(detail.phase == static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::allocate_ranges));
}

TEST_CASE("tensor_allocator_actions_initialize_tensors_invalid_view_source") {
  emel::tensor::allocator::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 1,
      .alloc_size = 0,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  c.tensors = tensors.data();
  c.tensor_count = 1;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.phase == static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::initialize_tensors));
}

TEST_CASE("tensor_allocator_actions_initialize_tensors_invalid_chunk") {
  emel::tensor::allocator::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 2,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.effective_sizes[0] = 16;
  c.chunk_count = 1;
  c.tensor_chunk_ids[0] = -1;
  c.tensor_offsets[0] = 0;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(detail.phase == static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::initialize_tensors));
}

TEST_CASE("tensor_allocator_actions_initialize_tensors_null_buffer") {
  emel::tensor::allocator::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 3,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.effective_sizes[0] = 16;
  c.chunk_count = 1;
  c.tensor_chunk_ids[0] = 0;
  c.tensor_offsets[0] = 0;
  c.chunk_sizes[0] = 16;
  c.allocated_buffers[0] = nullptr;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("tensor_allocator_actions_initialize_tensors_offset_out_of_range") {
  emel::tensor::allocator::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 4,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  void * buffer = std::malloc(8);
  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.effective_sizes[0] = 16;
  c.chunk_count = 1;
  c.tensor_chunk_ids[0] = 0;
  c.tensor_offsets[0] = 0;
  c.chunk_sizes[0] = 8;
  c.allocated_buffers[0] = buffer;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("tensor_allocator_actions_assemble_requires_chunk_sizes_capacity") {
  emel::tensor::allocator::action::context c{};
  c.chunk_count = 2;
  c.no_alloc = true;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 16;
  c.total_bytes = 32;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  std::array<int32_t, 1> sizes = {{0}};
  emel::tensor::allocator::action::run_assemble(
    emel::tensor::allocator::event::assemble{
      .error_out = &err,
      .detail_out = &detail,
      .chunk_sizes_out = sizes.data(),
      .chunk_sizes_out_count = static_cast<int32_t>(sizes.size()),
    },
    c);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_actions_assemble_sets_result_buffer_single_chunk") {
  emel::tensor::allocator::action::context c{};
  c.chunk_count = 1;
  c.no_alloc = false;
  c.chunk_sizes[0] = 16;
  c.total_bytes = 16;
  std::array<uint8_t, 16> buffer = {};
  c.allocated_buffers[0] = buffer.data();

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  void * result = nullptr;
  emel::tensor::allocator::action::run_assemble(
    emel::tensor::allocator::event::assemble{
      .error_out = &err,
      .detail_out = &detail,
      .result_buffer_out = &result,
    },
    c);

  CHECK(err == EMEL_OK);
  CHECK(result == buffer.data());
}

TEST_CASE("tensor_allocator_actions_assemble_sets_result_buffer_multiple_chunks") {
  emel::tensor::allocator::action::context c{};
  c.chunk_count = 2;
  c.no_alloc = false;
  c.chunk_sizes[0] = 8;
  c.chunk_sizes[1] = 8;
  c.total_bytes = 16;
  void * buffer0 = std::malloc(8);
  void * buffer1 = std::malloc(8);
  c.allocated_buffers[0] = buffer0;
  c.allocated_buffers[1] = buffer1;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  void * result = nullptr;
  emel::tensor::allocator::action::run_assemble(
    emel::tensor::allocator::event::assemble{
      .error_out = &err,
      .detail_out = &detail,
      .result_buffer_out = &result,
    },
    c);

  CHECK(err == EMEL_OK);
  CHECK(result != nullptr);
  std::free(buffer0);
  std::free(buffer1);
}

TEST_CASE("tensor_allocator_actions_partition_ranges_overflows_chunk_limit") {
  emel::tensor::allocator::action::context c{};
  c.tensor_count = emel::tensor::allocator::action::k_max_chunks + 1;
  c.max_buffer_size = 1;
  for (int32_t i = 0; i < c.tensor_count; ++i) {
    c.effective_sizes[i] = 2;
  }

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_partition_ranges(
    emel::tensor::allocator::event::partition_ranges{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_OK);
  CHECK(c.chunk_count > 0);
}

TEST_CASE("tensor_allocator_actions_initialize_tensors_skips_zero_sizes") {
  emel::tensor::allocator::action::context c{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 6,
      .alloc_size = 0,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  std::array<uint8_t, 16> buffer = {};
  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.effective_sizes[0] = 0;
  c.chunk_count = 1;
  c.tensor_chunk_ids[0] = 0;
  c.tensor_offsets[0] = 0;
  c.chunk_sizes[0] = 16;
  c.allocated_buffers[0] = buffer.data();

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_OK);
}

TEST_CASE("tensor_allocator_actions_release_reports_error_detail") {
  emel::tensor::allocator::action::context c{};
  c.chunk_count = 1;
  c.chunk_sizes[0] = 16;
  c.allocated_buffers[0] = nullptr;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::begin_release(
    emel::tensor::allocator::event::release{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_OK);
  CHECK(detail.phase == 0);
}

TEST_CASE("tensor_allocator_actions_on_unexpected_sets_error") {
  emel::tensor::allocator::action::context c{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::action::on_unexpected(
    emel::tensor::allocator::event::allocate_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(detail.status == EMEL_ERR_BACKEND);
}
