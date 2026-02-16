#include <array>
#include <limits>
#include <doctest/doctest.h>

#include "emel/tensor/allocator/actions.hpp"
#include "emel/tensor/allocator/events.hpp"
#include "emel/emel.h"

TEST_CASE("tensor_allocator_run_validate_rejects_invalid_inputs") {
  emel::tensor::allocator::action::context ctx{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  ctx.tensor_count = -1;
  emel::tensor::allocator::action::run_validate(
    emel::tensor::allocator::event::validate{
      .error_out = &err,
      .detail_out = &detail,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_run_validate_accepts_valid_inputs") {
  emel::tensor::allocator::action::context ctx{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;
  emel::tensor::allocator::event::tensor_desc tensor{};

  ctx.tensors = &tensor;
  ctx.tensor_count = 1;
  ctx.alignment = 16;
  ctx.max_buffer_size = 1024;

  emel::tensor::allocator::action::run_validate(
    emel::tensor::allocator::event::validate{
      .error_out = &err,
      .detail_out = &detail,
      .chunk_sizes_out = nullptr,
      .chunk_sizes_out_count = 0,
    },
    ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("tensor_allocator_run_scan_tensors_detects_duplicate_and_view_errors") {
  emel::tensor::allocator::action::context ctx{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;
  std::array<emel::tensor::allocator::event::tensor_desc, 2> tensors = {};

  tensors[0].tensor_id = 1;
  tensors[0].alloc_size = 8;
  tensors[1].tensor_id = 1;
  tensors[1].alloc_size = 8;
  ctx.tensors = tensors.data();
  ctx.tensor_count = static_cast<int32_t>(tensors.size());
  ctx.alignment = 16;

  emel::tensor::allocator::action::run_scan_tensors(
    emel::tensor::allocator::event::scan_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_run_scan_tensors_detects_invalid_view_source") {
  emel::tensor::allocator::action::context ctx{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;
  emel::tensor::allocator::event::tensor_desc tensor{};

  tensor.tensor_id = 1;
  tensor.alloc_size = 8;
  tensor.is_view = true;
  tensor.view_src_id = -1;
  ctx.tensors = &tensor;
  ctx.tensor_count = 1;
  ctx.alignment = 16;

  emel::tensor::allocator::action::run_scan_tensors(
    emel::tensor::allocator::event::scan_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_run_scan_tensors_detects_alignment_overflow") {
  emel::tensor::allocator::action::context ctx{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;
  emel::tensor::allocator::event::tensor_desc tensor{};

  tensor.tensor_id = 1;
  tensor.alloc_size = std::numeric_limits<int32_t>::max();
  ctx.tensors = &tensor;
  ctx.tensor_count = 1;
  ctx.alignment = 16;

  emel::tensor::allocator::action::run_scan_tensors(
    emel::tensor::allocator::event::scan_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("tensor_allocator_run_partition_ranges_splits_chunks") {
  emel::tensor::allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.tensor_count = 2;
  ctx.max_buffer_size = 8;
  ctx.effective_sizes[0] = 8;
  ctx.effective_sizes[1] = 8;

  emel::tensor::allocator::action::run_partition_ranges(
    emel::tensor::allocator::event::partition_ranges{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.chunk_count == 2);
  CHECK(ctx.total_bytes == 16);
}

TEST_CASE("tensor_allocator_run_allocate_ranges_handles_no_alloc_and_invalid") {
  emel::tensor::allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.no_alloc = true;
  ctx.chunk_count = 1;
  emel::tensor::allocator::action::run_allocate_ranges(
    emel::tensor::allocator::event::allocate_ranges{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);

  ctx.no_alloc = false;
  ctx.chunk_count = 1;
  ctx.chunk_sizes[0] = 0;
  err = EMEL_OK;
  emel::tensor::allocator::action::run_allocate_ranges(
    emel::tensor::allocator::event::allocate_ranges{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("tensor_allocator_run_initialize_tensors_reports_errors") {
  emel::tensor::allocator::action::context ctx{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;
  emel::tensor::allocator::event::tensor_desc tensor{};

  ctx.no_alloc = true;
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    ctx);
  CHECK(err == EMEL_OK);

  ctx.no_alloc = false;
  tensor.tensor_id = 1;
  tensor.alloc_size = 8;
  tensor.is_view = true;
  tensor.view_src_id = 42;
  ctx.tensors = &tensor;
  ctx.tensor_count = 1;
  err = EMEL_OK;
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  tensor.is_view = false;
  ctx.effective_sizes[0] = 8;
  ctx.tensor_chunk_ids[0] = -1;
  err = EMEL_OK;
  emel::tensor::allocator::action::run_initialize_tensors(
    emel::tensor::allocator::event::initialize_tensors{
      .error_out = &err,
      .detail_out = &detail,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("tensor_allocator_run_assemble_paths") {
  emel::tensor::allocator::action::context ctx{};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  ctx.chunk_count = 2;
  std::array<int32_t, 1> sizes = {{0}};
  emel::tensor::allocator::action::run_assemble(
    emel::tensor::allocator::event::assemble{
      .error_out = &err,
      .detail_out = &detail,
      .chunk_sizes_out = sizes.data(),
      .chunk_sizes_out_count = 1,
    },
    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.no_alloc = false;
  ctx.chunk_count = 1;
  ctx.chunk_sizes[0] = 8;
  std::array<uint8_t, 8> buffer = {};
  ctx.allocated_buffers[0] = buffer.data();
  void * result = nullptr;
  err = EMEL_OK;
  emel::tensor::allocator::action::run_assemble(
    emel::tensor::allocator::event::assemble{
      .error_out = &err,
      .detail_out = &detail,
      .result_buffer_out = &result,
      .chunk_sizes_out = sizes.data(),
      .chunk_sizes_out_count = 1,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(result == buffer.data());
}
