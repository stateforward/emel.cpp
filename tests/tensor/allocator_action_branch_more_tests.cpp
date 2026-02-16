#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/actions.hpp"

namespace {

using tensor_desc = emel::tensor::allocator::event::tensor_desc;

}  // namespace

TEST_CASE("tensor_allocator_actions_partition_ranges_success_single_chunk") {
  emel::tensor::allocator::action::context c{};
  c.tensor_count = 2;
  c.no_alloc = true;
  c.max_buffer_size = 64;
  c.effective_sizes[0] = 16;
  c.effective_sizes[1] = 8;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_partition_ranges(
    emel::tensor::allocator::event::partition_ranges{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_OK);
  CHECK(c.chunk_count == 1);
  CHECK(c.chunk_sizes[0] >= 24);
}

TEST_CASE("tensor_allocator_actions_allocate_ranges_no_alloc_success") {
  emel::tensor::allocator::action::context c{};
  c.no_alloc = true;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 8;
  c.total_bytes = 24;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  emel::tensor::allocator::action::run_allocate_ranges(
    emel::tensor::allocator::event::allocate_ranges{
      .error_out = &err,
      .detail_out = &detail,
    },
    c);

  CHECK(err == EMEL_OK);
  CHECK(c.total_bytes == 24);
}

TEST_CASE("tensor_allocator_actions_initialize_tensors_success") {
  emel::tensor::allocator::action::context c{};
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

  std::array<uint8_t, 16> buffer = {};
  c.tensors = tensors.data();
  c.tensor_count = static_cast<int32_t>(tensors.size());
  c.effective_sizes[0] = 16;
  c.chunk_count = 1;
  c.chunk_sizes[0] = 16;
  c.tensor_chunk_ids[0] = 0;
  c.tensor_offsets[0] = 0;
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

TEST_CASE("tensor_allocator_actions_assemble_no_alloc_copies_sizes") {
  emel::tensor::allocator::action::context c{};
  c.no_alloc = true;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 8;
  c.chunk_sizes[1] = 12;
  c.total_bytes = 20;

  int32_t err = EMEL_OK;
  emel_error_detail detail{};
  std::array<int32_t, 2> sizes = {{0, 0}};
  emel::tensor::allocator::action::run_assemble(
    emel::tensor::allocator::event::assemble{
      .error_out = &err,
      .detail_out = &detail,
      .chunk_sizes_out = sizes.data(),
      .chunk_sizes_out_count = static_cast<int32_t>(sizes.size()),
    },
    c);

  CHECK(err == EMEL_OK);
  CHECK(sizes[0] == 8);
  CHECK(sizes[1] == 12);
}
