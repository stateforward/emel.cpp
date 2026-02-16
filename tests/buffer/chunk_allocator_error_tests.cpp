#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/chunk_allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/emel.h"

TEST_CASE("buffer_chunk_allocator_allocate_requires_chunk_and_offset_outputs") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = EMEL_OK;
  uint64_t offset = 0;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 16,
    .chunk_out = nullptr,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  int32_t chunk = -1;
  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 16,
    .chunk_out = &chunk,
    .offset_out = nullptr,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_chunk_allocator_configure_rejects_invalid_alignment") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 0,
    .max_chunk_size = 0,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_chunk_allocator_allocate_rejects_zero_size") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = EMEL_OK;
  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t size = 0;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 0,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .aligned_size_out = &size,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_chunk_allocator_release_rejects_invalid_chunk") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = 0,
    .offset = 0,
    .size = 16,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_chunk_allocator_free_bytes_merges_previous_block") {
  emel::buffer::chunk_allocator::action::context ctx{};
  ctx.chunk_count = 1;
  ctx.request_chunk = 0;
  ctx.request_offset = 16;
  ctx.aligned_request_size = 16;

  auto & chunk = ctx.chunks[0];
  chunk.free_block_count = 2;
  chunk.free_blocks[0] = emel::buffer::chunk_allocator::action::free_block{
    .offset = 0,
    .size = 16,
  };
  chunk.free_blocks[1] = emel::buffer::chunk_allocator::action::free_block{
    .offset = 32,
    .size = 16,
  };

  CHECK(emel::buffer::chunk_allocator::action::detail::free_bytes(ctx));
  CHECK(chunk.free_block_count == 1);
  CHECK(chunk.free_blocks[0].offset == 0);
  CHECK(chunk.free_blocks[0].size >= 32);
}
