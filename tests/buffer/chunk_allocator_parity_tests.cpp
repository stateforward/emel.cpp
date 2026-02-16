#include "doctest/doctest.h"

#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/emel.h"

TEST_CASE("chunk_allocator_allows_allocation_above_max_chunk_size") {
  emel::buffer::chunk_allocator::sm allocator;
  int32_t err = EMEL_OK;
  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t aligned = 0;

  CHECK(allocator.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 24,
    .alignment = 8,
    .max_chunk_size = 16,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .aligned_size_out = &aligned,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);
  CHECK_EQ(chunk, 0);
  CHECK_EQ(offset, 0);
  CHECK_EQ(aligned, 24);
  CHECK_EQ(allocator.chunk_count(), 1);
}

TEST_CASE("chunk_allocator_last_chunk_grows_when_exhausted") {
  emel::buffer::chunk_allocator::sm allocator;
  const int32_t max_chunks = emel::buffer::chunk_allocator::action::k_max_chunks;
  int32_t err = EMEL_OK;

  for (int32_t i = 0; i < max_chunks + 1; ++i) {
    int32_t chunk = -1;
    uint64_t offset = 0;
    uint64_t aligned = 0;
    CHECK(allocator.process_event(emel::buffer::chunk_allocator::event::allocate{
      .size = 8,
      .alignment = 8,
      .max_chunk_size = 8,
      .chunk_out = &chunk,
      .offset_out = &offset,
      .aligned_size_out = &aligned,
      .error_out = &err,
    }));
    CHECK_EQ(err, EMEL_OK);
  }

  CHECK_EQ(allocator.chunk_count(), max_chunks);
}
