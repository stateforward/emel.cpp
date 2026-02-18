#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/emel.h"

TEST_CASE("chunk_allocator_testing_policy_configure_success_path") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 64,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("chunk_allocator_testing_policy_configure_error_path") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 0,
    .max_chunk_size = 64,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("chunk_allocator_testing_policy_allocate_success_path") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;
  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t aligned_size = 0;

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 16,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .aligned_size_out = &aligned_size,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(chunk >= 0);
  CHECK(aligned_size > 0);
}

TEST_CASE("chunk_allocator_testing_policy_allocate_error_path") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;
  int32_t chunk = -1;
  uint64_t offset = 0;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 0,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("chunk_allocator_testing_policy_release_success_path") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;
  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t aligned_size = 0;

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 16,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .aligned_size_out = &aligned_size,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = chunk,
    .offset = offset,
    .size = aligned_size,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("chunk_allocator_testing_policy_release_error_path") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = -1,
    .offset = 0,
    .size = 8,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("chunk_allocator_testing_policy_reset_path") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::reset{
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}
