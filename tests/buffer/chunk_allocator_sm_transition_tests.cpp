#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/buffer/chunk_allocator/sm.hpp"

namespace {

TEST_CASE("chunk_allocator_sm_configure_allocate_release_reset") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 1024,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t aligned_size = 0;
  err = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 64,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .aligned_size_out = &aligned_size,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(chunk >= 0);

  err = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = chunk,
    .offset = offset,
    .size = aligned_size,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::reset{
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("chunk_allocator_sm_rejects_invalid_requests") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 0,
    .max_chunk_size = 0,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);

  err = EMEL_OK;
  machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 64,
    .chunk_out = nullptr,
    .offset_out = nullptr,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);

  err = EMEL_OK;
  machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = -1,
    .offset = 0,
    .size = 0,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
