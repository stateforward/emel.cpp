#pragma once

#include <cstdint>
#include <limits>

namespace emel::buffer::chunk_allocator::event {

struct configure {
  uint64_t alignment = 16;
  uint64_t max_chunk_size = std::numeric_limits<uint64_t>::max() / 2;
  int32_t * error_out = nullptr;
};

struct validate_configure {
  int32_t * error_out = nullptr;
  const event::configure * request = nullptr;
};

struct apply_configure {
  int32_t * error_out = nullptr;
};

struct allocate {
  uint64_t size = 0;
  uint64_t alignment = 0;
  uint64_t max_chunk_size = 0;
  int32_t * chunk_out = nullptr;
  uint64_t * offset_out = nullptr;
  uint64_t * aligned_size_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_allocate {
  int32_t * error_out = nullptr;
  const event::allocate * request = nullptr;
};

struct select_block {
  int32_t * error_out = nullptr;
};

struct ensure_chunk {
  int32_t * error_out = nullptr;
};

struct commit_allocate {
  int32_t * error_out = nullptr;
};

struct release {
  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t size = 0;
  uint64_t alignment = 0;
  int32_t * error_out = nullptr;
};

struct validate_release {
  int32_t * error_out = nullptr;
  const event::release * request = nullptr;
};

struct merge_release {
  int32_t * error_out = nullptr;
};

struct reset {
  int32_t * error_out = nullptr;
};

struct apply_reset {
  int32_t * error_out = nullptr;
};

}  // namespace emel::buffer::chunk_allocator::event

namespace emel::buffer::chunk_allocator::events {

struct validate_configure_done {
  const event::configure * request = nullptr;
};
struct validate_configure_error {
  int32_t err = 0;
  const event::configure * request = nullptr;
};

struct apply_configure_done {
  const event::configure * request = nullptr;
};
struct apply_configure_error {
  int32_t err = 0;
  const event::configure * request = nullptr;
};

struct configure_done {
  int32_t * error_out = nullptr;
  const event::configure * request = nullptr;
};
struct configure_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::configure * request = nullptr;
};

struct validate_allocate_done {
  const event::allocate * request = nullptr;
};
struct validate_allocate_error {
  int32_t err = 0;
  const event::allocate * request = nullptr;
};

struct select_block_done {
  const event::allocate * request = nullptr;
};
struct select_block_error {
  int32_t err = 0;
  const event::allocate * request = nullptr;
};

struct ensure_chunk_done {
  const event::allocate * request = nullptr;
};
struct ensure_chunk_error {
  int32_t err = 0;
  const event::allocate * request = nullptr;
};

struct commit_allocate_done {
  const event::allocate * request = nullptr;
};
struct commit_allocate_error {
  int32_t err = 0;
  const event::allocate * request = nullptr;
};

struct allocate_done {
  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t size = 0;
  int32_t * error_out = nullptr;
  const event::allocate * request = nullptr;
};

struct allocate_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::allocate * request = nullptr;
};

struct validate_release_done {
  const event::release * request = nullptr;
};
struct validate_release_error {
  int32_t err = 0;
  const event::release * request = nullptr;
};

struct merge_release_done {
  const event::release * request = nullptr;
};
struct merge_release_error {
  int32_t err = 0;
  const event::release * request = nullptr;
};

struct release_done {
  int32_t * error_out = nullptr;
  const event::release * request = nullptr;
};
struct release_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::release * request = nullptr;
};

struct apply_reset_done {
  const event::reset * request = nullptr;
};
struct apply_reset_error {
  int32_t err = 0;
  const event::reset * request = nullptr;
};

struct reset_done {
  int32_t * error_out = nullptr;
  const event::reset * request = nullptr;
};
struct reset_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
  const event::reset * request = nullptr;
};

using bootstrap_event = event::allocate;

}  // namespace emel::buffer::chunk_allocator::events
