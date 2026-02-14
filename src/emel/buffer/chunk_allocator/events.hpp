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
};

struct apply_configure {
  int32_t * error_out = nullptr;
};

struct allocate {
  uint64_t size = 0;
  int32_t * chunk_out = nullptr;
  uint64_t * offset_out = nullptr;
  uint64_t * aligned_size_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate_allocate {
  int32_t * error_out = nullptr;
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
  int32_t * error_out = nullptr;
};

struct validate_release {
  int32_t * error_out = nullptr;
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

struct validate_configure_done {};
struct validate_configure_error {
  int32_t err = 0;
};

struct apply_configure_done {};
struct apply_configure_error {
  int32_t err = 0;
};

struct configure_done {};
struct configure_error {
  int32_t err = 0;
};

struct validate_allocate_done {};
struct validate_allocate_error {
  int32_t err = 0;
};

struct select_block_done {};
struct select_block_error {
  int32_t err = 0;
};

struct ensure_chunk_done {};
struct ensure_chunk_error {
  int32_t err = 0;
};

struct commit_allocate_done {};
struct commit_allocate_error {
  int32_t err = 0;
};

struct allocate_done {
  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct allocate_error {
  int32_t err = 0;
};

struct validate_release_done {};
struct validate_release_error {
  int32_t err = 0;
};

struct merge_release_done {};
struct merge_release_error {
  int32_t err = 0;
};

struct release_done {};
struct release_error {
  int32_t err = 0;
};

struct apply_reset_done {};
struct apply_reset_error {
  int32_t err = 0;
};

struct reset_done {};
struct reset_error {
  int32_t err = 0;
};

using bootstrap_event = event::allocate;

}  // namespace emel::buffer::chunk_allocator::events

