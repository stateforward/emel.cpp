#pragma once

#include <cstdint>

namespace emel::batch::splitter::event {

enum class split_mode : int32_t {
  simple = 0,
  equal = 1,
  seq = 2,
};

struct split {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t n_ubatch = 0;
  split_mode mode = split_mode::simple;
  const uint64_t * seq_masks = nullptr;
  const int32_t * seq_primary_ids = nullptr;
  bool equal_sequential = true;

  int32_t * ubatch_sizes_out = nullptr;
  int32_t ubatch_sizes_capacity = 0;
  int32_t * ubatch_count_out = nullptr;
  int32_t * total_outputs_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
  int32_t * error_out = nullptr;
};

struct normalize_batch {
  int32_t * error_out = nullptr;
};

struct create_ubatches {
  int32_t * error_out = nullptr;
};

struct publish {
  int32_t * error_out = nullptr;
};

}  // namespace emel::batch::splitter::event

namespace emel::batch::splitter::events {

struct validate_done {
  const event::split * request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::split * request = nullptr;
};

struct normalize_done {
  const event::split * request = nullptr;
};
struct normalize_error {
  int32_t err = 0;
  const event::split * request = nullptr;
};

struct split_done {
  const event::split * request = nullptr;
};
struct split_error {
  int32_t err = 0;
  const event::split * request = nullptr;
};

struct publish_done {
  const event::split * request = nullptr;
};
struct publish_error {
  int32_t err = 0;
  const event::split * request = nullptr;
};

struct splitting_done {
  const event::split * request = nullptr;
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
};

struct splitting_error {
  int32_t err = 0;
  const event::split * request = nullptr;
};

}  // namespace emel::batch::splitter::events
