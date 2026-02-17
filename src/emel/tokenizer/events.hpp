#pragma once

#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"

namespace emel::tokenizer::events {
struct tokenizer_done;
struct tokenizer_error;
}  // namespace emel::tokenizer::events

namespace emel::tokenizer::event {

struct tokenize {
  const emel::model::data::vocab * vocab = nullptr;
  std::string_view text = {};
  bool add_special = false;
  bool parse_special = false;
  int32_t * token_ids_out = nullptr;
  int32_t token_capacity = 0;
  int32_t * token_count_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::tokenizer_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::tokenizer_error &) = nullptr;
};

struct special_tokens_ready {
  const tokenize * request = nullptr;
};

struct partitioning_special_done {
  const tokenize * request = nullptr;
};
struct partitioning_special_error {
  const tokenize * request = nullptr;
  int32_t err = 0;
};

struct selecting_backend_done {
  const tokenize * request = nullptr;
};
struct selecting_backend_error {
  const tokenize * request = nullptr;
  int32_t err = 0;
};

struct applying_special_prefix_done {
  const tokenize * request = nullptr;
};
struct applying_special_prefix_error {
  const tokenize * request = nullptr;
  int32_t err = 0;
};

struct encoding_fragment_done {
  const tokenize * request = nullptr;
};
struct encoding_fragment_error {
  const tokenize * request = nullptr;
  int32_t err = 0;
};

struct next_fragment {
  const tokenize * request = nullptr;
};

struct applying_special_suffix_done {
  const tokenize * request = nullptr;
};
struct applying_special_suffix_error {
  const tokenize * request = nullptr;
  int32_t err = 0;
};

struct finalizing_done {
  const tokenize * request = nullptr;
};
struct finalizing_error {
  const tokenize * request = nullptr;
  int32_t err = 0;
};

}  // namespace emel::tokenizer::event

namespace emel::tokenizer::events {

struct tokenizer_done {
  const event::tokenize * request = nullptr;
  int32_t token_count = 0;
};

struct tokenizer_error {
  const event::tokenize * request = nullptr;
  int32_t err = 0;
};

}  // namespace emel::tokenizer::events
