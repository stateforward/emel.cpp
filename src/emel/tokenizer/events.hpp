#pragma once

#include <cstdint>
#include <string_view>

namespace emel::tokenizer::event {

struct tokenize {
  std::string_view text = {};
};

struct partitioning_special_done {};
struct partitioning_special_error {};

struct selecting_backend_done {};
struct selecting_backend_error {};

struct applying_special_prefix_done {};
struct applying_special_prefix_error {};

struct encoding_fragment_done {};
struct encoding_fragment_error {};

struct applying_special_suffix_done {};
struct applying_special_suffix_error {};

struct finalizing_done {};
struct finalizing_error {};

}  // namespace emel::tokenizer::event

namespace emel::tokenizer::events {

struct tokenizer_done {};
struct tokenizer_error {};

}  // namespace emel::tokenizer::events
