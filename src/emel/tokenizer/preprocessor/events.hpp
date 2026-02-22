#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace emel::tokenizer::preprocessor::events {
struct preprocess_done;
struct preprocess_error;
}  // namespace emel::tokenizer::preprocessor::events

namespace emel::tokenizer::preprocessor::event {

struct preprocess {
  const emel::model::data::vocab * vocab = nullptr;
  std::string_view text = {};
  bool parse_special = false;
  fragment * fragments_out = nullptr;
  size_t fragment_capacity = 0;
  size_t * fragment_count_out = nullptr;
  bool * preprocessed_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm,
                        const events::preprocess_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm,
                         const events::preprocess_error &) = nullptr;
};

}  // namespace emel::tokenizer::preprocessor::event

namespace emel::tokenizer::preprocessor::events {

struct preprocess_done {
  const event::preprocess * request = nullptr;
  size_t fragment_count = 0;
};

struct preprocess_error {
  const event::preprocess * request = nullptr;
  int32_t err = 0;
};

}  // namespace emel::tokenizer::preprocessor::events
