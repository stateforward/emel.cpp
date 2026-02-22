#pragma once

#include <cstdint>
#include <string_view>

#include "emel/model/data.hpp"

namespace emel::encoder::events {

struct encoding_done;
struct encoding_error;

}  // namespace emel::encoder::events

namespace emel::encoder::event {

struct encode {
  const emel::model::data::vocab * vocab = nullptr;
  std::string_view text = {};
  bool preprocessed = false;
  int32_t * token_ids = nullptr;
  int32_t token_capacity = 0;
  int32_t * token_count_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::encoding_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::encoding_error &) = nullptr;
};

}  // namespace emel::encoder::event

namespace emel::encoder::events {

struct encoding_done {
  const event::encode * request = nullptr;
  int32_t token_count = 0;
};

struct encoding_error {
  const event::encode * request = nullptr;
  int32_t err = 0;
};

}  // namespace emel::encoder::events
