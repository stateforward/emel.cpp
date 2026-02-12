#pragma once

#include <cstdint>

namespace emel::codec::encoder::event {

struct encode {
  const char * text = nullptr;
};

struct tokenized_done {};
struct tokenized_error {};

}  // namespace emel::codec::encoder::event

namespace emel::codec::encoder::events {

struct encoding_done {};
struct encoding_error {};

}  // namespace emel::codec::encoder::events
