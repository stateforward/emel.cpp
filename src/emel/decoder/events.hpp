#pragma once

#include <cstdint>

namespace emel::decoder::event {

struct decode {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
};

struct detokenized_done {};
struct detokenized_error {};

}  // namespace emel::decoder::event

namespace emel::decoder::events {

struct decoding_done {};
struct decoding_error {};

}  // namespace emel::decoder::events
