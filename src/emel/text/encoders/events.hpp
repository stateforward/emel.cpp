#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/emel.h"
#include "emel/model/data.hpp"

namespace emel::text::encoders::events {

struct encoding_done;
struct encoding_error;

}  // namespace emel::text::encoders::events

namespace emel::text::encoders::event {

inline const emel::model::data::vocab & default_encode_vocab() noexcept {
  static const emel::model::data::vocab vocab{};
  return vocab;
}

struct encode {
  const emel::model::data::vocab & vocab = default_encode_vocab();
  std::string_view text = {};
  bool preprocessed = false;
  std::span<int32_t> token_ids = {};
  int32_t * token_count_out = nullptr;
  int32_t * error_out = nullptr;
  void * owner_sm = nullptr;
  bool (*dispatch_done)(void * owner_sm, const events::encoding_done &) = nullptr;
  bool (*dispatch_error)(void * owner_sm, const events::encoding_error &) = nullptr;
};

struct encode_ctx {
  int32_t token_count = 0;
  int32_t err = EMEL_OK;
};

struct encode_runtime {
  const encode & request;
  encode_ctx & ctx;
};

}  // namespace emel::text::encoders::event

namespace emel::text::encoders::events {

struct encoding_done {
  const event::encode & request;
  int32_t token_count = 0;
};

struct encoding_error {
  const event::encode & request;
  int32_t err = EMEL_OK;
};

}  // namespace emel::text::encoders::events
