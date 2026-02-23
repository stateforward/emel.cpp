#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "emel/emel.h"

namespace emel::text::formatter {

struct format_request {
  std::string_view input = {};
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t * output_length_out = nullptr;
};

using format_fn = bool (*)(void * formatter_ctx,
                           const format_request & request,
                           int32_t * error_out);

inline bool format_raw(void *,
                       const format_request & request,
                       int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = EMEL_OK;
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  if ((request.output == nullptr && request.output_capacity > 0) ||
      request.input.size() > request.output_capacity) {
    if (error_out != nullptr) {
      *error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }

  if (request.input.empty()) {
    return true;
  }

  std::memcpy(request.output, request.input.data(), request.input.size());
  if (request.output_length_out != nullptr) {
    *request.output_length_out = request.input.size();
  }
  return true;
}

}  // namespace emel::text::formatter
