#pragma once

#include "emel/error/error.hpp"
#include "emel/io/loader/errors.hpp"
#include "emel/io/loader/events.hpp"

namespace emel::io::loader::detail {

struct runtime_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
};

struct load_tensor_runtime {
  const event::load_tensor &request;
  runtime_status &ctx;
};

} // namespace emel::io::loader::detail
