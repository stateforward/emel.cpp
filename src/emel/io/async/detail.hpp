#pragma once

#include "emel/error/error.hpp"
#include "emel/io/async/errors.hpp"
#include "emel/io/async/events.hpp"

namespace emel::io::async::detail {

struct load_window_status {
  bool ok = false;
  emel::error::type err = emel::error::cast(error::none);
};

struct load_window_runtime {
  const event::load_window &intent;
  load_window_status &status;
};

} // namespace emel::io::async::detail
