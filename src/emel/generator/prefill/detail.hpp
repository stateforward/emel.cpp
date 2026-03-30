#pragma once

#include "emel/generator/events.hpp"

namespace emel::generator::prefill::event {

struct run {
  const emel::generator::event::generate & request;
  emel::generator::event::generate_ctx & ctx;
};

}  // namespace emel::generator::prefill::event
