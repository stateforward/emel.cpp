#pragma once

#include "emel/generator/events.hpp"

namespace emel::generator::initializer::event {

struct run {
  const emel::generator::event::initialize & request;
  emel::generator::event::initialize_ctx & ctx;
};

}  // namespace emel::generator::initializer::event
