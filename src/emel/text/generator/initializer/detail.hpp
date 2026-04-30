#pragma once

#include "emel/text/generator/events.hpp"

namespace emel::text::generator::initializer::event {

struct run {
  const emel::text::generator::event::initialize & request;
  emel::text::generator::event::initialize_ctx & ctx;
};

}  // namespace emel::text::generator::initializer::event
