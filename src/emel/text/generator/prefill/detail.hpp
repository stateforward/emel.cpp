#pragma once

#include "emel/text/generator/events.hpp"

namespace emel::text::generator::prefill::event {

struct run {
  const emel::text::generator::event::generate & request;
  emel::text::generator::event::generate_ctx & ctx;
};

}  // namespace emel::text::generator::prefill::event
