#pragma once

namespace emel::text::generator::action {
struct context;
}

namespace emel::text::generator::initializer::action {

struct context {
  explicit context(emel::text::generator::action::context & generator_ref) noexcept
    : generator(generator_ref) {}

  emel::text::generator::action::context & generator;
};

}  // namespace emel::text::generator::initializer::action
