#pragma once

namespace emel::generator::action {
struct context;
}

namespace emel::generator::prefill::action {

struct context {
  explicit context(emel::generator::action::context & generator_ref) noexcept
    : generator(generator_ref) {}

  emel::generator::action::context & generator;
};

}  // namespace emel::generator::prefill::action
