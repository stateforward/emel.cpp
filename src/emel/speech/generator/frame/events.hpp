#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"

namespace emel::speech::generator::frame::events {

struct run_done;
struct run_error;

} // namespace emel::speech::generator::frame::events

namespace emel::speech::generator::frame::event {

struct run {
  run(std::span<const int32_t> encoded_tokens_ref,
      std::span<int32_t> generated_tokens_out_ref, int32_t &text_token_out_ref,
      emel::error::type &error_out_ref,
      const emel::callback<void(const events::run_done &)> on_done_ref = {},
      const emel::callback<void(const events::run_error &)> on_error_ref =
          {}) noexcept
      : encoded_tokens(encoded_tokens_ref),
        generated_tokens_out(generated_tokens_out_ref),
        text_token_out(text_token_out_ref), error_out(error_out_ref),
        on_done(on_done_ref), on_error(on_error_ref) {}

  const std::span<const int32_t> encoded_tokens;
  const std::span<int32_t> generated_tokens_out;
  int32_t &text_token_out;
  emel::error::type &error_out;
  const emel::callback<void(const events::run_done &)> on_done;
  const emel::callback<void(const events::run_error &)> on_error;
};

struct reset {
  explicit reset(emel::error::type &error_out_ref) noexcept
      : error_out(error_out_ref) {}

  emel::error::type &error_out;
};

} // namespace emel::speech::generator::frame::event

namespace emel::speech::generator::frame::events {

struct run_done {
  const event::run &request;
  int32_t text_token = -1;
};

struct run_error {
  const event::run &request;
  emel::error::type err = {};
};

} // namespace emel::speech::generator::frame::events
