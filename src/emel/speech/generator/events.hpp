#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"

namespace emel::speech::generator::events {

struct initialize_done;
struct initialize_error;
struct generation_done;
struct generation_error;
struct stream_frame_done;
struct stream_frame_error;
struct flush_done;
struct flush_error;

} // namespace emel::speech::generator::events

namespace emel::speech::generator::event {

struct initialize {
  explicit initialize(emel::error::type &error_out_ref) noexcept
      : error_out(error_out_ref) {}

  emel::error::type &error_out;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct generate {
  generate(const std::string_view text_ref, std::span<float> pcm_out_ref,
           int32_t &sample_count_out_ref,
           emel::error::type &error_out_ref) noexcept
      : text(text_ref), pcm_out(pcm_out_ref),
        sample_count_out(sample_count_out_ref), error_out(error_out_ref) {}

  std::string_view text = {};
  std::span<const float> reference_pcm = {};
  std::span<float> pcm_out = {};
  int32_t &sample_count_out;
  emel::error::type &error_out;
  emel::callback<void(const events::generation_done &)> on_done = {};
  emel::callback<void(const events::generation_error &)> on_error = {};
};

struct stream_frame {
  stream_frame(std::span<const float> pcm_in_ref, std::span<float> pcm_out_ref,
               int32_t &sample_count_out_ref, bool &produced_out_ref,
               emel::error::type &error_out_ref) noexcept
      : pcm_in(pcm_in_ref), pcm_out(pcm_out_ref),
        sample_count_out(sample_count_out_ref), produced_out(produced_out_ref),
        error_out(error_out_ref) {}

  std::span<const float> pcm_in = {};
  std::span<float> pcm_out = {};
  int32_t &sample_count_out;
  bool &produced_out;
  emel::error::type &error_out;
  emel::callback<void(const events::stream_frame_done &)> on_done = {};
  emel::callback<void(const events::stream_frame_error &)> on_error = {};
};

struct flush {
  flush(std::span<float> pcm_out_ref, int32_t &sample_count_out_ref,
        bool &complete_out_ref, emel::error::type &error_out_ref) noexcept
      : pcm_out(pcm_out_ref), sample_count_out(sample_count_out_ref),
        complete_out(complete_out_ref), error_out(error_out_ref) {}

  std::span<float> pcm_out = {};
  int32_t &sample_count_out;
  bool &complete_out;
  emel::error::type &error_out;
  emel::callback<void(const events::flush_done &)> on_done = {};
  emel::callback<void(const events::flush_error &)> on_error = {};
};

struct reset {
  emel::error::type &error_out;
};

struct initialize_ctx {
  emel::error::type err = {};
};

struct generate_ctx {
  emel::error::type err = {};
};

struct stream_frame_ctx {
  emel::error::type err = {};
};

struct flush_ctx {
  emel::error::type err = {};
};

struct initialize_run {
  const initialize &request;
  initialize_ctx &ctx;
};

struct generate_run {
  const generate &request;
  generate_ctx &ctx;
};

struct stream_frame_run {
  const stream_frame &request;
  stream_frame_ctx &ctx;
};

struct flush_run {
  const flush &request;
  flush_ctx &ctx;
};

} // namespace emel::speech::generator::event

namespace emel::speech::generator::events {

struct initialize_done {
  const event::initialize &request;
};

struct initialize_error {
  const event::initialize &request;
  emel::error::type err = {};
};

struct generation_done {
  const event::generate &request;
  int32_t sample_count = 0;
};

struct generation_error {
  const event::generate &request;
  emel::error::type err = {};
};

struct stream_frame_done {
  const event::stream_frame &request;
  int32_t sample_count = 0;
  bool produced = false;
};

struct stream_frame_error {
  const event::stream_frame &request;
  emel::error::type err = {};
};

struct flush_done {
  const event::flush &request;
  int32_t sample_count = 0;
  bool complete = false;
};

struct flush_error {
  const event::flush &request;
  emel::error::type err = {};
};

} // namespace emel::speech::generator::events
