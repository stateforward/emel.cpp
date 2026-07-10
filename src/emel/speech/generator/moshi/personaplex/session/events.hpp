#pragma once

#include <cstdint>
#include <span>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/generator/moshi/personaplex/session/errors.hpp"

namespace emel::speech::generator::moshi::personaplex::session::events {

struct initialize_done;
struct initialize_error;
struct advance_voice_done;
struct advance_voice_error;
struct advance_prompt_done;
struct advance_prompt_error;
struct live_frame_done;
struct live_frame_error;
struct begin_flush_done;
struct begin_flush_error;
struct flush_frame_done;
struct flush_frame_error;
struct finish_done;
struct finish_error;

} // namespace emel::speech::generator::moshi::personaplex::session::events

namespace emel::speech::generator::moshi::personaplex::session::event {

struct codec_storage {
  std::span<float> prepared = {};
  std::span<float> state = {};
  std::span<float> workspace = {};
  std::span<float> frame = {};
};

struct sampling_config {
  bool enabled = false;
  bool consume_forced_text = false;
  float audio_temperature = 0.0f;
  float text_temperature = 0.0f;
  int32_t audio_top_k = 0;
  int32_t text_top_k = 0;
  uint32_t seed = 0;
};

struct initialize {
  const emel::model::data &mimi_model;
  const emel::model::data &lm_model;
  const emel::model::data &voice_model;
  codec_storage encoder_storage = {};
  codec_storage decoder_storage = {};
  sampling_config sampling = {};
  int32_t max_blocks = 0;
  int32_t block_tokens = 0;
  emel::error::type &error_out;
};

struct advance_voice {
  emel::error::type &error_out;
};

struct advance_prompt {
  int32_t text_token = -1;
  emel::error::type &error_out;
};

struct frame_payload {
  std::span<const float> pcm = {};
  std::span<int32_t> input_codes_out = {};
  std::span<int32_t> output_codes_out = {};
  int32_t &text_token_out;
  bool &produced_out;
  std::span<float> pcm_out = {};
};

struct live_frame {
  frame_payload payload;
  emel::error::type &error_out;
};

struct begin_flush {
  emel::error::type &error_out;
};

struct flush_frame {
  frame_payload payload;
  emel::error::type &error_out;
};

struct finish {
  emel::error::type &error_out;
};

struct initialize_ctx {
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type child_err = 0;
  emel::error::type graph_err = 0;
  bool child_accepted = false;
  int32_t frame_samples = 0;
  int32_t mimi_n_q = 0;
};

struct phase_ctx {
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type child_err = 0;
  emel::error::type graph_err = 0;
  bool child_accepted = false;
  bool complete = false;
  int32_t remaining_frames = -1;
};

struct frame_ctx {
  emel::error::type err = emel::error::cast(error::none);
  emel::error::type child_err = 0;
  emel::error::type graph_err = 0;
  bool child_accepted = false;
  bool produced = false;
};

struct simple_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct initialize_run {
  const initialize &request;
  initialize_ctx &ctx;
};
struct advance_voice_run {
  const advance_voice &request;
  phase_ctx &ctx;
};
struct advance_prompt_run {
  const advance_prompt &request;
  phase_ctx &ctx;
};
struct live_frame_run {
  const live_frame &request;
  frame_ctx &ctx;
};
struct begin_flush_run {
  const begin_flush &request;
  simple_ctx &ctx;
};
struct flush_frame_run {
  const flush_frame &request;
  frame_ctx &ctx;
};
struct finish_run {
  const finish &request;
  simple_ctx &ctx;
};

} // namespace emel::speech::generator::moshi::personaplex::session::event

namespace emel::speech::generator::moshi::personaplex::session::events {

struct initialize_done {
  const event::initialize *request = nullptr;
  int32_t frame_samples = 0;
  int32_t n_q = 0;
};
struct initialize_error {
  const event::initialize *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};
struct advance_voice_done {
  const event::advance_voice *request = nullptr;
  bool complete = false;
  int32_t remaining_frames = -1;
};
struct advance_voice_error {
  const event::advance_voice *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};
struct advance_prompt_done {
  const event::advance_prompt *request = nullptr;
  bool complete = false;
  int32_t remaining_frames = -1;
};
struct advance_prompt_error {
  const event::advance_prompt *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};
struct live_frame_done {
  const event::live_frame *request = nullptr;
  bool produced = false;
};
struct live_frame_error {
  const event::live_frame *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};
struct begin_flush_done {
  const event::begin_flush *request = nullptr;
};
struct begin_flush_error {
  const event::begin_flush *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};
struct flush_frame_done {
  const event::flush_frame *request = nullptr;
  bool produced = false;
};
struct flush_frame_error {
  const event::flush_frame *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};
struct finish_done {
  const event::finish *request = nullptr;
};
struct finish_error {
  const event::finish *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::generator::moshi::personaplex::session::events
