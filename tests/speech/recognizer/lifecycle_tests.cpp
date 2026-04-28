#include "doctest/doctest.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/recognizer/sm.hpp"

namespace {

namespace recognizer = emel::speech::recognizer;

inline constexpr std::string_view k_backend_sha = "test-backend-sha";

struct initialize_error_capture {
  int32_t calls = 0;
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);
};

struct initialize_done_capture {
  int32_t calls = 0;
  const emel::speech::recognizer::event::initialize *request = nullptr;
};

struct recognition_done_capture {
  int32_t calls = 0;
  const emel::speech::recognizer::event::recognize *request = nullptr;
  int32_t transcript_size = 0;
  int32_t selected_token = 0;
  float confidence = 0.0f;
  int32_t encoder_frame_count = 0;
  int32_t encoder_width = 0;
  int32_t generated_token_count = 0;
  uint64_t encoder_digest = 0u;
  uint64_t decoder_digest = 0u;
};

struct recognition_error_capture {
  int32_t calls = 0;
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);
};

void record_initialize_done(
    initialize_done_capture &capture,
    const emel::speech::recognizer::events::initialize_done &ev) noexcept {
  ++capture.calls;
  capture.request = ev.request;
}

void record_initialize_error(
    initialize_error_capture &capture,
    const emel::speech::recognizer::events::initialize_error &ev) noexcept {
  ++capture.calls;
  capture.err = ev.err;
}

void record_recognition_done(
    recognition_done_capture &capture,
    const emel::speech::recognizer::events::recognition_done &ev) noexcept {
  ++capture.calls;
  capture.request = ev.request;
  capture.transcript_size = ev.transcript_size;
  capture.selected_token = ev.selected_token;
  capture.confidence = ev.confidence;
  capture.encoder_frame_count = ev.encoder_frame_count;
  capture.encoder_width = ev.encoder_width;
  capture.generated_token_count = ev.generated_token_count;
  capture.encoder_digest = ev.encoder_digest;
  capture.decoder_digest = ev.decoder_digest;
}

void record_recognition_error(
    recognition_error_capture &capture,
    const emel::speech::recognizer::events::recognition_error &ev) noexcept {
  ++capture.calls;
  capture.err = ev.err;
}

void noop_initialize_done(
    const emel::speech::recognizer::events::initialize_done &) noexcept {}

void noop_recognition_done(
    const emel::speech::recognizer::events::recognition_done &) noexcept {}

bool backend_supports_tokenizer(
    const recognizer::event::tokenizer_assets &assets) noexcept {
  return assets.model_json.data() != nullptr && !assets.model_json.empty() &&
         assets.sha256 == k_backend_sha;
}

bool backend_supports_model(const emel::model::data &) noexcept { return true; }

bool backend_recognition_ready(
    const recognizer::event::recognize &request) noexcept {
  return request.storage.encoder_workspace.data() != nullptr &&
         request.storage.encoder_state.data() != nullptr &&
         request.storage.decoder_workspace.data() != nullptr &&
         request.storage.logits.data() != nullptr &&
         request.storage.generated_tokens.data() != nullptr &&
         !request.storage.generated_tokens.empty();
}

void backend_encode(const recognizer::event::recognize &,
                    recognizer::event::recognize_ctx &ctx) noexcept {
  ctx.encoder_accepted = true;
  ctx.encoder_frame_count = 1;
  ctx.encoder_width = 8;
  ctx.encoder_digest = 1234u;
  ctx.err = emel::error::cast(recognizer::error::none);
}

void backend_decode(const recognizer::event::recognize &request,
                    recognizer::event::recognize_ctx &ctx) noexcept {
  request.storage.generated_tokens[0] = 67;
  ctx.generated_token_count = 1;
  ctx.decoder_accepted = true;
  ctx.selected_token = 67;
  ctx.confidence = 0.875f;
  ctx.decoder_digest = 5678u;
  ctx.err = emel::error::cast(recognizer::error::none);
}

void backend_detokenize(const recognizer::event::recognize &request,
                        recognizer::event::recognize_ctx &ctx) noexcept {
  request.transcript[0] = 'O';
  request.transcript[1] = 'K';
  ctx.transcript_size = 2;
  ctx.err = emel::error::cast(recognizer::error::none);
}

struct guard_test_tokenizer_supported {
  bool
  operator()(const recognizer::event::tokenizer_assets &assets) const noexcept {
    return backend_supports_tokenizer(assets);
  }
};

struct guard_test_model_supported {
  bool operator()(const emel::model::data &model) const noexcept {
    return backend_supports_model(model);
  }
};

struct guard_test_recognition_ready {
  bool operator()(const recognizer::event::recognize &request) const noexcept {
    return backend_recognition_ready(request);
  }
};

struct effect_test_encode {
  void operator()(const recognizer::event::recognize_run &runtime_ev,
                  recognizer::action::context &) const noexcept {
    backend_encode(runtime_ev.request, runtime_ev.ctx);
  }
};

struct effect_test_decode {
  void operator()(const recognizer::event::recognize_run &runtime_ev,
                  recognizer::action::context &) const noexcept {
    backend_decode(runtime_ev.request, runtime_ev.ctx);
  }
};

struct effect_test_detokenize {
  void operator()(const recognizer::event::recognize_run &runtime_ev,
                  recognizer::action::context &) const noexcept {
    backend_detokenize(runtime_ev.request, runtime_ev.ctx);
  }
};

struct test_route {
  using guard_tokenizer_supported = guard_test_tokenizer_supported;
  using guard_model_supported = guard_test_model_supported;
  using guard_recognition_ready = guard_test_recognition_ready;
  using effect_encode = effect_test_encode;
  using effect_decode = effect_test_decode;
  using effect_detokenize = effect_test_detokenize;
};

} // namespace

TEST_CASE("speech_recognizer_rejects_invalid_initialize") {
  emel::speech::recognizer::sm recognizer{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);
  initialize_error_capture capture{};

  emel::speech::recognizer::event::initialize initialize_ev{
      *model, emel::speech::recognizer::event::tokenizer_assets{}};
  initialize_ev.error_out = &err;
  initialize_ev.on_error = emel::callback<void(
      const emel::speech::recognizer::events::initialize_error &)>::
      from<initialize_error_capture, record_initialize_error>(&capture);

  CHECK_FALSE(recognizer.process_event(initialize_ev));
  CHECK(err ==
        emel::error::cast(emel::speech::recognizer::error::invalid_request));
  CHECK(capture.calls == 1);
  CHECK(capture.err ==
        emel::error::cast(emel::speech::recognizer::error::invalid_request));
}

TEST_CASE("speech_recognizer_rejects_unsupported_initialized_model") {
  emel::speech::recognizer::sm recognizer{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);

  emel::speech::recognizer::event::initialize initialize_ev{
      *model, emel::speech::recognizer::event::tokenizer_assets{
                  .model_json = "{}",
                  .sha256 = k_backend_sha,
              }};
  initialize_ev.error_out = &err;

  CHECK_FALSE(recognizer.process_event(initialize_ev));
  CHECK(err ==
        emel::error::cast(emel::speech::recognizer::error::unsupported_model));
}

TEST_CASE("speech_recognizer_rejects_initialize_after_initialize_error") {
  emel::speech::recognizer::sm<test_route> recognizer{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);
  initialize_error_capture first_error{};
  initialize_error_capture second_error{};

  emel::speech::recognizer::event::initialize invalid_initialize_ev{
      *model, emel::speech::recognizer::event::tokenizer_assets{}};
  invalid_initialize_ev.error_out = &err;
  invalid_initialize_ev.on_error = emel::callback<void(
      const emel::speech::recognizer::events::initialize_error &)>::
      from<initialize_error_capture, record_initialize_error>(&first_error);

  CHECK_FALSE(recognizer.process_event(invalid_initialize_ev));
  CHECK(err ==
        emel::error::cast(emel::speech::recognizer::error::invalid_request));
  CHECK(first_error.calls == 1);
  CHECK(recognizer.is(boost::sml::state<recognizer::state_errored>));

  err = emel::error::cast(emel::speech::recognizer::error::none);
  emel::speech::recognizer::event::initialize valid_initialize_ev{
      *model, emel::speech::recognizer::event::tokenizer_assets{
                  .model_json = "{}",
                  .sha256 = k_backend_sha,
              }};
  valid_initialize_ev.error_out = &err;
  valid_initialize_ev.on_error = emel::callback<void(
      const emel::speech::recognizer::events::initialize_error &)>::
      from<initialize_error_capture, record_initialize_error>(&second_error);

  CHECK_FALSE(recognizer.process_event(valid_initialize_ev));
  CHECK(err ==
        emel::error::cast(emel::speech::recognizer::error::invalid_request));
  CHECK(second_error.calls == 1);
  CHECK(second_error.err ==
        emel::error::cast(emel::speech::recognizer::error::invalid_request));
  CHECK(recognizer.is(boost::sml::state<recognizer::state_errored>));
}

TEST_CASE("speech_recognizer_runs_backend_route_from_public_events") {
  emel::speech::recognizer::sm<test_route> recognizer{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);
  initialize_done_capture initialize_capture{};

  emel::speech::recognizer::event::initialize initialize_ev{
      *model, emel::speech::recognizer::event::tokenizer_assets{
                  .model_json = "{}",
                  .sha256 = k_backend_sha,
              }};
  initialize_ev.error_out = &err;
  initialize_ev.on_done = emel::callback<void(
      const emel::speech::recognizer::events::initialize_done &)>::
      from<initialize_done_capture, record_initialize_done>(
          &initialize_capture);

  REQUIRE(recognizer.process_event(initialize_ev));
  CHECK(initialize_capture.calls == 1);
  CHECK(err == emel::error::cast(emel::speech::recognizer::error::none));

  std::array<float, 4> pcm{};
  std::array<float, 4> encoder_workspace{};
  std::array<float, 8> encoder_state{};
  std::array<float, 8> decoder_workspace{};
  std::array<float, 16> logits{};
  std::array<int32_t, 4> generated_tokens{};
  std::array<char, 8> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  int32_t generated_token_count = -1;
  recognition_done_capture recognition_capture{};

  emel::speech::recognizer::event::recognize recognize_ev{
      *model,
      emel::speech::recognizer::event::tokenizer_assets{
          .model_json = "{}",
          .sha256 = k_backend_sha,
      },
      std::span<const float>{pcm},
      16000,
      std::span<char>{transcript},
      transcript_size,
      selected_token,
      confidence,
      encoder_frames,
      encoder_width,
      encoder_digest,
      decoder_digest,
      generated_token_count};
  recognize_ev.storage = emel::speech::recognizer::event::runtime_storage{
      .encoder_workspace = std::span<float>{encoder_workspace},
      .encoder_state = std::span<float>{encoder_state},
      .decoder_workspace = std::span<float>{decoder_workspace},
      .logits = std::span<float>{logits},
      .generated_tokens = std::span<int32_t>{generated_tokens},
  };
  recognize_ev.error_out = &err;
  recognize_ev.on_done = emel::callback<void(
      const emel::speech::recognizer::events::recognition_done &)>::
      from<recognition_done_capture, record_recognition_done>(
          &recognition_capture);

  REQUIRE(recognizer.process_event(recognize_ev));
  CHECK(err == emel::error::cast(emel::speech::recognizer::error::none));
  CHECK(transcript_size == 2);
  CHECK(transcript[0] == 'O');
  CHECK(transcript[1] == 'K');
  CHECK(selected_token == 67);
  CHECK(confidence == doctest::Approx(0.875f));
  CHECK(encoder_frames == 1);
  CHECK(encoder_width == 8);
  CHECK(encoder_digest == 1234u);
  CHECK(decoder_digest == 5678u);
  CHECK(generated_token_count == 1);
  CHECK(recognition_capture.calls == 1);
  CHECK(recognition_capture.transcript_size == 2);
  CHECK(recognition_capture.generated_token_count == 1);
}

TEST_CASE("speech_recognizer_returns_to_ready_after_rejected_recognition") {
  emel::speech::recognizer::sm<test_route> recognizer{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);

  emel::speech::recognizer::event::initialize initialize_ev{
      *model, emel::speech::recognizer::event::tokenizer_assets{
                  .model_json = "{}",
                  .sha256 = k_backend_sha,
              }};
  initialize_ev.error_out = &err;
  REQUIRE(recognizer.process_event(initialize_ev));
  CHECK(err == emel::error::cast(emel::speech::recognizer::error::none));

  std::array<char, 8> invalid_transcript{};
  int32_t invalid_transcript_size = -1;
  int32_t invalid_token = -1;
  float invalid_confidence = -1.0f;
  int32_t invalid_frames = -1;
  int32_t invalid_width = -1;
  uint64_t invalid_encoder_digest = 1u;
  uint64_t invalid_decoder_digest = 1u;
  recognition_error_capture error_capture{};

  emel::speech::recognizer::event::recognize invalid_recognize_ev{
      *model,
      emel::speech::recognizer::event::tokenizer_assets{
          .model_json = "{}",
          .sha256 = k_backend_sha,
      },
      std::span<const float>{},
      16000,
      std::span<char>{invalid_transcript},
      invalid_transcript_size,
      invalid_token,
      invalid_confidence,
      invalid_frames,
      invalid_width,
      invalid_encoder_digest,
      invalid_decoder_digest};
  invalid_recognize_ev.error_out = &err;
  invalid_recognize_ev.on_error = emel::callback<void(
      const emel::speech::recognizer::events::recognition_error &)>::
      from<recognition_error_capture, record_recognition_error>(&error_capture);

  CHECK_FALSE(recognizer.process_event(invalid_recognize_ev));
  CHECK(err ==
        emel::error::cast(emel::speech::recognizer::error::invalid_request));
  CHECK(error_capture.calls == 1);

  std::array<float, 4> pcm{};
  std::array<float, 4> encoder_workspace{};
  std::array<float, 8> encoder_state{};
  std::array<float, 8> decoder_workspace{};
  std::array<float, 16> logits{};
  std::array<int32_t, 4> generated_tokens{};
  std::array<char, 8> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  int32_t generated_token_count = -1;

  emel::speech::recognizer::event::recognize valid_recognize_ev{
      *model,
      emel::speech::recognizer::event::tokenizer_assets{
          .model_json = "{}",
          .sha256 = k_backend_sha,
      },
      std::span<const float>{pcm},
      16000,
      std::span<char>{transcript},
      transcript_size,
      selected_token,
      confidence,
      encoder_frames,
      encoder_width,
      encoder_digest,
      decoder_digest,
      generated_token_count};
  valid_recognize_ev.storage = emel::speech::recognizer::event::runtime_storage{
      .encoder_workspace = std::span<float>{encoder_workspace},
      .encoder_state = std::span<float>{encoder_state},
      .decoder_workspace = std::span<float>{decoder_workspace},
      .logits = std::span<float>{logits},
      .generated_tokens = std::span<int32_t>{generated_tokens},
  };
  err = emel::error::cast(emel::speech::recognizer::error::none);
  valid_recognize_ev.error_out = &err;

  REQUIRE(recognizer.process_event(valid_recognize_ev));
  CHECK(err == emel::error::cast(emel::speech::recognizer::error::none));
  CHECK(transcript_size == 2);
  CHECK(transcript[0] == 'O');
  CHECK(transcript[1] == 'K');
  CHECK(selected_token == 67);
}

TEST_CASE("speech_recognizer_rejects_recognition_before_initialize") {
  emel::speech::recognizer::sm recognizer{};
  auto model = std::make_unique<emel::model::data>();
  std::array<float, 160> pcm{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  emel::error::type err =
      emel::error::cast(emel::speech::recognizer::error::none);
  recognition_error_capture capture{};

  emel::speech::recognizer::event::recognize recognize_ev{
      *model,
      "{}",
      std::span<const float>{pcm},
      16000,
      std::span<char>{transcript},
      transcript_size,
      selected_token,
      confidence,
      encoder_frames,
      encoder_width,
      encoder_digest,
      decoder_digest};
  recognize_ev.error_out = &err;
  recognize_ev.on_error = emel::callback<void(
      const emel::speech::recognizer::events::recognition_error &)>::
      from<recognition_error_capture, record_recognition_error>(&capture);

  CHECK_FALSE(recognizer.process_event(recognize_ev));
  CHECK(err ==
        emel::error::cast(emel::speech::recognizer::error::uninitialized));
  CHECK(capture.calls == 1);
  CHECK(capture.err ==
        emel::error::cast(emel::speech::recognizer::error::uninitialized));
}

TEST_CASE("speech_recognizer_guards_keep_generic_boundary_explicit") {
  namespace recognizer = emel::speech::recognizer;

  auto model = std::make_unique<emel::model::data>();
  recognizer::action::context ctx{};
  emel::error::type err = emel::error::cast(recognizer::error::none);
  initialize_error_capture initialize_capture{};

  recognizer::event::initialize invalid_initialize{
      *model, recognizer::event::tokenizer_assets{}};
  invalid_initialize.error_out = &err;
  recognizer::event::initialize_ctx invalid_initialize_ctx{};
  recognizer::event::initialize_run invalid_initialize_run{
      invalid_initialize, invalid_initialize_ctx};

  CHECK_FALSE(
      recognizer::guard::guard_valid_initialize{}(invalid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_invalid_initialize{}(invalid_initialize_run,
                                                      ctx));
  CHECK_FALSE(
      recognizer::guard::guard_initialize_tokenizer_supported<test_route>{}(
          invalid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_initialize_tokenizer_unsupported<test_route>{}(
      invalid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_has_initialize_error_out{}(
      invalid_initialize_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_no_initialize_error_out{}(
      invalid_initialize_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_has_initialize_done_callback{}(
      invalid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_no_initialize_done_callback{}(
      invalid_initialize_run, ctx));

  recognizer::event::initialize valid_initialize{*model, "{}"};
  valid_initialize.tokenizer.sha256 = k_backend_sha;
  valid_initialize.on_done =
      emel::callback<void(const recognizer::events::initialize_done &)>::from<
          &noop_initialize_done>();
  valid_initialize.on_error =
      emel::callback<void(const recognizer::events::initialize_error &)>::from<
          initialize_error_capture, record_initialize_error>(
          &initialize_capture);
  recognizer::event::initialize_ctx valid_initialize_ctx{};
  recognizer::event::initialize_run valid_initialize_run{valid_initialize,
                                                         valid_initialize_ctx};

  CHECK(recognizer::guard::guard_valid_initialize{}(valid_initialize_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_invalid_initialize{}(valid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_initialize_tokenizer_supported<test_route>{}(
      valid_initialize_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_initialize_tokenizer_unsupported<test_route>{}(
          valid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_initialize_model_supported<test_route>{}(
      valid_initialize_run, ctx));
  CHECK(recognizer::guard::
            guard_initialize_model_supported_and_route_storage_ready<
                test_route>{}(valid_initialize_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_initialize_unsupported_model<test_route>{}(
          valid_initialize_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_has_initialize_error_out{}(
      valid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_no_initialize_error_out{}(valid_initialize_run,
                                                           ctx));
  CHECK(recognizer::guard::guard_has_initialize_done_callback{}(
      valid_initialize_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_no_initialize_done_callback{}(
      valid_initialize_run, ctx));
  CHECK(recognizer::guard::guard_has_initialize_error_callback{}(
      valid_initialize_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_no_initialize_error_callback{}(
      valid_initialize_run, ctx));
}

TEST_CASE("speech_recognizer_recognize_guards_stay_model_family_free") {
  namespace recognizer = emel::speech::recognizer;

  auto model = std::make_unique<emel::model::data>();
  recognizer::action::context ctx{};
  std::array<float, 160> pcm{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  emel::error::type err = emel::error::cast(recognizer::error::none);
  recognition_error_capture recognition_capture{};

  recognizer::event::recognize valid_recognize{*model,
                                               "{}",
                                               std::span<const float>{pcm},
                                               16000,
                                               std::span<char>{transcript},
                                               transcript_size,
                                               selected_token,
                                               confidence,
                                               encoder_frames,
                                               encoder_width,
                                               encoder_digest,
                                               decoder_digest};
  valid_recognize.error_out = &err;
  valid_recognize.on_done =
      emel::callback<void(const recognizer::events::recognition_done &)>::from<
          &noop_recognition_done>();
  valid_recognize.on_error =
      emel::callback<void(const recognizer::events::recognition_error &)>::from<
          recognition_error_capture, record_recognition_error>(
          &recognition_capture);
  recognizer::event::recognize_ctx valid_ctx{};
  recognizer::event::recognize_run valid_run{valid_recognize, valid_ctx};

  CHECK(recognizer::guard::guard_valid_recognize{}(valid_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_invalid_recognize{}(valid_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_recognizer_route_ready<test_route>{}(
      valid_run, ctx));
  CHECK(recognizer::guard::guard_recognizer_route_unsupported<test_route>{}(
      valid_run, ctx));
  std::array<float, 4> encoder_workspace{};
  std::array<float, 8> encoder_state{};
  std::array<float, 8> decoder_workspace{};
  std::array<float, 16> logits{};
  std::array<int32_t, 4> generated_tokens{};
  valid_recognize.storage = recognizer::event::runtime_storage{
      .encoder_workspace = std::span<float>{encoder_workspace},
      .encoder_state = std::span<float>{encoder_state},
      .decoder_workspace = std::span<float>{decoder_workspace},
      .logits = std::span<float>{logits},
      .generated_tokens = std::span<int32_t>{generated_tokens},
  };
  CHECK(recognizer::guard::guard_recognizer_route_ready<test_route>{}(valid_run,
                                                                      ctx));
  CHECK_FALSE(
      recognizer::guard::guard_recognizer_route_unsupported<test_route>{}(
          valid_run, ctx));
  CHECK(recognizer::guard::guard_has_recognize_error_out{}(valid_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_no_recognize_error_out{}(valid_run, ctx));
  CHECK(recognizer::guard::guard_has_recognize_done_callback{}(valid_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_no_recognize_done_callback{}(valid_run, ctx));
  CHECK(
      recognizer::guard::guard_has_recognize_error_callback{}(valid_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_no_recognize_error_callback{}(valid_run, ctx));

  recognizer::event::recognize invalid_recognize{
      *model,         "{}",           std::span<const float>{},
      16000,          transcript,     transcript_size,
      selected_token, confidence,     encoder_frames,
      encoder_width,  encoder_digest, decoder_digest};
  recognizer::event::recognize_ctx invalid_ctx{};
  recognizer::event::recognize_run invalid_run{invalid_recognize, invalid_ctx};

  CHECK_FALSE(recognizer::guard::guard_valid_recognize{}(invalid_run, ctx));
  CHECK(recognizer::guard::guard_invalid_recognize{}(invalid_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_has_recognize_error_out{}(invalid_run, ctx));
  CHECK(recognizer::guard::guard_no_recognize_error_out{}(invalid_run, ctx));
  CHECK_FALSE(
      recognizer::guard::guard_has_recognize_done_callback{}(invalid_run, ctx));
  CHECK(
      recognizer::guard::guard_no_recognize_done_callback{}(invalid_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_has_recognize_error_callback{}(
      invalid_run, ctx));
  CHECK(
      recognizer::guard::guard_no_recognize_error_callback{}(invalid_run, ctx));

  CHECK_FALSE(recognizer::guard::guard_encoder_success{}(valid_run, ctx));
  CHECK(recognizer::guard::guard_encoder_failure{}(valid_run, ctx));
  valid_ctx.encoder_accepted = true;
  valid_ctx.err = emel::error::cast(recognizer::error::none);
  CHECK(recognizer::guard::guard_encoder_success{}(valid_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_encoder_failure{}(valid_run, ctx));
  valid_ctx.err = emel::error::cast(recognizer::error::backend);
  CHECK_FALSE(recognizer::guard::guard_encoder_success{}(valid_run, ctx));
  CHECK(recognizer::guard::guard_encoder_failure{}(valid_run, ctx));

  valid_ctx.err = emel::error::cast(recognizer::error::none);
  CHECK_FALSE(recognizer::guard::guard_decoder_success{}(valid_run, ctx));
  CHECK(recognizer::guard::guard_decoder_failure{}(valid_run, ctx));
  valid_ctx.decoder_accepted = true;
  CHECK(recognizer::guard::guard_decoder_success{}(valid_run, ctx));
  CHECK_FALSE(recognizer::guard::guard_decoder_failure{}(valid_run, ctx));
}

TEST_CASE("speech_recognizer_actions_store_and_emit_dispatch_outputs") {
  namespace recognizer = emel::speech::recognizer;

  auto model = std::make_unique<emel::model::data>();
  recognizer::action::context ctx{};
  emel::error::type err = emel::error::cast(recognizer::error::none);
  initialize_done_capture initialize_done{};
  initialize_error_capture initialize_error{};

  recognizer::event::initialize initialize_ev{*model, "{}"};
  initialize_ev.tokenizer.sha256 = k_backend_sha;
  initialize_ev.error_out = &err;
  initialize_ev.on_done =
      emel::callback<void(const recognizer::events::initialize_done &)>::from<
          initialize_done_capture, record_initialize_done>(&initialize_done);
  initialize_ev.on_error =
      emel::callback<void(const recognizer::events::initialize_error &)>::from<
          initialize_error_capture, record_initialize_error>(&initialize_error);
  recognizer::event::initialize_ctx initialize_ctx{};
  recognizer::event::initialize_run initialize_run{initialize_ev,
                                                   initialize_ctx};

  recognizer::action::effect_begin_initialize(initialize_run, ctx);
  CHECK(initialize_ctx.err == emel::error::cast(recognizer::error::none));
  recognizer::action::effect_reject_initialize(initialize_run, ctx);
  CHECK(initialize_ctx.err ==
        emel::error::cast(recognizer::error::invalid_request));
  recognizer::action::effect_mark_tokenizer_invalid(initialize_run, ctx);
  CHECK(initialize_ctx.err ==
        emel::error::cast(recognizer::error::tokenizer_invalid));
  recognizer::action::effect_mark_unsupported_model(initialize_run, ctx);
  CHECK(initialize_ctx.err ==
        emel::error::cast(recognizer::error::unsupported_model));
  recognizer::action::effect_mark_initialize_backend_error(initialize_run, ctx);
  CHECK(initialize_ctx.err == emel::error::cast(recognizer::error::backend));

  recognizer::action::effect_store_initialize_error(initialize_run, ctx);
  CHECK(err == emel::error::cast(recognizer::error::backend));
  initialize_ctx.err = emel::error::cast(recognizer::error::none);
  recognizer::action::effect_store_initialize_success(initialize_run, ctx);
  CHECK(err == emel::error::cast(recognizer::error::none));
  recognizer::action::effect_emit_initialize_done(initialize_run, ctx);
  CHECK(initialize_done.calls == 1);
  CHECK(initialize_done.request == &initialize_ev);
  initialize_ctx.err = emel::error::cast(recognizer::error::backend);
  recognizer::action::effect_emit_initialize_error(initialize_run, ctx);
  CHECK(initialize_error.calls == 1);
  CHECK(initialize_error.err == emel::error::cast(recognizer::error::backend));

  std::array<float, 160> pcm{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  int32_t generated_token_count = -1;
  recognition_done_capture recognition_done{};
  recognition_error_capture recognition_error{};

  recognizer::event::recognize recognize_ev{*model,
                                            "{}",
                                            std::span<const float>{pcm},
                                            16000,
                                            std::span<char>{transcript},
                                            transcript_size,
                                            selected_token,
                                            confidence,
                                            encoder_frames,
                                            encoder_width,
                                            encoder_digest,
                                            decoder_digest,
                                            generated_token_count};
  recognize_ev.error_out = &err;
  recognize_ev.on_done =
      emel::callback<void(const recognizer::events::recognition_done &)>::from<
          recognition_done_capture, record_recognition_done>(&recognition_done);
  recognize_ev.on_error =
      emel::callback<void(const recognizer::events::recognition_error &)>::from<
          recognition_error_capture, record_recognition_error>(
          &recognition_error);
  recognizer::event::recognize_ctx recognize_ctx{};
  recognizer::event::recognize_run recognize_run{recognize_ev, recognize_ctx};

  recognizer::action::effect_begin_recognize(recognize_run, ctx);
  CHECK(recognize_ctx.err == emel::error::cast(recognizer::error::none));
  CHECK_FALSE(recognize_ctx.encoder_accepted);
  CHECK_FALSE(recognize_ctx.decoder_accepted);
  CHECK(recognize_ctx.generated_token_count == 0);
  CHECK(transcript_size == 0);
  CHECK(generated_token_count == 0);

  recognizer::action::effect_reject_recognize(recognize_run, ctx);
  CHECK(recognize_ctx.err ==
        emel::error::cast(recognizer::error::invalid_request));
  CHECK(transcript_size == 0);
  recognizer::action::effect_mark_uninitialized(recognize_run, ctx);
  CHECK(recognize_ctx.err ==
        emel::error::cast(recognizer::error::uninitialized));
  recognizer::action::effect_mark_backend_error(recognize_run, ctx);
  CHECK(recognize_ctx.err == emel::error::cast(recognizer::error::backend));

  recognize_ctx.selected_token = 42;
  recognize_ctx.confidence = 0.875f;
  recognize_ctx.encoder_frame_count = 7;
  recognize_ctx.encoder_width = 384;
  recognize_ctx.encoder_digest = 1234u;
  recognize_ctx.decoder_digest = 5678u;
  recognize_ctx.generated_token_count = 9;
  recognize_ctx.transcript_size = 3;
  recognizer::action::effect_publish_recognition_outputs(recognize_run, ctx);
  CHECK(selected_token == 42);
  CHECK(confidence == doctest::Approx(0.875f));
  CHECK(encoder_frames == 7);
  CHECK(encoder_width == 384);
  CHECK(encoder_digest == 1234u);
  CHECK(decoder_digest == 5678u);
  CHECK(generated_token_count == 9);

  recognize_ctx.err = emel::error::cast(recognizer::error::none);
  recognizer::action::effect_store_recognize_success(recognize_run, ctx);
  CHECK(err == emel::error::cast(recognizer::error::none));
  recognize_ctx.err = emel::error::cast(recognizer::error::backend);
  recognizer::action::effect_store_recognize_error(recognize_run, ctx);
  CHECK(err == emel::error::cast(recognizer::error::backend));

  recognizer::action::effect_emit_recognize_done(recognize_run, ctx);
  CHECK(recognition_done.calls == 1);
  CHECK(recognition_done.request == &recognize_ev);
  CHECK(recognition_done.transcript_size == 3);
  CHECK(recognition_done.selected_token == 42);
  CHECK(recognition_done.confidence == doctest::Approx(0.875f));
  CHECK(recognition_done.encoder_frame_count == 7);
  CHECK(recognition_done.encoder_width == 384);
  CHECK(recognition_done.generated_token_count == 9);
  CHECK(recognition_done.encoder_digest == 1234u);
  CHECK(recognition_done.decoder_digest == 5678u);
  recognizer::action::effect_emit_recognize_error(recognize_run, ctx);
  CHECK(recognition_error.calls == 1);
  CHECK(recognition_error.err == emel::error::cast(recognizer::error::backend));

  recognizer::action::effect_on_unexpected(0, ctx);
}
