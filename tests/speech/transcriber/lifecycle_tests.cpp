#include "doctest/doctest.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/transcriber/sm.hpp"

namespace {

namespace transcriber = emel::speech::transcriber;

inline constexpr std::string_view k_backend_sha = "test-backend-sha";

struct initialize_error_capture {
  int32_t calls = 0;
  emel::error::type err = emel::error::cast(transcriber::error::none);
};

struct initialize_done_capture {
  int32_t calls = 0;
  const transcriber::event::initialize *request = nullptr;
};

struct recognition_done_capture {
  int32_t calls = 0;
  const transcriber::event::recognize *request = nullptr;
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
  emel::error::type err = emel::error::cast(transcriber::error::none);
};

void record_initialize_done(
    initialize_done_capture &capture,
    const transcriber::events::initialize_done &ev) noexcept {
  ++capture.calls;
  capture.request = ev.request;
}

void record_initialize_error(
    initialize_error_capture &capture,
    const transcriber::events::initialize_error &ev) noexcept {
  ++capture.calls;
  capture.err = ev.err;
}

void record_recognition_done(
    recognition_done_capture &capture,
    const transcriber::events::recognition_done &ev) noexcept {
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
    const transcriber::events::recognition_error &ev) noexcept {
  ++capture.calls;
  capture.err = ev.err;
}

// Synthetic injected dependencies: component-level kind selectors and contracts
// bound against a caller-owned model, mirroring what a variant binder would
// produce at the variant boundary. No variant headers or namespaces are used.
transcriber::dependencies
make_supported_dependencies(const emel::model::data &model) noexcept {
  transcriber::dependencies deps{};
  deps.encoder_kind = emel::speech::encoder::encoder_kind::whisper;
  deps.decoder_kind = emel::speech::decoder::decoder_kind::whisper;
  deps.tokenizer_kind = emel::speech::tokenizer::tokenizer_kind::whisper;
  deps.encoder_contract = emel::speech::encoder::execution_contract{
      .model = &model,
      .sample_rate = 16000,
      .mel_bin_count = 80,
      .embedding_length = 384,
      .feed_forward_length = 1536,
      .attention_head_count = 6,
      .encoder_block_count = 4,
  };
  deps.decoder_contract = emel::speech::decoder::execution_contract{
      .model = &model,
      .vocab_size = 51865,
      .embedding_length = 384,
      .decoder_block_count = 4,
  };
  deps.decode_policy = emel::speech::tokenizer::asr_decode_policy{};
  deps.tokenizer_sha256 = k_backend_sha;
  return deps;
}

transcriber::event::tokenizer_assets make_supported_assets() noexcept {
  return transcriber::event::tokenizer_assets{
      .model_json = "{}",
      .sha256 = k_backend_sha,
  };
}

} // namespace

TEST_CASE("speech_transcriber_rejects_invalid_initialize") {
  transcriber::sm engine{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err = emel::error::cast(transcriber::error::none);
  initialize_error_capture capture{};

  transcriber::event::initialize initialize_ev{
      *model, transcriber::event::tokenizer_assets{}};
  initialize_ev.error_out = &err;
  initialize_ev.on_error =
      emel::callback<void(const transcriber::events::initialize_error &)>::from<
          initialize_error_capture, record_initialize_error>(&capture);

  CHECK_FALSE(engine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::invalid_request));
  CHECK(capture.calls == 1);
  CHECK(capture.err == emel::error::cast(transcriber::error::invalid_request));
}

TEST_CASE("speech_transcriber_rejects_unsupported_tokenizer") {
  transcriber::sm engine{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err = emel::error::cast(transcriber::error::none);
  initialize_error_capture capture{};

  transcriber::event::initialize initialize_ev{*model, make_supported_assets()};
  initialize_ev.error_out = &err;
  initialize_ev.on_error =
      emel::callback<void(const transcriber::events::initialize_error &)>::from<
          initialize_error_capture, record_initialize_error>(&capture);

  CHECK_FALSE(engine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::tokenizer_invalid));
  CHECK(capture.calls == 1);
  CHECK(capture.err ==
        emel::error::cast(transcriber::error::tokenizer_invalid));
}

TEST_CASE("speech_transcriber_rejects_unsupported_initialized_model") {
  auto model = std::make_unique<emel::model::data>();
  transcriber::dependencies deps{};
  deps.tokenizer_kind = emel::speech::tokenizer::tokenizer_kind::whisper;
  deps.tokenizer_sha256 = k_backend_sha;
  transcriber::sm engine{deps};
  emel::error::type err = emel::error::cast(transcriber::error::none);

  transcriber::event::initialize initialize_ev{*model, make_supported_assets()};
  initialize_ev.error_out = &err;

  CHECK_FALSE(engine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::unsupported_model));
}

TEST_CASE("speech_transcriber_rejects_initialize_after_initialize_error") {
  transcriber::sm engine{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err = emel::error::cast(transcriber::error::none);
  initialize_error_capture first_error{};
  initialize_error_capture second_error{};

  transcriber::event::initialize invalid_initialize_ev{
      *model, transcriber::event::tokenizer_assets{}};
  invalid_initialize_ev.error_out = &err;
  invalid_initialize_ev.on_error =
      emel::callback<void(const transcriber::events::initialize_error &)>::from<
          initialize_error_capture, record_initialize_error>(&first_error);

  CHECK_FALSE(engine.process_event(invalid_initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::invalid_request));
  CHECK(first_error.calls == 1);
  CHECK(engine.is(stateforward::sml::state<transcriber::state_errored>));

  err = emel::error::cast(transcriber::error::none);
  transcriber::event::initialize valid_initialize_ev{*model,
                                                     make_supported_assets()};
  valid_initialize_ev.error_out = &err;
  valid_initialize_ev.on_error =
      emel::callback<void(const transcriber::events::initialize_error &)>::from<
          initialize_error_capture, record_initialize_error>(&second_error);

  CHECK_FALSE(engine.process_event(valid_initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::invalid_request));
  CHECK(second_error.calls == 1);
  CHECK(second_error.err ==
        emel::error::cast(transcriber::error::invalid_request));
  CHECK(engine.is(stateforward::sml::state<transcriber::state_errored>));
}

TEST_CASE("speech_transcriber_initializes_with_injected_contracts") {
  auto model = std::make_unique<emel::model::data>();
  const transcriber::dependencies deps = make_supported_dependencies(*model);
  transcriber::sm engine{deps};
  emel::error::type err = emel::error::cast(transcriber::error::none);
  initialize_done_capture done_capture{};

  transcriber::event::initialize initialize_ev{*model, make_supported_assets()};
  initialize_ev.error_out = &err;
  initialize_ev.on_done =
      emel::callback<void(const transcriber::events::initialize_done &)>::from<
          initialize_done_capture, record_initialize_done>(&done_capture);

  REQUIRE(engine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::none));
  CHECK(done_capture.calls == 1);
  CHECK(done_capture.request == &initialize_ev);
  CHECK(engine.is(stateforward::sml::state<transcriber::state_ready>));
}

TEST_CASE("speech_transcriber_reports_backend_error_when_component_rejects") {
  auto model = std::make_unique<emel::model::data>();
  const transcriber::dependencies deps = make_supported_dependencies(*model);
  transcriber::sm engine{deps};
  emel::error::type err = emel::error::cast(transcriber::error::none);

  transcriber::event::initialize initialize_ev{*model, make_supported_assets()};
  initialize_ev.error_out = &err;
  REQUIRE(engine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::none));

  std::array<float, 320> pcm{};
  std::array<float, 64> encoder_workspace{};
  std::array<float, 64> encoder_state{};
  std::array<float, 64> decoder_workspace{};
  std::array<float, 64> logits{};
  std::array<int32_t, 8> generated_tokens{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  int32_t generated_token_count = -1;
  recognition_error_capture error_capture{};

  transcriber::event::recognize recognize_ev{*model,
                                             make_supported_assets(),
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
  recognize_ev.storage = transcriber::event::runtime_storage{
      .encoder_workspace = std::span<float>{encoder_workspace},
      .encoder_state = std::span<float>{encoder_state},
      .decoder_workspace = std::span<float>{decoder_workspace},
      .logits = std::span<float>{logits},
      .generated_tokens = std::span<int32_t>{generated_tokens},
  };
  recognize_ev.error_out = &err;
  recognize_ev.on_error =
      emel::callback<void(const transcriber::events::recognition_error &)>::
          from<recognition_error_capture, record_recognition_error>(
              &error_capture);

  // The injected whisper-kind encoder component rejects the synthetic model
  // (no tensors), so the pipeline reports a backend error and returns to
  // ready.
  CHECK_FALSE(engine.process_event(recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::backend));
  CHECK(error_capture.calls == 1);
  CHECK(error_capture.err == emel::error::cast(transcriber::error::backend));
  CHECK(engine.is(stateforward::sml::state<transcriber::state_ready>));

  err = emel::error::cast(transcriber::error::none);
  transcriber::event::initialize reinitialize_ev{*model,
                                                 make_supported_assets()};
  reinitialize_ev.error_out = &err;
  CHECK(engine.process_event(reinitialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::none));
}

TEST_CASE("speech_transcriber_rejects_recognition_before_initialize") {
  auto model = std::make_unique<emel::model::data>();
  const transcriber::dependencies deps = make_supported_dependencies(*model);
  transcriber::sm engine{deps};
  std::array<float, 160> pcm{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  emel::error::type err = emel::error::cast(transcriber::error::none);
  recognition_error_capture capture{};

  transcriber::event::recognize recognize_ev{*model,
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
  recognize_ev.on_error =
      emel::callback<void(const transcriber::events::recognition_error &)>::
          from<recognition_error_capture, record_recognition_error>(&capture);

  CHECK_FALSE(engine.process_event(recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::uninitialized));
  CHECK(capture.calls == 1);
  CHECK(capture.err == emel::error::cast(transcriber::error::uninitialized));

  // The rejection must return to state_uninitialized, not promote the machine
  // to ready: a second pre-initialize recognize is rejected identically, so
  // the pipeline stays unreachable without a successful initialize.
  CHECK(engine.is(stateforward::sml::state<transcriber::state_uninitialized>));
  err = emel::error::cast(transcriber::error::none);
  CHECK_FALSE(engine.process_event(recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::uninitialized));
  CHECK(capture.calls == 2);
  CHECK(engine.is(stateforward::sml::state<transcriber::state_uninitialized>));

  // A subsequent initialize still succeeds from uninitialized, after which
  // recognize dispatches normally (the injected whisper-kind encoder rejects
  // the synthetic model with a backend error and returns to ready).
  err = emel::error::cast(transcriber::error::none);
  transcriber::event::initialize initialize_ev{*model, make_supported_assets()};
  initialize_ev.error_out = &err;
  REQUIRE(engine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::none));
  CHECK(engine.is(stateforward::sml::state<transcriber::state_ready>));

  err = emel::error::cast(transcriber::error::none);
  transcriber::event::recognize post_init_recognize_ev{
      *model,
      make_supported_assets(),
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
  post_init_recognize_ev.error_out = &err;
  CHECK_FALSE(engine.process_event(post_init_recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::backend));
  CHECK(engine.is(stateforward::sml::state<transcriber::state_ready>));
}

TEST_CASE(
    "speech_transcriber_recognition_after_failed_initialize_stays_errored") {
  transcriber::sm engine{};
  auto model = std::make_unique<emel::model::data>();
  emel::error::type err = emel::error::cast(transcriber::error::none);

  // Drive the machine into state_errored via an invalid initialize.
  transcriber::event::initialize invalid_initialize_ev{
      *model, transcriber::event::tokenizer_assets{}};
  invalid_initialize_ev.error_out = &err;
  CHECK_FALSE(engine.process_event(invalid_initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::invalid_request));
  REQUIRE(engine.is(stateforward::sml::state<transcriber::state_errored>));

  std::array<float, 160> pcm{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  recognition_error_capture capture{};

  transcriber::event::recognize recognize_ev{*model,
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
  recognize_ev.on_error =
      emel::callback<void(const transcriber::events::recognition_error &)>::
          from<recognition_error_capture, record_recognition_error>(&capture);

  // The rejection must return to state_errored, not promote the machine to
  // ready: repeated recognize dispatches keep rejecting, so the pipeline is
  // unreachable without a successful initialize.
  err = emel::error::cast(transcriber::error::none);
  CHECK_FALSE(engine.process_event(recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::uninitialized));
  CHECK(capture.calls == 1);
  CHECK(capture.err == emel::error::cast(transcriber::error::uninitialized));
  CHECK(engine.is(stateforward::sml::state<transcriber::state_errored>));

  err = emel::error::cast(transcriber::error::none);
  CHECK_FALSE(engine.process_event(recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::uninitialized));
  CHECK(capture.calls == 2);
  CHECK(engine.is(stateforward::sml::state<transcriber::state_errored>));
}

TEST_CASE("speech_transcriber_returns_to_ready_after_rejected_recognition") {
  auto model = std::make_unique<emel::model::data>();
  const transcriber::dependencies deps = make_supported_dependencies(*model);
  transcriber::sm engine{deps};
  emel::error::type err = emel::error::cast(transcriber::error::none);

  transcriber::event::initialize initialize_ev{*model, make_supported_assets()};
  initialize_ev.error_out = &err;
  REQUIRE(engine.process_event(initialize_ev));
  CHECK(err == emel::error::cast(transcriber::error::none));

  std::array<char, 32> invalid_transcript{};
  int32_t invalid_transcript_size = -1;
  int32_t invalid_token = -1;
  float invalid_confidence = -1.0f;
  int32_t invalid_frames = -1;
  int32_t invalid_width = -1;
  uint64_t invalid_encoder_digest = 1u;
  uint64_t invalid_decoder_digest = 1u;
  recognition_error_capture error_capture{};

  transcriber::event::recognize invalid_recognize_ev{
      *model,
      make_supported_assets(),
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
  invalid_recognize_ev.on_error =
      emel::callback<void(const transcriber::events::recognition_error &)>::
          from<recognition_error_capture, record_recognition_error>(
              &error_capture);

  CHECK_FALSE(engine.process_event(invalid_recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::invalid_request));
  CHECK(error_capture.calls == 1);
  CHECK(engine.is(stateforward::sml::state<transcriber::state_ready>));

  std::array<float, 320> pcm{};
  std::array<float, 64> encoder_workspace{};
  std::array<float, 64> encoder_state{};
  std::array<float, 64> decoder_workspace{};
  std::array<float, 64> logits{};
  std::array<int32_t, 8> generated_tokens{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;
  int32_t generated_token_count = -1;

  transcriber::event::recognize valid_recognize_ev{*model,
                                                   make_supported_assets(),
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
  valid_recognize_ev.storage = transcriber::event::runtime_storage{
      .encoder_workspace = std::span<float>{encoder_workspace},
      .encoder_state = std::span<float>{encoder_state},
      .decoder_workspace = std::span<float>{decoder_workspace},
      .logits = std::span<float>{logits},
      .generated_tokens = std::span<int32_t>{generated_tokens},
  };
  err = emel::error::cast(transcriber::error::none);
  valid_recognize_ev.error_out = &err;

  // The well-formed request still dispatches from ready (no unexpected-event
  // path): it reaches the injected encoder component, which rejects the
  // synthetic model with a backend error.
  CHECK_FALSE(engine.process_event(valid_recognize_ev));
  CHECK(err == emel::error::cast(transcriber::error::backend));
  CHECK(engine.is(stateforward::sml::state<transcriber::state_ready>));
}

TEST_CASE("speech_transcriber_guards_validate_injected_dependencies") {
  auto model = std::make_unique<emel::model::data>();
  auto other_model = std::make_unique<emel::model::data>();
  const transcriber::dependencies supported =
      make_supported_dependencies(*model);

  const transcriber::event::tokenizer_assets supported_assets =
      make_supported_assets();
  CHECK(transcriber::guard::guard_tokenizer_assets_supported{}(supported_assets,
                                                               supported));

  transcriber::dependencies unsupported_tokenizer = supported;
  unsupported_tokenizer.tokenizer_kind =
      emel::speech::tokenizer::tokenizer_kind::unsupported;
  CHECK_FALSE(transcriber::guard::guard_tokenizer_assets_supported{}(
      supported_assets, unsupported_tokenizer));

  // An out-of-range cast kind (neither `whisper` nor `unsupported`) must be
  // rejected: sm_any would otherwise clamp it to the Whisper facade.
  transcriber::dependencies unknown_tokenizer = supported;
  unknown_tokenizer.tokenizer_kind =
      static_cast<emel::speech::tokenizer::tokenizer_kind>(42);
  CHECK_FALSE(transcriber::guard::guard_tokenizer_assets_supported{}(
      supported_assets, unknown_tokenizer));

  const transcriber::event::tokenizer_assets mismatched_sha_assets{
      .model_json = "{}",
      .sha256 = "other-sha",
  };
  CHECK_FALSE(transcriber::guard::guard_tokenizer_assets_supported{}(
      mismatched_sha_assets, supported));

  transcriber::dependencies empty_sha_deps = supported;
  empty_sha_deps.tokenizer_sha256 = {};
  const transcriber::event::tokenizer_assets empty_sha_assets{
      .model_json = "{}",
      .sha256 = {},
  };
  CHECK_FALSE(transcriber::guard::guard_tokenizer_assets_supported{}(
      empty_sha_assets, empty_sha_deps));

  const transcriber::event::tokenizer_assets empty_json_assets{
      .model_json = {},
      .sha256 = k_backend_sha,
  };
  CHECK_FALSE(transcriber::guard::guard_tokenizer_assets_supported{}(
      empty_json_assets, supported));

  CHECK(
      transcriber::guard::guard_model_contracts_supported{}(*model, supported));

  transcriber::dependencies unsupported_encoder = supported;
  unsupported_encoder.encoder_kind =
      emel::speech::encoder::encoder_kind::unsupported;
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, unsupported_encoder));

  transcriber::dependencies unsupported_decoder = supported;
  unsupported_decoder.decoder_kind =
      emel::speech::decoder::decoder_kind::unsupported;
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, unsupported_decoder));

  // Out-of-range cast component kinds must be rejected, not clamped to Whisper.
  transcriber::dependencies unknown_encoder = supported;
  unknown_encoder.encoder_kind =
      static_cast<emel::speech::encoder::encoder_kind>(42);
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, unknown_encoder));

  transcriber::dependencies unknown_decoder = supported;
  unknown_decoder.decoder_kind =
      static_cast<emel::speech::decoder::decoder_kind>(42);
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, unknown_decoder));

  // Contracts bound against a different model than the request carries.
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *other_model, supported));

  transcriber::dependencies zero_encoder_embedding = supported;
  zero_encoder_embedding.encoder_contract.embedding_length = 0;
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, zero_encoder_embedding));

  transcriber::dependencies zero_vocab = supported;
  zero_vocab.decoder_contract.vocab_size = 0;
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, zero_vocab));

  transcriber::dependencies zero_decoder_embedding = supported;
  zero_decoder_embedding.decoder_contract.embedding_length = 0;
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, zero_decoder_embedding));

  // The decode phase slices the encoder-state buffer as frame_count x decoder
  // embedding_length, so contracts whose embedding lengths disagree are an
  // unsupported dependency combination.
  transcriber::dependencies mismatched_embedding = supported;
  mismatched_embedding.decoder_contract.embedding_length =
      supported.encoder_contract.embedding_length + 1;
  CHECK_FALSE(transcriber::guard::guard_model_contracts_supported{}(
      *model, mismatched_embedding));
}

TEST_CASE("speech_transcriber_encoder_success_requires_state_capacity") {
  auto model = std::make_unique<emel::model::data>();
  transcriber::action::context ctx{};
  ctx.deps = make_supported_dependencies(*model);

  std::array<float, 160> pcm{};
  std::array<float, 768> encoder_state{};
  std::array<char, 32> transcript{};
  int32_t transcript_size = -1;
  int32_t selected_token = -1;
  float confidence = -1.0f;
  int32_t encoder_frames = -1;
  int32_t encoder_width = -1;
  uint64_t encoder_digest = 1u;
  uint64_t decoder_digest = 1u;

  transcriber::event::recognize recognize_ev{*model,
                                             make_supported_assets(),
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
  recognize_ev.storage = transcriber::event::runtime_storage{
      .encoder_state = std::span<float>{encoder_state},
  };
  transcriber::event::recognize_ctx recognize_ctx{};
  transcriber::event::recognize_run recognize_run{recognize_ev, recognize_ctx};

  recognize_ctx.encoder_accepted = true;
  recognize_ctx.err = emel::error::cast(transcriber::error::none);

  // Two frames of the 384-wide contract fit the 768-float buffer.
  recognize_ctx.encoder_frame_count = 2;
  CHECK(transcriber::guard::guard_encoder_success{}(recognize_run, ctx));
  CHECK_FALSE(transcriber::guard::guard_encoder_failure{}(recognize_run, ctx));

  // Three frames would read past the caller's encoder-state span, so the
  // encode outcome must route to the backend-error path.
  recognize_ctx.encoder_frame_count = 3;
  CHECK_FALSE(transcriber::guard::guard_encoder_success{}(recognize_run, ctx));
  CHECK(transcriber::guard::guard_encoder_failure{}(recognize_run, ctx));

  // A negative frame count never counts as encoder success.
  recognize_ctx.encoder_frame_count = -1;
  CHECK_FALSE(transcriber::guard::guard_encoder_success{}(recognize_run, ctx));
}

TEST_CASE("speech_transcriber_actions_store_and_emit_dispatch_outputs") {
  auto model = std::make_unique<emel::model::data>();
  transcriber::action::context ctx{};
  emel::error::type err = emel::error::cast(transcriber::error::none);
  initialize_done_capture initialize_done{};
  initialize_error_capture initialize_error{};

  transcriber::event::initialize initialize_ev{*model, "{}"};
  initialize_ev.tokenizer.sha256 = k_backend_sha;
  initialize_ev.error_out = &err;
  initialize_ev.on_done =
      emel::callback<void(const transcriber::events::initialize_done &)>::from<
          initialize_done_capture, record_initialize_done>(&initialize_done);
  initialize_ev.on_error =
      emel::callback<void(const transcriber::events::initialize_error &)>::from<
          initialize_error_capture, record_initialize_error>(&initialize_error);
  transcriber::event::initialize_ctx initialize_ctx{};
  transcriber::event::initialize_run initialize_run{initialize_ev,
                                                    initialize_ctx};

  transcriber::action::effect_begin_initialize(initialize_run, ctx);
  CHECK(initialize_ctx.err == emel::error::cast(transcriber::error::none));
  transcriber::action::effect_reject_initialize(initialize_run, ctx);
  CHECK(initialize_ctx.err ==
        emel::error::cast(transcriber::error::invalid_request));
  transcriber::action::effect_mark_tokenizer_invalid(initialize_run, ctx);
  CHECK(initialize_ctx.err ==
        emel::error::cast(transcriber::error::tokenizer_invalid));
  transcriber::action::effect_mark_unsupported_model(initialize_run, ctx);
  CHECK(initialize_ctx.err ==
        emel::error::cast(transcriber::error::unsupported_model));
  transcriber::action::effect_mark_initialize_backend_error(initialize_run,
                                                            ctx);
  CHECK(initialize_ctx.err == emel::error::cast(transcriber::error::backend));

  transcriber::action::effect_store_initialize_error(initialize_run, ctx);
  CHECK(err == emel::error::cast(transcriber::error::backend));
  initialize_ctx.err = emel::error::cast(transcriber::error::none);
  transcriber::action::effect_store_initialize_success(initialize_run, ctx);
  CHECK(err == emel::error::cast(transcriber::error::none));
  transcriber::action::effect_emit_initialize_done(initialize_run, ctx);
  CHECK(initialize_done.calls == 1);
  CHECK(initialize_done.request == &initialize_ev);
  initialize_ctx.err = emel::error::cast(transcriber::error::backend);
  transcriber::action::effect_emit_initialize_error(initialize_run, ctx);
  CHECK(initialize_error.calls == 1);
  CHECK(initialize_error.err == emel::error::cast(transcriber::error::backend));

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

  transcriber::event::recognize recognize_ev{*model,
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
      emel::callback<void(const transcriber::events::recognition_done &)>::from<
          recognition_done_capture, record_recognition_done>(&recognition_done);
  recognize_ev.on_error =
      emel::callback<void(const transcriber::events::recognition_error &)>::
          from<recognition_error_capture, record_recognition_error>(
              &recognition_error);
  transcriber::event::recognize_ctx recognize_ctx{};
  transcriber::event::recognize_run recognize_run{recognize_ev, recognize_ctx};

  transcriber::action::effect_begin_recognize(recognize_run, ctx);
  CHECK(recognize_ctx.err == emel::error::cast(transcriber::error::none));
  CHECK_FALSE(recognize_ctx.encoder_accepted);
  CHECK_FALSE(recognize_ctx.decoder_accepted);
  CHECK_FALSE(recognize_ctx.detokenize_accepted);
  CHECK(recognize_ctx.generated_token_count == 0);
  CHECK(transcript_size == 0);
  CHECK(generated_token_count == 0);

  transcriber::action::effect_reject_recognize(recognize_run, ctx);
  CHECK(recognize_ctx.err ==
        emel::error::cast(transcriber::error::invalid_request));
  CHECK(transcript_size == 0);
  transcriber::action::effect_mark_uninitialized(recognize_run, ctx);
  CHECK(recognize_ctx.err ==
        emel::error::cast(transcriber::error::uninitialized));
  transcriber::action::effect_mark_backend_error(recognize_run, ctx);
  CHECK(recognize_ctx.err == emel::error::cast(transcriber::error::backend));

  recognize_ctx.selected_token = 42;
  recognize_ctx.confidence = 0.875f;
  recognize_ctx.encoder_frame_count = 7;
  recognize_ctx.encoder_width = 384;
  recognize_ctx.encoder_digest = 1234u;
  recognize_ctx.decoder_digest = 5678u;
  recognize_ctx.generated_token_count = 9;
  recognize_ctx.transcript_size = 3;
  recognize_ctx.detokenize_accepted = true;
  transcriber::action::effect_publish_recognition_outputs(recognize_run, ctx);
  CHECK(transcript_size == 3);
  CHECK(selected_token == 42);
  CHECK(confidence == doctest::Approx(0.875f));
  CHECK(encoder_frames == 7);
  CHECK(encoder_width == 384);
  CHECK(encoder_digest == 1234u);
  CHECK(decoder_digest == 5678u);
  CHECK(generated_token_count == 9);

  recognize_ctx.err = emel::error::cast(transcriber::error::none);
  transcriber::action::effect_store_recognize_success(recognize_run, ctx);
  CHECK(err == emel::error::cast(transcriber::error::none));
  recognize_ctx.err = emel::error::cast(transcriber::error::backend);
  transcriber::action::effect_store_recognize_error(recognize_run, ctx);
  CHECK(err == emel::error::cast(transcriber::error::backend));

  transcriber::action::effect_emit_recognize_done(recognize_run, ctx);
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
  transcriber::action::effect_emit_recognize_error(recognize_run, ctx);
  CHECK(recognition_error.calls == 1);
  CHECK(recognition_error.err ==
        emel::error::cast(transcriber::error::backend));

  transcriber::action::effect_on_unexpected(0, ctx);
}
