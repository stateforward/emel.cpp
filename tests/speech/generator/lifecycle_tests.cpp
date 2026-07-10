#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>

#include "../../allocation_tracker.hpp"
#include "emel/error/error.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/speech/generator/sm.hpp"

namespace {

namespace generator = emel::speech::generator;
namespace sml = stateforward::sml;

struct fake_initialize {
  emel::error::type *error_out = nullptr;
};

struct fake_voice_condition {
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
  bool *complete_out = nullptr;
  int32_t *remaining_frames_out = nullptr;
};

struct fake_prompt_begin {
  emel::error::type *error_out = nullptr;
};

struct fake_prompt_condition {
  int32_t text_token = -1;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
  bool *complete_out = nullptr;
  int32_t *remaining_frames_out = nullptr;
};

struct fake_encode {
  fake_encode(std::span<const float> pcm_ref,
              std::span<int32_t> tokens_out_ref) noexcept
      : pcm(pcm_ref), tokens_out(tokens_out_ref) {}

  std::span<const float> pcm;
  std::span<int32_t> tokens_out;
  emel::error::type *error_out = nullptr;
};

struct fake_predict {
  fake_predict(std::span<const int32_t> tokens_ref,
               std::span<int32_t> tokens_out_ref,
               int32_t &text_token_out_ref) noexcept
      : tokens(tokens_ref), tokens_out(tokens_out_ref),
        text_token_out(text_token_out_ref) {}

  std::span<const int32_t> tokens;
  std::span<int32_t> tokens_out;
  int32_t &text_token_out;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
  bool *produced_out = nullptr;
};

struct fake_decode {
  fake_decode(std::span<const int32_t> tokens_ref,
              std::span<float> pcm_out_ref) noexcept
      : tokens(tokens_ref), pcm_out(pcm_out_ref) {}

  std::span<const int32_t> tokens;
  std::span<float> pcm_out;
  emel::error::type *error_out = nullptr;
};

struct fake_encoder_actor {
  int32_t initialize_calls = 0;
  int32_t encode_calls = 0;
  bool initialize_accepted = true;
  bool encode_accepted = true;
  emel::error::type initialize_error = 0;
  emel::error::type encode_error = 0;

  bool process_event(const fake_initialize &request) noexcept {
    ++initialize_calls;
    *request.error_out = initialize_error;
    return initialize_accepted;
  }

  bool process_event(const fake_encode &request) noexcept {
    ++encode_calls;
    request.tokens_out[0] = 11;
    request.tokens_out[1] = 12;
    *request.error_out = encode_error;
    return encode_accepted;
  }
};

struct fake_decoder_actor {
  int32_t initialize_calls = 0;
  int32_t decode_calls = 0;
  bool initialize_accepted = true;
  bool decode_accepted = true;
  emel::error::type initialize_error = 0;
  emel::error::type decode_error = 0;

  bool process_event(const fake_initialize &request) noexcept {
    ++initialize_calls;
    *request.error_out = initialize_error;
    return initialize_accepted;
  }

  bool process_event(const fake_decode &request) noexcept {
    ++decode_calls;
    std::fill(request.pcm_out.begin(), request.pcm_out.end(), 0.25f);
    *request.error_out = decode_error;
    return decode_accepted;
  }
};

struct fake_runtime_actor {
  int32_t initialize_calls = 0;
  bool initialize_accepted = true;
  emel::error::type initialize_error = 0;

  bool process_event(const fake_initialize &request) noexcept {
    ++initialize_calls;
    *request.error_out = initialize_error;
    return initialize_accepted;
  }
};

struct fake_predictor_actor {
  int32_t initialize_calls = 0;
  int32_t conditioning_initialize_calls = 0;
  int32_t voice_condition_calls = 0;
  int32_t prompt_begin_calls = 0;
  int32_t prompt_condition_calls = 0;
  int32_t predict_calls = 0;
  int32_t rejected_initialize_call = 0;
  bool voice_accepted = true;
  bool prompt_begin_accepted = true;
  bool prompt_accepted = true;
  bool predict_accepted = true;
  emel::error::type initialize_error = 0;
  emel::error::type voice_error = 0;
  emel::error::type voice_graph_error = 0;
  emel::error::type prompt_begin_error = 0;
  emel::error::type prompt_error = 0;
  emel::error::type prompt_graph_error = 0;
  emel::error::type predict_error = 0;
  emel::error::type predict_graph_error = 0;
  bool voice_complete = true;
  bool prompt_complete = true;
  bool produce = true;

  bool process_event(const fake_initialize &request) noexcept {
    ++initialize_calls;
    *request.error_out = initialize_error;
    return rejected_initialize_call != initialize_calls;
  }

  bool process_event(const fake_prompt_begin &request) noexcept {
    ++prompt_begin_calls;
    *request.error_out = prompt_begin_error;
    return prompt_begin_accepted;
  }

  bool process_event(const fake_voice_condition &request) noexcept {
    ++voice_condition_calls;
    *request.error_out = voice_error;
    *request.graph_error_out = voice_graph_error;
    *request.complete_out = voice_complete;
    *request.remaining_frames_out = voice_complete ? 0 : 1;
    return voice_accepted;
  }

  bool process_event(const fake_prompt_condition &request) noexcept {
    ++prompt_condition_calls;
    *request.error_out = prompt_error;
    *request.graph_error_out = prompt_graph_error;
    *request.complete_out = prompt_complete;
    *request.remaining_frames_out = prompt_complete ? 0 : 1;
    return prompt_accepted;
  }

  bool process_event(const fake_predict &request) noexcept {
    ++predict_calls;
    request.tokens_out[0] = request.tokens[0] + 100;
    request.tokens_out[1] = request.tokens[1] + 100;
    request.text_token_out = 42;
    *request.error_out = predict_error;
    *request.graph_error_out = predict_graph_error;
    *request.produced_out = produce;
    return predict_accepted;
  }
};

struct fake_dependencies {
  using voice_condition_event = fake_voice_condition;
  using prompt_begin_event = fake_prompt_begin;
  using prompt_condition_event = fake_prompt_condition;
  using encode_event = fake_encode;
  using predict_event = fake_predict;
  using decode_event = fake_decode;

  emel::memory::streaming::sm &temporal_positions;
  emel::memory::streaming::sm &secondary_positions;
  fake_encoder_actor &encoder;
  fake_decoder_actor &decoder;
  fake_runtime_actor &runtime;
  fake_predictor_actor &predictor;
  fake_initialize encoder_initialize = {};
  fake_initialize decoder_initialize = {};
  fake_initialize runtime_initialize = {};
  fake_initialize predictor_initialize = {};
  fake_initialize conditioning_initialize = {};
  std::span<float> silence_pcm = {};
  std::span<int32_t> input_codes = {};
  std::span<int32_t> output_codes = {};
  int32_t frame_samples = 0;
  int32_t codebook_count = 0;
};

struct fixture {
  std::array<float, 4> silence{};
  std::array<int32_t, 2> input_codes{};
  std::array<int32_t, 2> output_codes{};
  emel::memory::streaming::sm temporal_positions{
      emel::memory::streaming::dependencies{.capacity = 16}};
  emel::memory::streaming::sm secondary_positions{
      emel::memory::streaming::dependencies{.capacity = 16}};
  fake_encoder_actor encoder{};
  fake_decoder_actor decoder{};
  fake_runtime_actor runtime{};
  fake_predictor_actor predictor{};
  fake_dependencies dependencies;
  generator::sm<fake_dependencies> machine;

  explicit fixture(const int32_t frame_samples = 4,
                   const int32_t codebook_count = 2)
      : dependencies{
            .temporal_positions = temporal_positions,
            .secondary_positions = secondary_positions,
            .encoder = encoder,
            .decoder = decoder,
            .runtime = runtime,
            .predictor = predictor,
            .silence_pcm = std::span<float>{silence},
            .input_codes = std::span<int32_t>{input_codes},
            .output_codes = std::span<int32_t>{output_codes},
            .frame_samples = frame_samples,
            .codebook_count = codebook_count,
        },
        machine{dependencies} {}

  bool initialize() {
    emel::error::type err =
        generator::action::error_code(generator::error::none);
    return machine.process_event(generator::event::initialize{err});
  }

  bool condition() {
    bool complete = false;
    int32_t remaining = -1;
    emel::error::type err =
        generator::action::error_code(generator::error::none);
    return machine.process_event(
        generator::event::condition{7, complete, remaining, err});
  }
};

struct callback_probe {
  int32_t done_calls = 0;
  int32_t error_calls = 0;
  emel::error::type last_error = 0;

  void initialize_done(const generator::events::initialize_done &) noexcept {
    ++done_calls;
  }
  void
  initialize_error(const generator::events::initialize_error &ev) noexcept {
    ++error_calls;
    last_error = ev.err;
  }
  void condition_done(const generator::events::condition_done &) noexcept {
    ++done_calls;
  }
  void condition_error(const generator::events::condition_error &ev) noexcept {
    ++error_calls;
    last_error = ev.err;
  }
  void
  generation_error(const generator::events::generation_error &ev) noexcept {
    ++error_calls;
    last_error = ev.err;
  }
  void stream_done(const generator::events::stream_frame_done &) noexcept {
    ++done_calls;
  }
  void stream_error(const generator::events::stream_frame_error &ev) noexcept {
    ++error_calls;
    last_error = ev.err;
  }
  void flush_done(const generator::events::flush_done &) noexcept {
    ++done_calls;
  }
  void flush_error(const generator::events::flush_error &ev) noexcept {
    ++error_calls;
    last_error = ev.err;
  }
};

void advance_to_ready(fixture &test) {
  REQUIRE(test.initialize());
  REQUIRE(test.condition());
  REQUIRE(test.condition());
}

} // namespace

TEST_CASE("speech_generator_initializes_injected_actor_composition") {
  auto test = std::make_unique<fixture>();

  REQUIRE(test->initialize());
  CHECK(test->machine.is(sml::state<generator::state_condition_voice>));
  CHECK(test->encoder.initialize_calls == 1);
  CHECK(test->decoder.initialize_calls == 1);
  CHECK(test->runtime.initialize_calls == 1);
  CHECK(test->predictor.initialize_calls == 2);
}

TEST_CASE("speech_generator_reports_initialize_outcomes_through_callbacks") {
  SUBCASE("success") {
    auto test = std::make_unique<fixture>();
    callback_probe probe{};
    emel::error::type err = -1;
    generator::event::initialize request{err};
    request.on_done = decltype(request.on_done)::from<
        callback_probe, &callback_probe::initialize_done>(&probe);

    REQUIRE(test->machine.process_event(request));
    CHECK(probe.done_calls == 1);
    CHECK(err == generator::action::error_code(generator::error::none));
  }

  SUBCASE("invalid composition") {
    auto test = std::make_unique<fixture>(0, 2);
    callback_probe probe{};
    emel::error::type err = 0;
    generator::event::initialize request{err};
    request.on_error = decltype(request.on_error)::from<
        callback_probe, &callback_probe::initialize_error>(&probe);

    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.error_calls == 1);
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::invalid_request));
    CHECK(test->machine.is(sml::state<generator::state_errored>));
  }

  SUBCASE("child initialization failure") {
    auto test = std::make_unique<fixture>();
    test->encoder.initialize_error = 9;
    callback_probe probe{};
    emel::error::type err = 0;
    generator::event::initialize request{err};
    request.on_error = decltype(request.on_error)::from<
        callback_probe, &callback_probe::initialize_error>(&probe);

    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.error_calls == 1);
    CHECK(probe.last_error == generator::action::error_code(
                                  generator::error::encoder_initialize_failed));
  }
}

TEST_CASE("speech_generator_conditions_without_model_family_contracts") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());

  REQUIRE(test->condition());
  CHECK(test->machine.is(sml::state<generator::state_condition_prompt>));
  CHECK(test->predictor.voice_condition_calls == 1);
  CHECK(test->predictor.prompt_begin_calls == 1);

  REQUIRE(test->condition());
  CHECK(test->machine.is(sml::state<generator::state_ready>));
  CHECK(test->predictor.prompt_condition_calls == 1);
}

TEST_CASE("speech_generator_models_condition_pending_and_error_results") {
  SUBCASE("bounded pending frames") {
    auto test = std::make_unique<fixture>();
    REQUIRE(test->initialize());
    callback_probe probe{};
    bool complete = true;
    int32_t remaining = -1;
    emel::error::type err = 0;
    generator::event::condition request{7, complete, remaining, err};
    request.on_done = decltype(request.on_done)::from<
        callback_probe, &callback_probe::condition_done>(&probe);

    test->predictor.voice_complete = false;
    REQUIRE(test->machine.process_event(request));
    CHECK_FALSE(complete);
    CHECK(remaining == 1);
    CHECK(probe.done_calls == 1);
    CHECK(test->machine.is(sml::state<generator::state_condition_voice>));

    test->predictor.voice_complete = true;
    REQUIRE(test->machine.process_event(request));
    CHECK(probe.done_calls == 2);
    CHECK(test->machine.is(sml::state<generator::state_condition_prompt>));

    test->predictor.prompt_complete = false;
    REQUIRE(test->machine.process_event(request));
    CHECK_FALSE(complete);
    CHECK(probe.done_calls == 3);

    test->predictor.prompt_complete = true;
    REQUIRE(test->machine.process_event(request));
    CHECK(complete);
    CHECK(probe.done_calls == 4);
    CHECK(test->machine.is(sml::state<generator::state_ready>));
  }

  SUBCASE("graph failure") {
    auto test = std::make_unique<fixture>();
    REQUIRE(test->initialize());
    test->predictor.voice_graph_error = 7;
    callback_probe probe{};
    bool complete = true;
    int32_t remaining = 0;
    emel::error::type err = 0;
    generator::event::condition request{7, complete, remaining, err};
    request.on_error = decltype(request.on_error)::from<
        callback_probe, &callback_probe::condition_error>(&probe);

    CHECK_FALSE(test->machine.process_event(request));
    CHECK_FALSE(complete);
    CHECK(remaining == -1);
    CHECK(probe.error_calls == 1);
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::conditioning_failed));
    CHECK(test->machine.is(sml::state<generator::state_errored>));
  }
}

TEST_CASE("speech_generator_keeps_offline_synthesis_failure_explicit") {
  auto test = std::make_unique<fixture>();
  advance_to_ready(*test);
  callback_probe probe{};
  std::array<float, 4> pcm_out{};
  int32_t sample_count = -1;
  emel::error::type err = 0;
  generator::event::generate request{"hello", std::span<float>{pcm_out},
                                     sample_count, err};
  request.on_error = decltype(request.on_error)::from<
      callback_probe, &callback_probe::generation_error>(&probe);

  CHECK_FALSE(test->machine.process_event(request));
  CHECK(sample_count == 0);
  CHECK(probe.error_calls == 1);
  CHECK(probe.last_error ==
        generator::action::error_code(generator::error::cutover_pending));
  CHECK(test->machine.is(sml::state<generator::state_ready>));
}

TEST_CASE("speech_generator_streams_through_injected_actor_pipeline") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());
  REQUIRE(test->condition());
  REQUIRE(test->condition());

  std::array<float, 4> pcm_in{};
  std::array<float, 4> pcm_out{};
  std::array<int32_t, 2> encoded{};
  std::array<int32_t, 2> generated{};
  int32_t text_token = -1;
  int32_t sample_count = 0;
  bool produced = false;
  emel::error::type err = generator::action::error_code(generator::error::none);
  generator::event::stream_frame request{std::span<const float>{pcm_in},
                                         std::span<float>{pcm_out},
                                         std::span<int32_t>{encoded},
                                         std::span<int32_t>{generated},
                                         text_token,
                                         sample_count,
                                         produced,
                                         err};

  REQUIRE(test->machine.process_event(request));
  CHECK(test->machine.is(sml::state<generator::state_ready>));
  CHECK(encoded == std::array<int32_t, 2>{11, 12});
  CHECK(generated == std::array<int32_t, 2>{111, 112});
  CHECK(text_token == 42);
  CHECK(sample_count == 4);
  CHECK(produced);
  CHECK(pcm_out[0] == doctest::Approx(0.25f));
}

TEST_CASE("speech_generator_streams_pending_frames_and_reports_failures") {
  auto test = std::make_unique<fixture>();
  advance_to_ready(*test);
  std::array<float, 4> pcm_in{};
  std::array<float, 4> pcm_out{};
  std::array<int32_t, 2> encoded{};
  std::array<int32_t, 2> generated{};
  int32_t text_token = -1;
  int32_t sample_count = -1;
  bool produced = true;
  emel::error::type err = 0;
  callback_probe probe{};
  generator::event::stream_frame request{std::span<const float>{pcm_in},
                                         std::span<float>{pcm_out},
                                         std::span<int32_t>{encoded},
                                         std::span<int32_t>{generated},
                                         text_token,
                                         sample_count,
                                         produced,
                                         err};

  SUBCASE("pending") {
    test->predictor.produce = false;
    request.on_done =
        decltype(request.on_done)::from<callback_probe,
                                        &callback_probe::stream_done>(&probe);
    REQUIRE(test->machine.process_event(request));
    CHECK_FALSE(produced);
    CHECK(sample_count == 0);
    CHECK(test->decoder.decode_calls == 0);
    CHECK(probe.done_calls == 1);
  }

  SUBCASE("invalid request") {
    request.pcm_in = request.pcm_in.first(3);
    request.on_error =
        decltype(request.on_error)::from<callback_probe,
                                         &callback_probe::stream_error>(&probe);
    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::invalid_request));
  }

  SUBCASE("encode failure") {
    test->encoder.encode_error = 3;
    request.on_error =
        decltype(request.on_error)::from<callback_probe,
                                         &callback_probe::stream_error>(&probe);
    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::encode_failed));
  }

  SUBCASE("predict failure") {
    test->predictor.predict_graph_error = 4;
    request.on_error =
        decltype(request.on_error)::from<callback_probe,
                                         &callback_probe::stream_error>(&probe);
    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::predict_failed));
  }

  SUBCASE("decode failure") {
    test->decoder.decode_accepted = false;
    request.on_error =
        decltype(request.on_error)::from<callback_probe,
                                         &callback_probe::stream_error>(&probe);
    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::decode_failed));
  }
}

TEST_CASE("speech_generator_flushes_silence_through_same_pipeline") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());
  REQUIRE(test->condition());
  REQUIRE(test->condition());

  std::array<float, 4> pcm_out{};
  std::array<int32_t, 2> encoded{};
  std::array<int32_t, 2> generated{};
  int32_t text_token = -1;
  int32_t sample_count = 0;
  bool complete = true;
  emel::error::type err = generator::action::error_code(generator::error::none);
  generator::event::flush request{std::span<float>{pcm_out},
                                  std::span<int32_t>{encoded},
                                  std::span<int32_t>{generated},
                                  text_token,
                                  sample_count,
                                  complete,
                                  err};

  REQUIRE(test->machine.process_event(request));
  CHECK(test->machine.is(sml::state<generator::state_flushing>));
  CHECK(sample_count == 4);
  CHECK_FALSE(complete);
  CHECK(encoded == std::array<int32_t, 2>{11, 12});
  CHECK(generated == std::array<int32_t, 2>{111, 112});
}

TEST_CASE("speech_generator_flushes_pending_frames_and_reports_failures") {
  auto test = std::make_unique<fixture>();
  advance_to_ready(*test);
  std::array<float, 4> pcm_out{};
  std::array<int32_t, 2> encoded{};
  std::array<int32_t, 2> generated{};
  int32_t text_token = -1;
  int32_t sample_count = -1;
  bool complete = true;
  emel::error::type err = 0;
  callback_probe probe{};
  generator::event::flush request{std::span<float>{pcm_out},
                                  std::span<int32_t>{encoded},
                                  std::span<int32_t>{generated},
                                  text_token,
                                  sample_count,
                                  complete,
                                  err};

  SUBCASE("pending") {
    test->predictor.produce = false;
    request.on_done =
        decltype(request.on_done)::from<callback_probe,
                                        &callback_probe::flush_done>(&probe);
    REQUIRE(test->machine.process_event(request));
    CHECK(sample_count == 0);
    CHECK_FALSE(complete);
    CHECK(probe.done_calls == 1);
  }

  SUBCASE("invalid request") {
    request.generated_tokens_out = request.generated_tokens_out.first(1);
    request.on_error =
        decltype(request.on_error)::from<callback_probe,
                                         &callback_probe::flush_error>(&probe);
    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::invalid_request));
  }

  SUBCASE("predict failure") {
    test->predictor.predict_accepted = false;
    request.on_error =
        decltype(request.on_error)::from<callback_probe,
                                         &callback_probe::flush_error>(&probe);
    CHECK_FALSE(test->machine.process_event(request));
    CHECK(probe.last_error ==
          generator::action::error_code(generator::error::predict_failed));
  }
}

TEST_CASE("speech_generator_duplex_dispatch_is_allocation_free") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());
  REQUIRE(test->condition());
  REQUIRE(test->condition());

  std::array<float, 4> pcm_in{};
  std::array<float, 4> pcm_out{};
  std::array<int32_t, 2> encoded{};
  std::array<int32_t, 2> generated{};
  int32_t text_token = -1;
  int32_t sample_count = 0;
  bool produced = false;
  emel::error::type err = generator::action::error_code(generator::error::none);
  generator::event::stream_frame request{std::span<const float>{pcm_in},
                                         std::span<float>{pcm_out},
                                         std::span<int32_t>{encoded},
                                         std::span<int32_t>{generated},
                                         text_token,
                                         sample_count,
                                         produced,
                                         err};
  size_t allocation_count = 0;
  bool accepted = true;
  {
    emel::test::allocation::allocation_scope allocations;
    for (int32_t iteration = 0; iteration < 32; ++iteration) {
      accepted = test->machine.process_event(request) && accepted;
    }
    allocation_count = allocations.allocations();
  }

  CHECK(accepted);
  CHECK(allocation_count == 0u);
}
