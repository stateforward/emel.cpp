#include <array>
#include <cstdint>
#include <limits>
#include <span>

#include <doctest/doctest.h>

#include "../../../allocation_tracker.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/tokenizer/moshi/sm.hpp"

namespace {

namespace moshi = emel::speech::tokenizer::moshi;
namespace sml = stateforward::sml;

int32_t error_code(const moshi::error value) noexcept {
  return static_cast<int32_t>(emel::error::cast(value));
}

struct standard_fixture {
  static constexpr int32_t codebooks = 3;
  static constexpr int32_t generated_codebooks = 2;
  static constexpr int32_t delayed_codebooks = 2;
  static constexpr int32_t rows = 4;

  explicit standard_fixture(const int32_t initial_delay_frames = 0)
      : machine(moshi::dependencies{
            .delays = std::span<const int32_t>{delays},
            .cache = std::span<int32_t>{cache},
            .codebooks = codebooks,
            .generated_audio_codebooks = generated_codebooks,
            .delayed_audio_codebooks = delayed_codebooks,
            .cache_rows = rows,
            .maximum_delay = 2,
            .initial_delay_frames = initial_delay_frames,
            .text_initial_token = 100,
            .audio_initial_token = 200,
            .token_zero = -1,
            .token_ungenerated = -2,
        }) {}

  bool initialize() {
    err = error_code(moshi::error::none);
    return machine.process_event(moshi::event::initialize{.error_out = err});
  }

  bool tokenize(const std::span<const int32_t> input = {}) {
    err = error_code(moshi::error::none);
    return machine.process_event(moshi::event::tokenize{
        .audio_tokens = input,
        .model_tokens_out = std::span<int32_t>{model_tokens},
        .error_out = err,
    });
  }

  bool detokenize(const int32_t text,
                  const std::span<const int32_t> generated) {
    err = error_code(moshi::error::none);
    return machine.process_event(moshi::event::detokenize{
        .text_token = text,
        .audio_tokens = generated,
        .text_token_out = text_out,
        .audio_tokens_out = std::span<int32_t>{audio_out},
        .produced_out = produced,
        .error_out = err,
    });
  }

  std::array<int32_t, codebooks> delays = {0, 1, 2};
  std::array<int32_t, rows * codebooks> cache = {};
  std::array<int32_t, codebooks> model_tokens = {};
  std::array<int32_t, delayed_codebooks> audio_out = {};
  int32_t text_out = -1;
  bool produced = false;
  int32_t err = error_code(moshi::error::none);
  moshi::sm machine;
};

} // namespace

TEST_CASE("speech Moshi tokenizer delays and restores codebook alignment") {
  standard_fixture state;
  REQUIRE(state.initialize());
  CHECK(state.machine.is(sml::state<moshi::state_ready>));

  REQUIRE(state.tokenize());
  CHECK(state.model_tokens == std::array<int32_t, 3>{100, 200, 200});
  const std::array<int32_t, 2> first_audio = {11, 12};
  REQUIRE(state.detokenize(10, first_audio));
  CHECK_FALSE(state.produced);

  REQUIRE(state.tokenize());
  CHECK(state.model_tokens == std::array<int32_t, 3>{10, 200, 200});
  const std::array<int32_t, 2> second_audio = {21, 22};
  REQUIRE(state.detokenize(20, second_audio));
  CHECK_FALSE(state.produced);

  REQUIRE(state.tokenize());
  CHECK(state.model_tokens == std::array<int32_t, 3>{20, 21, 200});
  const std::array<int32_t, 2> third_audio = {31, 32};
  REQUIRE(state.detokenize(30, third_audio));
  CHECK(state.produced);
  CHECK(state.text_out == 10);
  CHECK(state.audio_out == std::array<int32_t, 2>{21, 32});
}

TEST_CASE("speech Moshi tokenizer preserves complete provided frames") {
  standard_fixture state;
  REQUIRE(state.initialize());

  const std::array<int32_t, 3> first_input = {50, 51, 52};
  const std::array<int32_t, 3> second_input = {60, 61, 62};
  const std::array<int32_t, 3> third_input = {70, 71, 72};
  const std::array<int32_t, 2> generated = {91, 92};

  REQUIRE(state.tokenize(first_input));
  REQUIRE(state.detokenize(90, generated));
  REQUIRE(state.tokenize(second_input));
  REQUIRE(state.detokenize(90, generated));
  REQUIRE(state.tokenize(third_input));
  REQUIRE(state.detokenize(90, generated));

  CHECK(state.produced);
  CHECK(state.text_out == 60);
  CHECK(state.audio_out == std::array<int32_t, 2>{61, 62});
}

TEST_CASE("speech Moshi tokenizer accepts the PersonaPlex tail shape") {
  std::array<int32_t, 5> delays = {0, 1, 1, 1, 1};
  std::array<int32_t, 20> cache = {};
  moshi::sm machine{moshi::dependencies{
      .delays = std::span<const int32_t>{delays},
      .cache = std::span<int32_t>{cache},
      .codebooks = 5,
      .generated_audio_codebooks = 4,
      .delayed_audio_codebooks = 2,
      .cache_rows = 4,
      .maximum_delay = 1,
      .initial_delay_frames = 0,
      .text_initial_token = 100,
      .audio_initial_token = 200,
      .token_zero = -1,
      .token_ungenerated = -2,
  }};
  int32_t err = error_code(moshi::error::none);
  REQUIRE(machine.process_event(moshi::event::initialize{.error_out = err}));

  std::array<int32_t, 2> tail = {101, 102};
  std::array<int32_t, 5> model_tokens = {};
  REQUIRE(machine.process_event(moshi::event::tokenize{
      .audio_tokens = std::span<const int32_t>{tail},
      .model_tokens_out = std::span<int32_t>{model_tokens},
      .error_out = err,
  }));
  CHECK(model_tokens == std::array<int32_t, 5>{100, 200, 200, 200, 200});

  std::array<int32_t, 4> generated = {11, 12, 13, 14};
  std::array<int32_t, 2> output = {};
  int32_t text_out = -1;
  bool produced = false;
  REQUIRE(machine.process_event(moshi::event::detokenize{
      .text_token = 10,
      .audio_tokens = std::span<const int32_t>{generated},
      .text_token_out = text_out,
      .audio_tokens_out = std::span<int32_t>{output},
      .produced_out = produced,
      .error_out = err,
  }));
  CHECK_FALSE(produced);

  const std::array<int32_t, 2> next_tail = {111, 112};
  REQUIRE(machine.process_event(moshi::event::tokenize{
      .audio_tokens = std::span<const int32_t>{next_tail},
      .model_tokens_out = std::span<int32_t>{model_tokens},
      .error_out = err,
  }));
  REQUIRE(machine.process_event(moshi::event::detokenize{
      .text_token = 20,
      .audio_tokens = std::span<const int32_t>{generated},
      .text_token_out = text_out,
      .audio_tokens_out = std::span<int32_t>{output},
      .produced_out = produced,
      .error_out = err,
  }));
  const std::array<int32_t, 2> third_tail = {121, 122};
  REQUIRE(machine.process_event(moshi::event::tokenize{
      .audio_tokens = std::span<const int32_t>{third_tail},
      .model_tokens_out = std::span<int32_t>{model_tokens},
      .error_out = err,
  }));
  CHECK(model_tokens[3] == next_tail[0]);
  CHECK(model_tokens[4] == next_tail[1]);
}

TEST_CASE("speech Moshi tokenizer rejects invalid public audio tokens before "
          "caching") {
  standard_fixture state;
  REQUIRE(state.initialize());

  const std::array<int32_t, 3> invalid_input = {50, 200, 52};
  CHECK(state.tokenize(invalid_input));
  CHECK(state.err == error_code(moshi::error::request_shape));
  CHECK(state.machine.is(sml::state<moshi::state_ready>));

  REQUIRE(state.tokenize());
  CHECK(state.model_tokens == std::array<int32_t, 3>{100, 200, 200});
}

TEST_CASE("speech Moshi tokenizer rejects invalid generated tokens before "
          "commit") {
  standard_fixture state;
  REQUIRE(state.initialize());

  REQUIRE(state.tokenize());
  const std::array<int32_t, 2> invalid_audio = {11, 200};
  CHECK(state.detokenize(10, invalid_audio));
  CHECK(state.err == error_code(moshi::error::request_shape));
  CHECK(state.machine.is(sml::state<moshi::state_prepared_generated>));

  const std::array<int32_t, 2> valid_audio = {11, 12};
  CHECK(state.detokenize(100, valid_audio));
  CHECK(state.err == error_code(moshi::error::request_shape));
  CHECK(state.machine.is(sml::state<moshi::state_prepared_generated>));
}

TEST_CASE("speech Moshi tokenizer masks initial delayed audio explicitly") {
  standard_fixture state{2};
  REQUIRE(state.initialize());
  const std::array<int32_t, 2> generated = {11, 12};

  REQUIRE(state.tokenize());
  REQUIRE(state.detokenize(10, generated));
  CHECK_FALSE(state.produced);
  REQUIRE(state.tokenize());
  REQUIRE(state.detokenize(20, generated));
  CHECK_FALSE(state.produced);
  REQUIRE(state.tokenize());
  CHECK(state.model_tokens == std::array<int32_t, 3>{20, -1, 200});
  REQUIRE(state.detokenize(30, generated));
  CHECK_FALSE(state.produced);
}

TEST_CASE("speech Moshi tokenizer restores column-major voice cache") {
  standard_fixture state;
  REQUIRE(state.initialize());
  const std::array<int32_t, 2> generated = {41, 42};
  REQUIRE(state.tokenize());
  REQUIRE(state.detokenize(40, generated));
  REQUIRE(state.tokenize());
  REQUIRE(state.detokenize(40, generated));

  const std::array<int32_t, 12> column_major = {
      10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33,
  };
  REQUIRE(state.machine.process_event(moshi::event::restore_cache{
      .column_major_cache = std::span<const int32_t>{column_major},
      .offset = 2,
      .error_out = state.err,
  }));

  REQUIRE(state.tokenize());
  CHECK(state.model_tokens == std::array<int32_t, 3>{12, 22, 200});
}

TEST_CASE("speech Moshi tokenizer rejects invalid restored cache tokens") {
  standard_fixture state;
  REQUIRE(state.initialize());
  std::array<int32_t, 12> column_major{};
  column_major[0] = 100;
  CHECK(state.machine.process_event(moshi::event::restore_cache{
      .column_major_cache = std::span<const int32_t>{column_major},
      .offset = 0,
      .error_out = state.err,
  }));
  CHECK(state.err == error_code(moshi::error::request_shape));
}

TEST_CASE(
    "speech Moshi tokenizer does not publish restored ungenerated output") {
  standard_fixture state;
  REQUIRE(state.initialize());

  std::array<int32_t, 12> column_major{};
  column_major.fill(-2);
  REQUIRE(state.machine.process_event(moshi::event::restore_cache{
      .column_major_cache = std::span<const int32_t>{column_major},
      .offset = 3,
      .error_out = state.err,
  }));

  REQUIRE(state.tokenize());
  const std::array<int32_t, 2> generated = {41, 42};
  REQUIRE(state.detokenize(40, generated));
  CHECK_FALSE(state.produced);
}

TEST_CASE("speech Moshi tokenizer advances a provided prompt frame") {
  standard_fixture state;
  REQUIRE(state.initialize());
  const std::array<int32_t, 3> prompt = {50, 51, 52};
  REQUIRE(state.tokenize(prompt));
  REQUIRE(state.machine.process_event(
      moshi::event::advance{.error_out = state.err}));
  REQUIRE(state.tokenize());
  CHECK(state.model_tokens == std::array<int32_t, 3>{-2, 200, 200});
}

TEST_CASE("speech Moshi tokenizer models lifecycle and shape failures") {
  standard_fixture state;
  std::array<int32_t, 2> generated = {11, 12};

  CHECK(state.tokenize());
  CHECK(state.err == error_code(moshi::error::uninitialized));
  REQUIRE(state.initialize());
  CHECK(state.initialize());
  CHECK(state.err == error_code(moshi::error::already_initialized));

  std::array<int32_t, 1> invalid_input = {1};
  CHECK(state.tokenize(invalid_input));
  CHECK(state.err == error_code(moshi::error::request_shape));
  REQUIRE(state.tokenize());

  std::array<int32_t, 1> invalid_generated = {1};
  CHECK(state.detokenize(10, invalid_generated));
  CHECK(state.err == error_code(moshi::error::request_shape));
  REQUIRE(state.detokenize(10, generated));

  CHECK(
      state.machine.process_event(moshi::event::reset{.error_out = state.err}));
  CHECK(state.machine.is(sml::state<moshi::state_ready>));
}

TEST_CASE("speech Moshi tokenizer rejects invalid injected geometry") {
  std::array<int32_t, 1> delays = {0};
  std::array<int32_t, 1> cache = {};
  moshi::sm machine{moshi::dependencies{
      .delays = std::span<const int32_t>{delays},
      .cache = std::span<int32_t>{cache},
      .codebooks = 1,
      .generated_audio_codebooks = 0,
      .delayed_audio_codebooks = 0,
      .cache_rows = 1,
      .maximum_delay = 0,
      .initial_delay_frames = 0,
      .text_initial_token = 0,
      .audio_initial_token = 0,
      .token_zero = -1,
      .token_ungenerated = -2,
  }};
  int32_t err = error_code(moshi::error::none);
  CHECK(machine.process_event(moshi::event::initialize{.error_out = err}));
  CHECK(err == error_code(moshi::error::invalid_configuration));
  CHECK(machine.is(sml::state<moshi::state_uninitialized>));
}

TEST_CASE("speech Moshi tokenizer rejects invalid token bounds and sentinels") {
  const auto rejected = [](const int32_t text_initial_token,
                           const int32_t token_zero,
                           const int32_t token_ungenerated) {
    std::array<int32_t, 3> delays = {0, 1, 2};
    std::array<int32_t, 12> cache{};
    moshi::sm machine{moshi::dependencies{
        .delays = std::span<const int32_t>{delays},
        .cache = std::span<int32_t>{cache},
        .codebooks = 3,
        .generated_audio_codebooks = 2,
        .delayed_audio_codebooks = 2,
        .cache_rows = 4,
        .maximum_delay = 2,
        .initial_delay_frames = 0,
        .text_initial_token = text_initial_token,
        .audio_initial_token = 200,
        .token_zero = token_zero,
        .token_ungenerated = token_ungenerated,
    }};
    int32_t err = error_code(moshi::error::none);
    return !machine.process_event(moshi::event::initialize{.error_out = err}) ||
           err == error_code(moshi::error::invalid_configuration);
  };
  CHECK(rejected(0, -1, -2));
  CHECK(rejected(100, 0, -2));
  CHECK(rejected(100, -1, 0));
}

TEST_CASE("speech Moshi tokenizer dispatch remains allocation free") {
  standard_fixture state;
  REQUIRE(state.initialize());
  const std::array<int32_t, 2> generated = {11, 12};

  emel::test::allocation::allocation_scope allocations;
  for (int32_t frame = 0; frame < 32; ++frame) {
    REQUIRE(state.tokenize());
    REQUIRE(state.detokenize(frame, generated));
  }
  CHECK(allocations.allocations() == 0u);
}

TEST_CASE("speech Moshi tokenizer makes position overflow explicit") {
  standard_fixture state;
  REQUIRE(state.initialize());
  REQUIRE(state.tokenize());

  moshi::action::context context{moshi::dependencies{
      .delays = std::span<const int32_t>{state.delays},
      .cache = std::span<int32_t>{state.cache},
      .codebooks = standard_fixture::codebooks,
      .generated_audio_codebooks = standard_fixture::generated_codebooks,
      .delayed_audio_codebooks = standard_fixture::delayed_codebooks,
      .cache_rows = standard_fixture::rows,
      .maximum_delay = 2,
      .initial_delay_frames = 0,
      .text_initial_token = 100,
      .audio_initial_token = 200,
      .token_zero = -1,
      .token_ungenerated = -2,
  }};
  stateforward::sml::sm<moshi::model, stateforward::sml::testing> machine{
      context};
  machine.set_current_states(sml::state<moshi::state_prepared_generated>);
  context.offset = std::numeric_limits<int64_t>::max();

  std::array<int32_t, 2> generated = {11, 12};
  std::array<int32_t, 2> output = {};
  int32_t text_out = -1;
  bool produced = false;
  int32_t err = error_code(moshi::error::none);
  moshi::event::detokenize_ctx detokenize_ctx{};
  CHECK(machine.process_event(moshi::event::detokenize_run{
      .request =
          moshi::event::detokenize{
              .text_token = 10,
              .audio_tokens = std::span<const int32_t>{generated},
              .text_token_out = text_out,
              .audio_tokens_out = std::span<int32_t>{output},
              .produced_out = produced,
              .error_out = err,
          },
      .ctx = detokenize_ctx,
  }));
  CHECK(err == error_code(moshi::error::position_overflow));

  machine.set_current_states(sml::state<moshi::state_ready>);
  std::array<int32_t, 3> model_tokens = {};
  err = error_code(moshi::error::none);
  CHECK(machine.process_event(moshi::event::tokenize{
      .audio_tokens = {},
      .model_tokens_out = std::span<int32_t>{model_tokens},
      .error_out = err,
  }));
  CHECK(err == error_code(moshi::error::position_overflow));
}

TEST_CASE("speech Moshi tokenizer exposes unexpected events and recovers") {
  struct unknown_event {};

  standard_fixture state;
  REQUIRE(state.initialize());
  CHECK(state.machine.process_event(unknown_event{}));
  CHECK(state.machine.is(sml::state<moshi::state_errored>));

  CHECK(state.tokenize());
  CHECK(state.err == error_code(moshi::error::internal_error));
  REQUIRE(
      state.machine.process_event(moshi::event::reset{.error_out = state.err}));
  CHECK(state.machine.is(sml::state<moshi::state_ready>));
}
