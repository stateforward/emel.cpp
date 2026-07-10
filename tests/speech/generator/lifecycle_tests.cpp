#include "doctest/doctest.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>

#include "../../allocation_tracker.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/generator/sm.hpp"

namespace {

namespace generator = emel::speech::generator;
namespace sml = stateforward::sml;

struct callback_capture {
  int32_t initialize_done_calls = 0;
  int32_t initialize_error_calls = 0;
  int32_t generation_error_calls = 0;
  int32_t stream_error_calls = 0;
  int32_t flush_error_calls = 0;
  emel::error::type err = generator::action::error_code(generator::error::none);
};

void record_initialize_done(
    callback_capture &capture,
    const generator::events::initialize_done &) noexcept {
  ++capture.initialize_done_calls;
}

void record_initialize_error(
    callback_capture &capture,
    const generator::events::initialize_error &ev) noexcept {
  ++capture.initialize_error_calls;
  capture.err = ev.err;
}

void record_generation_error(
    callback_capture &capture,
    const generator::events::generation_error &ev) noexcept {
  ++capture.generation_error_calls;
  capture.err = ev.err;
}

void record_stream_error(
    callback_capture &capture,
    const generator::events::stream_frame_error &ev) noexcept {
  ++capture.stream_error_calls;
  capture.err = ev.err;
}

void record_flush_error(callback_capture &capture,
                        const generator::events::flush_error &ev) noexcept {
  ++capture.flush_error_calls;
  capture.err = ev.err;
}

struct fixture {
  emel::batch::planner::sm planner{};
  emel::memory::hybrid::sm memory{};
  emel::graph::sm graph{};
  emel::logits::sampler::sm sampler{};
  emel::kernel::sm kernel{};
  generator::dependencies dependencies{planner, memory, graph, sampler, kernel};
  generator::sm machine{dependencies};

  bool initialize(callback_capture *capture = nullptr) {
    emel::error::type err =
        generator::action::error_code(generator::error::none);
    generator::event::initialize ev{err};
    if (capture != nullptr) {
      ev.on_done =
          emel::callback<void(const generator::events::initialize_done &)>::
              from<callback_capture, record_initialize_done>(capture);
    }
    return machine.process_event(ev);
  }
};

static_assert(std::is_reference_v<decltype(generator::dependencies::planner)>);
static_assert(std::is_reference_v<decltype(generator::dependencies::memory)>);
static_assert(std::is_reference_v<decltype(generator::dependencies::graph)>);
static_assert(std::is_reference_v<decltype(generator::dependencies::sampler)>);
static_assert(std::is_reference_v<decltype(generator::dependencies::kernel)>);

} // namespace

TEST_CASE("speech_generator_initializes_with_injected_collaborators") {
  auto test = std::make_unique<fixture>();
  callback_capture capture{};

  REQUIRE(test->initialize(&capture));
  CHECK(capture.initialize_done_calls == 1);
  CHECK(test->machine.is(sml::state<generator::state_ready>));

  emel::error::type err = generator::action::error_code(generator::error::none);
  generator::event::initialize duplicate{err};
  duplicate.on_error =
      emel::callback<void(const generator::events::initialize_error &)>::from<
          callback_capture, record_initialize_error>(&capture);

  CHECK_FALSE(test->machine.process_event(duplicate));
  CHECK(err ==
        generator::action::error_code(generator::error::already_initialized));
  CHECK(capture.initialize_error_calls == 1);
  CHECK(capture.err == err);
  CHECK(test->machine.is(sml::state<generator::state_ready>));
}

TEST_CASE("speech_generator_exposes_synthesis_phases_without_hidden_cutover") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());

  std::array<float, 32> reference_pcm{};
  std::array<float, 64> pcm_out{};
  int32_t sample_count = 99;
  emel::error::type err = generator::action::error_code(generator::error::none);
  callback_capture capture{};
  generator::event::generate generate_ev{"Hello from the generic generator.",
                                         std::span<float>{pcm_out},
                                         sample_count, err};
  generate_ev.reference_pcm = std::span<const float>{reference_pcm};
  generate_ev.on_error =
      emel::callback<void(const generator::events::generation_error &)>::from<
          callback_capture, record_generation_error>(&capture);

  CHECK_FALSE(test->machine.process_event(generate_ev));
  CHECK(err ==
        generator::action::error_code(generator::error::cutover_pending));
  CHECK(sample_count == 0);
  CHECK(capture.generation_error_calls == 1);
  CHECK(capture.err == err);
  CHECK(test->machine.is(sml::state<generator::state_ready>));
}

TEST_CASE("speech_generator_exposes_duplex_and_flush_phases") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());

  std::array<float, 32> pcm_in{};
  std::array<float, 64> pcm_out{};
  int32_t sample_count = 99;
  bool produced = true;
  emel::error::type err = generator::action::error_code(generator::error::none);
  callback_capture capture{};
  generator::event::stream_frame stream_ev{std::span<const float>{pcm_in},
                                           std::span<float>{pcm_out},
                                           sample_count, produced, err};
  stream_ev.on_error =
      emel::callback<void(const generator::events::stream_frame_error &)>::from<
          callback_capture, record_stream_error>(&capture);

  CHECK_FALSE(test->machine.process_event(stream_ev));
  CHECK(err ==
        generator::action::error_code(generator::error::cutover_pending));
  CHECK(sample_count == 0);
  CHECK_FALSE(produced);
  CHECK(capture.stream_error_calls == 1);
  CHECK(test->machine.is(sml::state<generator::state_ready>));

  sample_count = 99;
  bool complete = true;
  generator::event::flush flush_ev{std::span<float>{pcm_out}, sample_count,
                                   complete, err};
  flush_ev.on_error =
      emel::callback<void(const generator::events::flush_error &)>::from<
          callback_capture, record_flush_error>(&capture);

  CHECK_FALSE(test->machine.process_event(flush_ev));
  CHECK(err ==
        generator::action::error_code(generator::error::cutover_pending));
  CHECK(sample_count == 0);
  CHECK_FALSE(complete);
  CHECK(capture.flush_error_calls == 1);
  CHECK(test->machine.is(sml::state<generator::state_ready>));
}

TEST_CASE("speech_generator_rejects_uninitialized_and_invalid_requests") {
  auto test = std::make_unique<fixture>();
  std::array<float, 8> pcm{};
  int32_t sample_count = 99;
  emel::error::type err = generator::action::error_code(generator::error::none);
  generator::event::generate before_initialize{"speech", std::span<float>{pcm},
                                               sample_count, err};

  CHECK_FALSE(test->machine.process_event(before_initialize));
  CHECK(err == generator::action::error_code(generator::error::uninitialized));
  CHECK(test->machine.is(sml::state<generator::state_uninitialized>));

  generator::event::reset reset_before_initialize{err};
  CHECK_FALSE(test->machine.process_event(reset_before_initialize));
  CHECK(err == generator::action::error_code(generator::error::uninitialized));
  CHECK(test->machine.is(sml::state<generator::state_uninitialized>));

  REQUIRE(test->initialize());
  generator::event::generate invalid_generate{"", std::span<float>{pcm},
                                              sample_count, err};
  CHECK_FALSE(test->machine.process_event(invalid_generate));
  CHECK(err ==
        generator::action::error_code(generator::error::invalid_request));

  bool produced = true;
  generator::event::stream_frame invalid_stream{std::span<const float>{},
                                                std::span<float>{pcm},
                                                sample_count, produced, err};
  CHECK_FALSE(test->machine.process_event(invalid_stream));
  CHECK(err ==
        generator::action::error_code(generator::error::invalid_request));

  bool complete = true;
  generator::event::flush invalid_flush{std::span<float>{}, sample_count,
                                        complete, err};
  CHECK_FALSE(test->machine.process_event(invalid_flush));
  CHECK(err ==
        generator::action::error_code(generator::error::invalid_request));
  CHECK(test->machine.is(sml::state<generator::state_ready>));
}

TEST_CASE("speech_generator_reset_recovers_from_unexpected_event") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());

  struct unknown_event {};
  test->machine.process_event(unknown_event{});
  CHECK(test->machine.is(sml::state<generator::state_errored>));

  emel::error::type err = generator::action::error_code(generator::error::none);
  generator::event::reset reset_ev{err};
  CHECK(test->machine.process_event(reset_ev));
  CHECK(err == generator::action::error_code(generator::error::none));
  CHECK(test->machine.is(sml::state<generator::state_ready>));
}

TEST_CASE("speech_generator_scaffold_dispatch_is_allocation_free") {
  auto test = std::make_unique<fixture>();
  REQUIRE(test->initialize());

  std::array<float, 64> pcm_out{};
  int32_t sample_count = 0;
  emel::error::type err = generator::action::error_code(generator::error::none);
  generator::event::generate generate_ev{
      "allocation probe", std::span<float>{pcm_out}, sample_count, err};
  bool all_rejected_as_pending = true;
  size_t allocation_count = 0u;
  {
    emel::test::allocation::allocation_scope allocations;
    for (int32_t dispatch = 0; dispatch < 32; ++dispatch) {
      all_rejected_as_pending = !test->machine.process_event(generate_ev) &&
                                err == generator::action::error_code(
                                           generator::error::cutover_pending) &&
                                all_rejected_as_pending;
    }
    allocation_count = allocations.allocations();
  }

  CHECK(all_rejected_as_pending);
  CHECK(allocation_count == 0u);
}
