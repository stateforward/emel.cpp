#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <span>

#include "../../allocation_tracker.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/speech/generator/frame/sm.hpp"
#include "emel/speech/generator/sm.hpp"

namespace {

namespace frame = emel::speech::generator::frame;
namespace generator = emel::speech::generator;
namespace sml = stateforward::sml;

constexpr size_t FRAME_SAMPLES = 4u;
constexpr size_t CODEBOOKS = 2u;
constexpr int32_t MAX_WAVEFRONT_TRANSITIONS = 3;
constexpr auto FAKE_WAVEFRONT_DISPATCH_BUDGET = std::chrono::milliseconds{100};

struct fake_workspace {};

struct fake_tokenize {
  std::span<const int32_t> audio_tokens;
  std::span<int32_t> model_tokens_out;
  int32_t &error_out;
};

struct fake_predict {
  fake_predict(std::span<const int32_t> tokens_ref,
               fake_workspace &workspace_ref, int32_t step_ref,
               int32_t outputs_ref) noexcept
      : tokens(tokens_ref), workspace(workspace_ref), step(step_ref),
        outputs(outputs_ref) {}
  std::span<const int32_t> tokens;
  fake_workspace &workspace;
  int32_t step;
  int32_t outputs;
  emel::error::type *error_out = nullptr;
};

struct fake_graph {
  fake_graph(fake_workspace &workspace_ref,
             std::span<const int32_t> tokens_ref) noexcept
      : workspace(workspace_ref), tokens(tokens_ref) {}
  fake_workspace &workspace;
  std::span<const int32_t> tokens;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
};

struct fake_sample {
  fake_sample(fake_workspace &workspace_ref,
              std::span<const int32_t> tokens_ref,
              std::span<int32_t> tokens_out_ref,
              int32_t &text_token_out_ref) noexcept
      : workspace(workspace_ref), tokens(tokens_ref),
        tokens_out(tokens_out_ref), text_token_out(text_token_out_ref) {}
  fake_workspace &workspace;
  std::span<const int32_t> tokens;
  std::span<int32_t> tokens_out;
  int32_t &text_token_out;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
};

struct fake_detokenize {
  int32_t text_token;
  std::span<const int32_t> audio_tokens;
  int32_t &text_token_out;
  std::span<int32_t> audio_tokens_out;
  bool &produced_out;
  int32_t &error_out;
};

struct overlap_probe {
  std::atomic<int32_t> active_workers = 0;
  std::atomic<int32_t> expected_workers = 0;
  std::atomic<bool> hold_workers = false;
  std::atomic<bool> release_workers = false;
  std::atomic<bool> overlap_observed = false;
};

struct sml_trace {
  std::array<uint8_t, 128> entries{};
  size_t size = 0u;
  int32_t transitions = 0;
  bool overflow = false;

  void record(const uint8_t value) noexcept {
    if (size < entries.size()) {
      entries[size++] = value;
    } else {
      overflow = true;
    }
  }

  void reset() noexcept {
    entries.fill(0u);
    size = 0u;
    transitions = 0;
    overflow = false;
  }

  template <class SM, class event_type>
  void log_process_event(const event_type &) noexcept {
    record(1u);
  }

  template <class SM, class guard_type, class event_type>
  void log_guard(const guard_type &, const event_type &,
                 const bool accepted) noexcept {
    record(accepted ? 3u : 2u);
  }

  template <class SM, class action_type, class event_type>
  void log_action(const action_type &, const event_type &) noexcept {
    record(4u);
  }

  template <class SM, class source_state, class destination_state>
  void log_state_change(const source_state &,
                        const destination_state &) noexcept {
    ++transitions;
    record(5u);
  }
};

struct fake_tokenizer_actor {
  overlap_probe *overlap = nullptr;
  bool accepted = true;
  bool produce = true;
  int32_t error = 0;

  bool process_event(const fake_tokenize &request) noexcept {
    while (overlap->hold_workers.load(std::memory_order_acquire) &&
           overlap->active_workers.load(std::memory_order_acquire) <
               overlap->expected_workers.load(std::memory_order_acquire)) {
      emel::policy::cpu_relax();
    }
    overlap->overlap_observed.store(
        overlap->hold_workers.load(std::memory_order_acquire),
        std::memory_order_release);
    overlap->release_workers.store(true, std::memory_order_release);
    std::copy(request.audio_tokens.begin(), request.audio_tokens.end(),
              request.model_tokens_out.begin());
    request.error_out = error;
    return accepted;
  }

  bool process_event(const fake_detokenize &request) noexcept {
    std::copy(request.audio_tokens.begin(), request.audio_tokens.end(),
              request.audio_tokens_out.begin());
    request.text_token_out = request.text_token;
    request.produced_out = produce;
    request.error_out = error;
    return accepted;
  }
};

struct fake_planner_actor {
  bool accepted = true;

  bool process_event(
      const emel::batch::planner::event::plan_request &request) noexcept {
    const std::array<int32_t, 1> steps{1};
    request.on_done(emel::batch::planner::events::plan_done{
        .request = &request,
        .step_sizes = steps.data(),
        .step_count = 1,
        .total_outputs = 1,
    });
    return accepted;
  }
};

struct fake_predictor_actor {
  bool accepted = true;
  emel::error::type error = 0;
  bool process_event(const fake_predict &request) noexcept {
    *request.error_out = error;
    return accepted;
  }
};

struct fake_graph_actor {
  bool accepted = true;
  bool process_event(const fake_graph &request) noexcept {
    *request.error_out = 0;
    *request.graph_error_out = 0;
    return accepted;
  }
};

struct fake_sampler_actor {
  bool accepted = true;
  bool process_event(const fake_sample &request) noexcept {
    for (size_t lane = 0u; lane < request.tokens_out.size(); ++lane) {
      request.tokens_out[lane] = request.tokens[lane] + 100;
    }
    request.text_token_out = request.tokens[0] + 1000;
    *request.error_out = 0;
    *request.graph_error_out = 0;
    return accepted;
  }
};

struct frame_dependencies {
  using tokenize_event = fake_tokenize;
  using predict_event = fake_predict;
  using graph_event = fake_graph;
  using sample_event = fake_sample;
  using detokenize_event = fake_detokenize;

  fake_planner_actor &planner;
  fake_tokenizer_actor &tokenizer;
  fake_predictor_actor &predictor;
  fake_graph_actor &graph;
  fake_sampler_actor &sampler;
  fake_workspace &prediction_workspace;
  std::span<int32_t> model_codes;
  std::span<int32_t> predicted_codes;
  int32_t codebook_count;
  emel::batch::planner::event::plan_mode frame_plan_mode;
  int32_t frame_plan_steps;
  int32_t frame_plan_token_count;
  bool frame_plan_output_all;
};

struct fake_encode {
  fake_encode(std::span<const float> pcm_ref,
              std::span<int32_t> tokens_ref) noexcept
      : pcm(pcm_ref), tokens(tokens_ref) {}
  std::span<const float> pcm;
  std::span<int32_t> tokens;
  emel::error::type *error_out = nullptr;
};

struct fake_decode {
  fake_decode(std::span<const int32_t> tokens_ref,
              std::span<float> pcm_ref) noexcept
      : tokens(tokens_ref), pcm(pcm_ref) {}
  std::span<const int32_t> tokens;
  std::span<float> pcm;
  emel::error::type *error_out = nullptr;
};

struct fake_reset {
  explicit fake_reset(emel::error::type &error_out_ref) noexcept
      : error_out(error_out_ref) {}

  emel::error::type &error_out;
};

struct fake_encoder_actor {
  overlap_probe *overlap = nullptr;
  bool accepted = true;
  emel::error::type error = 0;
  bool reset_accepted = true;
  emel::error::type reset_error = 0;
  int32_t reset_calls = 0;
  bool processed_since_reset = false;

  bool process_event(const fake_encode &request) noexcept {
    processed_since_reset = true;
    overlap->active_workers.fetch_add(1, std::memory_order_acq_rel);
    while (overlap->hold_workers.load(std::memory_order_acquire) &&
           !overlap->release_workers.load(std::memory_order_acquire)) {
      emel::policy::cpu_relax();
    }
    for (size_t lane = 0u; lane < request.tokens.size(); ++lane) {
      request.tokens[lane] =
          static_cast<int32_t>(request.pcm[0]) + static_cast<int32_t>(lane);
    }
    *request.error_out = error;
    overlap->active_workers.fetch_sub(1, std::memory_order_acq_rel);
    return accepted;
  }

  bool process_event(const fake_reset &request) noexcept {
    ++reset_calls;
    processed_since_reset = false;
    request.error_out = reset_error;
    return reset_accepted;
  }
};

struct fake_decoder_actor {
  overlap_probe *overlap = nullptr;
  bool accepted = true;
  emel::error::type error = 0;
  bool reset_accepted = true;
  emel::error::type reset_error = 0;
  int32_t reset_calls = 0;
  bool processed_since_reset = false;

  bool process_event(const fake_decode &request) noexcept {
    processed_since_reset = true;
    overlap->active_workers.fetch_add(1, std::memory_order_acq_rel);
    while (overlap->hold_workers.load(std::memory_order_acquire) &&
           !overlap->release_workers.load(std::memory_order_acquire)) {
      emel::policy::cpu_relax();
    }
    std::fill(request.pcm.begin(), request.pcm.end(),
              static_cast<float>(request.tokens[0]));
    *request.error_out = error;
    overlap->active_workers.fetch_sub(1, std::memory_order_acq_rel);
    return accepted;
  }

  bool process_event(const fake_reset &request) noexcept {
    ++reset_calls;
    processed_since_reset = false;
    request.error_out = reset_error;
    return reset_accepted;
  }
};

struct wavefront_dependencies {
  using generator_mode = generator::action::mode::wavefront;
  using wavefront_encode_event = fake_encode;
  using wavefront_encode_reset_event = fake_reset;
  using wavefront_middle_reset_event = frame::event::reset;
  using wavefront_decode_event = fake_decode;
  using wavefront_decode_reset_event = fake_reset;
  static constexpr size_t wavefront_frame_capacity = FRAME_SAMPLES;
  static constexpr size_t wavefront_codebook_capacity = CODEBOOKS;

  fake_encoder_actor &wavefront_encoder;
  frame::sm<frame_dependencies> &wavefront_middle;
  fake_decoder_actor &wavefront_decoder;
  generator::action::wavefront_stage_pool *stage_pool;
  generator::action::wavefront_diagnostics &stage_diagnostics;
  generator::action::wavefront_stage_mode stage_mode;
  int32_t frame_samples;
  int32_t codebook_count;
};

struct fixture {
  overlap_probe overlap{};
  std::array<int32_t, CODEBOOKS> model_codes{};
  std::array<int32_t, CODEBOOKS> predicted_codes{};
  fake_planner_actor planner{};
  fake_tokenizer_actor tokenizer{.overlap = &overlap};
  fake_predictor_actor predictor{};
  fake_graph_actor graph{};
  fake_sampler_actor sampler{};
  fake_workspace workspace{};
  frame_dependencies middle_dependencies{
      .planner = planner,
      .tokenizer = tokenizer,
      .predictor = predictor,
      .graph = graph,
      .sampler = sampler,
      .prediction_workspace = workspace,
      .model_codes = model_codes,
      .predicted_codes = predicted_codes,
      .codebook_count = static_cast<int32_t>(CODEBOOKS),
      .frame_plan_mode = emel::batch::planner::event::plan_mode::simple,
      .frame_plan_steps = 1,
      .frame_plan_token_count = 1,
      .frame_plan_output_all = true,
  };
  frame::sm<frame_dependencies> middle{middle_dependencies};
  fake_encoder_actor encoder{.overlap = &overlap};
  fake_decoder_actor decoder{.overlap = &overlap};
  generator::action::wavefront_stage_pool pool{};
  generator::action::wavefront_diagnostics diagnostics{};
  wavefront_dependencies dependencies;
  generator::sm<wavefront_dependencies> machine;
  int32_t last_text_token = -1;
  int32_t last_sample_count = 0;

  explicit fixture(
      const int32_t frame_samples = static_cast<int32_t>(FRAME_SAMPLES),
      const int32_t codebook_count = static_cast<int32_t>(CODEBOOKS),
      const generator::action::wavefront_stage_mode stage_mode =
          generator::action::wavefront_stage_mode::parallel)
      : dependencies{.wavefront_encoder = encoder,
                     .wavefront_middle = middle,
                     .wavefront_decoder = decoder,
                     .stage_pool = &pool,
                     .stage_diagnostics = diagnostics,
                     .stage_mode = stage_mode,
                     .frame_samples = frame_samples,
                     .codebook_count = codebook_count},
        machine{dependencies} {}

  bool
  push(const uint64_t sequence, bool &produced,
       generator::event::wavefront_attribution &output,
       std::array<float, FRAME_SAMPLES> &pcm_out,
       std::array<int32_t, CODEBOOKS> &tokens_out, emel::error::type &err,
       emel::callback<void(const generator::events::wavefront_frame_done &)>
           on_done = {}) {
    std::array<float, FRAME_SAMPLES> pcm_in{};
    pcm_in.fill(static_cast<float>(sequence + 1u));
    last_text_token = -1;
    last_sample_count = 0;
    generator::event::wavefront_frame request{
        pcm_in,
        pcm_out,
        tokens_out,
        {.sequence = sequence, .source = 99u},
        output,
        last_text_token,
        last_sample_count,
        produced,
        err};
    request.on_done = on_done;
    return machine.process_event(request);
  }

  bool
  flush(bool &produced, bool &complete,
        generator::event::wavefront_attribution &output,
        std::array<float, FRAME_SAMPLES> &pcm_out,
        std::array<int32_t, CODEBOOKS> &tokens_out, emel::error::type &err,
        emel::callback<void(const generator::events::wavefront_flush_done &)>
            on_done = {}) {
    last_text_token = -1;
    last_sample_count = 0;
    generator::event::wavefront_flush request{
        pcm_out,           tokens_out, output,   last_text_token,
        last_sample_count, produced,   complete, err};
    request.on_done = on_done;
    return machine.process_event(request);
  }
};

struct callback_probe {
  bool caller_active = false;
  int32_t calls = 0;
  bool all_calls_inside_dispatch = true;
  emel::error::type last_error = 0u;

  void done(const generator::events::wavefront_frame_done &) noexcept {
    ++calls;
    all_calls_inside_dispatch = all_calls_inside_dispatch && caller_active;
  }

  void flush_done(const generator::events::wavefront_flush_done &) noexcept {
    ++calls;
    all_calls_inside_dispatch = all_calls_inside_dispatch && caller_active;
  }

  void
  frame_error(const generator::events::wavefront_frame_error &ev) noexcept {
    ++calls;
    all_calls_inside_dispatch = all_calls_inside_dispatch && caller_active;
    last_error = ev.err;
  }

  void
  flush_error(const generator::events::wavefront_flush_error &ev) noexcept {
    ++calls;
    all_calls_inside_dispatch = all_calls_inside_dispatch && caller_active;
    last_error = ev.err;
  }
};

template <class machine_type>
concept accepts_internal_wavefront = requires(
    machine_type &machine, const generator::detail::wavefront_frame_run &ev) {
  machine.process_event(ev);
};

static_assert(
    !accepts_internal_wavefront<generator::sm<wavefront_dependencies>>);

struct production_wavefront_dependencies {
  using generator_mode = generator::action::mode::wavefront;
  using wavefront_encode_event = fake_encode;
  using wavefront_encode_reset_event = fake_reset;
  using wavefront_middle_reset_event = frame::event::reset;
  using wavefront_decode_event = fake_decode;
  using wavefront_decode_reset_event = fake_reset;
  static constexpr size_t wavefront_frame_capacity = FRAME_SAMPLES;
  static constexpr size_t wavefront_codebook_capacity = CODEBOOKS;

  fake_encoder_actor &wavefront_encoder;
  frame::sm<frame_dependencies> &wavefront_middle;
  fake_decoder_actor &wavefront_decoder;
  generator::action::wavefront_stage_pool *stage_pool;
  generator::action::wavefront_stage_mode stage_mode;
  int32_t frame_samples;
  int32_t codebook_count;
};

static_assert(generator::action::wavefront_dependencies<
              production_wavefront_dependencies>);

template <class dependencies_type>
concept has_wavefront_diagnostics =
    requires(dependencies_type &deps) { deps.stage_diagnostics; };

static_assert(!has_wavefront_diagnostics<production_wavefront_dependencies>);

emel::error::type error_code(const generator::error value) noexcept {
  return generator::action::error_code(value);
}

emel::error::type combined_error(const generator::error value,
                                 const generator::error qualifier) noexcept {
  return error_code(value) | error_code(qualifier);
}

struct wavefront_io {
  std::array<float, FRAME_SAMPLES> pcm{};
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  bool produced = false;
  bool complete = false;
  emel::error::type err = 0u;
};

void prime_frames(fixture &test, wavefront_io &io, const uint64_t frame_count) {
  for (uint64_t sequence = 0u; sequence < frame_count; ++sequence) {
    REQUIRE(
        test.push(sequence, io.produced, io.output, io.pcm, io.tokens, io.err));
  }
}

} // namespace

TEST_CASE(
    "speech_generator_wavefront_models_zero_through_many_frame_lifecycle") {
  SUBCASE("zero frames completes without output") {
    fixture test{};
    std::array<float, FRAME_SAMPLES> pcm{};
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    bool produced = true;
    bool complete = false;
    emel::error::type err = -1;
    REQUIRE(test.flush(produced, complete, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK(complete);
    CHECK(test.machine.is(sml::state<generator::state_wavefront_complete>));
  }

  for (const uint64_t frame_total : {1u, 2u, 3u, 9u}) {
    CAPTURE(frame_total);
    fixture test{};
    std::array<float, FRAME_SAMPLES> pcm{};
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    emel::error::type err = -1;
    uint64_t output_sequence = 0u;
    for (uint64_t sequence = 0u; sequence < frame_total; ++sequence) {
      bool produced = true;
      REQUIRE(test.push(sequence, produced, output, pcm, tokens, err));
      CHECK(produced == (sequence >= 2u));
      if (produced) {
        CHECK(output.sequence == output_sequence++);
        CHECK(output.source == 99u);
        CHECK(pcm[0] ==
              doctest::Approx(static_cast<float>(output.sequence + 101u)));
        CHECK(tokens[0] == static_cast<int32_t>(output.sequence + 101u));
        CHECK(tokens[1] == static_cast<int32_t>(output.sequence + 102u));
        CHECK(test.last_text_token ==
              static_cast<int32_t>(output.sequence + 1001u));
        CHECK(test.last_sample_count == static_cast<int32_t>(FRAME_SAMPLES));
      }
    }

    bool complete = false;
    while (!complete) {
      bool produced = false;
      REQUIRE(test.flush(produced, complete, output, pcm, tokens, err));
      if (produced) {
        CHECK(output.sequence == output_sequence++);
        CHECK(output.source == 99u);
        CHECK(pcm[0] ==
              doctest::Approx(static_cast<float>(output.sequence + 101u)));
        CHECK(tokens[0] == static_cast<int32_t>(output.sequence + 101u));
        CHECK(tokens[1] == static_cast<int32_t>(output.sequence + 102u));
        CHECK(test.last_text_token ==
              static_cast<int32_t>(output.sequence + 1001u));
        CHECK(test.last_sample_count == static_cast<int32_t>(FRAME_SAMPLES));
      }
    }
    CHECK(output_sequence == frame_total);
    CHECK(test.diagnostics.submissions.load() == test.diagnostics.joins.load());
  }
}

TEST_CASE("speech_generator_wavefront_emits_done_callbacks_for_each_public_"
          "lifecycle_phase") {
  for (const uint64_t frame_count : {0u, 1u, 2u, 3u}) {
    CAPTURE(frame_count);
    fixture test{};
    wavefront_io io{};
    callback_probe probe{};
    const auto frame_done =
        emel::callback<void(const generator::events::wavefront_frame_done &)>::
            from<callback_probe, &callback_probe::done>(&probe);
    const auto flush_done =
        emel::callback<void(const generator::events::wavefront_flush_done &)>::
            from<callback_probe, &callback_probe::flush_done>(&probe);

    probe.caller_active = true;
    for (uint64_t sequence = 0u; sequence < frame_count; ++sequence) {
      REQUIRE(test.push(sequence, io.produced, io.output, io.pcm, io.tokens,
                        io.err, frame_done));
    }
    while (!io.complete) {
      REQUIRE(test.flush(io.produced, io.complete, io.output, io.pcm, io.tokens,
                         io.err, flush_done));
    }
    probe.caller_active = false;

    CHECK(probe.calls > 0);
    CHECK(probe.all_calls_inside_dispatch);
    CHECK(test.machine.is(sml::state<generator::state_wavefront_complete>));
  }
}

TEST_CASE("speech_generator_wavefront_rejects_each_invalid_public_request_"
          "shape") {
  const auto check_invalid_frame = [](fixture &test,
                                      std::span<const float> pcm_in,
                                      std::span<float> pcm_out,
                                      std::span<int32_t> tokens_out,
                                      const uint64_t sequence) {
    generator::event::wavefront_attribution output{};
    int32_t text_token = 7;
    int32_t samples = 7;
    bool produced = true;
    emel::error::type err = 0u;
    const generator::event::wavefront_frame request{
        pcm_in, pcm_out,    tokens_out, {.sequence = sequence, .source = 99u},
        output, text_token, samples,    produced,
        err};

    CHECK_FALSE(test.machine.process_event(request));
    CHECK_FALSE(produced);
    CHECK(err == error_code(generator::error::invalid_request));
  };

  const auto check_invalid_flush = [](fixture &test, std::span<float> pcm_out,
                                      std::span<int32_t> tokens_out) {
    generator::event::wavefront_attribution output{};
    int32_t text_token = 7;
    int32_t samples = 7;
    bool produced = true;
    bool complete = true;
    emel::error::type err = 0u;
    const generator::event::wavefront_flush request{
        pcm_out, tokens_out, output,   text_token,
        samples, produced,   complete, err};

    CHECK_FALSE(test.machine.process_event(request));
    CHECK_FALSE(produced);
    CHECK_FALSE(complete);
    CHECK(err == error_code(generator::error::invalid_request));
  };

  std::array<float, FRAME_SAMPLES> pcm{};
  std::array<int32_t, CODEBOOKS> tokens{};
  SUBCASE("invalid persistent dimensions") {
    fixture zero_samples{0, static_cast<int32_t>(CODEBOOKS)};
    check_invalid_frame(zero_samples, pcm, pcm, tokens, 0u);

    fixture zero_codebooks{static_cast<int32_t>(FRAME_SAMPLES), 0};
    check_invalid_frame(zero_codebooks, pcm, pcm, tokens, 0u);

    fixture excess_samples{static_cast<int32_t>(FRAME_SAMPLES + 1u),
                           static_cast<int32_t>(CODEBOOKS)};
    check_invalid_frame(excess_samples, pcm, pcm, tokens, 0u);

    fixture excess_codebooks{static_cast<int32_t>(FRAME_SAMPLES),
                             static_cast<int32_t>(CODEBOOKS + 1u)};
    check_invalid_frame(excess_codebooks, pcm, pcm, tokens, 0u);
  }

  SUBCASE("invalid frame spans and attribution") {
    fixture narrow_input{};
    check_invalid_frame(narrow_input,
                        std::span<const float>{pcm}.first(FRAME_SAMPLES - 1u),
                        pcm, tokens, 0u);

    fixture narrow_output{};
    check_invalid_frame(narrow_output, pcm,
                        std::span<float>{pcm}.first(FRAME_SAMPLES - 1u), tokens,
                        0u);

    fixture narrow_tokens{};
    check_invalid_frame(narrow_tokens, pcm, pcm,
                        std::span<int32_t>{tokens}.first(CODEBOOKS - 1u), 0u);

    fixture invalid_sequence{};
    check_invalid_frame(invalid_sequence, pcm, pcm, tokens,
                        generator::event::k_invalid_wavefront_sequence);

    fixture out_of_order{};
    check_invalid_frame(out_of_order, pcm, pcm, tokens, 1u);
  }

  SUBCASE("invalid flush spans") {
    fixture narrow_output{};
    check_invalid_flush(
        narrow_output, std::span<float>{pcm}.first(FRAME_SAMPLES - 1u), tokens);

    fixture narrow_tokens{};
    check_invalid_flush(narrow_tokens, pcm,
                        std::span<int32_t>{tokens}.first(CODEBOOKS - 1u));
  }
}

TEST_CASE(
    "speech_generator_wavefront_observes_three_stage_overlap_and_callbacks") {
  fixture test{};
  std::array<float, FRAME_SAMPLES> pcm{};
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  bool produced = false;
  emel::error::type err = -1;
  REQUIRE(test.push(0u, produced, output, pcm, tokens, err));
  REQUIRE(test.push(1u, produced, output, pcm, tokens, err));

  test.overlap.expected_workers.store(2);
  test.overlap.release_workers.store(false);
  test.overlap.hold_workers.store(true);
  callback_probe probe{};
  std::array<float, FRAME_SAMPLES> pcm_in{};
  pcm_in.fill(3.0f);
  int32_t text_token = -1;
  int32_t samples = 0;
  generator::event::wavefront_frame request{
      pcm_in, pcm,        tokens,  {.sequence = 2u, .source = 99u},
      output, text_token, samples, produced,
      err};
  request.on_done =
      decltype(request.on_done)::from<callback_probe, &callback_probe::done>(
          &probe);
  probe.caller_active = true;
  REQUIRE(test.machine.process_event(request));
  probe.caller_active = false;

  CHECK(test.overlap.overlap_observed.load());
  CHECK(produced);
  CHECK(output.sequence == 0u);
  CHECK(probe.calls == 1);
  CHECK(probe.all_calls_inside_dispatch);
  CHECK(test.diagnostics.worker_entries.load() ==
        test.diagnostics.worker_exits.load());
  CHECK(test.diagnostics.submissions.load() == test.diagnostics.joins.load());
}

TEST_CASE("speech_generator_wavefront_serial_stage_mode_uses_no_workers") {
  fixture test{static_cast<int32_t>(FRAME_SAMPLES),
               static_cast<int32_t>(CODEBOOKS),
               generator::action::wavefront_stage_mode::serial};
  wavefront_io io{};
  REQUIRE(test.push(0u, io.produced, io.output, io.pcm, io.tokens, io.err));
  REQUIRE(test.push(1u, io.produced, io.output, io.pcm, io.tokens, io.err));
  REQUIRE(test.push(2u, io.produced, io.output, io.pcm, io.tokens, io.err));
  CHECK(test.diagnostics.submissions.load() == 0u);
  CHECK(test.diagnostics.joins.load() == 0u);
  CHECK(test.diagnostics.worker_entries.load() == 0u);
  CHECK(test.diagnostics.worker_exits.load() == 0u);

  REQUIRE(
      test.machine.process_event(generator::event::wavefront_reset{io.err}));
  CHECK(test.diagnostics.submissions.load() == 0u);
  CHECK(test.diagnostics.joins.load() == 0u);
  CHECK(test.diagnostics.worker_entries.load() == 0u);
  CHECK(test.diagnostics.worker_exits.load() == 0u);
}

TEST_CASE(
    "speech_generator_wavefront_drains_partial_submit_and_phase_failures") {
  SUBCASE("encode failure publishes no output") {
    fixture test{};
    test.encoder.error = 7;
    std::array<float, FRAME_SAMPLES> pcm{};
    pcm.fill(-9.0f);
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    bool produced = true;
    emel::error::type err = 0;
    CHECK_FALSE(test.push(0u, produced, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK(pcm[0] == -9.0f);
    CHECK(err == error_code(generator::error::encode_failed));
  }

  SUBCASE("middle and decode failures publish no output") {
    fixture middle_failure{};
    std::array<float, FRAME_SAMPLES> pcm{};
    pcm.fill(-9.0f);
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    bool produced = false;
    emel::error::type err = 0;
    REQUIRE(middle_failure.push(0u, produced, output, pcm, tokens, err));
    middle_failure.tokenizer.error = 8;
    CHECK_FALSE(middle_failure.push(1u, produced, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK(pcm[0] == -9.0f);
    CHECK(err == error_code(generator::error::middle_failed));

    fixture decode_failure{};
    REQUIRE(decode_failure.push(0u, produced, output, pcm, tokens, err));
    REQUIRE(decode_failure.push(1u, produced, output, pcm, tokens, err));
    decode_failure.decoder.error = 9;
    CHECK_FALSE(decode_failure.push(2u, produced, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK(pcm[0] == -9.0f);
    CHECK(err == error_code(generator::error::decode_failed));
  }

  SUBCASE("partial submit joins the accepted task then fails") {
    fixture test{};
    std::array<float, FRAME_SAMPLES> pcm{};
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    bool produced = false;
    emel::error::type err = 0;
    REQUIRE(test.push(0u, produced, output, pcm, tokens, err));
    REQUIRE(test.push(1u, produced, output, pcm, tokens, err));

    generator::action::wavefront_stage_pool::join_group blocker_group{};
    std::atomic<bool> blocker_started = false;
    std::atomic<bool> release_blocker = false;
    REQUIRE(test.pool.try_submit(blocker_group, [&]() noexcept {
      blocker_started.store(true, std::memory_order_release);
      while (!release_blocker.load(std::memory_order_acquire)) {
        emel::policy::cpu_relax();
      }
    }));
    while (!blocker_started.load(std::memory_order_acquire)) {
      emel::policy::cpu_relax();
    }
    const uint64_t submissions_before = test.diagnostics.submissions.load();
    const uint64_t joins_before = test.diagnostics.joins.load();
    CHECK_FALSE(test.push(2u, produced, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK(err == error_code(generator::error::stage_submit_failed));
    CHECK(test.diagnostics.submissions.load() - submissions_before == 1u);
    CHECK(test.diagnostics.joins.load() - joins_before == 1u);
    release_blocker.store(true, std::memory_order_release);
    CHECK(blocker_group.wait());
  }
}

TEST_CASE(
    "speech_generator_wavefront_preserves_failure_provenance_across_drains") {
  SUBCASE("child rejection and middle non-production are distinct") {
    fixture encode_rejection{};
    encode_rejection.encoder.accepted = false;
    std::array<float, FRAME_SAMPLES> pcm{};
    pcm.fill(-9.0f);
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    bool produced = true;
    emel::error::type err = 0u;
    CHECK_FALSE(encode_rejection.push(0u, produced, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK(err == combined_error(generator::error::encode_failed,
                                generator::error::unsupported_request));
    CHECK(pcm[0] == -9.0f);

    fixture non_production{};
    REQUIRE(non_production.push(0u, produced, output, pcm, tokens, err));
    non_production.tokenizer.produce = false;
    CHECK_FALSE(non_production.push(1u, produced, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK(err == error_code(generator::error::non_production_failed));
    CHECK(pcm[0] == -9.0f);
  }

  SUBCASE("middle and decode drain failures carry the drain qualifier") {
    fixture middle_drain{};
    std::array<float, FRAME_SAMPLES> pcm{};
    pcm.fill(-9.0f);
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    bool produced = false;
    bool complete = false;
    emel::error::type err = 0u;
    REQUIRE(middle_drain.push(0u, produced, output, pcm, tokens, err));
    middle_drain.tokenizer.error = 7;
    CHECK_FALSE(
        middle_drain.flush(produced, complete, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK_FALSE(complete);
    CHECK(err == combined_error(generator::error::middle_failed,
                                generator::error::drain_failed));
    CHECK(pcm[0] == -9.0f);

    fixture final_decode{};
    REQUIRE(final_decode.push(0u, produced, output, pcm, tokens, err));
    REQUIRE(final_decode.flush(produced, complete, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK_FALSE(complete);
    final_decode.decoder.error = 9;
    CHECK_FALSE(
        final_decode.flush(produced, complete, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK_FALSE(complete);
    CHECK(err == combined_error(generator::error::decode_failed,
                                generator::error::drain_failed));
    CHECK(pcm[0] == -9.0f);
  }

  SUBCASE("drain and final-decode submit failures carry the drain qualifier") {
    fixture drain{};
    std::array<float, FRAME_SAMPLES> pcm{};
    pcm.fill(-9.0f);
    std::array<int32_t, CODEBOOKS> tokens{};
    generator::event::wavefront_attribution output{};
    bool produced = false;
    bool complete = false;
    emel::error::type err = 0u;
    REQUIRE(drain.push(0u, produced, output, pcm, tokens, err));
    REQUIRE(drain.push(1u, produced, output, pcm, tokens, err));

    bool drain_accepted = true;
    generator::action::wavefront_stage_pool::join_group drain_group{};
    REQUIRE(drain.pool.try_submit(drain_group, [&]() noexcept {
      drain_accepted =
          drain.flush(produced, complete, output, pcm, tokens, err);
    }));
    REQUIRE(drain_group.wait());
    CHECK_FALSE(drain_accepted);
    CHECK_FALSE(produced);
    CHECK_FALSE(complete);
    CHECK(err == combined_error(generator::error::stage_submit_failed,
                                generator::error::drain_failed));
    CHECK(pcm[0] == -9.0f);

    fixture final_decode{};
    REQUIRE(final_decode.push(0u, produced, output, pcm, tokens, err));
    REQUIRE(final_decode.flush(produced, complete, output, pcm, tokens, err));
    CHECK_FALSE(produced);
    CHECK_FALSE(complete);

    bool final_accepted = true;
    generator::action::wavefront_stage_pool::join_group final_group{};
    REQUIRE(final_decode.pool.try_submit(final_group, [&]() noexcept {
      final_accepted =
          final_decode.flush(produced, complete, output, pcm, tokens, err);
    }));
    REQUIRE(final_group.wait());
    CHECK_FALSE(final_accepted);
    CHECK_FALSE(produced);
    CHECK_FALSE(complete);
    CHECK(err == combined_error(generator::error::stage_submit_failed,
                                generator::error::drain_failed));
    CHECK(pcm[0] == -9.0f);
  }
}

TEST_CASE("speech_generator_wavefront_classifies_child_failures_in_each_"
          "active_frame_phase") {
  const auto check_encode_rejection = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.encoder.accepted = false;

    CHECK_FALSE(test.push(primed_frames, io.produced, io.output, io.pcm,
                          io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK(io.err == combined_error(generator::error::encode_failed,
                                   generator::error::unsupported_request));
  };
  const auto check_encode_failure = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.encoder.error = 17u;

    CHECK_FALSE(test.push(primed_frames, io.produced, io.output, io.pcm,
                          io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK(io.err == error_code(generator::error::encode_failed));
  };
  const auto check_middle_failure = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.tokenizer.error = 19u;

    CHECK_FALSE(test.push(primed_frames, io.produced, io.output, io.pcm,
                          io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK(io.err == error_code(generator::error::middle_failed));
  };
  const auto check_decode_rejection = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.decoder.accepted = false;

    CHECK_FALSE(test.push(primed_frames, io.produced, io.output, io.pcm,
                          io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK(io.err == combined_error(generator::error::decode_failed,
                                   generator::error::unsupported_request));
  };
  const auto check_decode_failure = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.decoder.error = 23u;

    CHECK_FALSE(test.push(primed_frames, io.produced, io.output, io.pcm,
                          io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK(io.err == error_code(generator::error::decode_failed));
  };

  SUBCASE("fill one") {
    check_encode_rejection(1u);
    check_encode_failure(1u);
  }
  SUBCASE("steady even") {
    check_encode_rejection(2u);
    check_encode_failure(2u);
    check_middle_failure(2u);
    check_decode_rejection(2u);
    check_decode_failure(2u);
  }
  SUBCASE("steady odd") {
    check_encode_rejection(3u);
    check_encode_failure(3u);
    check_middle_failure(3u);
    check_decode_rejection(3u);
    check_decode_failure(3u);
  }
}

TEST_CASE("speech_generator_wavefront_classifies_child_failures_in_each_"
          "drain_phase") {
  const auto check_middle_failure = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.tokenizer.error = 29u;

    CHECK_FALSE(test.flush(io.produced, io.complete, io.output, io.pcm,
                           io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK_FALSE(io.complete);
    CHECK(io.err == combined_error(generator::error::middle_failed,
                                   generator::error::drain_failed));
  };
  const auto check_decode_rejection = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.decoder.accepted = false;

    CHECK_FALSE(test.flush(io.produced, io.complete, io.output, io.pcm,
                           io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK_FALSE(io.complete);
    CHECK(io.err == (error_code(generator::error::decode_failed) |
                     error_code(generator::error::unsupported_request) |
                     error_code(generator::error::drain_failed)));
  };
  const auto check_decode_failure = [](const uint64_t primed_frames) {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, primed_frames);
    test.decoder.error = 31u;

    CHECK_FALSE(test.flush(io.produced, io.complete, io.output, io.pcm,
                           io.tokens, io.err));
    CHECK_FALSE(io.produced);
    CHECK_FALSE(io.complete);
    CHECK(io.err == combined_error(generator::error::decode_failed,
                                   generator::error::drain_failed));
  };

  SUBCASE("drain model") { check_middle_failure(1u); }
  SUBCASE("drain even") {
    check_middle_failure(2u);
    check_decode_rejection(2u);
    check_decode_failure(2u);
  }
  SUBCASE("drain odd") {
    check_middle_failure(3u);
    check_decode_rejection(3u);
    check_decode_failure(3u);
  }

  SUBCASE("final decode lane zero") {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, 1u);
    REQUIRE(test.flush(io.produced, io.complete, io.output, io.pcm, io.tokens,
                       io.err));
    REQUIRE(test.machine.is(
        sml::state<generator::state_wavefront_final_decode_lane0>));
    test.decoder.accepted = false;
    CHECK_FALSE(test.flush(io.produced, io.complete, io.output, io.pcm,
                           io.tokens, io.err));
    CHECK(io.err == (error_code(generator::error::decode_failed) |
                     error_code(generator::error::unsupported_request) |
                     error_code(generator::error::drain_failed)));
  }

  SUBCASE("final decode lane one") {
    fixture test{};
    wavefront_io io{};
    prime_frames(test, io, 2u);
    REQUIRE(test.flush(io.produced, io.complete, io.output, io.pcm, io.tokens,
                       io.err));
    REQUIRE(test.machine.is(
        sml::state<generator::state_wavefront_final_decode_lane1>));
    test.decoder.error = 37u;
    CHECK_FALSE(test.flush(io.produced, io.complete, io.output, io.pcm,
                           io.tokens, io.err));
    CHECK(io.err == combined_error(generator::error::decode_failed,
                                   generator::error::drain_failed));
  }
}

TEST_CASE("speech_generator_wavefront_rejects_wrong_lifecycle_with_immediate_"
          "callbacks") {
  fixture test{};
  std::array<float, FRAME_SAMPLES> pcm{};
  pcm.fill(-9.0f);
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  bool produced = true;
  bool complete = false;
  emel::error::type err = 0u;
  REQUIRE(test.flush(produced, complete, output, pcm, tokens, err));
  REQUIRE(complete);

  std::array<float, FRAME_SAMPLES> pcm_in{};
  int32_t text_token = -1;
  int32_t samples = 0;
  callback_probe probe{};
  generator::event::wavefront_frame frame_request{
      pcm_in, pcm,        tokens,  {.sequence = 0u, .source = 99u},
      output, text_token, samples, produced,
      err};
  frame_request.on_error = decltype(frame_request.on_error)::from<
      callback_probe, &callback_probe::frame_error>(&probe);
  probe.caller_active = true;
  CHECK_FALSE(test.machine.process_event(frame_request));
  probe.caller_active = false;
  CHECK_FALSE(produced);
  CHECK(err == error_code(generator::error::unsupported_request));
  CHECK(probe.calls == 1);
  CHECK(probe.all_calls_inside_dispatch);
  CHECK(probe.last_error == err);
  CHECK(test.machine.is(sml::state<generator::state_wavefront_errored>));

  generator::event::wavefront_flush flush_request{
      pcm, tokens, output, text_token, samples, produced, complete, err};
  flush_request.on_error = decltype(flush_request.on_error)::from<
      callback_probe, &callback_probe::flush_error>(&probe);
  probe.caller_active = true;
  CHECK_FALSE(test.machine.process_event(flush_request));
  probe.caller_active = false;
  CHECK_FALSE(produced);
  CHECK_FALSE(complete);
  CHECK(err == error_code(generator::error::unsupported_request));
  CHECK(probe.calls == 2);
  CHECK(probe.all_calls_inside_dispatch);

  REQUIRE(test.machine.process_event(generator::event::wavefront_reset{err}));
  CHECK(test.machine.is(sml::state<generator::state_wavefront_fill0>));
  REQUIRE(test.push(0u, produced, output, pcm, tokens, err));
  CHECK_FALSE(produced);
  REQUIRE(test.flush(produced, complete, output, pcm, tokens, err));
  CHECK(test.machine.is(
      sml::state<generator::state_wavefront_final_decode_lane0>));
  CHECK_FALSE(test.push(1u, produced, output, pcm, tokens, err));
  CHECK_FALSE(produced);
  CHECK(err == error_code(generator::error::unsupported_request));
}

TEST_CASE("speech_generator_wavefront_detects_corrupt_lane_attribution") {
  fixture test{};
  generator::action::context<wavefront_dependencies> ctx{test.dependencies};
  ctx.encoded_lane0_attribution = {.sequence = 7u, .source = 99u};
  generator::detail::wavefront_run_ctx run_ctx{};
  run_ctx.all_submitted = true;
  run_ctx.joined = true;
  run_ctx.encode_accepted = true;
  std::array<float, FRAME_SAMPLES> pcm_in{};
  std::array<float, FRAME_SAMPLES> pcm_out{};
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  int32_t text_token = -1;
  int32_t samples = 0;
  bool produced = false;
  emel::error::type err = 0u;
  const generator::event::wavefront_frame request{
      pcm_in, pcm_out,    tokens,  {.sequence = 0u, .source = 99u},
      output, text_token, samples, produced,
      err};
  const generator::detail::wavefront_frame_run runtime_ev{request, run_ctx};
  using phase_fill0 = generator::guard::guard_wavefront_phase_succeeded<
      wavefront_dependencies, generator::action::lane_zero,
      generator::action::lane_zero, generator::action::lane_zero, true, false,
      false, false>;
  CHECK(generator::guard::guard_wavefront_attribution_failed<
        generator::detail::wavefront_frame_run, wavefront_dependencies,
        phase_fill0>{}(runtime_ev, ctx));
}

TEST_CASE("speech_generator_wavefront_reset_covers_every_stage_actor") {
  fixture test{};
  std::array<float, FRAME_SAMPLES> pcm{};
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  bool produced = false;
  emel::error::type err = 0u;

  REQUIRE(test.push(0u, produced, output, pcm, tokens, err));
  REQUIRE(test.push(1u, produced, output, pcm, tokens, err));
  CHECK(test.encoder.processed_since_reset);

  test.encoder.reset_error = 17u;
  CHECK_FALSE(
      test.machine.process_event(generator::event::wavefront_reset{err}));
  CHECK(err == error_code(generator::error::internal_error));
  CHECK(test.encoder.reset_calls == 1);
  CHECK(test.decoder.reset_calls == 1);
  CHECK_FALSE(test.encoder.processed_since_reset);
  CHECK_FALSE(test.decoder.processed_since_reset);
  CHECK(test.machine.is(sml::state<generator::state_wavefront_errored>));

  test.encoder.reset_error = 0u;
  REQUIRE(test.machine.process_event(generator::event::wavefront_reset{err}));
  CHECK(err == error_code(generator::error::none));
  CHECK(test.encoder.reset_calls == 2);
  CHECK(test.decoder.reset_calls == 2);
  CHECK(test.machine.is(sml::state<generator::state_wavefront_fill0>));
  REQUIRE(test.push(0u, produced, output, pcm, tokens, err));
  CHECK_FALSE(produced);

  emel::error::type middle_err = 9u;
  REQUIRE(test.middle.process_event(frame::event::reset{middle_err}));
  CHECK(middle_err == frame::action::error_code(frame::error::none));
}

TEST_CASE("speech_generator_wavefront_reset_classifies_each_fake_child_"
          "rejection_and_failure") {
  const auto check_reset_failure = [](fixture &test) {
    emel::error::type err = 0u;
    CHECK_FALSE(
        test.machine.process_event(generator::event::wavefront_reset{err}));
    CHECK(err == error_code(generator::error::internal_error));
    CHECK(test.machine.is(sml::state<generator::state_wavefront_errored>));
  };

  SUBCASE("encoder rejection") {
    fixture test{};
    test.encoder.reset_accepted = false;
    check_reset_failure(test);
  }
  SUBCASE("decoder rejection") {
    fixture test{};
    test.decoder.reset_accepted = false;
    check_reset_failure(test);
  }
  SUBCASE("decoder failure") {
    fixture test{};
    test.decoder.reset_error = 41u;
    check_reset_failure(test);
  }
}

TEST_CASE(
    "speech_generator_wavefront_reset_and_repeated_dispatch_allocate_zero") {
  fixture test{};
  std::array<float, FRAME_SAMPLES> pcm{};
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  bool produced = false;
  emel::error::type err = 0;

  size_t allocations = 0u;
  {
    emel::test::allocation::allocation_scope scope;
    for (uint64_t sequence = 0u; sequence < 64u; ++sequence) {
      REQUIRE(test.push(sequence, produced, output, pcm, tokens, err));
    }
    allocations = scope.allocations();
  }
  CHECK(allocations == 0u);
  CHECK(test.diagnostics.submissions.load() == test.diagnostics.joins.load());

  REQUIRE(test.machine.process_event(generator::event::wavefront_reset{err}));
  CHECK(test.encoder.reset_calls == 1);
  CHECK(test.decoder.reset_calls == 1);
  CHECK(test.machine.is(sml::state<generator::state_wavefront_fill0>));
  REQUIRE(test.push(0u, produced, output, pcm, tokens, err));
  CHECK_FALSE(produced);
}

TEST_CASE("speech_generator_wavefront_trace_is_deterministic_and_statically_"
          "bounded") {
  fixture test{};
  sml_trace logger{};
  using traced_machine =
      emel::sm<generator::model<wavefront_dependencies>,
               generator::action::context<wavefront_dependencies>,
               sml::logger<sml_trace>>;
  generator::action::context<wavefront_dependencies> ctx{test.dependencies};
  traced_machine machine{ctx, logger};
  std::array<float, FRAME_SAMPLES> pcm_in{};
  std::array<float, FRAME_SAMPLES> pcm_out{};
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  int32_t text_token = -1;
  int32_t samples = 0;
  bool produced = false;
  emel::error::type err = 0u;
  const generator::event::wavefront_frame request{
      pcm_in, pcm_out,    tokens,  {.sequence = 0u, .source = 99u},
      output, text_token, samples, produced,
      err};

  logger.reset();
  generator::detail::wavefront_run_ctx first_ctx{};
  REQUIRE(machine.process_event(
      generator::detail::wavefront_frame_run{request, first_ctx}));
  const auto first_trace = logger.entries;
  const size_t first_size = logger.size;
  const int32_t first_transitions = logger.transitions;
  CHECK(first_transitions <= MAX_WAVEFRONT_TRANSITIONS);
  CHECK_FALSE(logger.overflow);

  const generator::event::wavefront_reset reset_request{err};
  generator::detail::wavefront_reset_ctx reset_ctx{};
  REQUIRE(machine.process_event(
      generator::detail::event_wavefront_reset_run{reset_request, reset_ctx}));
  logger.reset();
  generator::detail::wavefront_run_ctx second_ctx{};
  REQUIRE(machine.process_event(
      generator::detail::wavefront_frame_run{request, second_ctx}));
  CHECK(logger.entries == first_trace);
  CHECK(logger.size == first_size);
  CHECK(logger.transitions == first_transitions);
  CHECK(logger.transitions <= MAX_WAVEFRONT_TRANSITIONS);
  CHECK_FALSE(logger.overflow);
}

TEST_CASE("speech_generator_wavefront_meets_the_fake_dispatch_time_budget") {
  fixture test{};
  std::array<float, FRAME_SAMPLES> pcm{};
  std::array<int32_t, CODEBOOKS> tokens{};
  generator::event::wavefront_attribution output{};
  bool produced = false;
  emel::error::type err = 0u;
  std::chrono::steady_clock::duration maximum_elapsed{};

  for (uint64_t sequence = 0u; sequence < 64u; ++sequence) {
    const auto start = std::chrono::steady_clock::now();
    const bool accepted =
        test.push(sequence, produced, output, pcm, tokens, err);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    REQUIRE(accepted);
    maximum_elapsed = std::max(maximum_elapsed, elapsed);
  }

  CHECK(maximum_elapsed < FAKE_WAVEFRONT_DISPATCH_BUDGET);
  CHECK(test.diagnostics.submissions.load() == test.diagnostics.joins.load());
}
