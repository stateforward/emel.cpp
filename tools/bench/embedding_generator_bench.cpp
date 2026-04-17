#include "bench_common.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/events.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "tests/embeddings/te_fixture_data.hpp"

namespace emel::bench {

namespace te_fixture = emel::tests::embeddings::te_fixture;

std::string text_embedding_generator_case_name() {
  return "embeddings/generator/steady_request/" + te_fixture::te_fixture_case_slug() +
      "_text_red_square_full_dim";
}

std::string image_embedding_generator_case_name() {
  return "embeddings/generator/steady_request/" + te_fixture::te_fixture_case_slug() +
      "_image_red_square_full_dim";
}

std::string audio_embedding_generator_case_name() {
  return "embeddings/generator/steady_request/" + te_fixture::te_fixture_case_slug() +
      "_audio_pure_tone_440hz_full_dim";
}

std::uint64_t benchmark_timestamp_now_ns() noexcept {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

bool benchmark_case_enabled(const char * case_name) {
  const char * filter = std::getenv("EMEL_BENCH_CASE_FILTER");
  if (filter == nullptr || filter[0] == '\0') {
    return true;
  }
  return std::string_view{case_name}.find(filter) != std::string_view::npos;
}

struct initialized_embedding_generator {
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm generator;

  explicit initialized_embedding_generator(const emel::model::data & model)
      : generator(model, conditioner, nullptr, emel::text::formatter::format_raw) {
    emel::error::type initialize_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::embeddings::generator::event::initialize initialize{
      &tokenizer,
      te_fixture::tokenizer_bind_dispatch,
      te_fixture::tokenizer_tokenize_dispatch,
    };
    initialize.preprocessor_variant =
        emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
    initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
    initialize.error_out = &initialize_error;
    if (!generator.process_event(initialize) ||
        initialize_error != emel::error::cast(emel::embeddings::generator::error::none)) {
      std::fprintf(stderr, "error: failed to initialize embedding generator benchmark case\n");
      std::abort();
    }
  }
};

void print_embedding_generator_bench_metadata() {
  std::printf("# embedding_generator_fixture: %s\n",
              te_fixture::te_fixture_path().lexically_relative(te_fixture::repo_root()).string().c_str());
  std::printf("# embedding_generator_fixture_slug: %s\n", te_fixture::te_fixture_case_slug().c_str());
  std::printf("# embedding_generator_vocab: %s\n",
              te_fixture::te_vocab_path().lexically_relative(te_fixture::repo_root()).string().c_str());
  std::printf("# embedding_generator_prompt: %s\n",
              te_fixture::te_prompt_path("red-square.txt")
                  .lexically_relative(te_fixture::repo_root())
                  .string()
                  .c_str());
  std::printf("# embedding_generator_contract_text: modality=text include_initialize=false "
              "truncate_dimension=0 preprocessor=wpm encoder=wpm formatter=raw\n");
  std::printf("# embedding_generator_contract_image: modality=image include_initialize=false "
              "truncate_dimension=0 payload=rgba_32x32_red_square\n");
  std::printf("# embedding_generator_contract_audio: modality=audio include_initialize=false "
              "truncate_dimension=0 payload=mono_f32_16khz_4000_pure_tone_440hz\n");
}

template <class fn_type>
result measure_embedding_case(const char * name, const config & cfg, fn_type && fn) {
  std::vector<double> total_samples = {};
  std::vector<double> prepare_samples = {};
  std::vector<double> encode_samples = {};
  std::vector<double> publish_samples = {};
  total_samples.reserve(cfg.runs);
  prepare_samples.reserve(cfg.runs);
  encode_samples.reserve(cfg.runs);
  publish_samples.reserve(cfg.runs);

  for (std::size_t run = 0; run < cfg.warmup_runs; ++run) {
    for (std::uint64_t i = 0; i < cfg.warmup_iterations; ++i) {
      (void) fn();
    }
  }

  for (std::size_t run = 0; run < cfg.runs; ++run) {
    std::uint64_t prepare_ns = 0u;
    std::uint64_t encode_ns = 0u;
    std::uint64_t publish_ns = 0u;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < cfg.iterations; ++i) {
      const auto timings = fn();
      prepare_ns += timings.prepare_ns;
      encode_ns += timings.encode_ns;
      publish_ns += timings.publish_ns;
    }
    const auto end = std::chrono::steady_clock::now();
    const auto duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_samples.push_back(static_cast<double>(duration_ns) / static_cast<double>(cfg.iterations));
    prepare_samples.push_back(static_cast<double>(prepare_ns) / static_cast<double>(cfg.iterations));
    encode_samples.push_back(static_cast<double>(encode_ns) / static_cast<double>(cfg.iterations));
    publish_samples.push_back(static_cast<double>(publish_ns) / static_cast<double>(cfg.iterations));
  }

  std::sort(total_samples.begin(), total_samples.end());
  std::sort(prepare_samples.begin(), prepare_samples.end());
  std::sort(encode_samples.begin(), encode_samples.end());
  std::sort(publish_samples.begin(), publish_samples.end());

  result out = {};
  out.name = name;
  out.ns_per_op = total_samples[total_samples.size() / 2];
  out.prepare_ns_per_op = prepare_samples[prepare_samples.size() / 2];
  out.encode_ns_per_op = encode_samples[encode_samples.size() / 2];
  out.publish_ns_per_op = publish_samples[publish_samples.size() / 2];
  out.iterations = cfg.iterations;
  out.runs = cfg.runs;
  return out;
}

void append_embedding_generator_cases(std::vector<result> & results, const config & cfg) {
  if (!te_fixture::te_assets_present()) {
    std::fprintf(stderr, "warning: skipping embedding generator benchmark because TE assets are missing\n");
    return;
  }

  const auto & fixture = te_fixture::cached_te_fixture();
  const std::string text_case_name = text_embedding_generator_case_name();
  const std::string image_case_name = image_embedding_generator_case_name();
  const std::string audio_case_name = audio_embedding_generator_case_name();
  const std::string prompt = te_fixture::read_text_file(te_fixture::te_prompt_path("red-square.txt"));
  const std::array messages = {
    emel::text::formatter::chat_message{.role = "user", .content = prompt},
  };
  const std::vector<uint8_t> image = te_fixture::make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto audio = te_fixture::make_sine_wave(440.0f);
  initialized_embedding_generator text_state{*fixture.model};
  initialized_embedding_generator image_state{*fixture.model};
  initialized_embedding_generator audio_state{*fixture.model};

  std::array<float, 1280> text_output = {};
  int32_t text_output_dimension = 0;
  emel::embeddings::generator::event::benchmark_stage_timings text_timings = {};
  emel::error::type text_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_text text_request{
    messages,
    text_output,
    text_output_dimension,
    text_timings,
  };
  text_request.error_out = &text_error;
  text_request.benchmark_time_now = benchmark_timestamp_now_ns;

  std::array<float, 1280> image_output = {};
  int32_t image_output_dimension = 0;
  emel::embeddings::generator::event::benchmark_stage_timings image_timings = {};
  emel::error::type image_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_image image_request{
    image,
    32,
    32,
    image_output,
    image_output_dimension,
    image_timings,
  };
  image_request.error_out = &image_error;
  image_request.benchmark_time_now = benchmark_timestamp_now_ns;

  std::array<float, 1280> audio_output = {};
  int32_t audio_output_dimension = 0;
  emel::embeddings::generator::event::benchmark_stage_timings audio_timings = {};
  emel::error::type audio_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::embed_audio audio_request{
    audio,
    16000,
    audio_output,
    audio_output_dimension,
    audio_timings,
  };
  audio_request.error_out = &audio_error;
  audio_request.benchmark_time_now = benchmark_timestamp_now_ns;

  auto text_fn = [&]() {
    text_error = emel::error::cast(emel::embeddings::generator::error::none);
    text_output_dimension = 0;
    if (!text_state.generator.process_event(text_request) ||
        text_error != emel::error::cast(emel::embeddings::generator::error::none) ||
        text_output_dimension != 1280) {
      std::fprintf(stderr, "error: failed to run text embedding generator benchmark case\n");
      std::abort();
    }
    return text_timings;
  };

  auto image_fn = [&]() {
    image_error = emel::error::cast(emel::embeddings::generator::error::none);
    image_output_dimension = 0;
    if (!image_state.generator.process_event(image_request) ||
        image_error != emel::error::cast(emel::embeddings::generator::error::none) ||
        image_output_dimension != 1280) {
      std::fprintf(stderr, "error: failed to run image embedding generator benchmark case\n");
      std::abort();
    }
    return image_timings;
  };

  auto audio_fn = [&]() {
    audio_error = emel::error::cast(emel::embeddings::generator::error::none);
    audio_output_dimension = 0;
    if (!audio_state.generator.process_event(audio_request) ||
        audio_error != emel::error::cast(emel::embeddings::generator::error::none) ||
        audio_output_dimension != 1280) {
      std::fprintf(stderr, "error: failed to run audio embedding generator benchmark case\n");
      std::abort();
    }
    return audio_timings;
  };

  if (benchmark_case_enabled(text_case_name.c_str())) {
    results.push_back(measure_embedding_case(text_case_name.c_str(), cfg, text_fn));
  }
  if (benchmark_case_enabled(image_case_name.c_str())) {
    results.push_back(measure_embedding_case(image_case_name.c_str(), cfg, image_fn));
  }
  if (benchmark_case_enabled(audio_case_name.c_str())) {
    results.push_back(measure_embedding_case(audio_case_name.c_str(), cfg, audio_fn));
  }
}

}  // namespace emel::bench
