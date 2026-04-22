#include "bench_common.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/events.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "embedding_compare_contract.hpp"
#include "embedding_generator_bench_helpers.hpp"
#include "embedding_variant_manifest.hpp"
#include "tests/embeddings/te_fixture_data.hpp"

namespace emel::bench {

namespace te_fixture = emel::tests::embeddings::te_fixture;

std::uint64_t benchmark_timestamp_now_ns() noexcept {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

[[noreturn]] void fail_embedding_variant_setup(const char * step, const char * detail) {
  std::fprintf(stderr, "error: embedding variant setup failed at %s (%s)\n", step, detail);
  std::abort();
}

const std::vector<embedding_variant_manifest> & maintained_embedding_variants() {
  static const std::vector<embedding_variant_manifest> variants = [] {
    std::vector<embedding_variant_manifest> loaded = {};
    std::string error = {};
    const std::filesystem::path directory =
        te_fixture::repo_root() / "tools" / "bench" / "embedding_variants";
    if (!load_embedding_variant_manifests(directory, loaded, &error)) {
      fail_embedding_variant_setup("load_embedding_variant_manifests", error.c_str());
    }
    return loaded;
  }();
  return variants;
}

bool embedding_variant_enabled(const embedding_variant_manifest & variant) {
  const char * variant_id = std::getenv("EMEL_BENCH_VARIANT_ID");
  if (variant_id != nullptr && variant_id[0] != '\0') {
    return variant.id == std::string_view{variant_id};
  }
  const char * filter = std::getenv("EMEL_BENCH_CASE_FILTER");
  if (filter == nullptr || filter[0] == '\0') {
    return true;
  }
  const std::string_view value{filter};
  return variant.id.find(value) != std::string::npos ||
      variant.case_name.find(value) != std::string::npos ||
      variant.compare_group.find(value) != std::string::npos ||
      variant.modality.find(value) != std::string::npos;
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

struct embedding_case_sample {
  emel::embeddings::generator::event::benchmark_stage_timings timings = {};
  std::uint64_t output_tokens = 0u;
  std::uint64_t output_dim = 0u;
  std::uint64_t output_checksum = 0u;
  std::vector<float> output_values = {};
};

struct measured_embedding_case {
  result summary = {};
  embedding_case_sample anchor = {};
};

template <class fn_type>
measured_embedding_case measure_embedding_case(const char * name, const config & cfg, fn_type && fn) {
  std::vector<double> total_samples = {};
  std::vector<double> prepare_samples = {};
  std::vector<double> encode_samples = {};
  std::vector<double> publish_samples = {};
  total_samples.reserve(cfg.runs);
  prepare_samples.reserve(cfg.runs);
  encode_samples.reserve(cfg.runs);
  publish_samples.reserve(cfg.runs);

  embedding_case_sample anchor = {};
  bool anchor_seen = false;

  for (std::size_t run = 0; run < cfg.warmup_runs; ++run) {
    for (std::uint64_t i = 0; i < cfg.warmup_iterations; ++i) {
      (void) fn(false);
    }
  }

  for (std::size_t run = 0; run < cfg.runs; ++run) {
    std::uint64_t prepare_ns = 0u;
    std::uint64_t encode_ns = 0u;
    std::uint64_t publish_ns = 0u;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < cfg.iterations; ++i) {
      const auto sample = fn(!anchor_seen);
      prepare_ns += sample.timings.prepare_ns;
      encode_ns += sample.timings.encode_ns;
      publish_ns += sample.timings.publish_ns;
      if (!anchor_seen) {
        anchor = sample;
        anchor_seen = true;
      } else if (anchor.output_tokens != sample.output_tokens ||
                 anchor.output_dim != sample.output_dim ||
                 anchor.output_checksum != sample.output_checksum) {
        std::fprintf(stderr, "error: embedding generator benchmark output anchors changed across iterations\n");
        std::abort();
      }
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

  measured_embedding_case out = {};
  out.summary.name = name;
  out.summary.ns_per_op = total_samples[total_samples.size() / 2];
  out.summary.prepare_ns_per_op = prepare_samples[prepare_samples.size() / 2];
  out.summary.encode_ns_per_op = encode_samples[encode_samples.size() / 2];
  out.summary.publish_ns_per_op = publish_samples[publish_samples.size() / 2];
  out.summary.output_tokens = anchor.output_tokens;
  out.summary.output_dim = anchor.output_dim;
  out.summary.output_checksum = anchor.output_checksum;
  out.summary.iterations = cfg.iterations;
  out.summary.runs = cfg.runs;
  out.anchor = std::move(anchor);
  return out;
}

void append_embedding_generator_cases(std::vector<result> & results,
                                      const config & cfg,
                                      std::vector<embedding_compare_record> * compare_records) {
  if (!te_fixture::te_assets_present()) {
    std::fprintf(stderr, "warning: skipping embedding generator benchmark because TE assets are missing\n");
    return;
  }

  const auto & fixture = te_fixture::cached_te_fixture();
  const std::string prompt = te_fixture::read_text_file(te_fixture::te_prompt_path("red-square.txt"));
  const std::array messages = {
    emel::text::formatter::chat_message{.role = "user", .content = prompt},
  };
  const std::vector<uint8_t> image = te_fixture::make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto audio = te_fixture::make_sine_wave(440.0f);
  initialized_embedding_generator text_state{*fixture.model};
  initialized_embedding_generator image_state{*fixture.model};
  initialized_embedding_generator audio_state{*fixture.model};
  const bool capture_compare_outputs = compare_records != nullptr;

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
  bool text_anchor_output_captured = false;
  bool image_anchor_output_captured = false;
  bool audio_anchor_output_captured = false;

  auto text_fn = [&](const bool capture_anchor_output) -> embedding_case_sample {
    text_error = emel::error::cast(emel::embeddings::generator::error::none);
    text_output_dimension = 0;
    if (!text_state.generator.process_event(text_request) ||
        text_error != emel::error::cast(emel::embeddings::generator::error::none) ||
        text_output_dimension != 1280) {
      std::fprintf(stderr, "error: failed to run text embedding generator benchmark case\n");
      std::abort();
    }
    embedding_case_sample sample = {};
    sample.timings = text_timings;
    sample.output_tokens = 1u;
    sample.output_dim = static_cast<std::uint64_t>(text_output_dimension);
    capture_embedding_case_output(sample,
                                  text_output.data(),
                                  text_output.size(),
                                  capture_compare_outputs,
                                  capture_anchor_output,
                                  text_anchor_output_captured);
    return sample;
  };

  auto image_fn = [&](const bool capture_anchor_output) -> embedding_case_sample {
    image_error = emel::error::cast(emel::embeddings::generator::error::none);
    image_output_dimension = 0;
    if (!image_state.generator.process_event(image_request) ||
        image_error != emel::error::cast(emel::embeddings::generator::error::none) ||
        image_output_dimension != 1280) {
      std::fprintf(stderr, "error: failed to run image embedding generator benchmark case\n");
      std::abort();
    }
    embedding_case_sample sample = {};
    sample.timings = image_timings;
    sample.output_tokens = 1u;
    sample.output_dim = static_cast<std::uint64_t>(image_output_dimension);
    capture_embedding_case_output(sample,
                                  image_output.data(),
                                  image_output.size(),
                                  capture_compare_outputs,
                                  capture_anchor_output,
                                  image_anchor_output_captured);
    return sample;
  };

  auto audio_fn = [&](const bool capture_anchor_output) -> embedding_case_sample {
    audio_error = emel::error::cast(emel::embeddings::generator::error::none);
    audio_output_dimension = 0;
    if (!audio_state.generator.process_event(audio_request) ||
        audio_error != emel::error::cast(emel::embeddings::generator::error::none) ||
        audio_output_dimension != 1280) {
      std::fprintf(stderr, "error: failed to run audio embedding generator benchmark case\n");
      std::abort();
    }
    embedding_case_sample sample = {};
    sample.timings = audio_timings;
    sample.output_tokens = 1u;
    sample.output_dim = static_cast<std::uint64_t>(audio_output_dimension);
    capture_embedding_case_output(sample,
                                  audio_output.data(),
                                  audio_output.size(),
                                  capture_compare_outputs,
                                  capture_anchor_output,
                                  audio_anchor_output_captured);
    return sample;
  };

  auto append_compare_record = [&](const embedding_variant_manifest & variant,
                                   measured_embedding_case & measured) {
    if (compare_records == nullptr) {
      return;
    }
    embedding_compare_record record = {};
    record.case_name = variant.case_name;
    record.compare_group = variant.compare_group;
    record.lane = "emel";
    record.backend_id = "emel.generator";
    record.backend_language = "cpp";
    record.comparison_mode = variant.comparison_mode;
    record.model_id = te_fixture::te_fixture_case_slug();
    record.fixture_id =
      te_fixture::te_fixture_path().lexically_relative(te_fixture::repo_root()).string();
    record.modality = variant.modality;
    record.ns_per_op = measured.summary.ns_per_op;
    record.prepare_ns_per_op = measured.summary.prepare_ns_per_op;
    record.encode_ns_per_op = measured.summary.encode_ns_per_op;
    record.publish_ns_per_op = measured.summary.publish_ns_per_op;
    record.output_tokens = measured.anchor.output_tokens;
    record.output_dim = measured.anchor.output_dim;
    record.output_checksum = measured.anchor.output_checksum;
    record.iterations = measured.summary.iterations;
    record.runs = measured.summary.runs;
    record.output_values = std::move(measured.anchor.output_values);
    record.note = variant.note;
    compare_records->push_back(std::move(record));
  };

  for (const embedding_variant_manifest & variant : maintained_embedding_variants()) {
    if (!embedding_variant_enabled(variant)) {
      continue;
    }
    if (variant.modality == "text" && variant.payload_id == "red_square_text_v1") {
      auto measured = measure_embedding_case(variant.case_name.c_str(), cfg, text_fn);
      results.push_back(measured.summary);
      append_compare_record(variant, measured);
      continue;
    }
    if (variant.modality == "image" && variant.payload_id == "red_square_image_v1") {
      auto measured = measure_embedding_case(variant.case_name.c_str(), cfg, image_fn);
      results.push_back(measured.summary);
      append_compare_record(variant, measured);
      continue;
    }
    if (variant.modality == "audio" && variant.payload_id == "pure_tone_440hz_audio_v1") {
      auto measured = measure_embedding_case(variant.case_name.c_str(), cfg, audio_fn);
      results.push_back(measured.summary);
      append_compare_record(variant, measured);
      continue;
    }
    fail_embedding_variant_setup("unsupported_embedding_payload", variant.id.c_str());
  }
}

}  // namespace emel::bench
