#include "bench_common.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "llama.h"
#include "mtmd.h"

namespace emel::bench {

namespace {

inline constexpr char k_text_arctic_case_name[] =
  "reference/text/steady_request/arctic_s_text_red_square_full_dim";
inline constexpr char k_text_gemma_case_name[] =
  "reference/text/steady_request/embeddinggemma_300m_text_red_square_full_dim";
inline constexpr char k_vision_case_name[] =
  "reference/vision/steady_request/lfm2_vl_450m_image_red_square_full_dim";
inline constexpr char k_audio_case_name[] =
  "reference/audio/steady_request/ultravox_v0_5_llama_3_2_1b_audio_pure_tone_440hz_full_dim";

inline constexpr char k_text_arctic_model_env[] = "EMEL_REFERENCE_TEXT_MODEL_ARCTIC_S";
inline constexpr char k_text_gemma_model_env[] = "EMEL_REFERENCE_TEXT_MODEL_EMBEDDINGGEMMA_300M";
inline constexpr char k_vision_model_env[] = "EMEL_REFERENCE_VISION_MODEL";
inline constexpr char k_vision_mmproj_env[] = "EMEL_REFERENCE_VISION_MMPROJ";
inline constexpr char k_audio_model_env[] = "EMEL_REFERENCE_AUDIO_MODEL";
inline constexpr char k_audio_mmproj_env[] = "EMEL_REFERENCE_AUDIO_MMPROJ";
inline constexpr char k_threads_env[] = "EMEL_REFERENCE_THREADS";
inline constexpr char k_legacy_mm_model_env[] = "EMEL_REFERENCE_MM_MODEL";
inline constexpr char k_legacy_mmproj_env[] = "EMEL_REFERENCE_MM_MMPROJ";
inline constexpr char k_legacy_mm_threads_env[] = "EMEL_REFERENCE_MM_THREADS";
inline constexpr char k_case_filter_env[] = "EMEL_BENCH_CASE_FILTER";

inline constexpr char k_text_prompt[] = "red square";
inline constexpr std::size_t k_image_width = 32u;
inline constexpr std::size_t k_image_height = 32u;
inline constexpr std::size_t k_audio_samples = 4000u;
inline constexpr float k_audio_frequency_hz = 440.0f;
inline constexpr float k_audio_sample_rate_hz = 16000.0f;
inline constexpr double k_pi = 3.14159265358979323846;
inline constexpr std::int32_t k_reference_context_tokens = 512;

struct mtmd_context_deleter {
  void operator()(mtmd_context * value) const noexcept {
    mtmd_free(value);
  }
};

struct mtmd_bitmap_deleter {
  void operator()(mtmd_bitmap * value) const noexcept {
    mtmd_bitmap_free(value);
  }
};

struct mtmd_input_chunks_deleter {
  void operator()(mtmd_input_chunks * value) const noexcept {
    mtmd_input_chunks_free(value);
  }
};

struct llama_model_deleter {
  void operator()(llama_model * value) const noexcept {
    llama_model_free(value);
  }
};

struct llama_context_deleter {
  void operator()(llama_context * value) const noexcept {
    llama_free(value);
  }
};

using mtmd_context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;
using mtmd_bitmap_ptr = std::unique_ptr<mtmd_bitmap, mtmd_bitmap_deleter>;
using mtmd_input_chunks_ptr = std::unique_ptr<mtmd_input_chunks, mtmd_input_chunks_deleter>;
using llama_model_ptr = std::unique_ptr<llama_model, llama_model_deleter>;
using llama_context_ptr = std::unique_ptr<llama_context, llama_context_deleter>;

struct text_reference_session {
  llama_model_ptr model = {};
  llama_context_ptr context = {};
  std::int32_t n_threads = 1;
};

struct multimodal_reference_session {
  llama_model_ptr model = {};
  mtmd_context_ptr context = {};
  std::int32_t n_threads = 1;
};

struct reference_stage_timings {
  std::uint64_t prepare_ns = 0u;
  std::uint64_t encode_ns = 0u;
  std::uint64_t publish_ns = 0u;
  std::uint64_t output_tokens = 0u;
  std::uint64_t output_dim = 0u;
  std::uint64_t output_checksum = 0u;
};

struct reference_case_state {
  std::vector<float> publish_buffer = {};
};

struct llama_batch_holder {
  explicit llama_batch_holder(const std::int32_t n_tokens)
    : batch(llama_batch_init(n_tokens, 0, 1)) {}

  ~llama_batch_holder() {
    llama_batch_free(batch);
  }

  llama_batch batch = {};
};

bool reference_case_enabled(const char * case_name) {
  const char * filter = std::getenv(k_case_filter_env);
  if (filter == nullptr || filter[0] == '\0') {
    return true;
  }
  return std::string_view{case_name}.find(filter) != std::string_view::npos;
}

std::string configured_env_value(const char * primary, const char * fallback = nullptr) {
  const char * value = std::getenv(primary);
  if (value != nullptr && value[0] != '\0') {
    return value;
  }
  if (fallback != nullptr) {
    value = std::getenv(fallback);
    if (value != nullptr && value[0] != '\0') {
      return value;
    }
  }
  return {};
}

bool configured_path_exists(const std::string & path) {
  return !path.empty() && std::filesystem::exists(std::filesystem::path(path));
}

std::int32_t read_thread_count() {
  const std::string configured = configured_env_value(k_threads_env, k_legacy_mm_threads_env);
  if (configured.empty()) {
    const auto detected = std::thread::hardware_concurrency();
    return detected == 0u ? 1 : static_cast<std::int32_t>(detected);
  }
  char * end = nullptr;
  const long parsed = std::strtol(configured.c_str(), &end, 10);
  if (end == configured.c_str() || parsed <= 0) {
    return 1;
  }
  return static_cast<std::int32_t>(parsed);
}

std::vector<unsigned char> make_red_square_rgb() {
  std::vector<unsigned char> image(k_image_width * k_image_height * 3u, 0u);
  for (std::size_t pixel = 0; pixel < k_image_width * k_image_height; ++pixel) {
    image[pixel * 3u + 0u] = 255u;
  }
  return image;
}

std::vector<float> make_pure_tone_audio() {
  std::vector<float> audio(k_audio_samples, 0.0f);
  for (std::size_t i = 0; i < audio.size(); ++i) {
    const double phase =
      2.0 * k_pi * static_cast<double>(k_audio_frequency_hz) *
      static_cast<double>(i) / static_cast<double>(k_audio_sample_rate_hz);
    audio[i] = static_cast<float>(std::sin(phase));
  }
  return audio;
}

std::uint64_t checksum_bytes(const std::uint8_t * bytes, const std::size_t count) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < count; ++i) {
    hash ^= static_cast<std::uint64_t>(bytes[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

void reference_log_sink(enum ggml_log_level, const char *, void *) {}

const mtmd_input_chunk * find_media_chunk(const mtmd_input_chunks * chunks,
                                          const mtmd_input_chunk_type expected_type) {
  const mtmd_input_chunk * match = nullptr;
  for (std::size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
    const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
    if (chunk == nullptr) {
      continue;
    }
    if (mtmd_input_chunk_get_type(chunk) != expected_type) {
      continue;
    }
    if (match != nullptr) {
      std::fprintf(stderr, "error: reference benchmark found multiple media chunks for one request\n");
      std::exit(1);
    }
    match = chunk;
  }
  return match;
}

void release_backend() {
  llama_backend_free();
}

void ensure_backend_ready() {
  static const bool ready = []() {
    llama_backend_init();
    llama_log_set(reference_log_sink, nullptr);
    mtmd_log_set(reference_log_sink, nullptr);
    std::atexit(release_backend);
    return true;
  }();
  (void) ready;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string_view prompt) {
  int32_t token_count = llama_tokenize(
    vocab,
    prompt.data(),
    static_cast<int32_t>(prompt.size()),
    nullptr,
    0,
    true,
    true);
  if (token_count == INT32_MIN) {
    std::fprintf(stderr, "error: reference text tokenize overflowed\n");
    std::exit(1);
  }
  if (token_count < 0) {
    token_count = -token_count;
  }
  std::vector<llama_token> tokens(static_cast<std::size_t>(token_count));
  const int32_t finalized = llama_tokenize(
    vocab,
    prompt.data(),
    static_cast<int32_t>(prompt.size()),
    tokens.data(),
    token_count,
    true,
    true);
  if (finalized < 0) {
    std::fprintf(stderr, "error: reference text tokenize failed on second pass\n");
    std::exit(1);
  }
  tokens.resize(static_cast<std::size_t>(finalized));
  return tokens;
}

void fill_single_sequence_batch(llama_batch & batch, const std::vector<llama_token> & tokens) {
  batch.n_tokens = static_cast<int32_t>(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = static_cast<llama_pos>(i);
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = static_cast<int8_t>(i + 1u == tokens.size());
  }
}

bool pooling_outputs_sequence_embedding(const llama_context * context) {
  switch (llama_pooling_type(context)) {
    case LLAMA_POOLING_TYPE_NONE:
    case LLAMA_POOLING_TYPE_UNSPECIFIED:
      return false;
    default:
      return true;
  }
}

text_reference_session load_text_session(const std::string & model_path) {
  ensure_backend_ready();
  text_reference_session session = {};
  session.n_threads = read_thread_count();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  session.model.reset(llama_model_load_from_file(model_path.c_str(), model_params));
  if (!session.model) {
    std::fprintf(stderr, "error: failed to load reference text model from %s\n", model_path.c_str());
    std::exit(1);
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = k_reference_context_tokens;
  context_params.n_batch = k_reference_context_tokens;
  context_params.n_ubatch = k_reference_context_tokens;
  context_params.n_seq_max = 1;
  context_params.n_threads = session.n_threads;
  context_params.n_threads_batch = session.n_threads;
  context_params.embeddings = true;
  context_params.no_perf = true;
  session.context.reset(llama_init_from_model(session.model.get(), context_params));
  if (!session.context) {
    std::fprintf(stderr, "error: failed to create reference text context for %s\n", model_path.c_str());
    std::exit(1);
  }

  if (llama_model_has_encoder(session.model.get()) && llama_model_has_decoder(session.model.get())) {
    std::fprintf(stderr,
                 "error: reference text benchmark does not support encoder-decoder embedding models: %s\n",
                 model_path.c_str());
    std::exit(1);
  }

  return session;
}

multimodal_reference_session load_multimodal_session(const std::string & model_path,
                                                     const std::string & mmproj_path) {
  ensure_backend_ready();
  multimodal_reference_session session = {};
  session.n_threads = read_thread_count();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  session.model.reset(llama_model_load_from_file(model_path.c_str(), model_params));
  if (!session.model) {
    std::fprintf(stderr, "error: failed to load reference multimodal text model from %s\n",
                 model_path.c_str());
    std::exit(1);
  }

  mtmd_context_params context_params = mtmd_context_params_default();
  context_params.use_gpu = false;
  context_params.print_timings = false;
  context_params.n_threads = session.n_threads;
  context_params.warmup = false;
  session.context.reset(
    mtmd_init_from_file(mmproj_path.c_str(), session.model.get(), context_params));
  if (!session.context) {
    std::fprintf(stderr, "error: failed to load reference multimodal projector from %s\n",
                 mmproj_path.c_str());
    std::exit(1);
  }

  return session;
}

reference_stage_timings run_text_case(text_reference_session & session,
                                      reference_case_state & state) {
  reference_stage_timings timings = {};

  const auto prepare_start = std::chrono::steady_clock::now();
  const llama_vocab * vocab = llama_model_get_vocab(session.model.get());
  if (vocab == nullptr) {
    std::fprintf(stderr, "error: reference text model returned null vocab\n");
    std::exit(1);
  }
  const std::vector<llama_token> tokens = tokenize_prompt(vocab, k_text_prompt);
  if (tokens.empty()) {
    std::fprintf(stderr, "error: reference text tokenize produced no tokens\n");
    std::exit(1);
  }
  llama_batch_holder batch_holder(static_cast<int32_t>(tokens.size()));
  fill_single_sequence_batch(batch_holder.batch, tokens);
  const auto prepare_end = std::chrono::steady_clock::now();
  timings.prepare_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(prepare_end - prepare_start).count());

  const auto encode_start = std::chrono::steady_clock::now();
  int32_t status = 0;
  if (llama_model_has_encoder(session.model.get())) {
    status = llama_encode(session.context.get(), batch_holder.batch);
  } else {
    llama_memory_clear(llama_get_memory(session.context.get()), true);
    status = llama_decode(session.context.get(), batch_holder.batch);
  }
  if (status != 0) {
    std::fprintf(stderr, "error: reference text encode failed with status %d\n", status);
    std::exit(1);
  }
  llama_synchronize(session.context.get());
  const auto encode_end = std::chrono::steady_clock::now();
  timings.encode_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(encode_end - encode_start).count());

  const std::uint64_t output_dim =
    static_cast<std::uint64_t>(llama_model_n_embd_out(session.model.get()));
  float * output = nullptr;
  if (pooling_outputs_sequence_embedding(session.context.get())) {
    output = llama_get_embeddings_seq(session.context.get(), 0);
  } else {
    output = llama_get_embeddings_ith(session.context.get(), 0);
  }
  if (output == nullptr) {
    std::fprintf(stderr, "error: reference text encode returned a null output buffer\n");
    std::exit(1);
  }

  const auto publish_start = std::chrono::steady_clock::now();
  state.publish_buffer.resize(static_cast<std::size_t>(output_dim));
  std::memcpy(state.publish_buffer.data(), output, state.publish_buffer.size() * sizeof(float));
  const auto publish_end = std::chrono::steady_clock::now();
  timings.publish_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(publish_end - publish_start).count());
  timings.output_tokens = 1u;
  timings.output_dim = output_dim;
  timings.output_checksum = checksum_bytes(
    reinterpret_cast<const std::uint8_t *>(state.publish_buffer.data()),
    state.publish_buffer.size() * sizeof(float));

  return timings;
}

reference_stage_timings run_image_case(multimodal_reference_session & session,
                                       reference_case_state & state) {
  reference_stage_timings timings = {};
  const std::vector<unsigned char> image = make_red_square_rgb();

  const auto prepare_start = std::chrono::steady_clock::now();
  mtmd_bitmap_ptr bitmap(
    mtmd_bitmap_init(static_cast<std::uint32_t>(k_image_width),
                     static_cast<std::uint32_t>(k_image_height),
                     image.data()));
  if (!bitmap) {
    std::fprintf(stderr, "error: failed to create reference image bitmap\n");
    std::exit(1);
  }
  const mtmd_bitmap * bitmaps[] = {bitmap.get()};
  mtmd_input_text text = {
    .text = mtmd_default_marker(),
    .add_special = false,
    .parse_special = true,
  };
  mtmd_input_chunks_ptr chunks(mtmd_input_chunks_init());
  if (!chunks) {
    std::fprintf(stderr, "error: failed to allocate reference image chunks\n");
    std::exit(1);
  }
  const std::int32_t tokenize_status =
    mtmd_tokenize(session.context.get(), chunks.get(), &text, bitmaps, 1u);
  if (tokenize_status != 0) {
    std::fprintf(stderr, "error: reference image tokenize failed with status %d\n", tokenize_status);
    std::exit(1);
  }
  const mtmd_input_chunk * chunk = find_media_chunk(chunks.get(), MTMD_INPUT_CHUNK_TYPE_IMAGE);
  if (chunk == nullptr) {
    std::fprintf(stderr, "error: reference image tokenize did not produce an image chunk\n");
    std::exit(1);
  }
  const auto prepare_end = std::chrono::steady_clock::now();
  timings.prepare_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(prepare_end - prepare_start).count());

  const auto encode_start = std::chrono::steady_clock::now();
  if (mtmd_encode_chunk(session.context.get(), chunk) != 0) {
    std::fprintf(stderr, "error: reference image encode failed\n");
    std::exit(1);
  }
  const auto encode_end = std::chrono::steady_clock::now();
  timings.encode_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(encode_end - encode_start).count());

  const std::uint64_t output_tokens = mtmd_input_chunk_get_n_tokens(chunk);
  const std::uint64_t output_dim = static_cast<std::uint64_t>(llama_model_n_embd_inp(session.model.get()));
  const std::size_t output_floats =
    static_cast<std::size_t>(output_tokens) * static_cast<std::size_t>(output_dim);
  float * output = mtmd_get_output_embd(session.context.get());
  if (output == nullptr) {
    std::fprintf(stderr, "error: reference image encode returned a null output buffer\n");
    std::exit(1);
  }
  const auto publish_start = std::chrono::steady_clock::now();
  state.publish_buffer.resize(output_floats);
  std::memcpy(state.publish_buffer.data(), output, output_floats * sizeof(float));
  const auto publish_end = std::chrono::steady_clock::now();
  timings.publish_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(publish_end - publish_start).count());
  timings.output_tokens = output_tokens;
  timings.output_dim = output_dim;
  timings.output_checksum = checksum_bytes(
    reinterpret_cast<const std::uint8_t *>(state.publish_buffer.data()),
    state.publish_buffer.size() * sizeof(float));

  return timings;
}

reference_stage_timings run_audio_case(multimodal_reference_session & session,
                                       reference_case_state & state) {
  reference_stage_timings timings = {};
  const std::vector<float> audio = make_pure_tone_audio();

  const auto prepare_start = std::chrono::steady_clock::now();
  mtmd_bitmap_ptr bitmap(mtmd_bitmap_init_from_audio(audio.size(), audio.data()));
  if (!bitmap) {
    std::fprintf(stderr, "error: failed to create reference audio bitmap\n");
    std::exit(1);
  }
  const mtmd_bitmap * bitmaps[] = {bitmap.get()};
  mtmd_input_text text = {
    .text = mtmd_default_marker(),
    .add_special = false,
    .parse_special = true,
  };
  mtmd_input_chunks_ptr chunks(mtmd_input_chunks_init());
  if (!chunks) {
    std::fprintf(stderr, "error: failed to allocate reference audio chunks\n");
    std::exit(1);
  }
  const std::int32_t tokenize_status =
    mtmd_tokenize(session.context.get(), chunks.get(), &text, bitmaps, 1u);
  if (tokenize_status != 0) {
    std::fprintf(stderr, "error: reference audio tokenize failed with status %d\n", tokenize_status);
    std::exit(1);
  }
  const mtmd_input_chunk * chunk = find_media_chunk(chunks.get(), MTMD_INPUT_CHUNK_TYPE_AUDIO);
  if (chunk == nullptr) {
    std::fprintf(stderr, "error: reference audio tokenize did not produce an audio chunk\n");
    std::exit(1);
  }
  const auto prepare_end = std::chrono::steady_clock::now();
  timings.prepare_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(prepare_end - prepare_start).count());

  const auto encode_start = std::chrono::steady_clock::now();
  if (mtmd_encode_chunk(session.context.get(), chunk) != 0) {
    std::fprintf(stderr, "error: reference audio encode failed\n");
    std::exit(1);
  }
  const auto encode_end = std::chrono::steady_clock::now();
  timings.encode_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(encode_end - encode_start).count());

  const std::uint64_t output_tokens = mtmd_input_chunk_get_n_tokens(chunk);
  const std::uint64_t output_dim = static_cast<std::uint64_t>(llama_model_n_embd_inp(session.model.get()));
  const std::size_t output_floats =
    static_cast<std::size_t>(output_tokens) * static_cast<std::size_t>(output_dim);
  float * output = mtmd_get_output_embd(session.context.get());
  if (output == nullptr) {
    std::fprintf(stderr, "error: reference audio encode returned a null output buffer\n");
    std::exit(1);
  }
  const auto publish_start = std::chrono::steady_clock::now();
  state.publish_buffer.resize(output_floats);
  std::memcpy(state.publish_buffer.data(), output, output_floats * sizeof(float));
  const auto publish_end = std::chrono::steady_clock::now();
  timings.publish_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(publish_end - publish_start).count());
  timings.output_tokens = output_tokens;
  timings.output_dim = output_dim;
  timings.output_checksum = checksum_bytes(
    reinterpret_cast<const std::uint8_t *>(state.publish_buffer.data()),
    state.publish_buffer.size() * sizeof(float));

  return timings;
}

template <class fn_type>
result measure_reference_case(const char * name, const config & cfg, fn_type && fn) {
  std::vector<double> total_samples = {};
  std::vector<double> prepare_samples = {};
  std::vector<double> encode_samples = {};
  std::vector<double> publish_samples = {};
  total_samples.reserve(cfg.runs);
  prepare_samples.reserve(cfg.runs);
  encode_samples.reserve(cfg.runs);
  publish_samples.reserve(cfg.runs);

  reference_stage_timings anchor = {};
  bool anchor_seen = false;

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
      const reference_stage_timings sample = fn();
      prepare_ns += sample.prepare_ns;
      encode_ns += sample.encode_ns;
      publish_ns += sample.publish_ns;
      if (!anchor_seen) {
        anchor = sample;
        anchor_seen = true;
      } else if (anchor.output_tokens != sample.output_tokens ||
                 anchor.output_dim != sample.output_dim ||
                 anchor.output_checksum != sample.output_checksum) {
        std::fprintf(stderr, "error: reference benchmark output anchors changed across iterations\n");
        std::exit(1);
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

  result out = {};
  out.name = name;
  out.ns_per_op = total_samples[total_samples.size() / 2];
  out.prepare_ns_per_op = prepare_samples[prepare_samples.size() / 2];
  out.encode_ns_per_op = encode_samples[encode_samples.size() / 2];
  out.publish_ns_per_op = publish_samples[publish_samples.size() / 2];
  out.output_tokens = anchor.output_tokens;
  out.output_dim = anchor.output_dim;
  out.output_checksum = anchor.output_checksum;
  out.iterations = cfg.iterations;
  out.runs = cfg.runs;
  return out;
}

void print_model_env_metadata(const char * label,
                              const char * primary_env,
                              const char * fallback_env = nullptr) {
  const std::string configured = configured_env_value(primary_env, fallback_env);
  const bool exists = configured_path_exists(configured);
  std::printf("# embedding_reference_%s_env: %s\n", label, primary_env);
  if (fallback_env != nullptr) {
    std::printf("# embedding_reference_%s_fallback_env: %s\n", label, fallback_env);
  }
  std::printf("# embedding_reference_%s_path: %s\n",
              label,
              configured.empty() ? "<unset>" : configured.c_str());
  std::printf("# embedding_reference_%s_asset: exists=%s\n",
              label,
              exists ? "true" : "false");
}

}  // namespace

void print_embedding_reference_bench_metadata() {
  std::printf("# embedding_reference_threads_env: %s\n", k_threads_env);
  std::printf("# embedding_reference_legacy_mm_threads_env: %s\n", k_legacy_mm_threads_env);
  print_model_env_metadata("text_model_arctic_s", k_text_arctic_model_env);
  print_model_env_metadata("text_model_embeddinggemma_300m", k_text_gemma_model_env);
  print_model_env_metadata("vision_model", k_vision_model_env, k_legacy_mm_model_env);
  print_model_env_metadata("vision_mmproj", k_vision_mmproj_env, k_legacy_mmproj_env);
  print_model_env_metadata("audio_model", k_audio_model_env);
  print_model_env_metadata("audio_mmproj", k_audio_mmproj_env);
  std::printf("# embedding_reference_contract_text_arctic_s: modality=text include_initialize=false "
              "payload=prompt_red_square pooling=model_default_or_last_token_fallback\n");
  std::printf("# embedding_reference_contract_text_embeddinggemma_300m: modality=text "
              "include_initialize=false payload=prompt_red_square pooling=model_default_or_last_token_fallback\n");
  std::printf("# embedding_reference_contract_image_lfm2_vl_450m: modality=image "
              "include_initialize=false payload=rgb_32x32_red_square marker_only_prompt=true\n");
  std::printf("# embedding_reference_contract_audio_ultravox_1b: modality=audio "
              "include_initialize=false payload=mono_f32_16khz_4000_pure_tone_440hz "
              "marker_only_prompt=true\n");
}

void append_embedding_reference_cases(std::vector<result> & results, const config & cfg) {
  const std::string arctic_model_path = configured_env_value(k_text_arctic_model_env);
  const std::string gemma_model_path = configured_env_value(k_text_gemma_model_env);
  const std::string vision_model_path =
    configured_env_value(k_vision_model_env, k_legacy_mm_model_env);
  const std::string vision_mmproj_path =
    configured_env_value(k_vision_mmproj_env, k_legacy_mmproj_env);
  const std::string audio_model_path = configured_env_value(k_audio_model_env);
  const std::string audio_mmproj_path = configured_env_value(k_audio_mmproj_env);

  if (configured_path_exists(arctic_model_path) && reference_case_enabled(k_text_arctic_case_name)) {
    text_reference_session session = load_text_session(arctic_model_path);
    std::printf("# embedding_reference_case_support_arctic_s: pooling=%d encoder=%s decoder=%s\n",
                static_cast<int>(llama_pooling_type(session.context.get())),
                llama_model_has_encoder(session.model.get()) ? "true" : "false",
                llama_model_has_decoder(session.model.get()) ? "true" : "false");
    reference_case_state state = {};
    results.push_back(measure_reference_case(k_text_arctic_case_name, cfg, [&]() {
      return run_text_case(session, state);
    }));
  }

  if (configured_path_exists(gemma_model_path) && reference_case_enabled(k_text_gemma_case_name)) {
    text_reference_session session = load_text_session(gemma_model_path);
    std::printf("# embedding_reference_case_support_embeddinggemma_300m: pooling=%d encoder=%s "
                "decoder=%s\n",
                static_cast<int>(llama_pooling_type(session.context.get())),
                llama_model_has_encoder(session.model.get()) ? "true" : "false",
                llama_model_has_decoder(session.model.get()) ? "true" : "false");
    reference_case_state state = {};
    results.push_back(measure_reference_case(k_text_gemma_case_name, cfg, [&]() {
      return run_text_case(session, state);
    }));
  }

  if (configured_path_exists(vision_model_path) &&
      configured_path_exists(vision_mmproj_path) &&
      reference_case_enabled(k_vision_case_name)) {
    multimodal_reference_session session = load_multimodal_session(vision_model_path, vision_mmproj_path);
    std::printf("# embedding_reference_case_support_lfm2_vl_450m: vision=%s audio=%s threads=%d\n",
                mtmd_support_vision(session.context.get()) ? "true" : "false",
                mtmd_support_audio(session.context.get()) ? "true" : "false",
                session.n_threads);
    if (mtmd_support_vision(session.context.get())) {
      reference_case_state state = {};
      results.push_back(measure_reference_case(k_vision_case_name, cfg, [&]() {
        return run_image_case(session, state);
      }));
    }
  }

  if (configured_path_exists(audio_model_path) &&
      configured_path_exists(audio_mmproj_path) &&
      reference_case_enabled(k_audio_case_name)) {
    multimodal_reference_session session = load_multimodal_session(audio_model_path, audio_mmproj_path);
    std::printf("# embedding_reference_case_support_ultravox_1b: vision=%s audio=%s threads=%d\n",
                mtmd_support_vision(session.context.get()) ? "true" : "false",
                mtmd_support_audio(session.context.get()) ? "true" : "false",
                session.n_threads);
    if (mtmd_support_audio(session.context.get())) {
      reference_case_state state = {};
      results.push_back(measure_reference_case(k_audio_case_name, cfg, [&]() {
        return run_audio_case(session, state);
      }));
    }
  }

  if (results.empty()) {
    std::fprintf(stderr,
                 "warning: skipping reference benchmark because no configured model assets matched "
                 "the approved baseline matrix\n");
  }
}

}  // namespace emel::bench
