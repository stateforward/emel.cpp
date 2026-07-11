#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "emel/batch/planner/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/speech/codec/mimi/any.hpp"
#include "emel/speech/generator/any.hpp"
#include "emel/speech/predictor/moshi/any.hpp"
#include "emel/speech/predictor/moshi/executor/any.hpp"
#include "emel/speech/tokenizer/moshi/any.hpp"

namespace {

namespace mimi = emel::speech::codec::mimi;
namespace generator = emel::speech::generator;
namespace predictor = emel::speech::predictor::moshi;
namespace runtime = emel::speech::predictor::moshi::executor;
namespace tokenizer = emel::speech::tokenizer::moshi;

struct runner_config {
  const char *mimi_path = nullptr;
  const char *lm_path = nullptr;
  const char *voice_path = nullptr;
  const char *input_path = nullptr;
  const char *output_path = nullptr;
  size_t target_output_frames = 0;
  uint32_t sampling_seed = 0;
  float audio_temperature = 0.0f;
  float text_temperature = 0.0f;
  int32_t audio_top_k = 0;
  int32_t text_top_k = 0;
  int32_t max_blocks = 0;
  int32_t block_tokens = 0;
  int32_t prompt_text_token = 0;
};

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

struct loaded_model {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = {};
};

struct streaming_cache {
  std::vector<uint16_t> key_cache = {};
  std::vector<uint16_t> value_cache = {};
  std::vector<size_t> layer_offsets = {};
};

struct cache_views {
  runtime::detail::temporal_kv_view temporal = {};
  runtime::detail::depformer_kv_view secondary = {};
};

struct personaplex_dependencies {
  using generator_mode = generator::duplex;
  using voice_condition_event = predictor::event::prefill_voice;
  using prompt_begin_event = predictor::event::begin_personaplex_prompt;
  using prompt_condition_event = predictor::event::prefill_personaplex_prompt;
  using encode_event = mimi::event::encode_frame;
  using tokenizer_initialize_event = tokenizer::event::initialize;
  using tokenize_event = tokenizer::event::tokenize;
  using predict_event = predictor::event::predict;
  using graph_event = predictor::event::execute;
  using sample_event = predictor::event::sample;
  using detokenize_event = tokenizer::event::detokenize;
  using capture_tokenizer_state_event =
      predictor::event::capture_tokenizer_state;
  using restore_tokenizer_state_event = tokenizer::event::restore_cache;
  using decode_event = mimi::event::decode_frame;

  emel::memory::streaming::sm &temporal_positions;
  emel::memory::streaming::sm &secondary_positions;
  emel::batch::planner::sm &planner;
  mimi::sm &encoder;
  tokenizer::sm &tokenizer;
  mimi::sm &decoder;
  predictor::sm<runtime::sm> &predictor;
  predictor::sm<runtime::sm> &graph;
  predictor::sm<runtime::sm> &sampler;
  predictor::event::predict::workspace &prediction_workspace;
  mimi::event::initialize encoder_initialize;
  mimi::event::initialize decoder_initialize;
  predictor::event::initialize predictor_initialize;
  predictor::event::load_voice conditioning_initialize;
  std::span<float> silence_pcm = {};
  std::span<int32_t> input_codes = {};
  std::span<const int32_t> tokenize_input_codes = {};
  std::span<int32_t> model_codes = {};
  std::span<int32_t> predicted_codes = {};
  std::span<int32_t> output_codes = {};
  std::span<int32_t> tokenizer_cache_snapshot = {};
  int32_t frame_samples = 0;
  int32_t codebook_count = 0;
  emel::batch::planner::event::plan_mode frame_plan_mode =
      emel::batch::planner::event::plan_mode::simple;
  int32_t frame_plan_steps = 0;
  int32_t frame_plan_token_count = 0;
  bool frame_plan_output_all = false;
};

bool read_binary_file(const std::filesystem::path &path,
                      std::vector<uint8_t> &bytes_out) {
  std::ifstream file{path, std::ios::binary};
  if (!file.good()) {
    return false;
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  bytes_out.resize(static_cast<size_t>(size));
  file.read(reinterpret_cast<char *>(bytes_out.data()), size);
  return file.good();
}

bool materialize_tensor_names(emel::model::data &model,
                              const std::vector<uint8_t> &file) {
  model.name_bytes_used = 0;
  for (uint32_t index = 0; index < model.n_tensors; ++index) {
    auto &tensor = model.tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    if (source_offset + length > file.size() ||
        static_cast<size_t>(model.name_bytes_used) + length >
            model.name_storage.size()) {
      return false;
    }
    std::copy_n(file.data() + source_offset, length,
                model.name_storage.data() + model.name_bytes_used);
    tensor.name_offset = model.name_bytes_used;
    model.name_bytes_used += static_cast<uint32_t>(length);
  }
  return true;
}

bool load_model(const std::filesystem::path &path, loaded_model &loaded) {
  loaded = {};
  loaded.model = std::make_unique<emel::model::data>();
  if (!read_binary_file(path, loaded.file_bytes)) {
    std::fprintf(stderr, "failed to read model: %s\n", path.string().c_str());
    return false;
  }

  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements{};
  if (!loader.process_event(emel::gguf::loader::event::probe{
          std::span<const uint8_t>{loaded.file_bytes},
          requirements,
          emel::gguf::loader::event::probe_done_fn::from<&noop_probe_done>(),
          emel::gguf::loader::event::probe_error_fn::from<&noop_probe_error>(),
      }) ||
      requirements.tensor_count > loaded.model->tensors.size()) {
    std::fprintf(stderr, "failed to probe model: %s\n", path.string().c_str());
    return false;
  }

  const uint64_t arena_bytes =
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  loaded.kv_arena.resize(static_cast<size_t>(arena_bytes));
  loaded.kv_entries.resize(requirements.kv_count);
  loaded.model->n_tensors = requirements.tensor_count;

  if (!loader.process_event(emel::gguf::loader::event::bind_storage{
          std::span<uint8_t>{loaded.kv_arena},
          std::span<emel::gguf::loader::kv_entry>{loaded.kv_entries},
          std::span<emel::model::data::tensor_record>{
              loaded.model->tensors.data(), loaded.model->n_tensors},
          emel::gguf::loader::event::bind_done_fn::from<&noop_bind_done>(),
          emel::gguf::loader::event::bind_error_fn::from<&noop_bind_error>(),
      }) ||
      !loader.process_event(emel::gguf::loader::event::parse{
          std::span<const uint8_t>{loaded.file_bytes},
          emel::gguf::loader::event::parse_done_fn::from<&noop_parse_done>(),
          emel::gguf::loader::event::parse_error_fn::from<&noop_parse_error>(),
      })) {
    std::fprintf(stderr, "failed to parse model: %s\n", path.string().c_str());
    return false;
  }

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{loaded.kv_arena},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{loaded.kv_entries},
  };
  if (!emel::model::detail::load_hparams_from_gguf(binding, *loaded.model)) {
    std::fprintf(stderr, "failed to load hparams: %s\n", path.string().c_str());
    return false;
  }
  loaded.model->weights_data = loaded.file_bytes.data();
  loaded.model->weights_size = loaded.file_bytes.size();
  return materialize_tensor_names(*loaded.model, loaded.file_bytes);
}

uint16_t read_u16_le(const uint8_t *data) {
  uint16_t value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

uint32_t read_u32_le(const uint8_t *data) {
  uint32_t value = 0;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

bool load_wav_24khz_mono_s16(const std::filesystem::path &path,
                             std::vector<float> &pcm_out) {
  std::vector<uint8_t> bytes = {};
  if (!read_binary_file(path, bytes) || bytes.size() < 44 ||
      std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    return false;
  }

  uint16_t format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits = 0;
  size_t data_offset = 0;
  size_t data_size = 0;
  size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    const uint8_t *chunk = bytes.data() + offset;
    const uint32_t chunk_size = read_u32_le(chunk + 4);
    const size_t payload = offset + 8;
    if (payload + chunk_size > bytes.size()) {
      return false;
    }
    if (std::memcmp(chunk, "fmt ", 4) == 0 && chunk_size >= 16) {
      format = read_u16_le(bytes.data() + payload);
      channels = read_u16_le(bytes.data() + payload + 2);
      sample_rate = read_u32_le(bytes.data() + payload + 4);
      bits = read_u16_le(bytes.data() + payload + 14);
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      data_offset = payload;
      data_size = chunk_size;
    }
    offset = payload + chunk_size + (chunk_size & 1u);
  }
  if (format != 1 || channels != 1 || sample_rate != 24000 || bits != 16 ||
      data_offset == 0 || data_size == 0) {
    return false;
  }

  const size_t samples = data_size / sizeof(int16_t);
  pcm_out.resize(samples);
  for (size_t index = 0; index < samples; ++index) {
    int16_t sample = 0;
    std::memcpy(&sample, bytes.data() + data_offset + index * sizeof(sample),
                sizeof(sample));
    pcm_out[index] = static_cast<float>(sample) / 32768.0f;
  }
  return true;
}

void write_u16_le(std::ofstream &out, const uint16_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_u32_le(std::ofstream &out, const uint32_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

bool write_wav_24khz_mono_s16(const std::filesystem::path &path,
                              const std::span<const float> pcm) {
  std::ofstream out{path, std::ios::binary};
  if (!out.good()) {
    return false;
  }
  const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * 2u);
  out.write("RIFF", 4);
  write_u32_le(out, 36u + data_bytes);
  out.write("WAVEfmt ", 8);
  write_u32_le(out, 16u);
  write_u16_le(out, 1u);
  write_u16_le(out, 1u);
  write_u32_le(out, 24000u);
  write_u32_le(out, 24000u * 2u);
  write_u16_le(out, 2u);
  write_u16_le(out, 16u);
  out.write("data", 4);
  write_u32_le(out, data_bytes);
  for (const float sample : pcm) {
    const float clipped = std::max(-1.0f, std::min(1.0f, sample));
    const int16_t s16 = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
    out.write(reinterpret_cast<const char *>(&s16), sizeof(s16));
  }
  return out.good();
}

bool prepare_streaming_cache(streaming_cache &cache, const int32_t layer_count,
                             const int32_t position_capacity,
                             const int32_t kv_dim) {
  if (layer_count <= 0 || position_capacity <= 0 || kv_dim <= 0) {
    return false;
  }
  const size_t per_layer =
      static_cast<size_t>(position_capacity) * static_cast<size_t>(kv_dim);
  cache.layer_offsets.resize(static_cast<size_t>(layer_count));
  for (int32_t layer = 0; layer < layer_count; ++layer) {
    cache.layer_offsets[static_cast<size_t>(layer)] =
        static_cast<size_t>(layer) * per_layer;
  }
  cache.key_cache.assign(static_cast<size_t>(layer_count) * per_layer, 0u);
  cache.value_cache.assign(cache.key_cache.size(), 0u);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 15) {
    std::fprintf(stderr,
                 "usage: %s <mimi.gguf> <moshi-lm.gguf> <voice.gguf> "
                 "<input-24k.wav> <output.wav> <target-output-frames> "
                 "<seed> <audio-temperature> <text-temperature> "
                 "<audio-top-k> <text-top-k> <max-blocks> <block-tokens> "
                 "<prompt-text-token>\n",
                 argv[0]);
    return 2;
  }

  char *target_end = nullptr;
  const unsigned long parsed_target = std::strtoul(argv[6], &target_end, 10);
  if (target_end == argv[6] || *target_end != '\0' || parsed_target == 0) {
    std::fprintf(stderr, "target output frames must be a positive integer\n");
    return 2;
  }
  char *seed_end = nullptr;
  const unsigned long parsed_seed = std::strtoul(argv[7], &seed_end, 10);
  char *audio_temperature_end = nullptr;
  const float parsed_audio_temperature =
      std::strtof(argv[8], &audio_temperature_end);
  char *text_temperature_end = nullptr;
  const float parsed_text_temperature =
      std::strtof(argv[9], &text_temperature_end);
  char *audio_top_k_end = nullptr;
  const long parsed_audio_top_k = std::strtol(argv[10], &audio_top_k_end, 10);
  char *text_top_k_end = nullptr;
  const long parsed_text_top_k = std::strtol(argv[11], &text_top_k_end, 10);
  char *max_blocks_end = nullptr;
  const long parsed_max_blocks = std::strtol(argv[12], &max_blocks_end, 10);
  char *block_tokens_end = nullptr;
  const long parsed_block_tokens = std::strtol(argv[13], &block_tokens_end, 10);
  char *prompt_text_token_end = nullptr;
  const long parsed_prompt_text_token =
      std::strtol(argv[14], &prompt_text_token_end, 10);
  if (seed_end == argv[7] || *seed_end != '\0' || parsed_seed == 0 ||
      parsed_seed > std::numeric_limits<uint32_t>::max() ||
      audio_temperature_end == argv[8] || *audio_temperature_end != '\0' ||
      parsed_audio_temperature <= 0.0f || text_temperature_end == argv[9] ||
      *text_temperature_end != '\0' || parsed_text_temperature <= 0.0f ||
      audio_top_k_end == argv[10] || *audio_top_k_end != '\0' ||
      parsed_audio_top_k <= 0 ||
      parsed_audio_top_k > std::numeric_limits<int32_t>::max() ||
      text_top_k_end == argv[11] || *text_top_k_end != '\0' ||
      parsed_text_top_k <= 0 ||
      parsed_text_top_k > std::numeric_limits<int32_t>::max() ||
      max_blocks_end == argv[12] || *max_blocks_end != '\0' ||
      parsed_max_blocks <= 0 ||
      parsed_max_blocks > std::numeric_limits<int32_t>::max() ||
      block_tokens_end == argv[13] || *block_tokens_end != '\0' ||
      parsed_block_tokens <= 0 ||
      parsed_block_tokens > std::numeric_limits<int32_t>::max() ||
      prompt_text_token_end == argv[14] || *prompt_text_token_end != '\0' ||
      parsed_prompt_text_token < std::numeric_limits<int32_t>::min() ||
      parsed_prompt_text_token > std::numeric_limits<int32_t>::max()) {
    std::fprintf(stderr, "invalid sampling configuration\n");
    return 2;
  }
  const runner_config config{
      .mimi_path = argv[1],
      .lm_path = argv[2],
      .voice_path = argv[3],
      .input_path = argv[4],
      .output_path = argv[5],
      .target_output_frames = static_cast<size_t>(parsed_target),
      .sampling_seed = static_cast<uint32_t>(parsed_seed),
      .audio_temperature = parsed_audio_temperature,
      .text_temperature = parsed_text_temperature,
      .audio_top_k = static_cast<int32_t>(parsed_audio_top_k),
      .text_top_k = static_cast<int32_t>(parsed_text_top_k),
      .max_blocks = static_cast<int32_t>(parsed_max_blocks),
      .block_tokens = static_cast<int32_t>(parsed_block_tokens),
      .prompt_text_token = static_cast<int32_t>(parsed_prompt_text_token),
  };

  loaded_model mimi_model = {};
  loaded_model lm_model = {};
  loaded_model voice_model = {};
  if (!load_model(config.mimi_path, mimi_model) ||
      !load_model(config.lm_path, lm_model) ||
      !load_model(config.voice_path, voice_model)) {
    return 1;
  }

  std::vector<float> input_pcm = {};
  if (!load_wav_24khz_mono_s16(config.input_path, input_pcm)) {
    std::fprintf(stderr, "input must be 24 kHz mono s16 WAV: %s\n",
                 config.input_path);
    return 1;
  }

  const int32_t frame_samples_i32 = static_cast<int32_t>(
      std::lround(static_cast<double>(mimi_model.model->mimi.sample_rate) /
                  static_cast<double>(mimi_model.model->mimi.frame_rate)));
  const int32_t public_n_q = lm_model.model->moshi_lm.inference_dep_q;
  if (frame_samples_i32 <= 0 || public_n_q <= 0 ||
      public_n_q != mimi_model.model->mimi.n_q ||
      mimi_model.model->mimi.card != lm_model.model->moshi_lm.card) {
    std::fprintf(stderr, "invalid PersonaPlex/Mimi model contract\n");
    return 1;
  }
  const size_t frame_samples = static_cast<size_t>(frame_samples_i32);
  const size_t input_frames =
      (input_pcm.size() + frame_samples - 1u) / frame_samples;
  if (input_frames > config.target_output_frames) {
    std::fprintf(stderr,
                 "input requires %zu frames, exceeding target output %zu\n",
                 input_frames, config.target_output_frames);
    return 1;
  }

  std::vector<float> encoder_prepared(
      mimi::prepared_arena_floats(*mimi_model.model));
  std::vector<float> encoder_state(mimi::state_arena_floats(*mimi_model.model));
  std::vector<float> encoder_workspace(
      mimi::workspace_arena_floats(*mimi_model.model));
  std::vector<float> encoder_frame(mimi::frame_arena_floats(*mimi_model.model));
  std::vector<float> decoder_prepared(
      mimi::prepared_arena_floats(*mimi_model.model));
  std::vector<float> decoder_state(mimi::state_arena_floats(*mimi_model.model));
  std::vector<float> decoder_workspace(
      mimi::workspace_arena_floats(*mimi_model.model));
  std::vector<float> decoder_frame(mimi::frame_arena_floats(*mimi_model.model));
  streaming_cache temporal_cache = {};
  streaming_cache depformer_cache = {};
  if (!prepare_streaming_cache(
          temporal_cache, lm_model.model->moshi_lm.num_layers,
          lm_model.model->moshi_lm.context, lm_model.model->moshi_lm.dim) ||
      !prepare_streaming_cache(depformer_cache,
                               lm_model.model->moshi_lm.depformer_num_layers,
                               lm_model.model->moshi_lm.depformer_context,
                               lm_model.model->moshi_lm.depformer_dim)) {
    std::fprintf(stderr, "failed to allocate Moshi streaming caches\n");
    return 1;
  }

  std::vector<float> pcm_frame(frame_samples, 0.0f);
  std::vector<float> output_pcm(config.target_output_frames * frame_samples,
                                0.0f);
  std::vector<int32_t> input_codes(static_cast<size_t>(public_n_q), -1);
  const int32_t model_codebook_count = lm_model.model->moshi_lm.n_q + 1;
  const auto delay_begin = lm_model.model->moshi_lm.delays.begin();
  const auto delay_end = delay_begin + lm_model.model->moshi_lm.delay_count;
  const int32_t maximum_delay = *std::max_element(delay_begin, delay_end);
  const int32_t tokenizer_cache_rows = maximum_delay + 3;
  std::vector<int32_t> model_codes(static_cast<size_t>(model_codebook_count),
                                   -1);
  std::vector<int32_t> predicted_codes(
      static_cast<size_t>(lm_model.model->moshi_lm.dep_q), -1);
  std::vector<int32_t> output_codes(static_cast<size_t>(public_n_q), -1);
  std::vector<int32_t> tokenizer_storage(
      static_cast<size_t>(tokenizer_cache_rows * model_codebook_count), -2);
  std::vector<int32_t> tokenizer_cache_snapshot(
      static_cast<size_t>(tokenizer_cache_rows * model_codebook_count), -2);

  emel::memory::streaming::sm temporal_positions{
      emel::memory::streaming::dependencies{
          .capacity = lm_model.model->moshi_lm.context}};
  emel::memory::streaming::sm secondary_positions{
      emel::memory::streaming::dependencies{
          .capacity = lm_model.model->moshi_lm.depformer_context}};
  cache_views cache_view_set{
      .temporal =
          {
              .key_cache = std::span<uint16_t>{temporal_cache.key_cache},
              .value_cache = std::span<uint16_t>{temporal_cache.value_cache},
              .layer_cache_offsets =
                  std::span<const size_t>{temporal_cache.layer_offsets},
              .layer_count = lm_model.model->moshi_lm.num_layers,
              .position_capacity = lm_model.model->moshi_lm.context,
              .kv_dim = lm_model.model->moshi_lm.dim,
          },
      .secondary =
          {
              .key_cache = std::span<uint16_t>{depformer_cache.key_cache},
              .value_cache = std::span<uint16_t>{depformer_cache.value_cache},
              .layer_cache_offsets =
                  std::span<const size_t>{depformer_cache.layer_offsets},
              .layer_count = lm_model.model->moshi_lm.depformer_num_layers,
              .position_capacity = lm_model.model->moshi_lm.depformer_context,
              .kv_dim = lm_model.model->moshi_lm.depformer_dim,
          },
  };
  emel::kernel::sm prediction_kernel{};
  emel::logits::sampler::sm prediction_sampler{};
  runtime::sm prediction_runtime{runtime::dependencies{
      .kv =
          runtime::kv_views{
              .temporal = cache_view_set.temporal,
              .depformer = cache_view_set.secondary,
              .temporal_positions = &temporal_positions,
              .depformer_positions = &secondary_positions,
          },
      .kernel = prediction_kernel,
      .sampler = &prediction_sampler,
      .policy =
          runtime::policies{
              .rms_norm_epsilon = 1.0e-8f,
              .zero_seed_state = 123459876u,
              .sampling_modulus = 2147483647u,
              .token_zero = -1,
          },
      .capacity =
          runtime::capacities{
              .hidden_dim = static_cast<uint64_t>(
                  std::max(lm_model.model->moshi_lm.dim,
                           lm_model.model->moshi_lm.depformer_dim)),
              .temporal_context =
                  static_cast<uint64_t>(lm_model.model->moshi_lm.context),
              .depformer_context = static_cast<uint64_t>(
                  lm_model.model->moshi_lm.depformer_context),
              .sampling_card = static_cast<uint64_t>(
                  std::max(lm_model.model->moshi_lm.text_card,
                           lm_model.model->moshi_lm.card)),
              .sampling_top_k = static_cast<uint64_t>(
                  std::max(config.audio_top_k, config.text_top_k)),
          },
  }};
  emel::memory::hybrid::sm prediction_memory{};
  predictor::sm token_predictor{predictor::dependencies<runtime::sm>{
      .memory = prediction_memory,
      .graph = prediction_runtime,
      .policy =
          predictor::policies{
              .token_zero = -1,
              .token_ungenerated = -2,
              .prediction_step_size = 1,
              .prediction_output_count = 1,
          },
  }};
  std::vector<float> prediction_temporal_state(
      static_cast<size_t>(lm_model.model->moshi_lm.dim));
  predictor::event::predict::workspace prediction_workspace{};
  prediction_workspace.temporal_state =
      std::span<float>{prediction_temporal_state};
  tokenizer::sm token_delay{tokenizer::dependencies{
      .delays = std::span<const int32_t>{delay_begin, delay_end},
      .cache = std::span<int32_t>{tokenizer_storage},
      .codebooks = model_codebook_count,
      .generated_audio_codebooks = lm_model.model->moshi_lm.dep_q,
      .delayed_audio_codebooks = lm_model.model->moshi_lm.inference_dep_q,
      .cache_rows = tokenizer_cache_rows,
      .maximum_delay = maximum_delay,
      .initial_delay_frames = 0,
      .text_initial_token = lm_model.model->moshi_lm.text_card,
      .audio_initial_token = lm_model.model->moshi_lm.card,
      .token_zero = -1,
      .token_ungenerated = -2,
  }};
  mimi::sm encoder{};
  mimi::sm decoder{};
  emel::batch::planner::sm frame_planner{};

  predictor::event::initialize predictor_initialize{*lm_model.model};
  predictor_initialize.max_sequences = 1;
  predictor_initialize.max_blocks = config.max_blocks;
  predictor_initialize.block_tokens = config.block_tokens;
  predictor_initialize.sequence_id = 0;
  predictor_initialize.codebook_capacity = model_codebook_count;
  predictor_initialize.delay_cache_row_capacity = tokenizer_cache_rows;
  predictor_initialize.sampling_enabled = true;
  predictor_initialize.sampling_consume_forced_text = true;
  predictor_initialize.sampling_audio_temperature = config.audio_temperature;
  predictor_initialize.sampling_text_temperature = config.text_temperature;
  predictor_initialize.sampling_audio_top_k = config.audio_top_k;
  predictor_initialize.sampling_text_top_k = config.text_top_k;
  predictor_initialize.sampling_seed = config.sampling_seed;

  personaplex_dependencies dependencies{
      .temporal_positions = temporal_positions,
      .secondary_positions = secondary_positions,
      .planner = frame_planner,
      .encoder = encoder,
      .tokenizer = token_delay,
      .decoder = decoder,
      .predictor = token_predictor,
      .graph = token_predictor,
      .sampler = token_predictor,
      .prediction_workspace = prediction_workspace,
      .encoder_initialize =
          mimi::event::initialize{*mimi_model.model,
                                  std::span<float>{encoder_prepared},
                                  std::span<float>{encoder_state},
                                  std::span<float>{encoder_workspace},
                                  std::span<float>{encoder_frame}},
      .decoder_initialize =
          mimi::event::initialize{*mimi_model.model,
                                  std::span<float>{decoder_prepared},
                                  std::span<float>{decoder_state},
                                  std::span<float>{decoder_workspace},
                                  std::span<float>{decoder_frame}},
      .predictor_initialize = predictor_initialize,
      .conditioning_initialize =
          predictor::event::load_voice{*voice_model.model},
      .silence_pcm = std::span<float>{pcm_frame},
      .input_codes = std::span<int32_t>{input_codes},
      .tokenize_input_codes = std::span<const int32_t>{input_codes},
      .model_codes = std::span<int32_t>{model_codes},
      .predicted_codes = std::span<int32_t>{predicted_codes},
      .output_codes = std::span<int32_t>{output_codes},
      .tokenizer_cache_snapshot = std::span<int32_t>{tokenizer_cache_snapshot},
      .frame_samples = frame_samples_i32,
      .codebook_count = public_n_q,
      .frame_plan_mode = emel::batch::planner::event::plan_mode::simple,
      .frame_plan_steps = 1,
      .frame_plan_token_count = 1,
      .frame_plan_output_all = true,
  };
  generator::sm<personaplex_dependencies> session{dependencies};
  emel::error::type session_err = emel::error::cast(generator::error::none);
  if (!session.process_event(generator::event::initialize{session_err})) {
    std::fprintf(stderr, "PersonaPlex session initialize failed err=%d\n",
                 static_cast<int>(session_err));
    return 1;
  }

  while (
      session.is(stateforward::sml::state<generator::state_condition_voice>)) {
    bool complete = false;
    int32_t remaining = -1;
    session_err = emel::error::cast(generator::error::none);
    (void)session.process_event(generator::event::condition{
        config.prompt_text_token, complete, remaining, session_err});
  }
  while (
      session.is(stateforward::sml::state<generator::state_condition_prompt>)) {
    bool complete = false;
    int32_t remaining = -1;
    session_err = emel::error::cast(generator::error::none);
    (void)session.process_event(generator::event::condition{
        config.prompt_text_token, complete, remaining, session_err});
  }
  if (!session.is(stateforward::sml::state<generator::state_ready>)) {
    std::fprintf(stderr, "PersonaPlex prefill failed err=%d\n",
                 static_cast<int>(session_err));
    return 1;
  }
  for (size_t frame_index = 0; frame_index < input_frames; ++frame_index) {
    std::fill(pcm_frame.begin(), pcm_frame.end(), 0.0f);
    const size_t begin = frame_index * frame_samples;
    const size_t copy_count = std::min(frame_samples, input_pcm.size() - begin);
    std::copy_n(input_pcm.data() + begin, copy_count, pcm_frame.data());
    int32_t text_token = -1;
    bool produced = false;
    int32_t sample_count = 0;
    session_err = emel::error::cast(generator::error::none);
    generator::event::stream_frame frame{
        std::span<const float>{pcm_frame},
        std::span<float>{output_pcm.data() + frame_index * frame_samples,
                         frame_samples},
        std::span<int32_t>{input_codes},
        std::span<int32_t>{output_codes},
        text_token,
        sample_count,
        produced,
        session_err};
    if (!session.process_event(frame)) {
      std::fprintf(stderr, "PersonaPlex stream frame=%zu failed err=%d\n",
                   frame_index, static_cast<int>(session_err));
      return 1;
    }
    std::fprintf(stderr, "EMEL_INPUT frame=%zu codes=", frame_index);
    for (int32_t index = 0; index < public_n_q; ++index) {
      std::fprintf(stderr, "%s%d", index == 0 ? "" : ",",
                   input_codes[static_cast<size_t>(index)]);
    }
    std::fprintf(stderr, "\nEMEL_OUTPUT frame=%zu text=%d codes=", frame_index,
                 text_token);
    for (int32_t index = 0; index < public_n_q; ++index) {
      std::fprintf(stderr, "%s%d", index == 0 ? "" : ",",
                   output_codes[static_cast<size_t>(index)]);
    }
    std::fprintf(stderr, "\n");
  }

  std::fill(pcm_frame.begin(), pcm_frame.end(), 0.0f);
  const size_t flush_steps = config.target_output_frames - input_frames;
  for (size_t index = 0; index < flush_steps; ++index) {
    const size_t frame_index = input_frames + index;
    int32_t text_token = -1;
    int32_t sample_count = 0;
    bool complete = false;
    session_err = emel::error::cast(generator::error::none);
    generator::event::flush frame{
        std::span<float>{output_pcm.data() + frame_index * frame_samples,
                         frame_samples},
        std::span<int32_t>{input_codes},
        std::span<int32_t>{output_codes},
        text_token,
        sample_count,
        complete,
        session_err};
    if (!session.process_event(frame)) {
      std::fprintf(stderr, "PersonaPlex flush frame=%zu failed err=%d\n",
                   frame_index, static_cast<int>(session_err));
      return 1;
    }
    std::fprintf(stderr, "EMEL_INPUT frame=%zu codes=", frame_index);
    for (int32_t codebook = 0; codebook < public_n_q; ++codebook) {
      std::fprintf(stderr, "%s%d", codebook == 0 ? "" : ",",
                   input_codes[static_cast<size_t>(codebook)]);
    }
    std::fprintf(stderr, "\nEMEL_OUTPUT frame=%zu text=%d codes=", frame_index,
                 text_token);
    for (int32_t codebook = 0; codebook < public_n_q; ++codebook) {
      std::fprintf(stderr, "%s%d", codebook == 0 ? "" : ",",
                   output_codes[static_cast<size_t>(codebook)]);
    }
    std::fprintf(stderr, "\n");
  }
  const bool session_finished =
      flush_steps == 0
          ? session.is(stateforward::sml::state<generator::state_ready>)
          : session.is(stateforward::sml::state<generator::state_flushing>);
  if (!session_finished) {
    std::fprintf(stderr, "PersonaPlex session failed err=%d\n",
                 static_cast<int>(session_err));
    return 1;
  }

  if (output_pcm.empty()) {
    std::fprintf(stderr, "no generated audio frames were produced\n");
    return 1;
  }

  double energy = 0.0;
  float peak = 0.0f;
  for (const float sample : output_pcm) {
    energy += static_cast<double>(sample) * static_cast<double>(sample);
    peak = std::max(peak, std::abs(sample));
  }
  const double rms = std::sqrt(energy / static_cast<double>(output_pcm.size()));
  if (!write_wav_24khz_mono_s16(config.output_path,
                                std::span<const float>{output_pcm})) {
    std::fprintf(stderr, "failed to write output wav: %s\n",
                 config.output_path);
    return 1;
  }

  std::printf("input_frames=%zu output_frames=%zu output_seconds=%.3f "
              "peak=%.6f rms=%.6f seed=%u output=%s\n",
              input_frames, output_pcm.size() / frame_samples,
              static_cast<double>(output_pcm.size()) / 24000.0, peak, rms,
              config.sampling_seed, config.output_path);
  return 0;
}
