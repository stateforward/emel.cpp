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

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/speech/codec/mimi/any.hpp"
#include "emel/speech/generator/sm.hpp"
#include "emel/speech/predictor/moshi/any.hpp"
#include "emel/speech/predictor/moshi/executor/any.hpp"

namespace {

namespace mimi = emel::speech::codec::mimi;
namespace generator = emel::speech::generator;
namespace predictor = emel::speech::predictor::moshi;
namespace runtime = emel::speech::predictor::moshi::executor;

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

struct cache_binding {
  runtime::detail::temporal_kv_view temporal = {};
  runtime::detail::depformer_kv_view secondary = {};
};

bool bind_temporal_cache(void *binding_ptr, const emel::model::data &,
                         const emel::memory::view::snapshot &, int32_t,
                         runtime::detail::temporal_kv_view &view) noexcept {
  view = static_cast<cache_binding *>(binding_ptr)->temporal;
  return true;
}

bool bind_secondary_cache(void *binding_ptr, const emel::model::data &,
                          const emel::memory::view::snapshot &, int32_t,
                          runtime::detail::depformer_kv_view &view) noexcept {
  view = static_cast<cache_binding *>(binding_ptr)->secondary;
  return true;
}

struct personaplex_dependencies {
  using generator_mode = generator::action::mode::duplex;
  using voice_condition_event = predictor::event::prefill_voice;
  using prompt_begin_event = predictor::event::begin_personaplex_prompt;
  using prompt_condition_event = predictor::event::prefill_personaplex_prompt;
  using encode_event = mimi::event::encode_frame;
  using predict_event = predictor::event::step;
  using decode_event = mimi::event::decode_frame;

  emel::memory::streaming::sm &temporal_positions;
  emel::memory::streaming::sm &secondary_positions;
  mimi::sm &encoder;
  mimi::sm &decoder;
  runtime::sm &runtime;
  predictor::sm &predictor;
  mimi::event::initialize encoder_initialize;
  mimi::event::initialize decoder_initialize;
  runtime::event::initialize runtime_initialize;
  predictor::event::initialize predictor_initialize;
  predictor::event::load_voice conditioning_initialize;
  std::span<float> silence_pcm = {};
  std::span<int32_t> input_codes = {};
  std::span<int32_t> output_codes = {};
  int32_t frame_samples = 0;
  int32_t codebook_count = 0;
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
  std::vector<int32_t> output_codes(static_cast<size_t>(public_n_q), -1);

  emel::memory::streaming::sm temporal_positions{
      emel::memory::streaming::dependencies{
          .capacity = lm_model.model->moshi_lm.context}};
  emel::memory::streaming::sm secondary_positions{
      emel::memory::streaming::dependencies{
          .capacity = lm_model.model->moshi_lm.depformer_context}};
  cache_binding cache_views{
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
  runtime::sm prediction_runtime{runtime::bind_kv_caches(
      runtime::bind_temporal_kv_cache(&cache_views, bind_temporal_cache),
      runtime::bind_depformer_kv_cache(&cache_views, bind_secondary_cache),
      temporal_positions, secondary_positions)};
  predictor::sm token_predictor{
      emel::memory::hybrid::kv_binding{},
      runtime::bind_graph_executor(prediction_runtime)};
  mimi::sm encoder{};
  mimi::sm decoder{};

  runtime::event::initialize runtime_initialize{*lm_model.model};
  runtime_initialize.sampling_enabled = true;
  runtime_initialize.sampling_consume_forced_text = true;
  runtime_initialize.sampling_audio_temperature = config.audio_temperature;
  runtime_initialize.sampling_text_temperature = config.text_temperature;
  runtime_initialize.sampling_audio_top_k = config.audio_top_k;
  runtime_initialize.sampling_text_top_k = config.text_top_k;
  runtime_initialize.sampling_seed = config.sampling_seed;
  predictor::event::initialize predictor_initialize{*lm_model.model};
  predictor_initialize.max_blocks = config.max_blocks;
  predictor_initialize.block_tokens = config.block_tokens;

  personaplex_dependencies dependencies{
      .temporal_positions = temporal_positions,
      .secondary_positions = secondary_positions,
      .encoder = encoder,
      .decoder = decoder,
      .runtime = prediction_runtime,
      .predictor = token_predictor,
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
      .runtime_initialize = runtime_initialize,
      .predictor_initialize = predictor_initialize,
      .conditioning_initialize =
          predictor::event::load_voice{*voice_model.model},
      .silence_pcm = std::span<float>{pcm_frame},
      .input_codes = std::span<int32_t>{input_codes},
      .output_codes = std::span<int32_t>{output_codes},
      .frame_samples = frame_samples_i32,
      .codebook_count = public_n_q,
  };
  generator::sm<personaplex_dependencies> session{dependencies};
  emel::error::type session_err =
      generator::action::error_code(generator::error::none);
  if (!session.process_event(generator::event::initialize{session_err})) {
    std::fprintf(stderr, "PersonaPlex session initialize failed err=%d\n",
                 static_cast<int>(session_err));
    return 1;
  }

  while (
      session.is(stateforward::sml::state<generator::state_condition_voice>)) {
    bool complete = false;
    int32_t remaining = -1;
    session_err = generator::action::error_code(generator::error::none);
    (void)session.process_event(generator::event::condition{
        config.prompt_text_token, complete, remaining, session_err});
  }
  while (
      session.is(stateforward::sml::state<generator::state_condition_prompt>)) {
    bool complete = false;
    int32_t remaining = -1;
    session_err = generator::action::error_code(generator::error::none);
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
    session_err = generator::action::error_code(generator::error::none);
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
    (void)session.process_event(frame);
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
    session_err = generator::action::error_code(generator::error::none);
    generator::event::flush frame{
        std::span<float>{output_pcm.data() + frame_index * frame_samples,
                         frame_samples},
        std::span<int32_t>{input_codes},
        std::span<int32_t>{output_codes},
        text_token,
        sample_count,
        complete,
        session_err};
    (void)session.process_event(frame);
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
  if (!session.is(stateforward::sml::state<generator::state_flushing>)) {
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
