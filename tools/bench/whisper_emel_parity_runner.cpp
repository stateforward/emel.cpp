#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
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
#include <string_view>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/whisper/any.hpp"
#include "emel/speech/decoder/whisper/any.hpp"
#include "emel/speech/decoder/whisper/sm.hpp"
#include "emel/speech/encoder/whisper/any.hpp"
#include "emel/speech/encoder/whisper/sm.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"

namespace {

using steady_clock = std::chrono::steady_clock;

uint64_t elapsed_ns(const steady_clock::time_point start,
                    const steady_clock::time_point end) noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count());
}

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

struct options {
  std::filesystem::path model = {};
  std::filesystem::path audio = {};
  std::filesystem::path output_dir = {};
  std::filesystem::path tokenizer = {};
};

std::vector<uint8_t> read_binary_file(const std::filesystem::path &path);

class mapped_binary_file {
public:
  mapped_binary_file() = default;
  mapped_binary_file(const mapped_binary_file &) = delete;
  mapped_binary_file &operator=(const mapped_binary_file &) = delete;

  ~mapped_binary_file() { close(); }

  bool open(const std::filesystem::path &path) {
    close();
#if defined(_WIN32)
    fallback_bytes_ = read_binary_file(path);
    return !fallback_bytes_.empty();
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      return false;
    }
    struct stat st = {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
      ::close(fd);
      return false;
    }
    size_ = static_cast<size_t>(st.st_size);
    void *mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
      data_ = nullptr;
      size_ = 0u;
      return false;
    }
    data_ = static_cast<const uint8_t *>(mapped);
    return true;
#endif
  }

  void close() noexcept {
#if defined(_WIN32)
    fallback_bytes_.clear();
#else
    if (data_ != nullptr) {
      ::munmap(const_cast<uint8_t *>(data_), size_);
      data_ = nullptr;
      size_ = 0u;
    }
#endif
  }

  std::span<const uint8_t> bytes() const noexcept {
#if defined(_WIN32)
    return std::span<const uint8_t>{fallback_bytes_};
#else
    return std::span<const uint8_t>{data_, size_};
#endif
  }

private:
#if defined(_WIN32)
  std::vector<uint8_t> fallback_bytes_ = {};
#else
  const uint8_t *data_ = nullptr;
  size_t size_ = 0u;
#endif
};

bool parse_options(const int argc, char **argv, options &opts) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg{argv[index]};
    if (arg == "--model" && index + 1 < argc) {
      opts.model = argv[++index];
    } else if (arg == "--audio" && index + 1 < argc) {
      opts.audio = argv[++index];
    } else if (arg == "--output-dir" && index + 1 < argc) {
      opts.output_dir = argv[++index];
    } else if (arg == "--tokenizer" && index + 1 < argc) {
      opts.tokenizer = argv[++index];
    } else {
      std::fprintf(stderr, "error: unknown or incomplete argument: %s\n",
                   argv[index]);
      return false;
    }
  }
  return !opts.model.empty() && !opts.audio.empty() &&
         !opts.output_dir.empty() && !opts.tokenizer.empty();
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return {};
  }
  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  if (size <= 0) {
    return {};
  }
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!stream) {
    return {};
  }
  return bytes;
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return {};
  }
  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  if (size <= 0) {
    return {};
  }
  stream.seekg(0, std::ios::beg);
  std::string text(static_cast<size_t>(size), '\0');
  stream.read(text.data(), size);
  if (!stream) {
    return {};
  }
  return text;
}

bool load_wav_16khz_mono(const std::filesystem::path &path,
                         std::vector<float> &pcm_out) {
  const auto bytes = read_binary_file(path);
  if (bytes.size() < 44u || std::memcmp(bytes.data(), "RIFF", 4u) != 0 ||
      std::memcmp(bytes.data() + 8u, "WAVE", 4u) != 0) {
    return false;
  }
  uint16_t channels = 0u;
  uint32_t sample_rate = 0u;
  uint16_t bits_per_sample = 0u;
  uint16_t audio_format = 0u;
  uint32_t data_offset = 0u;
  uint32_t data_size = 0u;
  size_t offset = 12u;
  while (offset + 8u <= bytes.size()) {
    const char *id = reinterpret_cast<const char *>(bytes.data() + offset);
    uint32_t chunk_size = 0u;
    std::memcpy(&chunk_size, bytes.data() + offset + 4u, sizeof(chunk_size));
    const size_t payload = offset + 8u;
    if (payload + chunk_size > bytes.size()) {
      return false;
    }
    if (std::memcmp(id, "fmt ", 4u) == 0 && chunk_size >= 16u) {
      std::memcpy(&audio_format, bytes.data() + payload, sizeof(audio_format));
      std::memcpy(&channels, bytes.data() + payload + 2u, sizeof(channels));
      std::memcpy(&sample_rate, bytes.data() + payload + 4u,
                  sizeof(sample_rate));
      std::memcpy(&bits_per_sample, bytes.data() + payload + 14u,
                  sizeof(bits_per_sample));
    } else if (std::memcmp(id, "data", 4u) == 0) {
      data_offset = static_cast<uint32_t>(payload);
      data_size = chunk_size;
    }
    offset = payload + chunk_size + (chunk_size & 1u);
  }
  if (audio_format != 1u || channels != 1u || sample_rate != 16000u ||
      bits_per_sample != 16u || data_offset == 0u || data_size == 0u) {
    return false;
  }
  const size_t samples = data_size / 2u;
  pcm_out.resize(samples);
  for (size_t index = 0; index < samples; ++index) {
    int16_t value = 0;
    std::memcpy(&value, bytes.data() + data_offset + index * 2u, sizeof(value));
    pcm_out[index] = static_cast<float>(value) / 32768.0f;
  }
  return true;
}

void materialize_tensor_names_from_file(
    emel::model::data &model, const std::span<const uint8_t> file_bytes) {
  model.name_bytes_used = 0u;
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    auto &tensor = model.tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    std::memcpy(model.name_storage.data() + model.name_bytes_used,
                file_bytes.data() + source_offset, length);
    tensor.name_offset = model.name_bytes_used;
    model.name_bytes_used += static_cast<uint32_t>(length);
  }
}

bool load_whisper_fixture_binding(
    const std::span<const uint8_t> file_bytes, std::vector<uint8_t> &kv_arena,
    std::vector<emel::gguf::loader::kv_entry> &kv_entries,
    emel::model::data &model_out) {
  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements = {};
  const auto on_probe_done =
      emel::gguf::loader::event::probe_done_fn::from<&noop_probe_done>();
  const auto on_probe_error =
      emel::gguf::loader::event::probe_error_fn::from<&noop_probe_error>();
  const auto on_bind_done =
      emel::gguf::loader::event::bind_done_fn::from<&noop_bind_done>();
  const auto on_bind_error =
      emel::gguf::loader::event::bind_error_fn::from<&noop_bind_error>();
  const auto on_parse_done =
      emel::gguf::loader::event::parse_done_fn::from<&noop_parse_done>();
  const auto on_parse_error =
      emel::gguf::loader::event::parse_error_fn::from<&noop_parse_error>();

  const emel::gguf::loader::event::probe probe{file_bytes, requirements,
                                               on_probe_done, on_probe_error};
  if (!loader.process_event(probe) || requirements.tensor_count == 0u ||
      requirements.tensor_count > model_out.tensors.size()) {
    return false;
  }

  kv_arena.resize(static_cast<size_t>(
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  kv_entries.resize(requirements.kv_count);
  model_out.n_tensors = requirements.tensor_count;

  const emel::gguf::loader::event::bind_storage bind{
      std::span<uint8_t>{kv_arena},
      std::span<emel::gguf::loader::kv_entry>{kv_entries},
      std::span<emel::model::data::tensor_record>{model_out.tensors.data(),
                                                  model_out.n_tensors},
      on_bind_done, on_bind_error};
  if (!loader.process_event(bind)) {
    return false;
  }

  const emel::gguf::loader::event::parse parse{file_bytes, on_parse_done,
                                               on_parse_error};
  return loader.process_event(parse);
}

uint64_t checksum_text(const std::string &text) {
  uint64_t hash = 1469598103934665603ull;
  for (const char c : text) {
    hash ^= static_cast<uint8_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

} // namespace

int main(int argc, char **argv) {
  const auto total_start = steady_clock::now();
  options opts{};
  if (!parse_options(argc, argv, opts)) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL --audio WAV --tokenizer JSON "
                 "--output-dir DIR\n",
                 argv[0]);
    return 2;
  }

  const auto model_load_start = steady_clock::now();
  mapped_binary_file mapped_model;
  if (!mapped_model.open(opts.model)) {
    std::fprintf(stderr, "error: failed to map model file\n");
    return 2;
  }
  std::vector<uint8_t> source_owned_gguf = {};
  std::span<const uint8_t> model_image = mapped_model.bytes();
  if (emel::model::whisper::is_legacy_lmgg_whisper(model_image)) {
    if (!emel::model::whisper::normalize_legacy_lmgg_to_gguf(
            model_image, source_owned_gguf)) {
      std::fprintf(stderr,
                   "error: failed to normalize legacy Whisper artifact\n");
      return 2;
    }
    model_image = std::span<const uint8_t>{source_owned_gguf};
  }
  const uint64_t model_load_ns =
      elapsed_ns(model_load_start, steady_clock::now());

  const std::string tokenizer_json = read_text_file(opts.tokenizer);
  if (tokenizer_json.empty()) {
    std::fprintf(stderr, "error: failed to load tokenizer JSON\n");
    return 2;
  }

  const auto audio_load_start = steady_clock::now();
  std::vector<float> pcm = {};
  const bool audio_loaded = load_wav_16khz_mono(opts.audio, pcm);
  const uint64_t audio_load_ns =
      elapsed_ns(audio_load_start, steady_clock::now());
  if (model_image.empty() || !audio_loaded) {
    std::fprintf(stderr, "error: failed to load model or 16 kHz mono WAV\n");
    return 2;
  }

  const auto binding_start = steady_clock::now();
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  auto model = std::make_unique<emel::model::data>();
  if (!load_whisper_fixture_binding(model_image, kv_arena, kv_entries,
                                    *model)) {
    std::fprintf(stderr, "error: failed to parse Whisper GGUF\n");
    return 2;
  }
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
  };
  if (!emel::model::detail::load_hparams_from_gguf(binding, *model)) {
    std::fprintf(stderr, "error: failed to load Whisper hparams\n");
    return 2;
  }
  model->weights_data = model_image.data();
  model->weights_size = model_image.size();
  model->weights_mapped = true;
  materialize_tensor_names_from_file(*model, model_image);
  const uint64_t binding_ns = elapsed_ns(binding_start, steady_clock::now());

  const auto initialize_start = steady_clock::now();
  emel::model::whisper::execution_contract model_contract = {};
  const auto &policy =
      emel::speech::tokenizer::whisper::tiny_asr_decode_policy();
  if (emel::model::whisper::build_execution_contract(*model,
                                                     model_contract) !=
          emel::error::cast(emel::model::loader::error::none) ||
      !emel::speech::tokenizer::whisper::validate_tiny_control_tokens(
          tokenizer_json) ||
      !emel::speech::tokenizer::whisper::is_tiny_asr_decode_policy_supported(
          policy)) {
    std::fprintf(stderr, "error: failed to initialize Whisper runtime\n");
    return 2;
  }
  const auto encoder_contract =
      emel::speech::encoder::whisper::bind_execution_contract(*model);
  const auto decoder_contract =
      emel::speech::decoder::whisper::bind_execution_contract(*model);
  const uint64_t contract_ns =
      elapsed_ns(initialize_start, steady_clock::now());

  const auto recognize_start = steady_clock::now();
  namespace whisper_encoder = emel::speech::encoder::whisper;
  namespace whisper_decoder = emel::speech::decoder::whisper;
  namespace whisper_tokenizer = emel::speech::tokenizer::whisper;
  std::vector<float> encoder_workspace(static_cast<size_t>(
      whisper_encoder::required_workspace_floats(pcm.size())));
  std::vector<float> encoder_state(static_cast<size_t>(
      whisper_encoder::required_encoder_output_floats(pcm.size())));
  std::vector<float> logits(static_cast<size_t>(whisper_decoder::vocab_size()));
  std::array<int32_t, 32> generated_tokens = {};
  std::vector<char> transcript(64u);
  int32_t transcript_size = 0;
  int32_t token = 0;
  float confidence = 0.0f;
  int32_t frames = 0;
  int32_t width = 0;
  uint64_t encoder_digest = 0u;
  uint64_t decoder_digest = 0u;
  emel::speech::encoder::whisper::sm encoder{};
  emel::speech::encoder::whisper::event::encode encode_ev{
      encoder_contract,
      std::span<const float>{pcm},
      16000,
      1,
      std::span<float>{encoder_workspace},
      std::span<float>{encoder_state},
      frames,
      width,
      encoder_digest};
  const auto encode_start = steady_clock::now();
  if (!encoder.process_event(encode_ev)) {
    std::fprintf(stderr, "error: Whisper encoder event failed\n");
    return 2;
  }
  const uint64_t encode_ns = elapsed_ns(encode_start, steady_clock::now());

  int32_t generated_token_count = 0;
  std::vector<float> decoder_workspace(static_cast<size_t>(
      whisper_decoder::required_workspace_floats(
          static_cast<uint64_t>(frames))));
  emel::speech::decoder::whisper::sm decoder{};
  emel::speech::decoder::whisper::event::decode decode_ev{
      decoder_contract,
      std::span<const float>{encoder_state},
      frames,
      policy,
      std::span<int32_t>{generated_tokens.data(), 4u},
      generated_token_count,
      std::span<float>{decoder_workspace},
      std::span<float>{logits},
      token,
      confidence,
      decoder_digest};
  const auto decode_start = steady_clock::now();
  if (!decoder.process_event(decode_ev)) {
    std::fprintf(stderr, "error: Whisper decoder event failed\n");
    return 2;
  }
  transcript_size = static_cast<int32_t>(
      whisper_tokenizer::decode_token_ids(
          tokenizer_json,
          std::span<const int32_t>{
              generated_tokens.data(),
              static_cast<size_t>(generated_token_count),
          },
          transcript.data(), static_cast<uint64_t>(transcript.size())));
  const uint64_t decode_ns = elapsed_ns(decode_start, steady_clock::now());
  const uint64_t recognize_ns =
      elapsed_ns(recognize_start, steady_clock::now());

  const auto publish_start = steady_clock::now();
  std::filesystem::create_directories(opts.output_dir);
  const std::string transcript_text{transcript.data(),
                                    transcript.data() +
                                        static_cast<size_t>(transcript_size)};
  const std::filesystem::path transcript_path =
      opts.output_dir / "transcript.txt";
  {
    std::ofstream output(transcript_path, std::ios::binary);
    output << transcript_text;
  }
  const uint64_t publish_ns = elapsed_ns(publish_start, steady_clock::now());
  const uint64_t total_ns = elapsed_ns(total_start, steady_clock::now());

  const uint64_t checksum = checksum_text(transcript_text);
  std::printf(
      "{\"schema\":\"whisper_compare/"
      "v1\",\"record_type\":\"result\",\"status\":\"ok\","
      "\"case_name\":\"emel/emel.speech.whisper.encoder_decoder\","
      "\"compare_group\":\"whisper/tiny/q8_0/phase99_440hz_16khz_mono\","
      "\"lane\":\"emel\",\"backend_id\":\"emel.speech.whisper.encoder_decoder\","
      "\"runtime_surface\":\"speech/encoder/whisper+speech/decoder/whisper+"
      "speech/tokenizer/whisper\","
      "\"decode_policy_language\":\"%s\","
      "\"decode_policy_task\":\"%s\","
      "\"decode_policy_timestamp_mode\":\"%s\","
      "\"decode_policy_suppress_translate\":%s,"
      "\"decode_policy_prompt_token_count\":%zu,"
      "\"backend_language\":\"cpp\",\"comparison_mode\":\"parity\","
      "\"model_id\":\"oxide_lab_whisper_tiny_q8_0\","
      "\"audio_fixture_id\":\"phase99_440hz_16khz_mono\","
      "\"transcript\":\"%s\",\"token_count\":%" PRIu64 ",\"selected_token\":%d,"
      "\"timestamp_metadata\":\"unsupported\",\"output_bytes\":%zu,"
      "\"output_checksum\":%" PRIu64 ",\"output_path\":\"%s\","
      "\"encoder_frames\":%d,\"encoder_width\":%d,"
      "\"encoder_digest\":%" PRIu64 ",\"decoder_digest\":%" PRIu64 ","
      "\"benchmark_mode\":\"single_thread_cpu\",\"thread_count\":1,"
      "\"cpu_only\":true,\"wall_time_ns\":%" PRIu64 ","
      "\"model_load_ns\":%" PRIu64 ",\"audio_load_ns\":%" PRIu64 ","
      "\"binding_ns\":%" PRIu64 ",\"contract_ns\":%" PRIu64 ","
      "\"encode_ns\":%" PRIu64 ",\"decode_ns\":%" PRIu64 ","
      "\"publish_ns\":%" PRIu64 "}\n",
      emel::speech::tokenizer::whisper::language_role_name(policy.language)
          .data(),
      emel::speech::tokenizer::whisper::task_role_name(policy.task).data(),
      emel::speech::tokenizer::whisper::timestamp_mode_name(policy.timestamps)
          .data(),
      policy.suppress_translate ? "true" : "false",
      policy.prompt_tokens.size(), transcript_text.c_str(),
      static_cast<uint64_t>(generated_token_count),
      token, transcript_text.size(), checksum, transcript_path.string().c_str(),
      frames, width, encoder_digest, decoder_digest, total_ns, model_load_ns,
      audio_load_ns, binding_ns, contract_ns, encode_ns, decode_ns, publish_ns);
  return 0;
}
