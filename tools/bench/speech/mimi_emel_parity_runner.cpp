// EMEL-lane Mimi codec parity runner.
//
// Loads an emel-enriched mimi-component GGUF (produced by
// tools/bench/moshi_gguf_convert.py), streams a 24 kHz mono s16 WAV through
// emel::speech::codec::mimi::sm one 80 ms frame at a time, and emits one
// `frame=<n> codes=<c0>,<c1>,...` line per frame on stdout. With
// --decode-out the same code stream is decoded back through the codec and
// the reconstructed PCM is written as raw little-endian f32.
//
// This runner is EMEL-owned end to end; the reference lane
// (moshi_reference_driver) is a separate executable per the reference
// policy. Comparison happens in tools/bench/mimi_compare.py.
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/speech/codec/mimi/any.hpp"

namespace {

namespace mimi = emel::speech::codec::mimi;

// initialize_done payload capture (emel::callback binds free functions)
int32_t g_frame_samples = 0;
int32_t g_n_q = 0;
void on_codec_initialized(const mimi::events::initialize_done &done) {
  g_frame_samples = done.frame_samples;
  g_n_q = done.n_q;
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
  std::filesystem::path decode_out = {};
};

bool parse_options(const int argc, char **argv, options &opts) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    const bool has_value = index + 1 < argc;
    if (arg == "--model" && has_value) {
      opts.model = argv[++index];
    } else if (arg == "--audio" && has_value) {
      opts.audio = argv[++index];
    } else if (arg == "--decode-out" && has_value) {
      opts.decode_out = argv[++index];
    } else {
      std::fprintf(stderr, "unknown or incomplete argument: %.*s\n",
                   static_cast<int>(arg.size()), arg.data());
      return false;
    }
  }
  return !opts.model.empty() && !opts.audio.empty();
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream.good()) {
    return {};
  }
  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(bytes.data()), size);
  return stream.good() ? bytes : std::vector<uint8_t>{};
}

bool load_wav_24khz_mono(const std::filesystem::path &path,
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
  if (audio_format != 1u || channels != 1u || sample_rate != 24000u ||
      bits_per_sample != 16u || data_offset == 0u || data_size == 0u) {
    return false;
  }
  const size_t samples = data_size / 2u;
  pcm_out.resize(samples);
  for (size_t index = 0; index < samples; ++index) {
    int16_t sample = 0;
    std::memcpy(&sample, bytes.data() + data_offset + index * 2u,
                sizeof(sample));
    pcm_out[index] = static_cast<float>(sample) / 32768.0f;
  }
  return true;
}

struct loaded_model {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = {};
};

bool load_enriched_mimi(const std::filesystem::path &path,
                        loaded_model &loaded) {
  loaded.model = std::make_unique<emel::model::data>();
  loaded.file_bytes = read_binary_file(path);
  if (loaded.file_bytes.empty()) {
    return false;
  }

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

  const emel::gguf::loader::event::probe probe{
      std::span<const uint8_t>{loaded.file_bytes}, requirements, on_probe_done,
      on_probe_error};
  if (!loader.process_event(probe) || requirements.tensor_count == 0u ||
      requirements.tensor_count > loaded.model->tensors.size()) {
    return false;
  }
  loaded.kv_arena.resize(static_cast<size_t>(
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  loaded.kv_entries.resize(requirements.kv_count);
  loaded.model->n_tensors = requirements.tensor_count;
  const emel::gguf::loader::event::bind_storage bind{
      std::span<uint8_t>{loaded.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{loaded.kv_entries},
      std::span<emel::model::data::tensor_record>{loaded.model->tensors.data(),
                                                  loaded.model->n_tensors},
      on_bind_done, on_bind_error};
  if (!loader.process_event(bind)) {
    return false;
  }
  const emel::gguf::loader::event::parse parse{
      std::span<const uint8_t>{loaded.file_bytes}, on_parse_done,
      on_parse_error};
  if (!loader.process_event(parse)) {
    return false;
  }

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{loaded.kv_arena},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{loaded.kv_entries},
  };
  if (!emel::model::detail::load_hparams_from_gguf(binding, *loaded.model)) {
    std::fprintf(stderr,
                 "model is not an emel-enriched moshi GGUF; convert it with "
                 "tools/bench/moshi_gguf_convert.py first\n");
    return false;
  }

  loaded.model->name_bytes_used = 0u;
  for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
    auto &tensor = loaded.model->tensors[index];
    if (loaded.model->name_bytes_used + tensor.name_length >
        loaded.model->name_storage.size()) {
      return false;
    }
    std::memcpy(
        loaded.model->name_storage.data() + loaded.model->name_bytes_used,
        loaded.file_bytes.data() + tensor.name_offset, tensor.name_length);
    tensor.name_offset = loaded.model->name_bytes_used;
    loaded.model->name_bytes_used += tensor.name_length;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  options opts{};
  if (!parse_options(argc, argv, opts)) {
    std::fprintf(stderr,
                 "usage: mimi_emel_parity_runner --model <enriched mimi "
                 "gguf> --audio <24khz mono s16 wav> [--decode-out "
                 "<raw f32 path>]\n");
    return 2;
  }

  loaded_model loaded{};
  if (!load_enriched_mimi(opts.model, loaded)) {
    std::fprintf(stderr, "failed to load model: %s\n",
                 opts.model.string().c_str());
    return 1;
  }

  std::vector<float> pcm = {};
  if (!load_wav_24khz_mono(opts.audio, pcm)) {
    std::fprintf(stderr, "failed to load 24 kHz mono s16 wav: %s\n",
                 opts.audio.string().c_str());
    return 1;
  }

  std::vector<float> prepared(
      mimi::detail::required_prepared_floats(*loaded.model));
  std::vector<float> state(mimi::detail::required_state_floats(*loaded.model));
  std::vector<float> workspace(
      mimi::detail::required_workspace_floats(*loaded.model));
  std::vector<float> frame(mimi::detail::required_frame_floats(*loaded.model));
  if (prepared.empty() || state.empty() || workspace.empty() || frame.empty()) {
    std::fprintf(stderr, "model does not describe a mimi codec component\n");
    return 1;
  }

  mimi::sm codec{};
  emel::error::type err = emel::error::cast(mimi::error::none);
  mimi::event::initialize init{
      *loaded.model, std::span<float>{prepared}, std::span<float>{state},
      std::span<float>{workspace}, std::span<float>{frame}};
  init.error_out = &err;
  init.on_done = emel::callback<void(
      const mimi::events::initialize_done &)>::from<&on_codec_initialized>();
  if (!codec.process_event(init)) {
    std::fprintf(stderr, "codec initialization failed: err=%" PRIu64 "\n",
                 static_cast<uint64_t>(err));
    return 1;
  }

  const auto frame_samples = static_cast<size_t>(g_frame_samples);
  const auto n_q = static_cast<size_t>(g_n_q);
  const size_t frames = pcm.size() / frame_samples;
  if (frames == 0u) {
    std::fprintf(stderr, "audio shorter than one 80 ms frame\n");
    return 1;
  }

  std::vector<int32_t> codes(n_q, -1);
  std::vector<std::vector<int32_t>> all_codes = {};
  all_codes.reserve(frames);
  for (size_t index = 0; index < frames; ++index) {
    const mimi::event::encode_frame encode_ev = [&] {
      mimi::event::encode_frame ev{
          std::span<const float>{pcm.data() + index * frame_samples,
                                 frame_samples},
          std::span<int32_t>{codes}};
      ev.error_out = &err;
      return ev;
    }();
    if (!codec.process_event(encode_ev)) {
      std::fprintf(stderr, "encode failed at frame %zu: err=%" PRIu64 "\n",
                   index, static_cast<uint64_t>(err));
      return 1;
    }
    std::printf("frame=%zu codes=", index);
    for (size_t stream = 0; stream < n_q; ++stream) {
      std::printf("%s%d", stream == 0 ? "" : ",", codes[stream]);
    }
    std::printf("\n");
    all_codes.push_back(codes);
  }

  if (!opts.decode_out.empty()) {
    if (!codec.process_event(mimi::event::reset_stream{})) {
      std::fprintf(stderr, "stream reset failed before decode pass\n");
      return 1;
    }
    std::vector<float> decoded(frame_samples, 0.0f);
    std::ofstream out(opts.decode_out, std::ios::binary);
    if (!out.good()) {
      std::fprintf(stderr, "failed to open decode output: %s\n",
                   opts.decode_out.string().c_str());
      return 1;
    }
    for (size_t index = 0; index < all_codes.size(); ++index) {
      const mimi::event::decode_frame decode_ev = [&] {
        mimi::event::decode_frame ev{std::span<const int32_t>{all_codes[index]},
                                     std::span<float>{decoded}};
        ev.error_out = &err;
        return ev;
      }();
      if (!codec.process_event(decode_ev)) {
        std::fprintf(stderr, "decode failed at frame %zu: err=%" PRIu64 "\n",
                     index, static_cast<uint64_t>(err));
        return 1;
      }
      out.write(reinterpret_cast<const char *>(decoded.data()),
                static_cast<std::streamsize>(decoded.size() * sizeof(float)));
    }
  }

  std::fprintf(stderr, "encoded %zu frames (n_q=%zu, frame_samples=%zu)\n",
               frames, n_q, frame_samples);
  return 0;
}
