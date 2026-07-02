// Reference-lane Mimi/Moshi driver.
//
// Compiles against moshi.cpp (pinned via tools/bench/moshi_reference_ref.txt)
// + ggml + SentencePiece and exposes deterministic frame-level subcommands
// for the parity scripts. This executable is the ONLY place the Moshi
// reference implementation runs; the EMEL lane (mimi_emel_parity_runner)
// never links or calls into it. Output format matches the EMEL runner:
// one `frame=<n> codes=<c0>,<c1>,...` line per frame.
//
// Subcommands:
//   encode --mimi <raw mimi gguf> --audio <24 kHz mono s16 wav> [--n-q N]
//   decode --mimi <raw mimi gguf> --codes <codes txt> --out <raw f32 pcm>
//          [--n-q N]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <moshi/moshi.h>

namespace {

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

struct options {
  std::string command = {};
  std::filesystem::path mimi = {};
  std::filesystem::path audio = {};
  std::filesystem::path codes = {};
  std::filesystem::path out = {};
  int n_q = 16;
};

bool parse_options(const int argc, char **argv, options &opts) {
  if (argc < 2) {
    return false;
  }
  opts.command = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string_view arg = argv[index];
    const bool has_value = index + 1 < argc;
    if (arg == "--mimi" && has_value) {
      opts.mimi = argv[++index];
    } else if (arg == "--audio" && has_value) {
      opts.audio = argv[++index];
    } else if (arg == "--codes" && has_value) {
      opts.codes = argv[++index];
    } else if (arg == "--out" && has_value) {
      opts.out = argv[++index];
    } else if (arg == "--n-q" && has_value) {
      opts.n_q = std::atoi(argv[++index]);
    } else {
      std::fprintf(stderr, "unknown or incomplete argument: %.*s\n",
                   static_cast<int>(arg.size()), arg.data());
      return false;
    }
  }
  return (opts.command == "encode" && !opts.mimi.empty() &&
          !opts.audio.empty()) ||
         (opts.command == "decode" && !opts.mimi.empty() &&
          !opts.codes.empty() && !opts.out.empty());
}

bool parse_codes_file(const std::filesystem::path &path, const int n_q,
                      std::vector<std::vector<int16_t>> &frames_out) {
  std::ifstream stream(path);
  if (!stream.good()) {
    return false;
  }
  std::string line = {};
  while (std::getline(stream, line)) {
    const auto codes_pos = line.find("codes=");
    if (codes_pos == std::string::npos) {
      continue;
    }
    std::vector<int16_t> frame = {};
    std::stringstream values(line.substr(codes_pos + 6u));
    std::string token = {};
    while (std::getline(values, token, ',')) {
      frame.push_back(static_cast<int16_t>(std::atoi(token.c_str())));
    }
    if (static_cast<int>(frame.size()) != n_q) {
      return false;
    }
    frames_out.push_back(std::move(frame));
  }
  return !frames_out.empty();
}

} // namespace

int main(int argc, char **argv) {
  options opts{};
  if (!parse_options(argc, argv, opts)) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  moshi_reference_driver encode --mimi <gguf> --audio <wav> "
                 "[--n-q N]\n"
                 "  moshi_reference_driver decode --mimi <gguf> --codes <txt> "
                 "--out <raw f32> [--n-q N]\n");
    return 2;
  }

  ggml_backend_t backend_cpu = ggml_backend_cpu_init();
  if (backend_cpu == nullptr) {
    std::fprintf(stderr, "failed to init ggml cpu backend\n");
    return 1;
  }
  moshi_context_t *moshi = moshi_alloc(backend_cpu, backend_cpu);
  if (moshi == nullptr) {
    std::fprintf(stderr, "failed to alloc moshi context\n");
    return 1;
  }
  mimi_codec_t *codec = mimi_alloc(moshi, opts.mimi.string().c_str(), opts.n_q);
  if (codec == nullptr) {
    std::fprintf(stderr, "failed to load mimi codec: %s\n",
                 opts.mimi.string().c_str());
    return 1;
  }
  const int frame_size = mimi_frame_size(codec);
  if (frame_size <= 0) {
    std::fprintf(stderr, "invalid mimi frame size\n");
    return 1;
  }

  if (opts.command == "encode") {
    std::vector<float> pcm = {};
    if (!load_wav_24khz_mono(opts.audio, pcm)) {
      std::fprintf(stderr, "failed to load 24 kHz mono s16 wav: %s\n",
                   opts.audio.string().c_str());
      return 1;
    }
    const size_t frames = pcm.size() / static_cast<size_t>(frame_size);
    if (frames == 0u) {
      std::fprintf(stderr, "audio shorter than one frame\n");
      return 1;
    }
    mimi_encode_context_t *encode_ctx = mimi_encode_alloc_context(codec);
    std::vector<int16_t> tokens(static_cast<size_t>(opts.n_q), -1);
    for (size_t index = 0; index < frames; ++index) {
      mimi_encode_send(encode_ctx,
                       pcm.data() + index * static_cast<size_t>(frame_size));
      mimi_encode_receive(encode_ctx, tokens.data());
      std::printf("frame=%zu codes=", index);
      for (size_t stream = 0; stream < tokens.size(); ++stream) {
        std::printf("%s%d", stream == 0 ? "" : ",",
                    static_cast<int>(tokens[stream]));
      }
      std::printf("\n");
    }
    std::fprintf(stderr, "encoded %zu frames (frame_size=%d)\n", frames,
                 frame_size);
    unref(encode_ctx);
  } else {
    std::vector<std::vector<int16_t>> frames = {};
    if (!parse_codes_file(opts.codes, opts.n_q, frames)) {
      std::fprintf(stderr, "failed to parse codes file: %s\n",
                   opts.codes.string().c_str());
      return 1;
    }
    mimi_decode_context_t *decode_ctx = mimi_decode_alloc_context(codec);
    std::ofstream out(opts.out, std::ios::binary);
    if (!out.good()) {
      std::fprintf(stderr, "failed to open output: %s\n",
                   opts.out.string().c_str());
      return 1;
    }
    std::vector<float> frame(static_cast<size_t>(frame_size), 0.0f);
    for (auto &tokens : frames) {
      mimi_decode_send(decode_ctx, tokens.data());
      mimi_decode_receive(decode_ctx, frame.data());
      out.write(reinterpret_cast<const char *>(frame.data()),
                static_cast<std::streamsize>(frame.size() * sizeof(float)));
    }
    std::fprintf(stderr, "decoded %zu frames (frame_size=%d)\n", frames.size(),
                 frame_size);
    unref(decode_ctx);
  }

  unref(codec);
  unref(moshi);
  ggml_backend_free(backend_cpu);
  return 0;
}
