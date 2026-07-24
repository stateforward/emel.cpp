#include "bench_cases.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/io/mmap/events.hpp"
#include "emel/io/mmap/sm.hpp"
#include "emel/kernel/matmul/sm.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/logits/sampler/sm.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/tensor/sm.hpp"
#include "emel/speech/predictor/moshi/executor/any.hpp"

namespace {

using emel::bench::config;
using emel::bench::measure_case;
using emel::bench::result;

struct gguf_capture {
  bool probe_done = false;
  bool probe_error = false;
  bool bind_done = false;
  bool bind_error = false;
  bool parse_done = false;
  bool parse_error = false;
  emel::gguf::loader::requirements requirements = {};
  emel::error::type err = emel::error::cast(emel::gguf::loader::error::none);
};

struct load_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
};

struct mapped_model {
  emel::io::mmap::sm mapper = {};
  bool map_done = false;
  bool map_error = false;
  bool release_done = false;
  bool release_error = false;
  uint32_t handle = emel::io::mmap::k_invalid_mapping_handle;
  const void *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
  emel::error::type err = emel::error::cast(emel::io::mmap::error::none);
};

struct lm_fixture {
  std::string model_path = {};
  mapped_model mapping = {};
  emel::model::data model_data = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  uint32_t gguf_tensor_count = 0u;
  std::vector<emel::model::tensor::effect_request> effect_requests = {};
  std::vector<emel::model::tensor::effect_result> effect_results = {};
  emel::gguf::loader::sm gguf_loader = {};
  emel::model::tensor::sm tensor_loader = {};
  emel::model::loader::sm model_loader = {};
  gguf_capture gguf = {};
  load_capture load = {};
};

uint64_t append_checksum(uint64_t checksum, const void *data,
                         const std::size_t bytes) noexcept {
  const auto *input = static_cast<const uint8_t *>(data);
  for (std::size_t index = 0u; index < bytes; ++index) {
    checksum ^= input[index];
    checksum *= 1099511628211ull;
  }
  return checksum;
}

template <class value_type>
uint64_t append_checksum(uint64_t checksum,
                         const std::span<const value_type> values) noexcept {
  return append_checksum(checksum, values.data(), values.size_bytes());
}

struct attention_route_snapshot {
  std::vector<float> temporal_state = {};
  std::vector<uint16_t> key_cache = {};
  std::vector<uint16_t> value_cache = {};
  std::vector<int32_t> audio_tokens = {};
  int32_t text_token = -1;

  bool operator==(const attention_route_snapshot &) const = default;

  uint64_t checksum() const noexcept {
    uint64_t value = 1469598103934665603ull;
    value = append_checksum(value, std::span<const float>{temporal_state});
    value = append_checksum(value, std::span<const uint16_t>{key_cache});
    value = append_checksum(value, std::span<const uint16_t>{value_cache});
    value = append_checksum(value, std::span<const int32_t>{audio_tokens});
    return append_checksum(value, &text_token, sizeof(text_token));
  }
};

template <class prepare_type, class run_type, class finish_type>
result measure_prepared_case(const char *name, const config &cfg,
                             prepare_type &&prepare, run_type &&run,
                             finish_type &&finish) {
  const auto runs = std::max<std::size_t>(cfg.runs, 1u);
  const auto iterations = std::max<uint64_t>(cfg.iterations, 1u);
  std::vector<double> samples;
  samples.reserve(runs);

  for (std::size_t warmup_run = 0u; warmup_run < cfg.warmup_runs;
       ++warmup_run) {
    for (uint64_t index = 0u; index < cfg.warmup_iterations; ++index) {
      auto prepared = prepare();
      run(*prepared);
    }
  }

  for (std::size_t measured_run = 0u; measured_run < runs; ++measured_run) {
    int64_t elapsed_ns = 0;
    for (uint64_t index = 0u; index < iterations; ++index) {
      auto prepared = prepare();
      const auto start = std::chrono::steady_clock::now();
      run(*prepared);
      const auto end = std::chrono::steady_clock::now();
      elapsed_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      finish(*prepared);
    }
    samples.push_back(static_cast<double>(elapsed_ns) /
                      static_cast<double>(iterations));
  }

  std::sort(samples.begin(), samples.end());
  double sum = 0.0;
  for (const double sample : samples) {
    sum += sample;
  }
  result measured;
  measured.name = name;
  measured.ns_per_op = emel::bench::select_reported_ns_per_op(samples);
  measured.ns_min_per_op = samples.front();
  measured.ns_mean_per_op = sum / static_cast<double>(samples.size());
  measured.ns_max_per_op = samples.back();
  measured.iterations = iterations;
  measured.runs = runs;
  return measured;
}

struct attention_executor_bench_fixture {
  static constexpr int32_t k_valid_positions = 250;

  attention_executor_bench_fixture(const emel::model::data &model_ref,
                                   const std::size_t lane_count_ref)
      : model(model_ref), lane_count(lane_count_ref),
        temporal_positions(emel::memory::streaming::dependencies{
            .capacity = model.moshi_lm.context}),
        depformer_positions(emel::memory::streaming::dependencies{
            .capacity = model.moshi_lm.depformer_context}),
        layer_offsets(static_cast<std::size_t>(model.moshi_lm.num_layers), 0u),
        key_cache(static_cast<std::size_t>(model.moshi_lm.num_layers) *
                  static_cast<std::size_t>(model.moshi_lm.context) *
                  static_cast<std::size_t>(model.moshi_lm.dim)),
        value_cache(key_cache.size()),
        depformer_layer_offsets(
            static_cast<std::size_t>(model.moshi_lm.depformer_num_layers), 0u),
        depformer_key_cache(
            static_cast<std::size_t>(model.moshi_lm.depformer_num_layers) *
            static_cast<std::size_t>(model.moshi_lm.depformer_context) *
            static_cast<std::size_t>(model.moshi_lm.depformer_dim)),
        depformer_value_cache(depformer_key_cache.size()),
        input_sequence(static_cast<std::size_t>(model.moshi_lm.n_q + 1), -1),
        input_embedding(static_cast<std::size_t>(model.moshi_lm.dim)),
        temporal_state(input_embedding.size()),
        audio_tokens(static_cast<std::size_t>(model.moshi_lm.dep_q), -1),
        matmul_policy{
            .parallel_matmul_lanes = nullptr,
            .kernel_kind = emel::kernel::detect_host_kind(),
            .active_lanes = 1u,
            .mode = emel::kernel::matmul::lane_mode::serial,
        },
        matmul(matmul_policy) {
    const std::size_t per_layer_cache =
        static_cast<std::size_t>(model.moshi_lm.context) *
        static_cast<std::size_t>(model.moshi_lm.dim);
    for (std::size_t index = 0u; index < layer_offsets.size(); ++index) {
      layer_offsets[index] = index * per_layer_cache;
    }
    const std::size_t depformer_per_layer_cache =
        static_cast<std::size_t>(model.moshi_lm.depformer_context) *
        static_cast<std::size_t>(model.moshi_lm.depformer_dim);
    for (std::size_t index = 0u; index < depformer_layer_offsets.size();
         ++index) {
      depformer_layer_offsets[index] = index * depformer_per_layer_cache;
    }
    if (lane_count > 1u) {
      attention_lanes.emplace(lane_count - 1u);
    }
    for (std::size_t index = 0u; index < input_embedding.size(); ++index) {
      input_embedding[index] =
          static_cast<float>(static_cast<int32_t>(index % 127u) - 63) / 128.0f;
    }
    input_sequence[0] = model.moshi_lm.text_padding_id;
    initialize_position(temporal_positions);
    initialize_position(depformer_positions);
    executor =
        std::make_unique<emel::speech::predictor::moshi::executor::sm>(
            emel::speech::predictor::moshi::executor::dependencies{
                .kv =
                    emel::speech::predictor::moshi::executor::kv_views{
                        .temporal =
                            {
                                .key_cache = std::span<uint16_t>{key_cache},
                                .value_cache = std::span<uint16_t>{value_cache},
                                .layer_cache_offsets =
                                    std::span<const std::size_t>{layer_offsets},
                                .layer_count = model.moshi_lm.num_layers,
                                .position_capacity = model.moshi_lm.context,
                                .kv_dim = model.moshi_lm.dim,
                            },
                        .depformer =
                            {
                                .key_cache =
                                    std::span<uint16_t>{depformer_key_cache},
                                .value_cache =
                                    std::span<uint16_t>{depformer_value_cache},
                                .layer_cache_offsets =
                                    std::span<const std::size_t>{
                                        depformer_layer_offsets},
                                .layer_count =
                                    model.moshi_lm.depformer_num_layers,
                                .position_capacity =
                                    model.moshi_lm.depformer_context,
                                .kv_dim = model.moshi_lm.depformer_dim,
                            },
                        .temporal_positions = &temporal_positions,
                        .depformer_positions = &depformer_positions,
                    },
                .kernel = kernel,
                .matmul = matmul,
                .matmul_lane_mode = emel::kernel::matmul::lane_mode::serial,
                .attention_lanes =
                    attention_lanes ? &*attention_lanes : nullptr,
                .active_attention_lanes = lane_count,
                .sampler = nullptr,
                .policy =
                    {
                        .rms_norm_epsilon = 1.0e-8f,
                        .zero_seed_state = 123459876u,
                        .sampling_modulus = 2147483647u,
                        .token_zero = -1,
                    },
                .capacity =
                    {
                        .hidden_dim =
                            static_cast<uint64_t>(
                                std::max(model.moshi_lm.dim,
                                         model.moshi_lm.depformer_dim)),
                        .temporal_context =
                            static_cast<uint64_t>(model.moshi_lm.context),
                        .depformer_context =
                            static_cast<uint64_t>(
                                model.moshi_lm.depformer_context),
                        .sampling_card =
                            static_cast<uint64_t>(
                                std::max(model.moshi_lm.text_card,
                                         model.moshi_lm.card)),
                        .sampling_top_k = 1u,
                    },
            });
    emel::error::type err = emel::error::cast(
        emel::speech::predictor::moshi::executor::error::none);
    emel::speech::predictor::moshi::executor::event::initialize initialize{
        model};
    initialize.error_out = &err;
    if (!executor->process_event(initialize)) {
      std::fprintf(stderr,
                   "error: Moshi executor attention benchmark init failed "
                   "lanes=%zu err=%d\n",
                   lane_count, static_cast<int>(err));
      std::exit(1);
    }
    prime_temporal_window();
  }

  static void initialize_position(emel::memory::streaming::sm &position) {
    int32_t err = static_cast<int32_t>(
        emel::error::cast(emel::memory::streaming::error::none));
    if (!position.process_event(
            emel::memory::streaming::event::initialize{err}) ||
        err != static_cast<int32_t>(
                   emel::error::cast(emel::memory::streaming::error::none))) {
      std::fprintf(stderr,
                   "error: Moshi executor attention position init failed\n");
      std::exit(1);
    }
  }

  void prime_temporal_window() noexcept {
    for (int32_t index = 0; index < k_valid_positions - 1; ++index) {
      emel::memory::streaming::advance_result result{};
      int32_t err = static_cast<int32_t>(
          emel::error::cast(emel::memory::streaming::error::none));
      if (!temporal_positions.process_event(
              emel::memory::streaming::event::advance{result, err}) ||
          err != static_cast<int32_t>(
                     emel::error::cast(emel::memory::streaming::error::none))) {
        std::exit(1);
      }
    }
  }

  void run_prediction() noexcept {
    emel::error::type err = emel::error::cast(
        emel::speech::predictor::moshi::executor::error::none);
    std::fill(temporal_state.begin(), temporal_state.end(), 0.0f);
    std::fill(audio_tokens.begin(), audio_tokens.end(), -1);
    text_token = -1;
    emel::memory::view::snapshot memory{};
    memory.max_sequences = 1;
    memory.sequence_active[0] = 1;
    memory.sequence_length_values[0] = k_valid_positions;
    memory.sequence_kv_block_count[0] = 1;
    memory.sequence_kv_blocks[0][0] = 0;
    emel::speech::predictor::moshi::event::graph_step step{
        model, memory, std::span<const int32_t>{input_sequence},
        std::span<int32_t>{audio_tokens}, text_token};
    step.phase = emel::speech::predictor::moshi::event::graph_phase::prediction;
    step.input_embedding = std::span<const float>{input_embedding};
    step.temporal_state = std::span<float>{temporal_state};
    step.error_out = &err;
    if (!executor->process_event(step)) {
      std::fprintf(stderr,
                   "error: Moshi executor attention prediction failed "
                   "lanes=%zu err=%d\n",
                   lane_count, static_cast<int>(err));
      std::exit(1);
    }
  }

  attention_route_snapshot capture_snapshot() const {
    return attention_route_snapshot{
        .temporal_state = temporal_state,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .audio_tokens = audio_tokens,
        .text_token = text_token,
    };
  }

  const emel::model::data &model;
  std::size_t lane_count = 1u;
  emel::memory::streaming::sm temporal_positions;
  emel::memory::streaming::sm depformer_positions;
  std::vector<std::size_t> layer_offsets = {};
  std::vector<uint16_t> key_cache = {};
  std::vector<uint16_t> value_cache = {};
  std::vector<std::size_t> depformer_layer_offsets = {};
  std::vector<uint16_t> depformer_key_cache = {};
  std::vector<uint16_t> depformer_value_cache = {};
  std::vector<int32_t> input_sequence = {};
  std::vector<float> input_embedding = {};
  std::vector<float> temporal_state = {};
  std::vector<int32_t> audio_tokens = {};
  int32_t text_token = -1;
  emel::kernel::sm kernel = {};
  emel::kernel::matmul::execution_policy matmul_policy;
  emel::kernel::matmul::sm matmul;
  std::optional<emel::kernel::matmul::lane_pool> attention_lanes = {};
  std::unique_ptr<emel::speech::predictor::moshi::executor::sm> executor = {};
};

bool bench_enabled() {
  const char *flag = std::getenv("EMEL_BENCH_SPEECH_LM_MOSHI");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

std::filesystem::path bench_root_path() {
#ifdef EMEL_BENCH_REPO_ROOT
  return std::filesystem::path(EMEL_BENCH_REPO_ROOT);
#else
  std::filesystem::path path = std::filesystem::path(__FILE__).parent_path();
  return path.parent_path().parent_path().parent_path();
#endif
}

std::filesystem::path
first_existing_path(const std::span<const std::filesystem::path> candidates) {
  for (const auto &path : candidates) {
    if (!path.empty() && std::filesystem::exists(path)) {
      return path;
    }
  }
  return {};
}

std::filesystem::path personaplex_lm_model_path() {
  for (const char *env_name :
       {"EMEL_PERSONAPLEX_LM_MODEL", "EMEL_MOSHI_LM_MODEL",
        "EMEL_BENCH_SPEECH_LM_MOSHI_MODEL",
        "EMEL_MOSHI_REFERENCE_MODEL_EMEL"}) {
    const char *value = std::getenv(env_name);
    if (value != nullptr && value[0] != '\0') {
      return std::filesystem::path{value};
    }
  }

  const auto root = bench_root_path();
  const std::array candidates{
      root / "build" / "moshi_reference" / "model-q4_k-emel.gguf",
      root.parent_path() / "companion" / "zig-out" /
          "personaplex-emel-converted" / "Codes4Fun" /
          "personaplex-7b-v1-q4_k-GGUF" / "model-q4_k.gguf",
      root.parent_path().parent_path() / "companion" / "zig-out" /
          "personaplex-emel-converted" / "Codes4Fun" /
          "personaplex-7b-v1-q4_k-GGUF" / "model-q4_k.gguf",
      root.parent_path().parent_path().parent_path() / "companion" / "zig-out" /
          "personaplex-emel-converted" / "Codes4Fun" /
          "personaplex-7b-v1-q4_k-GGUF" / "model-q4_k.gguf",
  };
  return first_existing_path(std::span<const std::filesystem::path>{
      candidates.data(), candidates.size()});
}

void on_probe_done(void *owner,
                   const emel::gguf::loader::events::probe_done &ev) {
  auto &fixture = *static_cast<lm_fixture *>(owner);
  fixture.gguf.probe_done = true;
  fixture.gguf.requirements = ev.requirements_out;
}

void on_probe_error(void *owner,
                    const emel::gguf::loader::events::probe_error &ev) {
  auto &fixture = *static_cast<lm_fixture *>(owner);
  fixture.gguf.probe_error = true;
  fixture.gguf.err = ev.err;
}

void on_bind_done(void *owner, const emel::gguf::loader::events::bind_done &) {
  static_cast<lm_fixture *>(owner)->gguf.bind_done = true;
}

void on_bind_error(void *owner,
                   const emel::gguf::loader::events::bind_error &ev) {
  auto &fixture = *static_cast<lm_fixture *>(owner);
  fixture.gguf.bind_error = true;
  fixture.gguf.err = ev.err;
}

void on_parse_done(void *owner,
                   const emel::gguf::loader::events::parse_done &) {
  static_cast<lm_fixture *>(owner)->gguf.parse_done = true;
}

void on_parse_error(void *owner,
                    const emel::gguf::loader::events::parse_error &ev) {
  auto &fixture = *static_cast<lm_fixture *>(owner);
  fixture.gguf.parse_error = true;
  fixture.gguf.err = ev.err;
}

void on_load_done(void *owner, const emel::model::loader::events::load_done &) {
  static_cast<lm_fixture *>(owner)->load.done = true;
}

void on_load_error(void *owner,
                   const emel::model::loader::events::load_error &ev) {
  auto &fixture = *static_cast<lm_fixture *>(owner);
  fixture.load.error = true;
  fixture.load.err = ev.err;
}

void on_map_done(void *owner,
                 const emel::io::mmap::events::map_tensor_done &ev) {
  auto &mapping = *static_cast<mapped_model *>(owner);
  mapping.map_done = true;
  mapping.handle = ev.handle;
  mapping.buffer = ev.buffer;
  mapping.buffer_bytes = ev.buffer_bytes;
}

void on_map_error(void *owner,
                  const emel::io::mmap::events::map_tensor_error &ev) {
  auto &mapping = *static_cast<mapped_model *>(owner);
  mapping.map_error = true;
  mapping.err = ev.err;
}

void on_release_done(void *owner,
                     const emel::io::mmap::events::release_mapping_done &) {
  static_cast<mapped_model *>(owner)->release_done = true;
}

void on_release_error(void *owner,
                      const emel::io::mmap::events::release_mapping_error &ev) {
  auto &mapping = *static_cast<mapped_model *>(owner);
  mapping.release_error = true;
  mapping.err = ev.err;
}

bool map_file(mapped_model &mapping, const std::string &path) {
  std::error_code ec;
  const auto file_size = std::filesystem::file_size(path, ec);
  if (ec || file_size == 0u) {
    return false;
  }

  emel::io::mmap::event::map_tensor_request request = {};
  request.tensor_id = 0;
  request.file_offset = 0u;
  request.byte_size = static_cast<uint64_t>(file_size);
  request.file_path = path;
  emel::io::mmap::event::map_tensor map{request};
  map.on_done = {&mapping, on_map_done};
  map.on_error = {&mapping, on_map_error};
  return mapping.mapper.process_event(map) && mapping.map_done &&
         !mapping.map_error && mapping.buffer != nullptr &&
         mapping.buffer_bytes == static_cast<uint64_t>(file_size);
}

void release_file(mapped_model &mapping) {
  if (mapping.handle == emel::io::mmap::k_invalid_mapping_handle) {
    return;
  }
  emel::io::mmap::event::release_mapping release{0, mapping.handle};
  release.on_done = {&mapping, on_release_done};
  release.on_error = {&mapping, on_release_error};
  static_cast<void>(mapping.mapper.process_event(release));
  mapping.handle = emel::io::mmap::k_invalid_mapping_handle;
  mapping.buffer = nullptr;
  mapping.buffer_bytes = 0u;
}

bool copy_tensor_names(const std::span<const uint8_t> file_image,
                       emel::model::data &model_data) {
  model_data.name_bytes_used = 0u;
  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    auto &tensor = model_data.tensors[index];
    const size_t name_offset = static_cast<size_t>(tensor.name_offset);
    const size_t name_length = static_cast<size_t>(tensor.name_length);
    if (name_offset + name_length > file_image.size() ||
        model_data.name_bytes_used + name_length >
            model_data.name_storage.size()) {
      return false;
    }
    if (name_length > 0u) {
      std::memcpy(model_data.name_storage.data() + model_data.name_bytes_used,
                  file_image.data() + name_offset, name_length);
    }
    tensor.name_offset = model_data.name_bytes_used;
    model_data.name_bytes_used += static_cast<uint32_t>(name_length);
  }
  return true;
}

bool prebind_gguf_storage(lm_fixture &fixture) {
  const std::span<const uint8_t> file_image{
      static_cast<const uint8_t *>(fixture.mapping.buffer),
      static_cast<size_t>(fixture.mapping.buffer_bytes),
  };
  fixture.gguf = {};
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&fixture,
                                                               on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{
      &fixture, on_probe_error};
  const emel::gguf::loader::event::probe probe_ev{
      file_image, requirements, probe_done_cb, probe_error_cb};
  if (!fixture.gguf_loader.process_event(probe_ev) ||
      !fixture.gguf.probe_done || fixture.gguf.probe_error ||
      requirements.tensor_count >
          static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return false;
  }

  const uint64_t arena_bytes =
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  fixture.kv_arena.resize(static_cast<size_t>(arena_bytes));
  fixture.kv_entries.resize(requirements.kv_count);
  fixture.gguf_tensor_count = requirements.tensor_count;
  return true;
}

emel::error::type run_parse_model(void *owner,
                                  const emel::model::loader::event::load &req) {
  auto &fixture = *static_cast<lm_fixture *>(owner);
  if (req.file_image == nullptr || req.file_size == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }
  const std::span<const uint8_t> file_image{
      static_cast<const uint8_t *>(req.file_image),
      static_cast<size_t>(req.file_size),
  };

  fixture.gguf = {};
  const emel::gguf::loader::event::bind_done_fn bind_done_cb{&fixture,
                                                             on_bind_done};
  const emel::gguf::loader::event::bind_error_fn bind_error_cb{&fixture,
                                                               on_bind_error};
  const emel::gguf::loader::event::bind_storage bind_ev{
      std::span<uint8_t>{fixture.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  fixture.gguf_tensor_count},
      bind_done_cb,
      bind_error_cb,
  };
  if (!fixture.gguf_loader.process_event(bind_ev) || !fixture.gguf.bind_done ||
      fixture.gguf.bind_error) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  fixture.gguf = {};
  const emel::gguf::loader::event::parse_done_fn parse_done_cb{&fixture,
                                                               on_parse_done};
  const emel::gguf::loader::event::parse_error_fn parse_error_cb{
      &fixture, on_parse_error};
  const emel::gguf::loader::event::parse parse_ev{file_image, parse_done_cb,
                                                  parse_error_cb};
  if (!fixture.gguf_loader.process_event(parse_ev) ||
      !fixture.gguf.parse_done || fixture.gguf.parse_error) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  req.model_data.n_tensors = fixture.gguf_tensor_count;
  if (!copy_tensor_names(file_image, req.model_data)) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena.data(),
                                        fixture.kv_arena.size()},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{
              fixture.kv_entries.data(), fixture.kv_entries.size()},
  };
  return emel::model::detail::load_hparams_from_gguf(binding, req.model_data)
             ? emel::error::cast(emel::model::loader::error::none)
             : emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type run_map_layers(void *,
                                 const emel::model::loader::event::load &req) {
  req.model_data.n_layers = req.model_data.params.n_layer;
  return req.model_data.n_layers > 0
             ? emel::error::cast(emel::model::loader::error::none)
             : emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type
run_validate_structure(void *, const emel::model::loader::event::load &req) {
  if (req.model_data.n_tensors == 0u ||
      req.model_data.moshi_component_id !=
          emel::model::data::moshi_component::lm ||
      req.model_data.weights_data == nullptr ||
      req.model_data.weights_size == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
run_validate_architecture(void *, const emel::model::loader::event::load &req) {
  return emel::model::validate_execution_contract(req.model_data);
}

bool load_contract_once(lm_fixture &fixture) {
  fixture.load = {};
  std::destroy_at(&fixture.model_data);
  std::construct_at(&fixture.model_data);
  fixture.effect_requests.resize(emel::model::data::k_max_tensors);
  fixture.effect_results.resize(emel::model::data::k_max_tensors);

  emel::model::loader::event::parse_model_fn parse_model{&fixture,
                                                         run_parse_model};
  emel::model::loader::event::load load_ev{fixture.model_data, parse_model};
  load_ev.model_path = fixture.model_path;
  load_ev.file_image = fixture.mapping.buffer;
  load_ev.file_size = fixture.mapping.buffer_bytes;
  load_ev.tensor_loader = &fixture.tensor_loader;
  load_ev.effect_requests = std::span{fixture.effect_requests};
  load_ev.effect_results = std::span{fixture.effect_results};
  load_ev.map_layers = {nullptr, run_map_layers};
  load_ev.validate_structure = {nullptr, run_validate_structure};
  load_ev.validate_architecture_impl = {nullptr, run_validate_architecture};
  load_ev.on_done = {&fixture, on_load_done};
  load_ev.on_error = {&fixture, on_load_error};
  return fixture.model_loader.process_event(load_ev) && fixture.load.done &&
         !fixture.load.error;
}

std::string fixture_id_from_path(const std::filesystem::path &path) {
  const auto parent = path.parent_path().filename().string();
  const auto filename = path.filename().string();
  if (parent.empty()) {
    return filename;
  }
  return parent + "/" + filename;
}

} // namespace

namespace emel::bench {

void append_emel_speech_lm_moshi_cases(std::vector<result> &results,
                                       const config &cfg) {
  if (!bench_enabled()) {
    return;
  }

  const auto model_path = personaplex_lm_model_path();
  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    std::fprintf(stderr, "error: missing PersonaPlex Moshi LM GGUF; set "
                         "EMEL_MOSHI_LM_MODEL, EMEL_PERSONAPLEX_LM_MODEL, "
                         "or EMEL_MOSHI_REFERENCE_MODEL_EMEL\n");
    std::exit(1);
  }

  auto fixture = std::make_unique<lm_fixture>();
  fixture->model_path = model_path.string();
  if (!map_file(fixture->mapping, fixture->model_path) ||
      !prebind_gguf_storage(*fixture) || !load_contract_once(*fixture)) {
    std::fprintf(stderr,
                 "error: speech_lm_moshi contract setup failed for %s\n",
                 fixture->model_path.c_str());
    release_file(fixture->mapping);
    std::exit(1);
  }

  auto fn = [&]() {
    if (!load_contract_once(*fixture)) {
      std::fprintf(stderr,
                   "error: speech_lm_moshi contract reload failed for %s\n",
                   fixture->model_path.c_str());
      std::exit(1);
    }
  };
  auto measured = measure_case(
      "speech_lm_moshi/load_contract/personaplex_q4_k_lm", cfg, fn);
  measured.lane = "emel";
  measured.backend_id = "emel";
  measured.backend_language = "c++";
  measured.model_id = "Codes4Fun/personaplex-7b-v1-q4_k-GGUF";
  measured.fixture_id = fixture_id_from_path(model_path);
  measured.workload_id = "moshi_lm_load_contract";
  measured.note = "component=lm contract=moshi_execution dtype=q4_k";
  results.push_back(std::move(measured));

  auto append_attention_result = [&](const char *name, const int32_t lanes,
                                     const attention_route_snapshot
                                         *reference) {
    attention_route_snapshot measured_snapshot;
    auto attention_result = measure_prepared_case(
        name, cfg,
        [&]() {
          return std::make_unique<attention_executor_bench_fixture>(
              fixture->model_data, static_cast<std::size_t>(lanes));
        },
        [](attention_executor_bench_fixture &attention) {
          attention.run_prediction();
        },
        [&](const attention_executor_bench_fixture &attention) {
          measured_snapshot = attention.capture_snapshot();
        });
    const uint64_t measured_checksum = measured_snapshot.checksum();
    if (reference != nullptr && measured_snapshot != *reference) {
      std::fprintf(stderr,
                   "error: measured %d-lane owning Moshi attention route is "
                   "not exact\n",
                   lanes);
      std::exit(1);
    }
    attention_result.lane = "emel";
    attention_result.backend_id = "emel";
    attention_result.backend_language = "c++";
    attention_result.thread_count = lanes;
    attention_result.thread_contract =
        "owning_moshi_executor_same_rtc_attention_head_ranges";
    attention_result.model_id = "Codes4Fun/personaplex-7b-v1-q4_k-GGUF";
    attention_result.fixture_id = fixture_id_from_path(model_path);
    attention_result.workload_id =
        "moshi_executor_prediction_attention_capacity_3000_start_valid_250";
    attention_result.output_checksum = measured_checksum;
    attention_result.output_dim =
        static_cast<uint64_t>(measured_snapshot.temporal_state.size());
    attention_result.note =
        "component=moshi_executor_prediction only_delta=attention_lanes "
        "nonattention_matmul=serial_identical capacity=3000 "
        "start_valid_positions=250 measured_checksum=temporal_state+kv+outputs "
        "iterations=" +
        std::to_string(attention_result.iterations) +
        " runs=" + std::to_string(attention_result.runs);
    results.push_back(std::move(attention_result));
    return measured_snapshot;
  };
  const attention_route_snapshot serial_snapshot =
      append_attention_result("speech_lm_moshi/executor_prediction_attention/"
                              "personaplex_capacity_3000_valid_250_lanes_1",
                              1, nullptr);
  append_attention_result("speech_lm_moshi/executor_prediction_attention/"
                          "personaplex_capacity_3000_valid_250_lanes_2",
                          2, &serial_snapshot);
  append_attention_result("speech_lm_moshi/executor_prediction_attention/"
                          "personaplex_capacity_3000_valid_250_lanes_4",
                          4, &serial_snapshot);
  append_attention_result("speech_lm_moshi/executor_prediction_attention/"
                          "personaplex_capacity_3000_valid_250_lanes_8",
                          8, &serial_snapshot);

  release_file(fixture->mapping);
}

void append_reference_speech_lm_moshi_cases(std::vector<result> &,
                                            const config &) {}

} // namespace emel::bench
