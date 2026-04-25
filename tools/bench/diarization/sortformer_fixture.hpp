#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/diarization/request/detail.hpp"
#include "emel/diarization/sortformer/output/detail.hpp"
#include "emel/diarization/sortformer/pipeline/detail.hpp"
#include "emel/diarization/sortformer/pipeline/sm.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/sortformer/detail.hpp"
#include "emel/model/weight_loader/errors.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace emel::bench::diarization::sortformer_fixture {

namespace output_detail = emel::diarization::sortformer::output::detail;
namespace pipeline = emel::diarization::sortformer::pipeline;
namespace pipeline_detail = emel::diarization::sortformer::pipeline::detail;
namespace request_detail = emel::diarization::request::detail;

inline constexpr const char * k_case_name =
    "diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono";
inline constexpr const char * k_profile_case_name =
    "diarization/sortformer/profile_ami_en2002b_mix_headset_137.00_152.04_16khz_mono";
inline constexpr const char * k_model_id = "diar_streaming_sortformer_4spk_v2_1_gguf";
inline constexpr const char * k_fixture_id = "ami_en2002b_mix_headset_137.00_152.04_16khz_mono";
inline constexpr const char * k_profile_id =
    "source=ami ihm/test path=EN2002b.Mix-Headset.wav window=137.00-152.04s";
inline constexpr std::string_view k_model_rel_path =
    "tests/models/diar_streaming_sortformer_4spk-v2.1.gguf";
inline constexpr std::string_view k_audio_rel_path =
    "tests/fixtures/diarization/ami_en2002b_mix_headset_137.00_152.04_16khz_mono.wav";
inline constexpr std::string_view k_audio_meta_rel_path =
    "tests/fixtures/diarization/ami_en2002b_mix_headset_137.00_152.04_16khz_mono.json";
inline constexpr std::string_view k_baseline_rel_path =
    "tests/fixtures/diarization/ami_en2002b_mix_headset_137.00_152.04_16khz_mono.baseline.txt";

inline std::filesystem::path repo_root_path() {
#ifdef EMEL_BENCH_REPO_ROOT
  return std::filesystem::path(EMEL_BENCH_REPO_ROOT);
#elif defined(EMEL_TEST_REPO_ROOT)
  return std::filesystem::path(EMEL_TEST_REPO_ROOT);
#else
  return std::filesystem::current_path();
#endif
}

inline std::filesystem::path resolve_repo_path(const std::string_view rel_path) {
  return repo_root_path() / std::filesystem::path(rel_path);
}

inline uint32_t read_u32_le(const std::span<const uint8_t> bytes) {
  uint32_t value = 0u;
  for (size_t i = 0u; i < sizeof(uint32_t); ++i) {
    value |= static_cast<uint32_t>(bytes[i]) << (i * 8u);
  }
  return value;
}

inline uint64_t read_u64_le(const std::span<const uint8_t> bytes) {
  uint64_t value = 0u;
  for (size_t i = 0u; i < sizeof(uint64_t); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8u);
  }
  return value;
}

inline bool read_file_bytes(const std::filesystem::path & path, std::vector<uint8_t> & out) {
  out.clear();

  std::FILE * file = std::fopen(path.string().c_str(), "rb");
  if (file == nullptr) {
    return false;
  }

  const bool seek_end_ok = std::fseek(file, 0, SEEK_END) == 0;
  const long file_size = seek_end_ok ? std::ftell(file) : -1L;
  const bool seek_start_ok = file_size >= 0L && std::fseek(file, 0, SEEK_SET) == 0;
  if (!seek_end_ok || file_size < 0L || !seek_start_ok) {
    std::fclose(file);
    return false;
  }

  out.resize(static_cast<size_t>(file_size));
  const size_t read_size = out.empty() ? 0u : std::fread(out.data(), 1u, out.size(), file);
  std::fclose(file);
  return read_size == out.size();
}

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

struct weight_capture {
  bool bind_done = false;
  bool bind_error = false;
  bool plan_done = false;
  bool plan_error = false;
  bool apply_done = false;
  bool apply_error = false;
  uint32_t effect_count = 0u;
  emel::error::type err = emel::error::cast(emel::model::weight_loader::error::none);
};

struct load_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0u;
  uint64_t bytes_done = 0u;
  bool used_mmap = false;
};

struct model_fixture {
  emel::model::data model = {};
  emel::model::sortformer::detail::execution_contract contract = {};
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::vector<emel::model::weight_loader::effect_request> effect_requests = {};
  std::vector<emel::model::weight_loader::effect_result> effect_results = {};
  emel::gguf::loader::sm gguf_loader = {};
  emel::model::weight_loader::sm weight_loader = {};
  emel::model::loader::sm model_loader = {};
  gguf_capture gguf = {};
  weight_capture weight = {};
  load_capture load = {};
  bool ready = false;
};

struct pcm_fixture {
  std::vector<float> pcm = {};
  int32_t sample_rate = 0;
  bool ready = false;
};

struct expected_output_baseline {
  int32_t segment_count = 0;
  std::uint64_t output_checksum = 0u;
  bool ready = false;
};

inline emel::model::detail::kv_binding kv_binding_from_fixture(const model_fixture & fixture) {
  return emel::model::detail::kv_binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena.data(), fixture.kv_arena.size()},
      .entries = std::span<const emel::gguf::loader::kv_entry>{fixture.kv_entries.data(),
                                                               fixture.kv_entries.size()},
  };
}

inline void reset_gguf_capture(model_fixture & fixture) { fixture.gguf = {}; }
inline void reset_weight_capture(model_fixture & fixture) { fixture.weight = {}; }
inline void reset_load_capture(model_fixture & fixture) { fixture.load = {}; }

inline void on_probe_done(void * owner, const emel::gguf::loader::events::probe_done & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.gguf.probe_done = true;
  fixture.gguf.probe_error = false;
  fixture.gguf.requirements = ev.requirements_out;
}

inline void on_probe_error(void * owner, const emel::gguf::loader::events::probe_error & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.gguf.probe_error = true;
  fixture.gguf.err = ev.err;
}

inline void on_bind_done(void * owner, const emel::gguf::loader::events::bind_done &) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.gguf.bind_done = true;
  fixture.gguf.bind_error = false;
}

inline void on_bind_error(void * owner, const emel::gguf::loader::events::bind_error & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.gguf.bind_error = true;
  fixture.gguf.err = ev.err;
}

inline void on_parse_done(void * owner, const emel::gguf::loader::events::parse_done &) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.gguf.parse_done = true;
  fixture.gguf.parse_error = false;
}

inline void on_parse_error(void * owner, const emel::gguf::loader::events::parse_error & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.gguf.parse_error = true;
  fixture.gguf.err = ev.err;
}

inline void on_weight_bind_done(void * owner,
                                const emel::model::weight_loader::events::bind_done &) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.weight.bind_done = true;
  fixture.weight.bind_error = false;
}

inline void on_weight_bind_error(void * owner,
                                 const emel::model::weight_loader::events::bind_error & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.weight.bind_error = true;
  fixture.weight.err = ev.err;
}

inline void on_weight_plan_done(void * owner,
                                const emel::model::weight_loader::events::plan_done & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.weight.plan_done = true;
  fixture.weight.plan_error = false;
  fixture.weight.effect_count = ev.effect_count;
}

inline void on_weight_plan_error(void * owner,
                                 const emel::model::weight_loader::events::plan_error & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.weight.plan_error = true;
  fixture.weight.err = ev.err;
}

inline void on_weight_apply_done(void * owner,
                                 const emel::model::weight_loader::events::apply_done &) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.weight.apply_done = true;
  fixture.weight.apply_error = false;
}

inline void on_weight_apply_error(void * owner,
                                  const emel::model::weight_loader::events::apply_error & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.weight.apply_error = true;
  fixture.weight.err = ev.err;
}

inline void on_load_done(void * owner, const emel::model::loader::events::load_done & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.load.done = true;
  fixture.load.error = false;
  fixture.load.err = emel::error::cast(emel::model::loader::error::none);
  fixture.load.bytes_total = ev.bytes_total;
  fixture.load.bytes_done = ev.bytes_done;
  fixture.load.used_mmap = ev.used_mmap;
}

inline void on_load_error(void * owner, const emel::model::loader::events::load_error & ev) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  fixture.load.error = true;
  fixture.load.err = ev.err;
}

inline emel::error::type map_gguf_error(const emel::error::type err) {
  using gguf_error = emel::gguf::loader::error;
  using model_error = emel::model::loader::error;

  switch (err) {
    case emel::error::cast(gguf_error::none):
      return emel::error::cast(model_error::none);
    case emel::error::cast(gguf_error::invalid_request):
      return emel::error::cast(model_error::invalid_request);
    case emel::error::cast(gguf_error::model_invalid):
      return emel::error::cast(model_error::model_invalid);
    case emel::error::cast(gguf_error::capacity):
      return emel::error::cast(model_error::backend_error);
    case emel::error::cast(gguf_error::parse_failed):
      return emel::error::cast(model_error::parse_failed);
    case emel::error::cast(gguf_error::internal_error):
      return emel::error::cast(model_error::internal_error);
    case emel::error::cast(gguf_error::untracked):
    default:
      return emel::error::cast(model_error::untracked);
  }
}

inline emel::error::type map_weight_loader_error(const emel::error::type err) {
  using model_error = emel::model::loader::error;
  using weight_error = emel::model::weight_loader::error;

  switch (err) {
    case emel::error::cast(weight_error::none):
      return emel::error::cast(model_error::none);
    case emel::error::cast(weight_error::invalid_request):
      return emel::error::cast(model_error::invalid_request);
    case emel::error::cast(weight_error::capacity):
    case emel::error::cast(weight_error::backend_error):
    case emel::error::cast(weight_error::out_of_memory):
      return emel::error::cast(model_error::backend_error);
    case emel::error::cast(weight_error::model_invalid):
      return emel::error::cast(model_error::model_invalid);
    case emel::error::cast(weight_error::internal_error):
      return emel::error::cast(model_error::internal_error);
    case emel::error::cast(weight_error::untracked):
    default:
      return emel::error::cast(model_error::untracked);
  }
}

inline bool copy_tensor_names(const std::span<const uint8_t> file_image,
                              emel::model::data & model_data) {
  model_data.name_bytes_used = 0u;

  for (uint32_t i = 0u; i < model_data.n_tensors; ++i) {
    auto & tensor = model_data.tensors[i];
    const size_t name_offset = static_cast<size_t>(tensor.name_offset);
    const size_t name_length = static_cast<size_t>(tensor.name_length);
    if (name_offset + name_length > file_image.size() ||
        model_data.name_bytes_used + name_length > model_data.name_storage.size()) {
      return false;
    }

    const uint32_t copied_offset = model_data.name_bytes_used;
    if (name_length > 0u) {
      std::memcpy(model_data.name_storage.data() + copied_offset,
                  file_image.data() + name_offset,
                  name_length);
    }

    model_data.name_bytes_used += static_cast<uint32_t>(name_length);
    tensor.name_offset = copied_offset;
  }
  return true;
}

inline bool try_parse_block_index(const std::string_view name, int32_t & block_index_out) {
  constexpr std::string_view k_prefix = "blk.";
  if (!name.starts_with(k_prefix)) {
    return false;
  }

  size_t cursor = k_prefix.size();
  if (cursor >= name.size()) {
    return false;
  }

  int32_t value = 0;
  bool saw_digit = false;
  while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
    saw_digit = true;
    value = (value * 10) + static_cast<int32_t>(name[cursor] - '0');
    ++cursor;
  }

  if (!saw_digit || cursor >= name.size() || name[cursor] != '.') {
    return false;
  }

  block_index_out = value;
  return true;
}

inline bool try_parse_sortformer_layer_index(const std::string_view name,
                                             const std::string_view prefix,
                                             int32_t & layer_index_out) {
  if (!name.starts_with(prefix) || prefix.size() >= name.size()) {
    return false;
  }

  size_t cursor = prefix.size();
  int32_t value = 0;
  bool saw_digit = false;
  while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
    saw_digit = true;
    value = (value * 10) + static_cast<int32_t>(name[cursor] - '0');
    ++cursor;
  }

  if (!saw_digit || cursor >= name.size() || name[cursor] != '.') {
    return false;
  }

  layer_index_out = value;
  return true;
}

inline emel::error::type populate_model_metadata(model_fixture & fixture,
                                                 emel::model::data & model_data) {
  return emel::model::detail::load_hparams_from_gguf(kv_binding_from_fixture(fixture), model_data)
             ? emel::error::cast(emel::model::loader::error::none)
             : emel::error::cast(emel::model::loader::error::model_invalid);
}

inline emel::error::type run_emel_parse_model(void * owner,
                                              const emel::model::loader::event::load & req) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  if (req.file_image == nullptr || req.file_size == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  const std::span<const uint8_t> file_image{
      static_cast<const uint8_t *>(req.file_image),
      static_cast<size_t>(req.file_size),
  };

  reset_gguf_capture(fixture);
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&fixture, on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{&fixture, on_probe_error};
  const emel::gguf::loader::event::probe probe_ev{
      file_image,
      requirements,
      probe_done_cb,
      probe_error_cb,
  };
  if (!fixture.gguf_loader.process_event(probe_ev) || !fixture.gguf.probe_done ||
      fixture.gguf.probe_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  if (requirements.tensor_count > static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const uint64_t arena_bytes =
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  fixture.kv_arena.resize(static_cast<size_t>(arena_bytes));
  fixture.kv_entries.resize(requirements.kv_count);

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::bind_done_fn bind_done_cb{&fixture, on_bind_done};
  const emel::gguf::loader::event::bind_error_fn bind_error_cb{&fixture, on_bind_error};
  const emel::gguf::loader::event::bind_storage bind_ev{
      std::span<uint8_t>{fixture.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  requirements.tensor_count},
      bind_done_cb,
      bind_error_cb,
  };
  if (!fixture.gguf_loader.process_event(bind_ev) || !fixture.gguf.bind_done ||
      fixture.gguf.bind_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::parse_done_fn parse_done_cb{&fixture, on_parse_done};
  const emel::gguf::loader::event::parse_error_fn parse_error_cb{&fixture, on_parse_error};
  const emel::gguf::loader::event::parse parse_ev{
      file_image,
      parse_done_cb,
      parse_error_cb,
  };
  if (!fixture.gguf_loader.process_event(parse_ev) || !fixture.gguf.parse_done ||
      fixture.gguf.parse_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  req.model_data.n_tensors = requirements.tensor_count;
  if (!copy_tensor_names(file_image, req.model_data)) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  return populate_model_metadata(fixture, req.model_data);
}

inline emel::error::type run_emel_load_weights(void * owner,
                                               const emel::model::loader::event::load & req,
                                               uint64_t & bytes_total,
                                               uint64_t & bytes_done,
                                               bool & used_mmap) {
  auto & fixture = *static_cast<model_fixture *>(owner);
  if (req.model_data.n_tensors == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  fixture.effect_requests.resize(req.model_data.n_tensors);
  fixture.effect_results.resize(req.model_data.n_tensors);

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::bind_storage bind_ev{
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  req.model_data.n_tensors},
  };
  bind_ev.on_done = {&fixture, on_weight_bind_done};
  bind_ev.on_error = {&fixture, on_weight_bind_error};
  if (!fixture.weight_loader.process_event(bind_ev) || !fixture.weight.bind_done ||
      fixture.weight.bind_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::plan_load plan_ev{
      std::span<emel::model::weight_loader::effect_request>{fixture.effect_requests},
  };
  plan_ev.on_done = {&fixture, on_weight_plan_done};
  plan_ev.on_error = {&fixture, on_weight_plan_error};
  if (!fixture.weight_loader.process_event(plan_ev) || !fixture.weight.plan_done ||
      fixture.weight.plan_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  const uint32_t effect_count = fixture.weight.effect_count;
  for (uint32_t i = 0u; i < effect_count; ++i) {
    fixture.effect_results[i] = emel::model::weight_loader::effect_result{
        .kind = fixture.effect_requests[i].kind,
        .handle = fixture.effect_requests[i].target,
        .err = emel::error::cast(emel::model::weight_loader::error::none),
    };
  }

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::apply_effect_results apply_ev{
      std::span<const emel::model::weight_loader::effect_result>{fixture.effect_results.data(),
                                                                 effect_count},
  };
  apply_ev.on_done = {&fixture, on_weight_apply_done};
  apply_ev.on_error = {&fixture, on_weight_apply_error};
  if (!fixture.weight_loader.process_event(apply_ev) || !fixture.weight.apply_done ||
      fixture.weight.apply_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  req.model_data.weights_data = req.file_image;
  req.model_data.weights_size = req.file_size;
  req.model_data.weights_mapped = false;
  req.model_data.weights_split_count = 1u;
  req.model_data.weights_split_offsets[0] = 0u;
  req.model_data.weights_split_sizes[0] = req.file_size;
  bytes_total = req.file_size;
  bytes_done = req.file_size;
  used_mmap = false;
  return emel::error::cast(emel::model::loader::error::none);
}

inline emel::error::type run_emel_map_layers(void *,
                                             const emel::model::loader::event::load & req) {
  int32_t max_block_index = -1;
  for (uint32_t i = 0u; i < req.model_data.n_tensors; ++i) {
    const auto name = emel::model::tensor_name_view(req.model_data, req.model_data.tensors[i]);
    int32_t block_index = -1;
    if ((emel::model::try_parse_block_index(name, block_index) ||
         try_parse_sortformer_layer_index(name, "enc.l", block_index) ||
         try_parse_sortformer_layer_index(name, "te.l", block_index)) &&
        block_index > max_block_index) {
      max_block_index = block_index;
    }
  }

  if (max_block_index >= 0) {
    req.model_data.n_layers = max_block_index + 1;
    return emel::error::cast(emel::model::loader::error::none);
  }
  if (req.model_data.params.n_layer > 0) {
    req.model_data.n_layers = req.model_data.params.n_layer;
    return emel::error::cast(emel::model::loader::error::none);
  }
  return emel::error::cast(emel::model::loader::error::model_invalid);
}

inline emel::error::type run_emel_validate_structure(
    void *,
    const emel::model::loader::event::load & req) {
  if (req.model_data.n_tensors == 0u || req.model_data.n_layers <= 0 ||
      req.model_data.weights_data == nullptr || req.model_data.weights_size == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return emel::error::cast(emel::model::loader::error::none);
}

inline emel::error::type run_emel_validate_architecture(
    void *,
    const emel::model::loader::event::load & req) {
  return emel::model::validate_execution_contract(req.model_data);
}

inline bool prepare(model_fixture & fixture) {
  fixture.ready = false;
  fixture.contract = {};
  if (!read_file_bytes(resolve_repo_path(k_model_rel_path), fixture.file_bytes)) {
    return false;
  }

  reset_load_capture(fixture);
  emel::model::loader::event::parse_model_fn parse_model{&fixture, run_emel_parse_model};
  emel::model::loader::event::load load_ev{fixture.model, parse_model};
  load_ev.model_path = k_model_rel_path;
  load_ev.file_image = fixture.file_bytes.data();
  load_ev.file_size = fixture.file_bytes.size();
  load_ev.load_weights = {&fixture, run_emel_load_weights};
  load_ev.map_layers = {nullptr, run_emel_map_layers};
  load_ev.validate_structure = {nullptr, run_emel_validate_structure};
  load_ev.validate_architecture_impl = {nullptr, run_emel_validate_architecture};
  load_ev.on_done = {&fixture, on_load_done};
  load_ev.on_error = {&fixture, on_load_error};
  if (!fixture.model_loader.process_event(load_ev) || !fixture.load.done || fixture.load.error) {
    return false;
  }

  if (emel::model::sortformer::detail::build_execution_contract(fixture.model, fixture.contract) !=
      emel::error::cast(emel::model::loader::error::none)) {
    fixture.contract = {};
    return false;
  }

  fixture.ready = fixture.contract.model == &fixture.model;
  return fixture.ready;
}

inline bool find_chunk(const std::span<const uint8_t> bytes,
                       const std::array<char, 4> chunk_id,
                       std::span<const uint8_t> & chunk_out) {
  size_t cursor = 12u;
  while (cursor + 8u <= bytes.size()) {
    const std::span<const uint8_t> id = bytes.subspan(cursor, 4u);
    const uint32_t chunk_size = read_u32_le(bytes.subspan(cursor + 4u, 4u));
    const size_t payload_begin = cursor + 8u;
    const size_t payload_end = payload_begin + static_cast<size_t>(chunk_size);
    if (payload_end > bytes.size()) {
      return false;
    }
    if (std::memcmp(id.data(), chunk_id.data(), chunk_id.size()) == 0) {
      chunk_out = bytes.subspan(payload_begin, static_cast<size_t>(chunk_size));
      return true;
    }
    cursor = payload_end + (chunk_size & 1u);
  }
  return false;
}

inline bool prepare(pcm_fixture & fixture) {
  fixture.ready = false;
  fixture.pcm.clear();
  fixture.sample_rate = 0;

  std::vector<uint8_t> file_bytes = {};
  if (!read_file_bytes(resolve_repo_path(k_audio_rel_path), file_bytes) || file_bytes.size() < 12u) {
    return false;
  }

  const std::span<const uint8_t> bytes{file_bytes.data(), file_bytes.size()};
  if (std::memcmp(bytes.data(), "RIFF", 4u) != 0 || std::memcmp(bytes.data() + 8u, "WAVE", 4u) !=
                                                   0) {
    return false;
  }

  std::span<const uint8_t> fmt_chunk = {};
  std::span<const uint8_t> data_chunk = {};
  if (!find_chunk(bytes, {'f', 'm', 't', ' '}, fmt_chunk) ||
      !find_chunk(bytes, {'d', 'a', 't', 'a'}, data_chunk) || fmt_chunk.size() < 16u ||
      (data_chunk.size() % 2u) != 0u) {
    return false;
  }

  const uint16_t audio_format = static_cast<uint16_t>(read_u32_le(fmt_chunk.first(4u)) & 0xffffu);
  const uint16_t channel_count =
      static_cast<uint16_t>(read_u32_le(fmt_chunk.subspan(2u, 4u)) & 0xffffu);
  const uint32_t sample_rate = read_u32_le(fmt_chunk.subspan(4u, 4u));
  const uint16_t bits_per_sample =
      static_cast<uint16_t>(read_u32_le(fmt_chunk.subspan(14u, 4u)) & 0xffffu);
  if (audio_format != 1u || channel_count != 1u ||
      sample_rate != static_cast<uint32_t>(request_detail::k_sample_rate) ||
      bits_per_sample != 16u) {
    return false;
  }

  fixture.pcm.resize(data_chunk.size() / 2u);
  for (size_t i = 0u; i < fixture.pcm.size(); ++i) {
    const uint16_t lo = data_chunk[(i * 2u)];
    const uint16_t hi = data_chunk[(i * 2u) + 1u];
    const int16_t sample = static_cast<int16_t>((hi << 8u) | lo);
    fixture.pcm[i] = static_cast<float>(sample) / 32768.0f;
  }

  fixture.sample_rate = static_cast<int32_t>(sample_rate);
  fixture.ready = !fixture.pcm.empty();
  return fixture.ready;
}

inline bool prepare(expected_output_baseline & baseline) {
  baseline = {};

  std::vector<uint8_t> file_bytes = {};
  if (!read_file_bytes(resolve_repo_path(k_baseline_rel_path), file_bytes) || file_bytes.empty()) {
    return false;
  }

  std::string file_text(reinterpret_cast<const char *>(file_bytes.data()), file_bytes.size());
  size_t cursor = 0u;
  while (cursor < file_text.size()) {
    const size_t line_end = file_text.find('\n', cursor);
    const std::string_view line =
        line_end == std::string::npos
            ? std::string_view{file_text.data() + cursor, file_text.size() - cursor}
            : std::string_view{file_text.data() + cursor, line_end - cursor};

    if (const size_t equals = line.find('='); equals != std::string::npos) {
      const std::string_view key = line.substr(0u, equals);
      const std::string value(line.substr(equals + 1u));
      if (key == "segment_count") {
        baseline.segment_count = std::atoi(value.c_str());
      } else if (key == "output_checksum") {
        baseline.output_checksum = static_cast<std::uint64_t>(std::strtoull(value.c_str(),
                                                                            nullptr,
                                                                            10));
      }
    }

    if (line_end == std::string::npos) {
      break;
    }
    cursor = line_end + 1u;
  }

  baseline.ready = baseline.segment_count > 0 && baseline.output_checksum != 0u;
  return baseline.ready;
}

inline emel::model::sortformer::detail::execution_contract make_contract(
    const emel::model::data & model) noexcept {
  emel::model::sortformer::detail::execution_contract contract = {};
  (void) emel::model::sortformer::detail::build_execution_contract(model, contract);
  return contract;
}

inline std::uint64_t compute_checksum(std::span<const output_detail::segment_record> segments,
                                      const int32_t segment_count) noexcept {
  std::uint64_t checksum = 1469598103934665603ull;
  for (int32_t i = 0; i < segment_count; ++i) {
    const auto & segment = segments[static_cast<size_t>(i)];
    checksum ^= static_cast<std::uint64_t>(segment.speaker + 1);
    checksum *= 1099511628211ull;
    checksum ^= static_cast<std::uint64_t>(segment.start_frame + 1);
    checksum *= 1099511628211ull;
    checksum ^= static_cast<std::uint64_t>(segment.end_frame + 1);
    checksum *= 1099511628211ull;
  }
  return checksum;
}

struct run_result {
  std::vector<float> probabilities =
      std::vector<float>(static_cast<size_t>(pipeline_detail::k_required_probability_value_count));
  std::array<output_detail::segment_record, pipeline_detail::k_max_segment_count> segments = {};
  int32_t frame_count = -1;
  int32_t probability_count = -1;
  int32_t segment_count = -1;
  emel::error::type err = pipeline_detail::to_error(pipeline::error::unexpected);
  bool accepted = false;
};

inline void reset(run_result & result) noexcept {
  std::fill(result.probabilities.begin(), result.probabilities.end(), 0.0f);
  std::fill(result.segments.begin(), result.segments.end(), output_detail::segment_record{});
  result.frame_count = -1;
  result.probability_count = -1;
  result.segment_count = -1;
  result.err = pipeline_detail::to_error(pipeline::error::unexpected);
  result.accepted = false;
}

inline pipeline::event::run make_run_event(
    const emel::model::sortformer::detail::execution_contract & contract,
    std::span<const float> pcm,
    const int32_t sample_rate,
    std::span<float> probabilities,
    std::span<output_detail::segment_record> segments,
    int32_t & frame_count,
    int32_t & probability_count,
    int32_t & segment_count,
    emel::error::type & err) noexcept {
  return pipeline::event::run{contract,
                              pcm,
                              sample_rate,
                              request_detail::k_channel_count,
                              probabilities,
                              segments,
                              frame_count,
                              probability_count,
                              segment_count,
                              err};
}

inline bool run_pipeline(
    pipeline::sm & machine,
    const emel::model::sortformer::detail::execution_contract & contract,
    std::span<const float> pcm,
    const int32_t sample_rate,
    run_result & result) {
  reset(result);
  auto request = make_run_event(contract,
                                pcm,
                                sample_rate,
                                result.probabilities,
                                result.segments,
                                result.frame_count,
                                result.probability_count,
                                result.segment_count,
                                result.err);
  result.accepted = machine.process_event(request);
  return result.accepted;
}

}  // namespace emel::bench::diarization::sortformer_fixture
