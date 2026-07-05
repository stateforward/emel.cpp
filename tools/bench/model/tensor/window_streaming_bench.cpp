// weight_streaming suite: compares three ways of running a model whose
// weights may not fit in RAM, all against the LFM2.5-230M-Q8_0 maintained
// fixture at max_tokens 32 (plus a first-token case):
//
//   emel_window - EMEL-owned streaming: the tensor window actor maps the GGUF
//                 as the copy source and streams per-layer weight slots ahead
//                 of decode under an explicit byte budget.
//   emel_mmap   - EMEL-owned baseline: identical whole-file mapping bound in
//                 passthrough (no slots); the resident route demand-pages via
//                 the OS page cache.
//   llama_mmap  - reference lane only: llama.cpp with its default mmap
//                 loading at 8 threads. It drives the comparison result and
//                 never touches the EMEL lanes, per the split-lane contract.
//
// The suite is opt-in: rows are emitted only when EMEL_BENCH_WEIGHT_STREAMING
// is set (scripts/bench.sh --suite=weight_streaming sets it), so default
// snapshot runs carry no baseline requirement. EMEL_BENCH_MEMORY_MAX (bytes)
// selects the emel_window budget; unset defaults to 35% of the fixture size.

#include "bench_cases.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/memory/view.hpp"
#include "emel/gguf/loader/any.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/io/loader/sm.hpp"
#include "emel/io/mmap/sm.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/io/source/any.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/tensor/sm.hpp"
#include "emel/model/tensor/window/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/generator/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

#include "../../../generation_formatter_contract.hpp"

// Reference lane only: llama.cpp drives the comparison result and never
// touches the EMEL lanes, per the split-lane benchmark contract.
#include "llama.h"

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace {

using emel::bench::config;
using emel::bench::measure_case;
using emel::bench::result;

namespace window = emel::model::tensor::window;

constexpr std::string_view k_fixture_rel = "tests/models/LFM2.5-230M-Q8_0.gguf";
constexpr std::string_view k_model_id = "LFM2.5-230M-Q8_0.gguf";
constexpr std::string_view k_prompt = "hello";
constexpr int32_t k_max_tokens = 32;
constexpr size_t k_output_capacity = 8192u;

bool weight_streaming_enabled() {
  const char *flag = std::getenv("EMEL_BENCH_WEIGHT_STREAMING");
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

std::filesystem::path fixture_path() { return bench_root_path() / k_fixture_rel; }

void report_missing_fixture() {
  std::fprintf(stderr,
               "warning: skipping missing weight_streaming fixture %.*s\n",
               static_cast<int>(k_fixture_rel.size()), k_fixture_rel.data());
}

uint64_t memory_max_bytes_from_env() {
  const char *value = std::getenv("EMEL_BENCH_MEMORY_MAX");
  if (value == nullptr || value[0] == '\0') {
    return 0u;
  }
  return static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
}

//------------------------------------------------------------------------------//
// Linux-only informational counters (zero elsewhere).

uint64_t process_read_bytes() {
#if defined(__linux__)
  std::FILE *io = std::fopen("/proc/self/io", "r");
  if (io == nullptr) {
    return 0u;
  }
  char line[128] = {};
  uint64_t value = 0u;
  while (std::fgets(line, sizeof(line), io) != nullptr) {
    if (std::sscanf(line, "read_bytes: %" SCNu64, &value) == 1) {
      break;
    }
  }
  std::fclose(io);
  return value;
#else
  return 0u;
#endif
}

uint64_t process_major_faults() {
#if defined(__linux__)
  struct rusage usage = {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0u;
  }
  return static_cast<uint64_t>(usage.ru_majflt);
#else
  return 0u;
#endif
}

//------------------------------------------------------------------------------//
// EMEL fixture: self-contained copy of the generation-bench load pipeline
// (gguf probe/bind/parse via the gguf loader, then the model loader with the
// caller-provided image), kept bench-local per the shared-lane conventions.

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

struct emel_fixture {
  emel::model::data model_data = {};
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  uint32_t gguf_tensor_count = 0u;
  std::vector<emel::model::tensor::effect_request> effect_requests = {};
  std::vector<emel::model::tensor::effect_result> effect_results = {};
  std::vector<emel::io::event::tensor_load_span> io_load_spans = {};
  emel::gguf::loader::sm gguf_loader = {};
  emel::io::read::sm io_read = {};
  emel::io::staged_read::sm io_staged_read = {};
  emel::io::loader::sm io_loader{
      {.io_read = &io_read, .io_staged_read = &io_staged_read}};
  emel::model::tensor::sm tensor_loader = {};
  emel::model::loader::sm model_loader = {};
  gguf_capture gguf = {};
  load_capture load = {};
};

void on_probe_done(void *owner, const emel::gguf::loader::events::probe_done &ev) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.probe_done = true;
  fixture.gguf.requirements = ev.requirements_out;
}

void on_probe_error(void *owner, const emel::gguf::loader::events::probe_error &ev) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.probe_error = true;
  fixture.gguf.err = ev.err;
}

void on_bind_done(void *owner, const emel::gguf::loader::events::bind_done &) {
  static_cast<emel_fixture *>(owner)->gguf.bind_done = true;
}

void on_bind_error(void *owner, const emel::gguf::loader::events::bind_error &ev) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.bind_error = true;
  fixture.gguf.err = ev.err;
}

void on_parse_done(void *owner, const emel::gguf::loader::events::parse_done &) {
  static_cast<emel_fixture *>(owner)->gguf.parse_done = true;
}

void on_parse_error(void *owner, const emel::gguf::loader::events::parse_error &ev) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  fixture.gguf.parse_error = true;
  fixture.gguf.err = ev.err;
}

void on_load_done(void *owner, const emel::model::loader::events::load_done &) {
  static_cast<emel_fixture *>(owner)->load.done = true;
}

void on_load_error(void *owner, const emel::model::loader::events::load_error &ev) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  fixture.load.error = true;
  fixture.load.err = ev.err;
}

bool copy_tensor_names(const std::span<const uint8_t> file_image,
                       emel::model::data &model_data) {
  model_data.name_bytes_used = 0u;
  for (uint32_t i = 0u; i < model_data.n_tensors; ++i) {
    auto &tensor = model_data.tensors[i];
    const size_t name_offset = static_cast<size_t>(tensor.name_offset);
    const size_t name_length = static_cast<size_t>(tensor.name_length);
    if (name_offset + name_length > file_image.size() ||
        model_data.name_bytes_used + name_length >
            model_data.name_storage.size()) {
      return false;
    }
    const uint32_t copied_offset = model_data.name_bytes_used;
    if (name_length > 0u) {
      std::memcpy(model_data.name_storage.data() + copied_offset,
                  file_image.data() + name_offset, name_length);
    }
    model_data.name_bytes_used += static_cast<uint32_t>(name_length);
    tensor.name_offset = copied_offset;
  }
  return true;
}

bool prebind_gguf_storage(emel_fixture &fixture,
                          const std::span<const uint8_t> file_image) {
  fixture.gguf = {};
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&fixture,
                                                               on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{
      &fixture, on_probe_error};
  const emel::gguf::loader::event::probe probe_ev{
      file_image,
      requirements,
      probe_done_cb,
      probe_error_cb,
  };
  if (!fixture.gguf_loader.process_event(probe_ev) || !fixture.gguf.probe_done ||
      fixture.gguf.probe_error) {
    return false;
  }
  if (requirements.tensor_count >
      static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return false;
  }
  const uint64_t arena_bytes =
      emel::gguf::loader::required_kv_arena_bytes(requirements);
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
  auto &fixture = *static_cast<emel_fixture *>(owner);
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
  const emel::gguf::loader::event::parse parse_ev{
      file_image,
      parse_done_cb,
      parse_error_cb,
  };
  if (!fixture.gguf_loader.process_event(parse_ev) || !fixture.gguf.parse_done ||
      fixture.gguf.parse_error) {
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
  int32_t max_block_index = -1;
  for (uint32_t i = 0u; i < req.model_data.n_tensors; ++i) {
    int32_t block_index = -1;
    if (emel::model::try_parse_block_index(
            emel::model::tensor_name_view(req.model_data,
                                          req.model_data.tensors[i]),
            block_index) &&
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

emel::error::type run_validate_structure(void *,
                                         const emel::model::loader::event::load &req) {
  if (req.model_data.n_tensors == 0u || req.model_data.n_layers <= 0 ||
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

bool load_model_from_image(emel_fixture &fixture, const std::string &model_path,
                           const void *file_image, const uint64_t file_size) {
  fixture.load = {};
  // model::data is multi-megabyte: reset in place, never via a stack temporary.
  std::destroy_at(&fixture.model_data);
  std::construct_at(&fixture.model_data);
  fixture.effect_requests.resize(emel::model::data::k_max_tensors);
  fixture.effect_results.resize(emel::model::data::k_max_tensors);
  fixture.io_load_spans.resize(emel::model::data::k_max_tensors);

  emel::model::loader::event::parse_model_fn parse_model{&fixture,
                                                         run_parse_model};
  emel::model::loader::event::load load_ev{fixture.model_data, parse_model};
  load_ev.model_path = model_path;
  load_ev.file_image = file_image;
  load_ev.file_size = file_size;
  load_ev.tensor_loader = &fixture.tensor_loader;
  load_ev.effect_requests = std::span{fixture.effect_requests};
  load_ev.effect_results = std::span{fixture.effect_results};
  load_ev.io_load_spans = std::span<emel::io::event::tensor_load_span>{
      fixture.io_load_spans.data(), fixture.io_load_spans.size()};
  load_ev.map_layers = {nullptr, run_map_layers};
  load_ev.validate_structure = {nullptr, run_validate_structure};
  load_ev.validate_architecture_impl = {nullptr, run_validate_architecture};
  load_ev.on_done = {&fixture, on_load_done};
  load_ev.on_error = {&fixture, on_load_error};
  if (!fixture.model_loader.process_event(load_ev) || !fixture.load.done ||
      fixture.load.error) {
    return false;
  }
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena.data(),
                                        fixture.kv_arena.size()},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{
              fixture.kv_entries.data(), fixture.kv_entries.size()},
  };
  if (!emel::model::detail::load_vocab_from_gguf(binding,
                                                 fixture.model_data.vocab_data)) {
    return false;
  }
  fixture.model_data.params.n_vocab =
      static_cast<int32_t>(fixture.model_data.vocab_data.n_tokens);
  return true;
}


//------------------------------------------------------------------------------//
// Formatter binding: resolve the fixture's chat template contract so the
// conditioner formats prompts exactly like the maintained generation bench.

std::string_view kv_key_view(const emel_fixture &fixture,
                             const emel::gguf::loader::kv_entry &entry) {
  if (static_cast<size_t>(entry.key_offset) +
          static_cast<size_t>(entry.key_length) >
      fixture.kv_arena.size()) {
    return {};
  }
  return std::string_view{
      reinterpret_cast<const char *>(fixture.kv_arena.data() + entry.key_offset),
      entry.key_length,
  };
}

std::span<const uint8_t> kv_value_view(const emel_fixture &fixture,
                                       const emel::gguf::loader::kv_entry &entry) {
  if (static_cast<size_t>(entry.value_offset) +
          static_cast<size_t>(entry.value_length) >
      fixture.kv_arena.size()) {
    return {};
  }
  return std::span<const uint8_t>{fixture.kv_arena.data() + entry.value_offset,
                                  entry.value_length};
}

uint64_t read_u64_le_bytes(const std::span<const uint8_t> bytes) {
  uint64_t value = 0u;
  for (size_t i = 0u; i < sizeof(uint64_t); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (i * 8u);
  }
  return value;
}

bool decode_kv_string(const emel_fixture &fixture,
                      const emel::gguf::loader::kv_entry &entry,
                      std::string_view &value_out) {
  namespace constants = emel::gguf::loader::constants;
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  if (entry.value_type != constants::gguf_type_string ||
      bytes.size() < sizeof(uint64_t)) {
    return false;
  }
  const uint64_t length = read_u64_le_bytes(bytes.first(sizeof(uint64_t)));
  if (length > bytes.size() - sizeof(uint64_t)) {
    return false;
  }
  value_out = std::string_view{
      reinterpret_cast<const char *>(bytes.data() + sizeof(uint64_t)),
      static_cast<size_t>(length)};
  return true;
}

emel::tools::generation_formatter_contract::formatter_binding
resolve_formatter_binding(const emel_fixture &fixture) {
  std::string_view primary_template = {};
  for (const auto &entry : fixture.kv_entries) {
    if (kv_key_view(fixture, entry) == "tokenizer.chat_template") {
      (void)decode_kv_string(fixture, entry, primary_template);
      break;
    }
  }
  uint32_t named_template_count = 0u;
  for (const auto &entry : fixture.kv_entries) {
    const std::string_view key = kv_key_view(fixture, entry);
    if (key.starts_with("tokenizer.chat_template.") &&
        key != "tokenizer.chat_template") {
      named_template_count += 1u;
    }
  }
  return emel::tools::generation_formatter_contract::
      resolve_primary_template_binding(primary_template, named_template_count);
}

//------------------------------------------------------------------------------//
// Streaming extents: per layer, one extent per present matmul role, in the
// canonical role order the generator's streamed rebase walks.

constexpr std::array<std::string_view, 9> k_stream_role_names = {
    "attn_q.weight",           "attn_k.weight",
    "attn_v.weight",           "attn_output.weight",
    "ffn_gate.weight",         "ffn_down.weight",
    "ffn_up.weight",           "shortconv.in_proj.weight",
    "shortconv.out_proj.weight",
};

const emel::model::data::tensor_record *
find_layer_tensor(const emel::model::data &model_data, const int32_t layer,
                  const std::string_view role) {
  const std::string name =
      "blk." + std::to_string(layer) + "." + std::string(role);
  for (uint32_t idx = 0u; idx < model_data.n_tensors; ++idx) {
    const auto &tensor = model_data.tensors[idx];
    if (emel::model::tensor_name_view(model_data, tensor) == name) {
      return &tensor;
    }
  }
  return nullptr;
}

struct stream_extents {
  std::vector<window::detail::weight_extent> extents = {};
  std::vector<uint16_t> layer_weight_counts = {};
};

bool build_stream_extents(const emel::model::data &model_data,
                          stream_extents &out) {
  out.extents.clear();
  out.layer_weight_counts.clear();
  int32_t tensor_id = 0;
  for (int32_t layer = 0; layer < model_data.n_layers; ++layer) {
    uint16_t count = 0u;
    for (const std::string_view role : k_stream_role_names) {
      const auto *tensor = find_layer_tensor(model_data, layer, role);
      if (tensor == nullptr) {
        continue;
      }
      out.extents.push_back(window::detail::weight_extent{
          .tensor_id = tensor_id++,
          .file_offset = tensor->file_offset,
          .byte_size = tensor->data_size,
          .slot_offset = 0u,
      });
      count += 1u;
    }
    if (count == 0u) {
      return false;
    }
    out.layer_weight_counts.push_back(count);
  }
  return !out.extents.empty();
}

//------------------------------------------------------------------------------//
// Window rig (owner side of the tensor window actor).

struct window_rig {
  emel::io::mmap::sm io_mmap{};
  std::array<emel::io::staged_read::sm, window::detail::k_max_window_slots>
      io_staged{};
  window::detail::stream_io_pool io_pool{};
  window::sm machine;
  bool streaming_active = false;
  const void *source_base = nullptr;
  uint64_t source_bytes = 0u;

  window_rig() : machine{make_context()} {}

  window::context make_context() noexcept {
    window::context ctx{};
    ctx.io_mmap = &io_mmap;
    ctx.io_staged = io_staged;
    ctx.io_pool = &io_pool;
    return ctx;
  }

  static void on_bind_done(void *object,
                           const window::events::bind_window_done &ev) noexcept {
    auto &rig = *static_cast<window_rig *>(object);
    rig.streaming_active = ev.streaming_active;
    rig.source_base = ev.source_base;
    rig.source_bytes = ev.source_bytes;
  }

  // Owner-side slot arena, allocated here before any machine dispatch (the
  // machine never allocates): four slots at the largest aligned layer span.
  std::vector<uint8_t> slot_arena = {};

  bool bind(const std::string &file_path, const uint64_t file_size,
            const stream_extents &extents, const uint64_t budget_bytes) {
    constexpr uint64_t k_align = window::detail::k_slot_alignment_bytes;
    uint64_t max_layer_bytes = 0u;
    size_t cursor = 0u;
    for (const uint16_t count : extents.layer_weight_counts) {
      uint64_t layer_bytes = 0u;
      for (uint16_t index = 0; index < count; ++index) {
        layer_bytes += (extents.extents[cursor].byte_size + (k_align - 1u)) &
                       ~(k_align - 1u);
        ++cursor;
      }
      max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
    }
    slot_arena.assign(4u * max_layer_bytes + k_align, 0u);
    const auto raw = reinterpret_cast<uintptr_t>(slot_arena.data());
    const uintptr_t aligned =
        (raw + (k_align - 1u)) & ~static_cast<uintptr_t>(k_align - 1u);
    const std::span<uint8_t> storage{reinterpret_cast<uint8_t *>(aligned),
                                     slot_arena.size() -
                                         static_cast<size_t>(aligned - raw)};

    const window::event::bind_window_request request{
        .file_path = file_path,
        .file_size_bytes = file_size,
        .extents = extents.extents,
        .layer_weight_counts = extents.layer_weight_counts,
        .budget_bytes = budget_bytes,
        .slot_storage = storage,
        .window_slots = 4u,
        .prefetch_depth = 2u,
        .stage_chunk_bytes = window::detail::k_default_stream_chunk_bytes,
    };
    window::event::bind_window bind_request{request};
    bind_request.on_done = {this, &window_rig::on_bind_done};
    return machine.process_event(bind_request);
  }
};

//------------------------------------------------------------------------------//
// EMEL generation session (raw prompt, preselected argmax; formatter_mode=raw
// on both comparison sides so the lanes stay comparable without the chat
// template machinery).

struct initialize_capture {
  bool done = false;
  bool error = false;
};

struct generation_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = 0u;
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct emel_session {
  emel::model::data model_data = {};
  emel::tools::generation_formatter_contract::formatter_binding formatter_binding = {};
  emel::text::tokenizer::sm tokenizer = {};
  emel::text::conditioner::sm conditioner = {};
  std::unique_ptr<emel::text::generator::sm> generator = {};
  initialize_capture initialize = {};
  generation_capture generation = {};
};

void on_initialize_done(void *owner,
                        const emel::text::generator::events::initialize_done &) {
  static_cast<emel_session *>(owner)->initialize.done = true;
}

void on_initialize_error(void *owner,
                         const emel::text::generator::events::initialize_error &) {
  static_cast<emel_session *>(owner)->initialize.error = true;
}

void on_generation_done(void *owner,
                        const emel::text::generator::events::generation_done &ev) {
  auto &session = *static_cast<emel_session *>(owner);
  session.generation.done = true;
  session.generation.tokens_generated = ev.tokens_generated;
  session.generation.output_length = ev.output_length;
}

void on_generation_error(void *owner,
                         const emel::text::generator::events::generation_error &ev) {
  auto &session = *static_cast<emel_session *>(owner);
  session.generation.error = true;
  session.generation.err = ev.err;
}

bool tokenizer_bind_dispatch(void *tokenizer_sm,
                             const emel::text::tokenizer::event::bind &ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(void *tokenizer_sm,
                                 const emel::text::tokenizer::event::tokenize &ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

emel::text::tokenizer::preprocessor::preprocessor_kind
session_preprocessor_variant(const emel::model::data &model_data) {
  using preprocessor_kind = emel::text::tokenizer::preprocessor::preprocessor_kind;
  using tokenizer_model = emel::model::data::tokenizer_model;
  switch (model_data.vocab_data.tokenizer_model_id) {
  case tokenizer_model::SPM:
    return preprocessor_kind::spm;
  case tokenizer_model::BPE:
    return preprocessor_kind::bpe;
  case tokenizer_model::WPM:
    return preprocessor_kind::wpm;
  case tokenizer_model::UGM:
    return preprocessor_kind::ugm;
  case tokenizer_model::RWKV:
    return preprocessor_kind::rwkv;
  case tokenizer_model::PLAMO2:
    return preprocessor_kind::plamo2;
  default:
    return preprocessor_kind::fallback;
  }
}

emel::text::encoders::encoder_kind
session_encoder_variant(const emel::model::data &model_data) {
  using encoder_kind = emel::text::encoders::encoder_kind;
  using tokenizer_model = emel::model::data::tokenizer_model;
  switch (model_data.vocab_data.tokenizer_model_id) {
  case tokenizer_model::SPM:
    return encoder_kind::spm;
  case tokenizer_model::BPE:
    return encoder_kind::bpe;
  case tokenizer_model::WPM:
    return encoder_kind::wpm;
  case tokenizer_model::UGM:
    return encoder_kind::ugm;
  case tokenizer_model::RWKV:
    return encoder_kind::rwkv;
  case tokenizer_model::PLAMO2:
    return encoder_kind::plamo2;
  default:
    return encoder_kind::fallback;
  }
}

void prepare_session(const emel_fixture &fixture, emel_session &session,
                     window_rig *rig) {
  session.model_data = fixture.model_data;
  session.formatter_binding = resolve_formatter_binding(fixture);
  if (rig != nullptr) {
    session.generator = std::make_unique<emel::text::generator::sm>(
        session.model_data, session.conditioner, rig->machine,
        rig->streaming_active, session.formatter_binding.formatter_ctx,
        session.formatter_binding.format_prompt);
  } else {
    session.generator = std::make_unique<emel::text::generator::sm>(
        session.model_data, session.conditioner,
        session.formatter_binding.formatter_ctx,
        session.formatter_binding.format_prompt);
  }
}

bool initialize_session(emel_session &session, const int32_t max_tokens) {
  const int32_t prompt_capacity = 64;
  const int32_t decode_capacity = std::max<int32_t>(4, max_tokens);
  session.initialize = {};
  emel::text::generator::event::initialize request{
      &session.tokenizer,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      std::span<emel::logits::sampler::fn>{},
  };
  request.preprocessor_variant = session_preprocessor_variant(session.model_data);
  request.encoder_variant = session_encoder_variant(session.model_data);
  request.add_special = false;
  request.parse_special = false;
  // The window lane streams via the preselected-argmax streamed decode rows:
  // an active tensor window routes the serial slot-consuming variants, so
  // this mode is valid for both the mmap-resident and the window lane.
  request.selection_mode =
      emel::text::generator::selection_mode::preselected_argmax;
  request.max_prompt_tokens = prompt_capacity;
  request.max_generated_tokens = decode_capacity;
  // Whole memory-contract blocks for the session budget, capped at the model
  // context window per the emel::memory::view geometry contract.
  request.max_blocks = std::max<int32_t>(
      1, emel::memory::view::blocks_for_tokens(
             emel::memory::view::DEFAULT_BLOCK_TOKENS,
             std::min<int32_t>(prompt_capacity + decode_capacity,
                               session.model_data.params.n_ctx)));
  request.block_tokens = emel::memory::view::DEFAULT_BLOCK_TOKENS;
  request.strip_leading_space = false;
  request.on_done = {&session, on_initialize_done};
  request.on_error = {&session, on_initialize_error};
  return session.generator->process_event(request) && session.initialize.done &&
         !session.initialize.error;
}

struct generation_run {
  std::array<char, k_output_capacity> output = {};
  size_t output_length = 0u;
  int32_t tokens_generated = 0;
};

bool run_generate(emel_session &session, const int32_t max_tokens,
                  generation_run &run) {
  session.generation = {};
  run.output_length = 0u;
  std::array<emel::text::formatter::chat_message, 1> messages = {
      emel::text::formatter::chat_message{.role = "user", .content = k_prompt},
  };
  emel::text::generator::event::generate request{
      std::span<const emel::text::formatter::chat_message>{messages},
      max_tokens,
      std::span<char>{run.output.data(), run.output.size()},
      run.output_length,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.on_done = {&session, on_generation_done};
  request.on_error = {&session, on_generation_error};
  if (!session.generator->process_event(request) || !session.generation.done ||
      session.generation.error) {
    return false;
  }
  run.tokens_generated = session.generation.tokens_generated;
  run.output_length = session.generation.output_length;
  return true;
}

//------------------------------------------------------------------------------//
// EMEL lanes.

struct metadata_mapping {
  emel::io::mmap::sm io_mmap{};
  uint32_t handle = emel::io::mmap::k_invalid_mapping_handle;
  const void *base = nullptr;
  uint64_t bytes = 0u;

  static void on_done(void *object,
                      const emel::io::mmap::events::map_tensor_done &ev) noexcept {
    auto &self = *static_cast<metadata_mapping *>(object);
    self.handle = ev.handle;
    self.base = ev.buffer;
    self.bytes = ev.buffer_bytes;
  }

  bool map(const std::string &path, const uint64_t file_size) {
    const emel::io::mmap::event::map_tensor_request request{
        .tensor_id = -3,
        .file_index = 0u,
        .file_offset = 0u,
        .byte_size = file_size,
        .file_path = path,
    };
    emel::io::mmap::event::map_tensor map_request{request};
    map_request.on_done = {this, &metadata_mapping::on_done};
    return io_mmap.process_event(map_request) && base != nullptr;
  }

  ~metadata_mapping() {
    if (handle != emel::io::mmap::k_invalid_mapping_handle) {
      emel::io::mmap::event::release_mapping release{-3, handle};
      (void)io_mmap.process_event(release);
    }
  }
};

struct emel_lane_setup {
  std::unique_ptr<emel_fixture> fixture = {};
  std::unique_ptr<window_rig> rig = {};
  uint64_t file_size = 0u;
  uint64_t budget_bytes = 0u;
  bool ok = false;
};

// Loads metadata from a transient heap image, builds the streaming extents,
// binds the window (mapping the GGUF as the copy source), then reloads the
// model against the mapping so every tensor record points at paged bytes.
void prepare_emel_lane(emel_lane_setup &setup, const bool streamed) {
  const std::string model_path = fixture_path().string();
  setup.fixture = std::make_unique<emel_fixture>();
  emel_fixture &fixture = *setup.fixture;

  std::error_code size_error{};
  const uint64_t file_size = std::filesystem::file_size(fixture_path(), size_error);
  if (size_error || file_size == 0u) {
    return;
  }
  setup.file_size = file_size;

  // Metadata pass through a transient whole-file mapping: no anonymous copy of
  // the model, so a cgroup MemoryMax cap only has to cover session buffers.
  metadata_mapping metadata{};
  if (!metadata.map(model_path, file_size)) {
    return;
  }
  const std::span<const uint8_t> metadata_image{
      static_cast<const uint8_t *>(metadata.base),
      static_cast<size_t>(metadata.bytes)};
  if (!prebind_gguf_storage(fixture, metadata_image)) {
    return;
  }
  if (!load_model_from_image(fixture, model_path, metadata.base,
                             metadata.bytes)) {
    return;
  }

  stream_extents extents{};
  if (!build_stream_extents(fixture.model_data, extents)) {
    return;
  }

  const uint64_t requested = memory_max_bytes_from_env();
  setup.budget_bytes =
      streamed ? (requested != 0u ? requested : (setup.file_size * 35u) / 100u)
               : 0u;  // passthrough: unlimited budget, no slots

  setup.rig = std::make_unique<window_rig>();
  if (!setup.rig->bind(model_path, setup.file_size, extents,
                       setup.budget_bytes)) {
    return;
  }
  if (streamed != setup.rig->streaming_active) {
    return;
  }

  // Rebind everything to the window's mapping (the metadata mapping is
  // released when this function returns).
  if (!load_model_from_image(fixture, model_path, setup.rig->source_base,
                             setup.rig->source_bytes)) {
    return;
  }
  setup.ok = true;
}

void append_emel_lane_cases(std::vector<result> &results, const config &cfg,
                            const bool streamed) {
  const char *lane_tag = streamed ? "emel_window" : "emel_mmap";
  if (!std::filesystem::exists(fixture_path())) {
    report_missing_fixture();
    return;
  }

  emel_lane_setup setup{};
  prepare_emel_lane(setup, streamed);
  if (!setup.ok) {
    std::fprintf(stderr, "warning: weight_streaming %s lane setup failed\n",
                 lane_tag);
    return;
  }

  const std::array<int32_t, 2> token_cases = {1, k_max_tokens};
  for (const int32_t max_tokens : token_cases) {
    auto session = std::make_unique<emel_session>();
    prepare_session(*setup.fixture, *session,
                    streamed ? setup.rig.get() : nullptr);
    if (!initialize_session(*session, max_tokens)) {
      std::fprintf(stderr,
                   "warning: weight_streaming %s session init failed\n",
                   lane_tag);
      return;
    }

    generation_run latest{};
    uint64_t bytes_read_delta = 0u;
    uint64_t major_faults_delta = 0u;
    volatile size_t sink = 0u;
    auto fn = [&]() {
      const uint64_t read_before = process_read_bytes();
      const uint64_t faults_before = process_major_faults();
      generation_run run{};
      if (!run_generate(*session, max_tokens, run)) {
        std::fprintf(stderr,
                     "error: weight_streaming %s generate failed (err=%u done=%d)\n",
                     lane_tag, session->generation.err,
                     session->generation.done ? 1 : 0);
        std::exit(1);
      }
      bytes_read_delta = process_read_bytes() - read_before;
      major_faults_delta = process_major_faults() - faults_before;
      latest = run;
      sink ^= run.output_length;
    };

    const std::string case_name =
        std::string("weight_streaming/") + (streamed ? "window" : "mmap") +
        "/lfm2_5_230m_q8_0_prompt_hello_" +
        (max_tokens == 1 ? "first_token"
                         : ("max_tokens_" + std::to_string(max_tokens)));
    results.push_back(measure_case(case_name.c_str(), cfg, fn));
    result &record = results.back();
    record.compare_group = case_name;
    record.lane = lane_tag;
    record.backend_id =
        streamed ? "emel.tensor_window_stream" : "emel.mmap_resident";
    record.backend_language = "cpp";
    record.comparison_mode = "streaming_vs_reference";
    record.model_id = std::string(k_model_id);
    record.fixture_id = std::string(k_fixture_rel);
    record.prompt_id = "hello";
    record.formatter_mode = "chat_template";
    record.max_output_tokens = static_cast<uint64_t>(max_tokens);
    record.comparable = true;
    record.output_tokens = static_cast<uint64_t>(latest.tokens_generated);
    record.output_bytes = static_cast<uint64_t>(latest.output_length);
    record.note = "memory_max=" + std::to_string(setup.budget_bytes) +
                  " streaming_active=" +
                  (setup.rig->streaming_active ? "1" : "0") +
                  " bytes_read=" + std::to_string(bytes_read_delta) +
                  " major_faults=" + std::to_string(major_faults_delta);
  }
}

//------------------------------------------------------------------------------//
// Reference lane (llama.cpp, mmap loading, 8 threads).

using llama_model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using llama_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

int32_t reference_threads() {
  if (const char *threads_env = std::getenv("EMEL_BENCH_REFERENCE_THREADS");
      threads_env != nullptr && threads_env[0] != '\0') {
    return std::max<int32_t>(1, std::atoi(threads_env));
  }
  return 8;
}

void append_reference_lane_cases(std::vector<result> &results,
                                 const config &cfg) {
  if (!std::filesystem::exists(fixture_path())) {
    report_missing_fixture();
    return;
  }

  const std::string model_path = fixture_path().string();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.use_mmap = true;
  llama_model_ptr model{
      llama_model_load_from_file(model_path.c_str(), model_params),
      llama_model_free};
  if (model == nullptr) {
    std::fprintf(stderr,
                 "warning: weight_streaming llama_mmap model load failed\n");
    return;
  }
  const llama_vocab *vocab = llama_model_get_vocab(model.get());
  const int32_t vocab_size = vocab != nullptr ? llama_vocab_n_tokens(vocab) : 0;
  if (vocab_size <= 0) {
    return;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 0;
  ctx_params.n_batch = 512;
  ctx_params.n_ubatch = 512;
  ctx_params.n_seq_max = 1;
  ctx_params.n_threads = reference_threads();
  ctx_params.n_threads_batch = reference_threads();
  ctx_params.embeddings = false;
  llama_context_ptr ctx{llama_init_from_model(model.get(), ctx_params),
                        llama_free};
  if (ctx == nullptr) {
    return;
  }

  // The EMEL lanes format the chat prompt through the fixture's template
  // contract (with an assistant generation prompt), so the reference lane
  // must tokenize the same formatted stream - raw k_prompt would put a
  // different token workload in the same compare group. Mirrors the
  // maintained generation bench's reference prompt path exactly, including
  // tokenizing with add_special/parse_special disabled.
  const auto reference_formatter = emel::tools::generation_formatter_contract::
      resolve_reference_formatter_info(model.get());
  std::string formatted_prompt = {};
  if (!emel::tools::generation_formatter_contract::
          format_reference_single_user_prompt(reference_formatter, k_prompt,
                                              formatted_prompt)) {
    std::fprintf(stderr,
                 "warning: weight_streaming reference formatter unsupported\n");
    return;
  }
  std::array<llama_token, 256> prompt_tokens = {};
  const int32_t prompt_count = llama_tokenize(
      vocab, formatted_prompt.data(),
      static_cast<int32_t>(formatted_prompt.size()), prompt_tokens.data(),
      static_cast<int32_t>(prompt_tokens.size()),
      /*add_special=*/false, /*parse_special=*/false);
  if (prompt_count <= 0) {
    return;
  }

  const std::array<int32_t, 2> token_cases = {1, k_max_tokens};
  for (const int32_t max_tokens : token_cases) {
    uint64_t bytes_read_delta = 0u;
    uint64_t major_faults_delta = 0u;
    int32_t latest_tokens = 0;
    volatile int32_t sink = 0;
    auto fn = [&]() {
      const uint64_t read_before = process_read_bytes();
      const uint64_t faults_before = process_major_faults();
      llama_memory_clear(llama_get_memory(ctx.get()), false);
      llama_batch prompt_batch =
          llama_batch_get_one(prompt_tokens.data(), prompt_count);
      if (llama_decode(ctx.get(), prompt_batch) != 0) {
        std::fprintf(stderr,
                     "error: weight_streaming llama_mmap decode failed\n");
        std::exit(1);
      }
      int32_t generated = 0;
      for (int32_t step = 0; step < max_tokens; ++step) {
        const float *logits = llama_get_logits(ctx.get());
        if (logits == nullptr) {
          std::exit(1);
        }
        int32_t best = 0;
        for (int32_t idx = 1; idx < vocab_size; ++idx) {
          if (logits[idx] > logits[best]) {
            best = idx;
          }
        }
        generated += 1;
        if (llama_vocab_is_eog(vocab, static_cast<llama_token>(best))) {
          break;
        }
        llama_token next_token = static_cast<llama_token>(best);
        llama_batch decode_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx.get(), decode_batch) != 0) {
          std::exit(1);
        }
      }
      latest_tokens = generated;
      bytes_read_delta = process_read_bytes() - read_before;
      major_faults_delta = process_major_faults() - faults_before;
      sink ^= generated;
    };

    const std::string suffix =
        std::string("/lfm2_5_230m_q8_0_prompt_hello_") +
        (max_tokens == 1 ? "first_token"
                         : ("max_tokens_" + std::to_string(max_tokens)));
    const std::string window_name = "weight_streaming/window" + suffix;
    result record = measure_case(window_name.c_str(), cfg, fn);
    record.compare_group = window_name;
    record.lane = "llama";
    record.backend_id = "llama.cpp";
    record.backend_language = "cpp";
    record.comparison_mode = "streaming_vs_reference";
    record.model_id = std::string(k_model_id);
    record.fixture_id = std::string(k_fixture_rel);
    record.prompt_id = "hello";
    record.formatter_mode = "chat_template";
    record.max_output_tokens = static_cast<uint64_t>(max_tokens);
    record.comparable = true;
    record.output_tokens = static_cast<uint64_t>(latest_tokens);
    record.note = std::string("threads=") + std::to_string(reference_threads()) +
                  " use_mmap=1 bytes_read=" + std::to_string(bytes_read_delta) +
                  " major_faults=" + std::to_string(major_faults_delta);
    // The same reference measurement baselines both EMEL lanes.
    result mmap_twin = record;
    mmap_twin.name = "weight_streaming/mmap" + suffix;
    mmap_twin.compare_group = mmap_twin.name;
    results.push_back(record);
    results.push_back(mmap_twin);
  }
}

}  // namespace

namespace emel::bench {

void append_emel_weight_streaming_cases(std::vector<result> &results,
                                        const config &cfg) {
  if (!weight_streaming_enabled()) {
    return;
  }
  append_emel_lane_cases(results, cfg, /*streamed=*/false);
  append_emel_lane_cases(results, cfg, /*streamed=*/true);
}

void append_reference_weight_streaming_cases(std::vector<result> &results,
                                             const config &cfg) {
  if (!weight_streaming_enabled()) {
    return;
  }
  append_reference_lane_cases(results, cfg);
}

}  // namespace emel::bench
