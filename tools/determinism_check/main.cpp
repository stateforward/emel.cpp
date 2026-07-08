// Determinism gate for the maintained text generation path
// (docs/determinism.md).
//
// Loads one maintained GGUF fixture through the production machines
// (gguf/model loaders, tokenizer, generator) and runs the SAME generate()
// request repeatedly, in both maintained selection modes:
//   - preselected_argmax: the production fused argmax decode route
//   - sample_logits: the materialized logits route with an externally
//     injected argmax sampler callback that also FNV-1a hashes every
//     materialized logits vector (the per-step logits checksum) and every
//     selected token id (the token-stream checksum)
// Every repeat, and a run on a freshly constructed session, must produce a
// bitwise-identical token stream (output text + token count) and identical
// logits checksums. Any mismatch prints the differing evidence and exits
// non-zero. scripts/check_determinism.sh additionally runs this binary in two
// separate processes and compares the emitted `determinism_evidence` lines,
// proving cross-process (fresh address space) determinism.
//
// Everything is driven through public state-machine process_event(...) calls;
// no kernel detail.hpp/actions.hpp helper is touched. The load recipe is the
// architecture-generic path (load_hparams_from_gguf), so it works for lfm2.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "../bench/model_load_strategy.hpp"
#include "../generation_route_policy.hpp"
#include "emel/error/error.hpp"
#include "emel/gguf/loader/any.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/io/events.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/io/source/any.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/generation/any.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/tensor/errors.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"
#include "emel/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/generator/errors.hpp"
#include "emel/text/generator/events.hpp"
#include "emel/text/generator/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

constexpr size_t k_output_capacity = 8192u;
constexpr uint64_t k_fnv_offset = 1469598103934665603ull;
constexpr uint64_t k_fnv_prime = 1099511628211ull;

uint64_t fnv1a_bytes(uint64_t hash, const void *data, const size_t size) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t idx = 0; idx < size; ++idx) {
    hash ^= static_cast<uint64_t>(bytes[idx]);
    hash *= k_fnv_prime;
  }
  return hash;
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

struct load_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
};

struct initialize_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::text::generator::error::none);
};

struct generation_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::text::generator::error::none);
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct emel_fixture {
  std::unique_ptr<emel::model::data> model_data =
      std::make_unique<emel::model::data>();
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  uint64_t gguf_tensor_data_bytes = 0u;
  std::vector<uint8_t> read_copy_storage = {};
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

// Externally injected sampler seam: deterministic argmax selection (strict
// `>` scan, ties resolve to the lowest token id, matching the runtime's fused
// argmax) plus FNV-1a checksums over the materialized logits bytes and the
// selected token ids. This is the only RNG-capable seam in the runtime and it
// is caller-owned, so the harness both selects tokens and observes logits
// through the public sampler contract.
struct logits_probe {
  uint64_t logits_hash = k_fnv_offset;
  uint64_t token_hash = k_fnv_offset;
  int32_t steps = 0;

  void reset() {
    logits_hash = k_fnv_offset;
    token_hash = k_fnv_offset;
    steps = 0;
  }
};

emel::error::type probe_argmax_sampler(void *owner, int32_t &candidate_ids,
                                       float &candidate_scores,
                                       int32_t &candidate_count,
                                       int32_t &selected_token_out) {
  auto &probe = *static_cast<logits_probe *>(owner);
  const int32_t *ids = &candidate_ids;
  const float *scores = &candidate_scores;
  int32_t best_index = 0;
  float best_score = scores[0];
  for (int32_t idx = 1; idx < candidate_count; ++idx) {
    if (scores[idx] > best_score) {
      best_score = scores[idx];
      best_index = idx;
    }
  }
  selected_token_out = ids[best_index];
  probe.logits_hash =
      fnv1a_bytes(probe.logits_hash, scores,
                  static_cast<size_t>(candidate_count) * sizeof(float));
  probe.token_hash = fnv1a_bytes(probe.token_hash, &selected_token_out,
                                 sizeof(selected_token_out));
  probe.steps += 1;
  return emel::error::cast(emel::logits::sampler::error::none);
}

// One generation session: its own tokenizer/conditioner/generator, output
// buffer, and logits probe, referencing the single shared model_data.
struct session {
  emel::text::tokenizer::sm tokenizer = {};
  emel::text::conditioner::sm conditioner = {};
  emel::model::generation::contract generation_contract = {};
  emel::text::generator::matmul::lane_pool<7u, 128u, 1048576u> parallel_matmul_lanes = {};
  std::unique_ptr<emel::text::generator::sm> generator = {};
  std::array<emel::logits::sampler::fn, 1> samplers = {};
  logits_probe probe = {};
  initialize_capture initialize = {};
  generation_capture generation = {};
  std::array<char, k_output_capacity> output = {};
  size_t output_length = 0u;
};

// Per-run evidence that must be bitwise identical across repeats, fresh
// sessions, and separate processes.
struct run_evidence {
  int32_t tokens_generated = 0;
  uint64_t output_fnv = 0u;
  uint64_t logits_fnv = 0u;
  uint64_t token_fnv = 0u;
  int32_t sampler_steps = 0;
  std::string output_text = {};

  bool operator==(const run_evidence &other) const {
    return tokens_generated == other.tokens_generated &&
           output_fnv == other.output_fnv && logits_fnv == other.logits_fnv &&
           token_fnv == other.token_fnv &&
           sampler_steps == other.sampler_steps &&
           output_text == other.output_text;
  }
};

// ---- gguf / model-loader callbacks (architecture-generic) ----

void on_probe_done(void *owner,
                   const emel::gguf::loader::events::probe_done &ev) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.gguf.probe_done = true;
  f.gguf.probe_error = false;
  f.gguf.requirements = ev.requirements_out;
}
void on_probe_error(void *owner,
                    const emel::gguf::loader::events::probe_error &ev) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.gguf.probe_error = true;
  f.gguf.err = ev.err;
}
void on_bind_done(void *owner, const emel::gguf::loader::events::bind_done &) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.gguf.bind_done = true;
  f.gguf.bind_error = false;
}
void on_bind_error(void *owner,
                   const emel::gguf::loader::events::bind_error &ev) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.gguf.bind_error = true;
  f.gguf.err = ev.err;
}
void on_parse_done(void *owner,
                   const emel::gguf::loader::events::parse_done &) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.gguf.parse_done = true;
  f.gguf.parse_error = false;
}
void on_parse_error(void *owner,
                    const emel::gguf::loader::events::parse_error &ev) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.gguf.parse_error = true;
  f.gguf.err = ev.err;
}
void on_load_done(void *owner, const emel::model::loader::events::load_done &) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.load.done = true;
  f.load.error = false;
  f.load.err = emel::error::cast(emel::model::loader::error::none);
}
void on_load_error(void *owner,
                   const emel::model::loader::events::load_error &ev) {
  auto &f = *static_cast<emel_fixture *>(owner);
  f.load.error = true;
  f.load.err = ev.err;
}

void on_initialize_done(
    void *owner, const emel::text::generator::events::initialize_done &) {
  auto &s = *static_cast<session *>(owner);
  s.initialize.done = true;
  s.initialize.error = false;
}
void on_initialize_error(
    void *owner, const emel::text::generator::events::initialize_error &ev) {
  auto &s = *static_cast<session *>(owner);
  s.initialize.error = true;
  s.initialize.err = ev.err;
}
void on_generation_done(
    void *owner, const emel::text::generator::events::generation_done &ev) {
  auto &s = *static_cast<session *>(owner);
  s.generation.done = true;
  s.generation.error = false;
  s.generation.tokens_generated = ev.tokens_generated;
  s.generation.output_length = ev.output_length;
}
void on_generation_error(
    void *owner, const emel::text::generator::events::generation_error &ev) {
  auto &s = *static_cast<session *>(owner);
  s.generation.error = true;
  s.generation.err = ev.err;
  s.generation.tokens_generated = ev.tokens_generated;
  s.generation.output_length = ev.output_length;
}

bool tokenizer_bind_dispatch(void *tokenizer_sm,
                             const emel::text::tokenizer::event::bind &ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)
      ->process_event(ev);
}
bool tokenizer_tokenize_dispatch(
    void *tokenizer_sm, const emel::text::tokenizer::event::tokenize &ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)
      ->process_event(ev);
}

emel::error::type map_gguf_error(const emel::error::type err) {
  using ge = emel::gguf::loader::error;
  using me = emel::model::loader::error;
  switch (err) {
  case emel::error::cast(ge::none):
    return emel::error::cast(me::none);
  case emel::error::cast(ge::invalid_request):
    return emel::error::cast(me::invalid_request);
  case emel::error::cast(ge::model_invalid):
    return emel::error::cast(me::model_invalid);
  case emel::error::cast(ge::capacity):
    return emel::error::cast(me::backend_error);
  case emel::error::cast(ge::parse_failed):
    return emel::error::cast(me::parse_failed);
  case emel::error::cast(ge::internal_error):
    return emel::error::cast(me::internal_error);
  default:
    return emel::error::cast(me::untracked);
  }
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

emel::model::detail::kv_binding
kv_binding_from_fixture(const emel_fixture &fixture) {
  return emel::model::detail::kv_binding{
      .arena = std::span<const uint8_t>{fixture.kv_arena.data(),
                                        fixture.kv_arena.size()},
      .entries =
          std::span<const emel::gguf::loader::kv_entry>{
              fixture.kv_entries.data(), fixture.kv_entries.size()},
  };
}

// Architecture-generic metadata: resolves architecture from the gguf and calls
// the registered load_hparams (lfm2, qwen3, llama, ...). No model-family gate.
emel::error::type populate_model_metadata(const emel_fixture &fixture,
                                          emel::model::data &model_data) {
  return emel::model::detail::load_hparams_from_gguf(
             kv_binding_from_fixture(fixture), model_data)
             ? emel::error::cast(emel::model::loader::error::none)
             : emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type prebind_emel_gguf_storage(emel_fixture &fixture) {
  if (fixture.file_bytes.empty()) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }
  const std::span<const uint8_t> file_image{fixture.file_bytes.data(),
                                            fixture.file_bytes.size()};
  fixture.gguf_tensor_count = 0u;
  fixture.gguf = {};
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&fixture,
                                                               on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{
      &fixture, on_probe_error};
  const emel::gguf::loader::event::probe probe_ev{
      file_image, requirements, probe_done_cb, probe_error_cb};
  if (!fixture.gguf_loader.process_event(probe_ev) ||
      !fixture.gguf.probe_done || fixture.gguf.probe_error) {
    return map_gguf_error(fixture.gguf.err);
  }
  if (requirements.tensor_count >
      static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  const uint64_t arena_bytes =
      emel::gguf::loader::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }
  fixture.kv_arena.resize(static_cast<size_t>(arena_bytes));
  fixture.kv_entries.resize(requirements.kv_count);
  fixture.gguf_tensor_count = requirements.tensor_count;
  fixture.gguf_tensor_data_bytes = requirements.tensor_data_bytes;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
run_emel_parse_model(void *owner, const emel::model::loader::event::load &req) {
  auto &fixture = *static_cast<emel_fixture *>(owner);
  if (req.file_image == nullptr || req.file_size == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }
  const std::span<const uint8_t> file_image{
      static_cast<const uint8_t *>(req.file_image),
      static_cast<size_t>(req.file_size)};

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
      bind_done_cb, bind_error_cb};
  if (!fixture.gguf_loader.process_event(bind_ev) || !fixture.gguf.bind_done ||
      fixture.gguf.bind_error) {
    return map_gguf_error(fixture.gguf.err);
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
    return map_gguf_error(fixture.gguf.err);
  }

  req.model_data.n_tensors = fixture.gguf_tensor_count;
  if (!copy_tensor_names(file_image, req.model_data)) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }
  return populate_model_metadata(fixture, req.model_data);
}

emel::error::type
run_emel_map_layers(void *, const emel::model::loader::event::load &req) {
  int32_t max_block_index = -1;
  for (uint32_t index = 0u; index < req.model_data.n_tensors; ++index) {
    int32_t block_index = -1;
    if (emel::model::try_parse_block_index(
            emel::model::tensor_name_view(req.model_data,
                                          req.model_data.tensors[index]),
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

emel::error::type
run_emel_validate_structure(void *,
                            const emel::model::loader::event::load &req) {
  if (req.model_data.n_tensors == 0u || req.model_data.n_layers <= 0 ||
      req.model_data.weights_data == nullptr ||
      req.model_data.weights_size == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
run_emel_validate_architecture(void *,
                               const emel::model::loader::event::load &req) {
  return emel::model::validate_execution_contract(req.model_data);
}

emel::text::tokenizer::preprocessor::preprocessor_kind
generation_preprocessor_variant(const emel::model::data &model_data) {
  using pk = emel::text::tokenizer::preprocessor::preprocessor_kind;
  using tm = emel::model::data::tokenizer_model;
  switch (model_data.vocab_data.tokenizer_model_id) {
  case tm::SPM:
    return pk::spm;
  case tm::BPE:
    return pk::bpe;
  case tm::WPM:
    return pk::wpm;
  case tm::UGM:
    return pk::ugm;
  case tm::RWKV:
    return pk::rwkv;
  case tm::PLAMO2:
    return pk::plamo2;
  default:
    return pk::fallback;
  }
}

emel::text::encoders::encoder_kind
generation_encoder_variant(const emel::model::data &model_data) {
  using ek = emel::text::encoders::encoder_kind;
  using tm = emel::model::data::tokenizer_model;
  switch (model_data.vocab_data.tokenizer_model_id) {
  case tm::SPM:
    return ek::spm;
  case tm::BPE:
    return ek::bpe;
  case tm::WPM:
    return ek::wpm;
  case tm::UGM:
    return ek::ugm;
  case tm::RWKV:
    return ek::rwkv;
  case tm::PLAMO2:
    return ek::plamo2;
  default:
    return ek::fallback;
  }
}

bool prepare_emel_fixture(emel_fixture &fixture,
                          const std::string &model_path) {
  if (emel::io::source::load_file_bytes(model_path, fixture.file_bytes) !=
      emel::error::cast(emel::io::read::error::none)) {
    std::fprintf(stderr, "load: source file load failed (%s)\n",
                 model_path.c_str());
    return false;
  }
  if (prebind_emel_gguf_storage(fixture) !=
      emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stderr, "load: prebind gguf failed\n");
    return false;
  }
  fixture.load = {};
  fixture.effect_requests.resize(emel::model::data::k_max_tensors);
  fixture.effect_results.resize(emel::model::data::k_max_tensors);
  fixture.io_load_spans.resize(emel::model::data::k_max_tensors);
  emel::model::loader::event::parse_model_fn parse_model{&fixture,
                                                         run_emel_parse_model};
  emel::model::loader::event::load load_ev{*fixture.model_data, parse_model};
  load_ev.model_path = model_path;
  load_ev.file_image = fixture.file_bytes.data();
  load_ev.file_size = fixture.file_bytes.size();
  load_ev.tensor_loader = &fixture.tensor_loader;
  load_ev.effect_requests = std::span{fixture.effect_requests};
  load_ev.effect_results = std::span{fixture.effect_results};
  load_ev.io_load_spans = std::span<emel::io::event::tensor_load_span>{
      fixture.io_load_spans.data(), fixture.io_load_spans.size()};
  emel::tools::bind_model_load_io_strategy(load_ev, fixture.io_loader);
  if (load_ev.io_strategy ==
          emel::io::loader::event::strategy_kind::read_copy ||
      load_ev.io_strategy ==
          emel::io::loader::event::strategy_kind::staged_read) {
    fixture.read_copy_storage.resize(
        static_cast<size_t>(fixture.gguf_tensor_data_bytes));
    load_ev.read_copy_storage = std::span<uint8_t>{fixture.read_copy_storage};
  }
  load_ev.map_layers = {nullptr, run_emel_map_layers};
  load_ev.validate_structure = {nullptr, run_emel_validate_structure};
  load_ev.validate_architecture_impl = {nullptr,
                                        run_emel_validate_architecture};
  load_ev.on_done = {&fixture, on_load_done};
  load_ev.on_error = {&fixture, on_load_error};
  if (!fixture.model_loader.process_event(load_ev) || !fixture.load.done ||
      fixture.load.error) {
    std::fprintf(stderr, "load: model_loader failed done=%d error=%d err=%d\n",
                 fixture.load.done ? 1 : 0, fixture.load.error ? 1 : 0,
                 fixture.load.err);
    return false;
  }
  return true;
}

bool initialize_session(session &s, const emel::model::data &model,
                        const emel::text::generator::selection_mode mode,
                        int32_t prompt_capacity, int32_t tokens) {
  const int32_t decode_capacity = std::max<int32_t>(4, tokens);
  const int32_t session_tokens =
      std::min<int32_t>(prompt_capacity + decode_capacity, model.params.n_ctx);
  const int32_t block_capacity = std::max<int32_t>(
      1, emel::memory::view::blocks_for_tokens(
             emel::memory::view::DEFAULT_BLOCK_TOKENS, session_tokens));
  s.initialize = {};
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  std::span<emel::logits::sampler::fn> sampler_span = {};
  if (mode == emel::text::generator::selection_mode::sample_logits) {
    s.samplers[0] = emel::logits::sampler::fn{&s.probe, probe_argmax_sampler};
    sampler_span = std::span<emel::logits::sampler::fn>{s.samplers};
  }
  emel::text::generator::event::initialize request{
      &s.tokenizer, tokenizer_bind_dispatch, tokenizer_tokenize_dispatch,
      sampler_span};
  request.preprocessor_variant = generation_preprocessor_variant(model);
  request.encoder_variant = generation_encoder_variant(model);
  request.add_special = false;
  request.parse_special = false;
  request.selection_mode = mode;
  request.max_prompt_tokens = prompt_capacity;
  request.max_generated_tokens = decode_capacity;
  request.max_blocks = block_capacity;
  request.block_tokens = emel::memory::view::DEFAULT_BLOCK_TOKENS;
  request.strip_leading_space = false;
  request.error_out = &error_out;
  request.on_done = {&s, on_initialize_done};
  request.on_error = {&s, on_initialize_error};
  const bool accepted = s.generator->process_event(request);
  return accepted && s.initialize.done && !s.initialize.error &&
         error_out == emel::error::cast(emel::text::generator::error::none);
}

bool run_generate(session &s, const std::string_view prompt, int32_t tokens,
                  run_evidence &evidence_out) {
  s.generation = {};
  s.output_length = 0u;
  s.probe.reset();
  emel::error::type error_out =
      emel::error::cast(emel::text::generator::error::none);
  std::array<emel::text::formatter::chat_message, 1> messages = {
      {{.role = "user", .content = prompt}}};
  emel::text::generator::event::generate request{
      std::span<const emel::text::formatter::chat_message>{messages}, tokens,
      std::span<char>{s.output}, s.output_length};
  request.add_generation_prompt = false;
  request.enable_thinking = false;
  request.error_out = &error_out;
  request.on_done = {&s, on_generation_done};
  request.on_error = {&s, on_generation_error};
  const bool accepted = s.generator->process_event(request);
  const bool ok =
      accepted && s.generation.done && !s.generation.error &&
      error_out == emel::error::cast(emel::text::generator::error::none);
  if (!ok) {
    return false;
  }
  evidence_out.tokens_generated = s.generation.tokens_generated;
  evidence_out.output_text.assign(s.output.data(), s.generation.output_length);
  evidence_out.output_fnv =
      fnv1a_bytes(k_fnv_offset, s.output.data(), s.generation.output_length);
  evidence_out.logits_fnv = s.probe.logits_hash;
  evidence_out.token_fnv = s.probe.token_hash;
  evidence_out.sampler_steps = s.probe.steps;
  return true;
}

std::unique_ptr<session>
make_session(const emel::model::data &model,
             const emel::text::generator::selection_mode mode,
             int32_t prompt_capacity, int32_t tokens) {
  auto s = std::make_unique<session>();
  if (emel::model::generation::build_contract(model, s->generation_contract) !=
      emel::error::cast(emel::model::loader::error::none)) {
    return nullptr;
  }
  const auto matmul_policy =
      emel::text::generator::matmul::make_auto_execution_policy(
          s->parallel_matmul_lanes);
  s->generator = std::make_unique<emel::text::generator::sm>(
      emel::text::generator::dependencies{
          .generation_contract = s->generation_contract,
          .conditioner = s->conditioner,
          .matmul_policy = matmul_policy,
          .runtime_policy =
              emel::tools::generation_route::make_current_runtime_policy(model),
          .formatter_ctx = nullptr,
          .format_prompt = emel::text::formatter::format_raw,
      });
  if (!initialize_session(*s, model, mode, prompt_capacity, tokens)) {
    return nullptr;
  }
  return s;
}

void print_evidence(const char *mode_name, const char *run_label,
                    const run_evidence &evidence) {
  std::printf("determinism_evidence mode=%s run=%s tokens_generated=%d "
              "output_fnv=0x%016llx logits_fnv=0x%016llx token_fnv=0x%016llx "
              "sampler_steps=%d\n",
              mode_name, run_label, evidence.tokens_generated,
              static_cast<unsigned long long>(evidence.output_fnv),
              static_cast<unsigned long long>(evidence.logits_fnv),
              static_cast<unsigned long long>(evidence.token_fnv),
              evidence.sampler_steps);
}

bool check_mode(const emel::model::data &model,
                const emel::text::generator::selection_mode mode,
                const char *mode_name, const std::string_view prompt,
                const int32_t prompt_capacity, const int32_t tokens,
                const int32_t repeats, run_evidence &reference_out) {
  auto persistent = make_session(model, mode, prompt_capacity, tokens);
  if (persistent == nullptr) {
    std::fprintf(stderr, "FAILED: initialize session mode=%s\n", mode_name);
    return false;
  }

  run_evidence reference = {};
  if (!run_generate(*persistent, prompt, tokens, reference)) {
    std::fprintf(stderr, "FAILED: generate mode=%s run=1\n", mode_name);
    return false;
  }
  print_evidence(mode_name, "1", reference);
  if (reference.tokens_generated <= 0 || reference.output_text.empty()) {
    std::fprintf(stderr,
                 "FAILED: mode=%s generated no output "
                 "(tokens_generated=%d output_bytes=%zu)\n",
                 mode_name, reference.tokens_generated,
                 reference.output_text.size());
    return false;
  }

  bool deterministic = true;
  for (int32_t repeat = 2; repeat <= repeats; ++repeat) {
    run_evidence evidence = {};
    if (!run_generate(*persistent, prompt, tokens, evidence)) {
      std::fprintf(stderr, "FAILED: generate mode=%s run=%d\n", mode_name,
                   repeat);
      return false;
    }
    char run_label[16] = {};
    std::snprintf(run_label, sizeof(run_label), "%d", repeat);
    print_evidence(mode_name, run_label, evidence);
    if (!(evidence == reference)) {
      std::fprintf(stderr,
                   "FAILED: mode=%s run=%d diverged from run=1 "
                   "(output \"%.128s\" vs \"%.128s\")\n",
                   mode_name, repeat, evidence.output_text.c_str(),
                   reference.output_text.c_str());
      deterministic = false;
    }
  }

  // A freshly constructed session (new generator, new KV blocks, new
  // buffers) must reproduce the persistent session's stream bit for bit.
  auto fresh = make_session(model, mode, prompt_capacity, tokens);
  if (fresh == nullptr) {
    std::fprintf(stderr, "FAILED: initialize fresh session mode=%s\n",
                 mode_name);
    return false;
  }
  run_evidence fresh_evidence = {};
  if (!run_generate(*fresh, prompt, tokens, fresh_evidence)) {
    std::fprintf(stderr, "FAILED: generate mode=%s run=fresh\n", mode_name);
    return false;
  }
  print_evidence(mode_name, "fresh", fresh_evidence);
  if (!(fresh_evidence == reference)) {
    std::fprintf(stderr,
                 "FAILED: mode=%s fresh session diverged from run=1 "
                 "(output \"%.128s\" vs \"%.128s\")\n",
                 mode_name, fresh_evidence.output_text.c_str(),
                 reference.output_text.c_str());
    deterministic = false;
  }

  reference_out = reference;
  return deterministic;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "Usage: %s <model_path> [repeats=3] [tokens=16] [prompt]\n",
                 argc > 0 ? argv[0] : "emel_determinism_check");
    return 1;
  }
  const std::string model_path = argv[1];
  const int32_t repeats =
      argc > 2 ? std::clamp(static_cast<int32_t>(std::atoi(argv[2])), 2, 64)
               : 3;
  const int32_t tokens = argc > 3 ? std::max(1, std::atoi(argv[3])) : 16;
  // Keep enough prompt tokens for the tool route policy to engage parallel
  // prefill; on models with n_embd >= 1024, parallel decode GEMV lanes engage
  // as well.
  const std::string_view prompt =
      argc > 4 ? std::string_view{argv[4]}
               : std::string_view{"The history of artificial intelligence "
                                  "began with early experiments in symbolic "
                                  "reasoning and"};
  const int32_t prompt_capacity = 64;

  auto fixture = std::make_unique<emel_fixture>();
  if (!prepare_emel_fixture(*fixture, model_path)) {
    std::fprintf(stderr, "FAILED: prepare_emel_fixture\n");
    return 1;
  }
  if (!emel::model::detail::load_vocab_from_gguf(
          kv_binding_from_fixture(*fixture), fixture->model_data->vocab_data)) {
    std::fprintf(stderr, "FAILED: load_vocab_from_gguf\n");
    return 1;
  }
  fixture->model_data->params.n_vocab =
      static_cast<int32_t>(fixture->model_data->vocab_data.n_tokens);

  const emel::model::data &model = *fixture->model_data;
  std::printf(
      "# model=%s arch=%.*s n_layer=%d n_embd=%d n_vocab=%d "
      "repeats=%d tokens=%d\n",
      model_path.c_str(),
      static_cast<int>(emel::model::architecture_name_view(model).size()),
      emel::model::architecture_name_view(model).data(), model.params.n_layer,
      model.params.n_embd, model.params.n_vocab, repeats, tokens);

  bool all_deterministic = true;
  run_evidence argmax_reference = {};
  run_evidence sampled_reference = {};
  all_deterministic =
      check_mode(model,
                 emel::text::generator::selection_mode::preselected_argmax,
                 "preselected_argmax", prompt, prompt_capacity, tokens, repeats,
                 argmax_reference) &&
      all_deterministic;
  all_deterministic =
      check_mode(model, emel::text::generator::selection_mode::sample_logits,
                 "sample_logits", prompt, prompt_capacity, tokens, repeats,
                 sampled_reference) &&
      all_deterministic;

  // Cross-route agreement: the fused argmax route and the materialized
  // sample_logits route (argmax sampler callback) must select the same
  // token stream and render the same output bytes.
  if (argmax_reference.tokens_generated != sampled_reference.tokens_generated ||
      argmax_reference.output_text != sampled_reference.output_text) {
    std::fprintf(stderr,
                 "FAILED: preselected_argmax and sample_logits routes "
                 "diverged (tokens %d vs %d, output \"%.128s\" vs "
                 "\"%.128s\")\n",
                 argmax_reference.tokens_generated,
                 sampled_reference.tokens_generated,
                 argmax_reference.output_text.c_str(),
                 sampled_reference.output_text.c_str());
    all_deterministic = false;
  }

  if (!all_deterministic) {
    std::printf("determinism_check: FAIL\n");
    return 1;
  }
  std::printf("determinism_check: PASS repeats=%d tokens=%d\n", repeats,
              tokens);
  return 0;
}
