// Decode-wavefront real-model eval.
//
// Measures what the decode-wavefront's scheduling mechanism (thread-pool
// inter-op parallelism) does on REAL LFM2.5 decode: N independent generators of
// the SAME loaded model (shared read-only weights, per-generator KV/activation
// state) each run a real generate(). We compare running the N generate() calls
// sequentially vs forking them across the wavefront's thread_pool_scheduler.
//
// Everything is driven through public state-machine process_event(...) calls;
// no kernel detail.hpp/actions.hpp helper is touched. The load recipe is the
// architecture-generic path (load_hparams_from_gguf), so it works for lfm2.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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

// The wavefront's lane pool (same type as
// src/emel/text/generator/decode_wavefront/context.hpp).
using lane_pool = emel::policy::thread_pool_scheduler<8u, 16u, 128u>;

constexpr size_t k_output_capacity = 8192u;

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

// One independent decode lane: its own tokenizer/conditioner/generator and
// output buffer, all referencing the single shared model_data.
struct lane_session {
  emel::text::tokenizer::sm tokenizer = {};
  emel::text::conditioner::sm conditioner = {};
  emel::model::generation::contract generation_contract = {};
  emel::text::generator::matmul::lane_pool<7u, 128u, 1048576u> parallel_matmul_lanes = {};
  std::unique_ptr<emel::text::generator::sm> generator = {};
  initialize_capture initialize = {};
  generation_capture generation = {};
  std::array<char, k_output_capacity> output = {};
  size_t output_length = 0u;
  std::string reference_text = {};
  bool last_ok = false;
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
  auto &s = *static_cast<lane_session *>(owner);
  s.initialize.done = true;
  s.initialize.error = false;
}
void on_initialize_error(
    void *owner, const emel::text::generator::events::initialize_error &ev) {
  auto &s = *static_cast<lane_session *>(owner);
  s.initialize.error = true;
  s.initialize.err = ev.err;
}
void on_generation_done(
    void *owner, const emel::text::generator::events::generation_done &ev) {
  auto &s = *static_cast<lane_session *>(owner);
  s.generation.done = true;
  s.generation.error = false;
  s.generation.tokens_generated = ev.tokens_generated;
  s.generation.output_length = ev.output_length;
}
void on_generation_error(
    void *owner, const emel::text::generator::events::generation_error &ev) {
  auto &s = *static_cast<lane_session *>(owner);
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

bool initialize_lane(lane_session &s, const emel::model::data &model,
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
  emel::text::generator::event::initialize request{
      &s.tokenizer, tokenizer_bind_dispatch, tokenizer_tokenize_dispatch,
      std::span<emel::logits::sampler::fn>{}};
  request.preprocessor_variant = generation_preprocessor_variant(model);
  request.encoder_variant = generation_encoder_variant(model);
  request.add_special = false;
  request.parse_special = false;
  request.selection_mode =
      emel::text::generator::selection_mode::preselected_argmax;
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

bool run_generate(lane_session &s, const std::string_view prompt,
                  int32_t tokens) {
  s.generation = {};
  s.output_length = 0u;
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
  s.last_ok =
      accepted && s.generation.done && !s.generation.error &&
      error_out == emel::error::cast(emel::text::generator::error::none);
  return s.last_ok;
}

std::string lane_output_text(const lane_session &s) {
  return std::string{s.output.data(), s.generation.output_length};
}

void run_sequential(const std::span<std::unique_ptr<lane_session>> active,
                    const std::string_view prompt, int32_t tokens) {
  for (auto &s : active) {
    run_generate(*s, prompt, tokens);
  }
}

void run_parallel(lane_pool &pool,
                  const std::span<std::unique_ptr<lane_session>> active,
                  const std::string_view prompt, int32_t tokens) {
  lane_pool::join_group group{};
  emel::policy::fork_join_start_gate gate{};
  size_t submitted_lanes = 0u;
  for (auto &s : active) {
    lane_session *lane = s.get();
    const bool submitted =
        pool.try_submit(group, [lane, prompt, tokens, &gate]() noexcept {
          gate.arrive_and_wait();
          run_generate(*lane, prompt, tokens);
        });
    submitted_lanes += submitted ? 1u : 0u;
  }
  gate.open_after_arrivals(submitted_lanes);
  (void)group.wait();
}

double ns_per_pass(const std::chrono::steady_clock::time_point t0,
                   const std::chrono::steady_clock::time_point t1,
                   int32_t iters) {
  const double ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  return ns / static_cast<double>(iters);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(
        stderr, "Usage: %s <model_path> [max_lanes=8] [tokens=32] [iters=20]\n",
        argc > 0 ? argv[0] : "emel_decode_wavefront_eval");
    return 1;
  }
  const std::string model_path = argv[1];
  const int32_t max_lanes =
      argc > 2 ? std::clamp(static_cast<int32_t>(std::atoi(argv[2])), 1, 8) : 8;
  const int32_t tokens = argc > 3 ? std::max(1, std::atoi(argv[3])) : 32;
  const int32_t iters = argc > 4 ? std::max(1, std::atoi(argv[4])) : 20;
  constexpr std::string_view k_prompt =
      "The history of artificial intelligence began";
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

  const emel::model::data &shared_model = *fixture->model_data;
  std::printf("# model=%s arch=%.*s n_layer=%d n_embd=%d n_head=%d "
              "n_head_kv=%d n_vocab=%d\n",
              model_path.c_str(),
              static_cast<int>(
                  emel::model::architecture_name_view(shared_model).size()),
              emel::model::architecture_name_view(shared_model).data(),
              shared_model.params.n_layer, shared_model.params.n_embd,
              shared_model.params.n_head, shared_model.params.n_head_kv,
              shared_model.params.n_vocab);

  // Build N independent generators sharing the one loaded model.
  std::vector<std::unique_ptr<lane_session>> sessions;
  sessions.reserve(static_cast<size_t>(max_lanes));
  for (int32_t lane = 0; lane < max_lanes; ++lane) {
    auto s = std::make_unique<lane_session>();
    if (emel::model::generation::build_contract(shared_model,
                                                s->generation_contract) !=
        emel::error::cast(emel::model::loader::error::none)) {
      std::fprintf(stderr, "FAILED: build_generation_contract lane=%d\n", lane);
      return 1;
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
                emel::tools::generation_route::make_current_runtime_policy(
                    shared_model),
            .formatter_ctx = nullptr,
            .format_prompt = emel::text::formatter::format_raw,
        });
    if (!initialize_lane(*s, shared_model, prompt_capacity, tokens)) {
      std::fprintf(stderr, "FAILED: initialize_lane lane=%d\n", lane);
      return 1;
    }
    sessions.push_back(std::move(s));
  }

  // Single-lane decode confirmation.
  if (!run_generate(*sessions[0], k_prompt, tokens)) {
    std::fprintf(stderr, "FAILED: single-lane generate\n");
    return 1;
  }
  std::printf("# single_lane_ok tokens_generated=%d output=\"%.*s\"\n",
              sessions[0]->generation.tokens_generated,
              static_cast<int>(std::min<size_t>(
                  sessions[0]->generation.output_length, 200u)),
              sessions[0]->output.data());
  std::printf("# prompt=\"%.*s\" tokens=%d iters=%d threads(pool)=8\n",
              static_cast<int>(k_prompt.size()), k_prompt.data(), tokens,
              iters);
  std::printf("# (decode timing excludes model load + generator init; "
              "model weights shared read-only across lanes)\n");

  lane_pool pool;
  const std::array<int32_t, 4> lane_counts = {1, 2, 4, 8};
  for (const int32_t n : lane_counts) {
    if (n > max_lanes) {
      continue;
    }
    const std::span<std::unique_ptr<lane_session>> active{
        sessions.data(), static_cast<size_t>(n)};
    // Warmup.
    run_sequential(active, k_prompt, tokens);
    // Reference outputs (sequential).
    run_sequential(active, k_prompt, tokens);
    for (auto &s : active) {
      s->reference_text = lane_output_text(*s);
    }
    bool all_ok = true;
    for (auto &s : active) {
      all_ok = all_ok && s->last_ok && s->generation.tokens_generated > 0;
    }

    const auto seq0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < iters; ++i) {
      run_sequential(active, k_prompt, tokens);
    }
    const auto seq1 = std::chrono::steady_clock::now();

    const auto par0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < iters; ++i) {
      run_parallel(pool, active, k_prompt, tokens);
    }
    const auto par1 = std::chrono::steady_clock::now();

    // Determinism: parallel outputs must match the sequential reference.
    bool deterministic = true;
    for (auto &s : active) {
      deterministic = deterministic && s->last_ok &&
                      lane_output_text(*s) == s->reference_text;
    }

    const double seq_ns = ns_per_pass(seq0, seq1, iters);
    const double par_ns = ns_per_pass(par0, par1, iters);
    const double seq_ms = seq_ns / 1.0e6;
    const double par_ms = par_ns / 1.0e6;
    const double speedup = par_ns > 0.0 ? seq_ns / par_ns : 0.0;
    const double total_tokens = static_cast<double>(n) * tokens;
    const double seq_tok_s = total_tokens / (seq_ns / 1.0e9);
    const double par_tok_s = total_tokens / (par_ns / 1.0e9);
    std::printf("lanes=%d tokens=%d seq_ms=%.3f par_ms=%.3f speedup=%.2fx "
                "seq_tok_s=%.1f par_tok_s=%.1f deterministic=%s%s\n",
                n, tokens, seq_ms, par_ms, speedup, seq_tok_s, par_tok_s,
                deterministic ? "yes" : "no", all_ok ? "" : " [GEN_FAIL]");
  }
  return 0;
}
