#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "emel/io/mmap/sm.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/logits/sampler/sm.hpp"
#include "emel/model/tensor/window/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/generator/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

// Streamed decode route coverage: the generator with an owner-bound streaming
// window must (a) produce token-for-token identical output to the resident
// route when the window streams the same weight bytes, and (b) genuinely
// consume slot bytes - a window streaming model B's matmul weights under a
// generator whose in-memory records hold model A's must reproduce resident-B
// output, not resident-A.

namespace {

namespace window = emel::model::tensor::window;

constexpr int32_t k_n_embd = 4;
constexpr int32_t k_n_layer = 4;
constexpr int32_t k_ffn_dim = 4;
constexpr size_t k_matmuls_per_layer = 7;  // q,k,v,output,gate,down,up

struct callback_tracker {
  bool initialize_done_called = false;
  bool initialize_error_called = false;
  bool generate_done_called = false;
  bool generate_error_called = false;
  int32_t tokens_generated = -1;
  size_t output_length = 0;
  emel::error::type err = 0u;
};

void on_initialize_done(void * owner,
                        const emel::text::generator::events::initialize_done &) {
  static_cast<callback_tracker *>(owner)->initialize_done_called = true;
}

void on_initialize_error(void * owner,
                         const emel::text::generator::events::initialize_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->initialize_error_called = true;
  tracker->err = ev.err;
}

void on_generate_done(void * owner,
                      const emel::text::generator::events::generation_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_done_called = true;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
}

void on_generate_error(void * owner,
                       const emel::text::generator::events::generation_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_error_called = true;
  tracker->err = ev.err;
}

int32_t add_token(emel::model::data::vocab & vocab, const char * text) {
  const uint32_t length = static_cast<uint32_t>(std::strlen(text));
  const uint32_t offset = vocab.token_bytes_used;
  std::memcpy(vocab.token_storage.data() + offset, text, length);
  const uint32_t id = vocab.n_tokens;
  vocab.entries[id].text_offset = offset;
  vocab.entries[id].text_length = length;
  vocab.entries[id].score = 0.0f;
  vocab.entries[id].type = 0;
  vocab.token_bytes_used += length;
  vocab.n_tokens = id + 1;
  return static_cast<int32_t>(id);
}

bool tokenizer_bind_dispatch(void * tokenizer_sm,
                             const emel::text::tokenizer::event::bind & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(void * tokenizer_sm,
                                 const emel::text::tokenizer::event::tokenize & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

emel::error::type sampler_select_argmax(int32_t & candidate_ids,
                                        float & candidate_scores,
                                        int32_t & candidate_count,
                                        int32_t & selected_token_out) {
  int32_t best_index = 0;
  float best_score = (&candidate_scores)[0];
  for (int32_t idx = 1; idx < candidate_count; ++idx) {
    if ((&candidate_scores)[idx] > best_score) {
      best_score = (&candidate_scores)[idx];
      best_index = idx;
    }
  }
  selected_token_out = (&candidate_ids)[best_index];
  return 0u;
}

// A four-layer llama-style toy model. Only the ffn gate row 0 differs between
// the two variants: +10 makes the ffn inject ~10 into hidden channel 1 (argmax
// "hello"), -10 silences the ffn (argmax "world"). Every other matmul weight
// is zero so attention contributes nothing and the decision is carried
// entirely by streamed matmul bytes.
struct prepared_model {
  emel::model::data data = {};
  std::vector<std::vector<float>> tensor_storage = {};
  int32_t hello_id = -1;
  int32_t world_id = -1;
};

void build_model(prepared_model & prepared, const float gate_sign) {
  prepared.tensor_storage.reserve(64);
  auto & data = prepared.data;
  data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(data.vocab_data, "hello");
  prepared.world_id = add_token(data.vocab_data, "world");
  data.params.n_vocab = static_cast<int32_t>(data.vocab_data.n_tokens);
  data.params.n_embd = k_n_embd;
  data.params.n_head = 1;
  data.params.n_head_kv = 1;
  data.params.n_ctx = 8;
  data.params.n_rot = 2;
  data.params.n_layer = k_n_layer;
  data.n_layers = k_n_layer;
  std::memcpy(data.architecture_name.data(), "llama", 5u);

  uint32_t tensor_index = 0u;
  const auto add_name = [&](emel::model::data::tensor_record & tensor,
                            const std::string_view name) {
    tensor.name_offset = data.name_bytes_used;
    tensor.name_length = static_cast<uint32_t>(name.size());
    std::memcpy(data.name_storage.data() + data.name_bytes_used, name.data(),
                name.size());
    data.name_bytes_used += static_cast<uint32_t>(name.size());
  };
  const auto add_vector = [&](const std::string_view name,
                              const std::vector<float> & values) {
    auto & tensor = data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 1;
    tensor.dims[0] = static_cast<int64_t>(values.size());
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };
  const auto add_matrix = [&](const std::string_view name, const int32_t rows,
                              const int32_t cols,
                              const std::vector<float> & values) {
    auto & tensor = data.tensors[tensor_index++];
    add_name(tensor, name);
    prepared.tensor_storage.push_back(values);
    tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::f32);
    tensor.n_dims = 2;
    tensor.dims[0] = cols;
    tensor.dims[1] = rows;
    tensor.data = prepared.tensor_storage.back().data();
    tensor.data_size = static_cast<uint64_t>(values.size() * sizeof(float));
  };

  add_matrix("token_embd.weight", 2, k_n_embd,
             {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  add_vector("output_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
  // hello scores hidden channel 1, world scores channel 0.
  add_matrix("output.weight", 2, k_n_embd,
             {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});

  const std::vector<float> zeros16(16, 0.0f);
  std::vector<float> gate(16, 0.0f);
  gate[0] = gate_sign * 10.0f;  // gate row 0 reads hidden channel 0
  std::vector<float> up(16, 0.0f);
  up[0] = 1.0f;  // up row 0 reads hidden channel 0
  std::vector<float> down(16, 0.0f);
  down[4] = 1.0f;  // down row 1 (channel 1) reads ffn lane 0

  for (int32_t layer = 0; layer < k_n_layer; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    add_vector(prefix + "attn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix(prefix + "attn_q.weight", k_n_embd, k_n_embd, zeros16);
    add_matrix(prefix + "attn_k.weight", k_n_embd, k_n_embd, zeros16);
    add_matrix(prefix + "attn_v.weight", k_n_embd, k_n_embd, zeros16);
    add_matrix(prefix + "attn_output.weight", k_n_embd, k_n_embd, zeros16);
    add_vector(prefix + "ffn_norm.weight", {1.0f, 1.0f, 1.0f, 1.0f});
    add_matrix(prefix + "ffn_gate.weight", k_ffn_dim, k_n_embd, gate);
    add_matrix(prefix + "ffn_down.weight", k_n_embd, k_ffn_dim, down);
    add_matrix(prefix + "ffn_up.weight", k_ffn_dim, k_n_embd, up);
  }
  data.n_tensors = tensor_index;
  data.weights_data = data.tensors.data();
  data.weights_size = 1u;
}

emel::model::data::tensor_record * find_named_tensor(prepared_model & prepared,
                                                     const std::string_view name) {
  for (uint32_t idx = 0; idx < prepared.data.n_tensors; ++idx) {
    auto & tensor = prepared.data.tensors[idx];
    const std::string_view tensor_name(
        prepared.data.name_storage.data() + tensor.name_offset, tensor.name_length);
    if (tensor_name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

// Writes the per-layer matmul weights of `source` to a temp file in the
// canonical stream role order, producing the extents for bind_window.
struct stream_weight_file {
  std::filesystem::path path{};
  std::string path_str{};
  uint64_t file_size = 0u;
  std::vector<window::detail::weight_extent> extents{};
  std::vector<uint16_t> layer_weight_counts{};

  stream_weight_file(prepared_model & source, const std::string_view tag) {
    path = std::filesystem::temp_directory_path() /
           (std::string{"emel_stream_gen_"} + std::string{tag} + ".bin");
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    REQUIRE(out.good());
    const std::array<std::string_view, k_matmuls_per_layer> roles = {
        "attn_q.weight",     "attn_k.weight",   "attn_v.weight",
        "attn_output.weight", "ffn_gate.weight", "ffn_down.weight",
        "ffn_up.weight",
    };
    uint64_t offset = 0u;
    int32_t tensor_id = 0;
    for (int32_t layer = 0; layer < k_n_layer; ++layer) {
      layer_weight_counts.push_back(static_cast<uint16_t>(k_matmuls_per_layer));
      for (const std::string_view role : roles) {
        const std::string name = "blk." + std::to_string(layer) + "." + std::string(role);
        auto * tensor = find_named_tensor(source, name);
        REQUIRE(tensor != nullptr);
        out.write(static_cast<const char *>(tensor->data),
                  static_cast<std::streamsize>(tensor->data_size));
        extents.push_back(window::detail::weight_extent{
            .tensor_id = tensor_id++,
            .file_offset = offset,
            .byte_size = tensor->data_size,
            .slot_offset = 0u,
        });
        offset += tensor->data_size;
      }
    }
    out.close();
    file_size = offset;
    path_str = path.string();
  }

  ~stream_weight_file() { std::filesystem::remove(path); }
};

struct bound_window {
  emel::io::mmap::sm io_mmap{};
  std::array<emel::io::staged_read::sm, window::detail::k_max_window_slots> io_staged{};
  window::detail::stream_io_pool io_pool{};
  window::sm machine;
  bool streaming_active = false;

  bound_window() : machine{make_context()} {}

  window::action::context make_context() noexcept {
    window::action::context ctx{};
    ctx.io_mmap = &io_mmap;
    ctx.io_staged = io_staged;
    ctx.io_pool = &io_pool;
    return ctx;
  }

  // Owner-provided slot arena: two slots at the toy model's aligned layer
  // span, alive for the rig's lifetime.
  alignas(window::detail::k_slot_alignment_bytes)
      std::array<uint8_t, 2u * 7u * 64u> slot_arena{};

  bool bind(const stream_weight_file & file, const uint64_t budget_bytes) {
    const window::event::bind_window_request request{
        .file_path = file.path_str,
        .file_size_bytes = file.file_size,
        .extents = file.extents,
        .layer_weight_counts = file.layer_weight_counts,
        .budget_bytes = budget_bytes,
        .slot_storage = std::span<uint8_t>{slot_arena},
        .window_slots = 2u,
        .prefetch_depth = 1u,
        .stage_chunk_bytes = window::detail::k_default_stream_chunk_bytes,
    };
    window::event::bind_window bind_request{request};
    bind_request.on_done = {this, &bound_window::on_bind_done};
    return machine.process_event(bind_request);
  }

  static void on_bind_done(void * object,
                           const window::events::bind_window_done & ev) noexcept {
    static_cast<bound_window *>(object)->streaming_active = ev.streaming_active;
  }
};

// Streaming budget for the toy file: total = 4 layers x 7 x 64B = 1792 aligned
// bytes; two 448-byte slots fit in 1000 while the total does not.
constexpr uint64_t k_streaming_budget = 1000u;

struct generation_run {
  std::array<char, 64> output = {};
  size_t output_length = 0;
  int32_t tokens_generated = -1;
  bool ok = false;

  std::string_view text() const noexcept {
    return std::string_view(output.data(), output_length);
  }
};

// Dispatches one generate on an already-initialized generator so tests can
// run repeated generations on the same machine.
generation_run dispatch_generate(emel::text::generator::sm &generator) {
  generation_run run{};
  callback_tracker tracker{};

  static constexpr std::array<emel::text::formatter::chat_message, 1> k_messages = {
      emel::text::formatter::chat_message{.role = "user", .content = "hello"},
  };
  emel::text::generator::event::generate generate{
      std::span<const emel::text::formatter::chat_message>{k_messages},
      2,
      std::span<char>{run.output.data(), run.output.size()},
      run.output_length,
  };
  generate.add_generation_prompt = false;
  generate.enable_thinking = false;
  generate.on_done =
      emel::callback<void(const emel::text::generator::events::generation_done &)>(
          &tracker, on_generate_done);
  generate.on_error =
      emel::callback<void(const emel::text::generator::events::generation_error &)>(
          &tracker, on_generate_error);
  run.ok = generator.process_event(generate) && tracker.generate_done_called;
  run.tokens_generated = tracker.tokens_generated;
  run.output_length = tracker.output_length;
  return run;
}

generation_run
run_generation(emel::text::generator::sm &generator,
               emel::text::tokenizer::sm &tokenizer,
               std::span<emel::logits::sampler::fn> samplers,
               const emel::text::generator::selection_mode mode =
                   emel::text::generator::selection_mode::sample_logits) {
  generation_run run{};
  callback_tracker tracker{};

  emel::text::generator::event::initialize initialize{
      &tokenizer,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      samplers,
  };
  initialize.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
  initialize.encoder_variant = emel::text::encoders::encoder_kind::bpe;
  initialize.add_special = false;
  initialize.parse_special = false;
  initialize.selection_mode = mode;
  initialize.max_prompt_tokens = 8;
  initialize.max_generated_tokens = 4;
  initialize.max_blocks = 8;
  initialize.block_tokens = 4;
  initialize.strip_leading_space = false;
  initialize.on_done =
      emel::callback<void(const emel::text::generator::events::initialize_done &)>(
          &tracker, on_initialize_done);
  initialize.on_error =
      emel::callback<void(const emel::text::generator::events::initialize_error &)>(
          &tracker, on_initialize_error);
  if (!generator.process_event(initialize) || !tracker.initialize_done_called) {
    return run;
  }

  return dispatch_generate(generator);
}

struct generator_rig {
  prepared_model prepared{};
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  std::array<emel::logits::sampler::fn, 1> samplers = {
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };

  explicit generator_rig(const float gate_sign) { build_model(prepared, gate_sign); }
};

}  // namespace

TEST_CASE("generator streamed decode matches resident output token for token") {
  auto rig = std::make_unique<generator_rig>(+1.0f);
  stream_weight_file file{rig->prepared, "parity"};

  auto resident = std::make_unique<emel::text::generator::sm>(rig->prepared.data, rig->conditioner);
  const generation_run resident_run =
      run_generation(*resident, rig->tokenizer, rig->samplers);
  REQUIRE(resident_run.ok);

  auto window_rig = std::make_unique<bound_window>();
  REQUIRE(window_rig->bind(file, k_streaming_budget));
  REQUIRE(window_rig->streaming_active);

  auto rig_streamed = std::make_unique<generator_rig>(+1.0f);
  auto streamed = std::make_unique<emel::text::generator::sm>(rig_streamed->prepared.data,
                                     rig_streamed->conditioner,
                                     window_rig->machine,
                                     window_rig->streaming_active);
  const generation_run streamed_run =
      run_generation(*streamed, rig_streamed->tokenizer, rig_streamed->samplers);
  REQUIRE(streamed_run.ok);

  CHECK(streamed_run.tokens_generated == resident_run.tokens_generated);
  CHECK(streamed_run.text() == resident_run.text());
}

TEST_CASE("generator streamed decode consumes slot bytes not resident records") {
  // Resident baselines for both weight variants must disagree, or the
  // engagement proof below would be vacuous.
  auto rig_a = std::make_unique<generator_rig>(+1.0f);
  auto resident_a = std::make_unique<emel::text::generator::sm>(rig_a->prepared.data, rig_a->conditioner);
  const generation_run run_a = run_generation(*resident_a, rig_a->tokenizer, rig_a->samplers);
  REQUIRE(run_a.ok);

  auto rig_b = std::make_unique<generator_rig>(-1.0f);
  auto resident_b = std::make_unique<emel::text::generator::sm>(rig_b->prepared.data, rig_b->conditioner);
  const generation_run run_b = run_generation(*resident_b, rig_b->tokenizer, rig_b->samplers);
  REQUIRE(run_b.ok);
  REQUIRE(run_a.text() != run_b.text());

  // Stream model B's matmul bytes under a generator holding model A's records:
  // the decode must follow the streamed bytes (B), proving slot consumption.
  auto rig_b_file = std::make_unique<generator_rig>(-1.0f);
  stream_weight_file file_b{rig_b_file->prepared, "engagement"};
  auto window_rig = std::make_unique<bound_window>();
  REQUIRE(window_rig->bind(file_b, k_streaming_budget));
  REQUIRE(window_rig->streaming_active);

  auto rig_mixed = std::make_unique<generator_rig>(+1.0f);
  auto streamed = std::make_unique<emel::text::generator::sm>(rig_mixed->prepared.data,
                                     rig_mixed->conditioner,
                                     window_rig->machine,
                                     window_rig->streaming_active);
  const generation_run mixed_run =
      run_generation(*streamed, rig_mixed->tokenizer, rig_mixed->samplers);
  REQUIRE(mixed_run.ok);

  // v1 streams the decode step only: the first generated token comes from the
  // resident prefill (model A memory), the second from streamed decode and
  // must follow the window's file bytes (model B).
  REQUIRE(run_a.text() == "hellohello");
  REQUIRE(run_b.text() == "worldworld");
  CHECK(mixed_run.text() == "helloworld");
  CHECK(mixed_run.text() != run_a.text());
}

TEST_CASE("generator streamed decode restores resident weight views between runs") {
  // Control: repeated generations on a resident machine are deterministic, so
  // any second-run divergence below is attributable to streamed-state leakage.
  auto rig_a = std::make_unique<generator_rig>(+1.0f);
  auto resident_a = std::make_unique<emel::text::generator::sm>(rig_a->prepared.data,
                                                                rig_a->conditioner);
  const generation_run resident_first =
      run_generation(*resident_a, rig_a->tokenizer, rig_a->samplers);
  REQUIRE(resident_first.ok);
  const generation_run resident_second = dispatch_generate(*resident_a);
  REQUIRE(resident_second.ok);
  REQUIRE(resident_second.text() == resident_first.text());

  // Mixed rig: model A resident records, model B streamed bytes. Each run's
  // prefill is resident (model A -> "hello") and its decode is streamed
  // (model B -> "world"). If the streamed decode left the block weight views
  // pointing at the shared per-role stream records instead of restoring the
  // pristine per-layer tensors, the second run's resident prefill reads the
  // last acquired slot's clone and diverges.
  auto rig_b_file = std::make_unique<generator_rig>(-1.0f);
  stream_weight_file file_b{rig_b_file->prepared, "restore"};
  auto window_rig = std::make_unique<bound_window>();
  REQUIRE(window_rig->bind(file_b, k_streaming_budget));
  REQUIRE(window_rig->streaming_active);

  auto rig_mixed = std::make_unique<generator_rig>(+1.0f);
  auto streamed = std::make_unique<emel::text::generator::sm>(rig_mixed->prepared.data,
                                     rig_mixed->conditioner,
                                     window_rig->machine,
                                     window_rig->streaming_active);
  const generation_run first_run =
      run_generation(*streamed, rig_mixed->tokenizer, rig_mixed->samplers);
  REQUIRE(first_run.ok);
  REQUIRE(first_run.text() == "helloworld");

  const generation_run second_run = dispatch_generate(*streamed);
  REQUIRE(second_run.ok);
  CHECK(second_run.text() == "helloworld");
}

TEST_CASE("generator preselected streamed decode consumes slot bytes not "
          "resident records") {
  constexpr auto k_preselected =
      emel::text::generator::selection_mode::preselected_argmax;
  const std::span<emel::logits::sampler::fn> no_samplers{};

  auto rig_a = std::make_unique<generator_rig>(+1.0f);
  auto resident_a = std::make_unique<emel::text::generator::sm>(
      rig_a->prepared.data, rig_a->conditioner);
  const generation_run run_a =
      run_generation(*resident_a, rig_a->tokenizer, no_samplers, k_preselected);
  REQUIRE(run_a.ok);

  auto rig_b = std::make_unique<generator_rig>(-1.0f);
  auto resident_b = std::make_unique<emel::text::generator::sm>(
      rig_b->prepared.data, rig_b->conditioner);
  const generation_run run_b =
      run_generation(*resident_b, rig_b->tokenizer, no_samplers, k_preselected);
  REQUIRE(run_b.ok);
  REQUIRE(run_a.text() != run_b.text());

  // Stream model B's matmul bytes under a generator holding model A's
  // records: the preselected streamed decode rows must follow the streamed
  // bytes (B), proving the window lane streams on this family too.
  auto rig_b_file = std::make_unique<generator_rig>(-1.0f);
  stream_weight_file file_b{rig_b_file->prepared, "preselected_engagement"};
  auto window_rig = std::make_unique<bound_window>();
  REQUIRE(window_rig->bind(file_b, k_streaming_budget));
  REQUIRE(window_rig->streaming_active);

  auto rig_mixed = std::make_unique<generator_rig>(+1.0f);
  auto streamed = std::make_unique<emel::text::generator::sm>(
      rig_mixed->prepared.data, rig_mixed->conditioner, window_rig->machine,
      window_rig->streaming_active);
  const generation_run mixed_run = run_generation(
      *streamed, rig_mixed->tokenizer, no_samplers, k_preselected);
  REQUIRE(mixed_run.ok);

  REQUIRE(run_a.text() == "hellohello");
  REQUIRE(run_b.text() == "worldworld");
  CHECK(mixed_run.text() == "helloworld");
  CHECK(mixed_run.text() != run_a.text());
}

TEST_CASE(
    "generator streamed decode fails cleanly when the window cannot serve") {
  // An owner wiring stream_active against a window that is not bound routes
  // the streamed decode rows, whose per-layer acquire fails with the typed
  // stream-acquire code through the modeled compute-error path: generation
  // reports failure instead of decoding from resident records or crashing.
  auto rig = std::make_unique<generator_rig>(+1.0f);
  auto window_rig = std::make_unique<bound_window>(); // never bound

  auto streamed = std::make_unique<emel::text::generator::sm>(
      rig->prepared.data, rig->conditioner, window_rig->machine,
      /*stream_active=*/true);
  const generation_run run =
      run_generation(*streamed, rig->tokenizer, rig->samplers);
  CHECK_FALSE(run.ok);
}

TEST_CASE("generator preselected streamed decode fails cleanly when the window "
          "cannot serve") {
  // Preselected twin of the case above: the per-layer acquire failure must
  // route the same typed compute-error path (and restore the resident weight
  // views) on the preselected-argmax streamed decode driver.
  auto rig = std::make_unique<generator_rig>(+1.0f);
  auto window_rig = std::make_unique<bound_window>(); // never bound

  auto streamed = std::make_unique<emel::text::generator::sm>(
      rig->prepared.data, rig->conditioner, window_rig->machine,
      /*stream_active=*/true);
  const generation_run run =
      run_generation(*streamed, rig->tokenizer, rig->samplers,
                     emel::text::generator::selection_mode::preselected_argmax);
  CHECK_FALSE(run.ok);
}

TEST_CASE("generator passthrough window keeps the resident route engaged") {
  auto rig = std::make_unique<generator_rig>(+1.0f);
  stream_weight_file file{rig->prepared, "passthrough"};

  auto window_rig = std::make_unique<bound_window>();
  REQUIRE(window_rig->bind(file, /*budget=*/0u));
  REQUIRE_FALSE(window_rig->streaming_active);

  auto resident = std::make_unique<emel::text::generator::sm>(rig->prepared.data, rig->conditioner);
  const generation_run resident_run =
      run_generation(*resident, rig->tokenizer, rig->samplers);
  REQUIRE(resident_run.ok);

  auto rig_pt = std::make_unique<generator_rig>(+1.0f);
  auto passthrough = std::make_unique<emel::text::generator::sm>(rig_pt->prepared.data,
                                        rig_pt->conditioner,
                                        window_rig->machine,
                                        window_rig->streaming_active);
  const generation_run passthrough_run =
      run_generation(*passthrough, rig_pt->tokenizer, rig_pt->samplers);
  REQUIRE(passthrough_run.ok);

  CHECK(passthrough_run.text() == resident_run.text());
}
