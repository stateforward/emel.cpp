#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/model/architecture/detail.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/moshi/detail.hpp"

namespace {

namespace moshi = emel::model::moshi::detail;

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

std::filesystem::path repo_root() {
#ifdef EMEL_TEST_REPO_ROOT
  return std::filesystem::path{EMEL_TEST_REPO_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path moshi_fixture_path(const std::string_view name) {
  return repo_root() / "tests" / "models" / name;
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());
  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  REQUIRE(size > 0);
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(stream.good());
  return bytes;
}

void materialize_tensor_names_from_file(
    emel::model::data &model, const std::vector<uint8_t> &file_bytes) {
  model.name_bytes_used = 0u;
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    auto &tensor = model.tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    REQUIRE(source_offset + length <= file_bytes.size());
    REQUIRE(static_cast<size_t>(model.name_bytes_used) + length <=
            model.name_storage.size());
    std::memcpy(model.name_storage.data() + model.name_bytes_used,
                file_bytes.data() + source_offset, length);
    tensor.name_offset = model.name_bytes_used;
    model.name_bytes_used += static_cast<uint32_t>(length);
  }
}

bool parse_gguf_binding(const std::vector<uint8_t> &file_bytes,
                        std::vector<uint8_t> &kv_arena,
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

  const emel::gguf::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes}, requirements, on_probe_done,
      on_probe_error};
  if (!loader.process_event(probe) || requirements.tensor_count == 0u ||
      requirements.tensor_count > model_out.tensors.size()) {
    return false;
  }

  // The bind guard requires non-empty storage spans even for raw moshi.cpp
  // caches that carry zero KV entries, so reserve at least one slot.
  kv_arena.resize(std::max<size_t>(
      1u,
      static_cast<size_t>(
          emel::gguf::loader::detail::required_kv_arena_bytes(requirements))));
  kv_entries.resize(std::max<uint32_t>(1u, requirements.kv_count));
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

  const emel::gguf::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes}, on_parse_done, on_parse_error};
  return loader.process_event(parse);
}

struct loaded_moshi_fixture {
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::unique_ptr<emel::model::data> model = {};

  emel::model::detail::kv_binding binding() const {
    return emel::model::detail::kv_binding{
        .arena = std::span<const uint8_t>{kv_arena},
        .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
    };
  }
};

loaded_moshi_fixture load_fixture_or_skip(const std::string_view name) {
  const auto fixture_path = moshi_fixture_path(name);
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping moshi fixture test because fixture is missing: "
            << fixture_path.string());
    return {};
  }

  loaded_moshi_fixture loaded{};
  loaded.model = std::make_unique<emel::model::data>();
  loaded.file_bytes = read_binary_file(fixture_path);
  REQUIRE(parse_gguf_binding(loaded.file_bytes, loaded.kv_arena,
                             loaded.kv_entries, *loaded.model));
  REQUIRE(emel::model::detail::load_hparams_from_gguf(loaded.binding(),
                                                      *loaded.model));
  loaded.model->weights_data = loaded.file_bytes.data();
  loaded.model->weights_size = loaded.file_bytes.size();
  materialize_tensor_names_from_file(*loaded.model, loaded.file_bytes);
  return loaded;
}

// In-memory kv_binding builder for hparam negative cases, mirroring
// tests/model/loader/lifecycle_tests.cpp helpers.
template <class value_type>
void append_scalar(std::vector<uint8_t> &bytes, const value_type value) {
  using unsigned_type = std::make_unsigned_t<value_type>;
  const unsigned_type raw = static_cast<unsigned_type>(value);
  for (size_t i = 0u; i < sizeof(value_type); ++i) {
    bytes.push_back(static_cast<uint8_t>((raw >> (i * 8u)) & 0xffu));
  }
}

struct kv_store {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  emel::model::detail::kv_binding binding() const {
    return emel::model::detail::kv_binding{
        .arena = std::span<const uint8_t>{arena},
        .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
    };
  }

  void append(const std::string_view key, const uint32_t value_type,
              const std::span<const uint8_t> value_bytes) {
    emel::gguf::loader::kv_entry entry = {};
    entry.key_offset = static_cast<uint32_t>(arena.size());
    entry.key_length = static_cast<uint32_t>(key.size());
    arena.insert(arena.end(), key.begin(), key.end());
    entry.value_offset = static_cast<uint32_t>(arena.size());
    entry.value_length = static_cast<uint32_t>(value_bytes.size());
    entry.value_type = value_type;
    arena.insert(arena.end(), value_bytes.begin(), value_bytes.end());
    entries.push_back(entry);
  }

  void string(const std::string_view key, const std::string_view value) {
    std::vector<uint8_t> encoded = {};
    append_scalar<uint64_t>(encoded, static_cast<uint64_t>(value.size()));
    encoded.insert(encoded.end(), value.begin(), value.end());
    append(key, emel::gguf::loader::detail::constants::gguf_type_string,
           std::span<const uint8_t>{encoded});
  }

  void u32(const std::string_view key, const uint32_t value) {
    std::vector<uint8_t> encoded = {};
    append_scalar<uint32_t>(encoded, value);
    append(key, emel::gguf::loader::detail::constants::gguf_type_uint32,
           std::span<const uint8_t>{encoded});
  }

  void boolean(const std::string_view key, const bool value) {
    const std::array<uint8_t, 1> encoded = {
        static_cast<uint8_t>(value ? 1u : 0u)};
    append(key, emel::gguf::loader::detail::constants::gguf_type_bool,
           std::span<const uint8_t>{encoded});
  }

  void f32(const std::string_view key, const float value) {
    std::vector<uint8_t> encoded = {};
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    append_scalar<uint32_t>(encoded, bits);
    append(key, emel::gguf::loader::detail::constants::gguf_type_float32,
           std::span<const uint8_t>{encoded});
  }

  void i32_array(const std::string_view key,
                 const std::span<const int32_t> values) {
    std::vector<uint8_t> encoded = {};
    append_scalar<uint32_t>(
        encoded, emel::gguf::loader::detail::constants::gguf_type_int32);
    append_scalar<uint64_t>(encoded, static_cast<uint64_t>(values.size()));
    for (const int32_t value : values) {
      append_scalar<int32_t>(encoded, value);
    }
    append(key, emel::gguf::loader::detail::constants::gguf_type_array,
           std::span<const uint8_t>{encoded});
  }
};

struct lm_kv_options {
  std::string_view skip_key = {};
  std::string_view gating = "silu";
  int32_t dep_q = 4;
  uint32_t delay_count = 5u;
  uint32_t n_q = 4u;
};

kv_store make_lm_kv(const lm_kv_options &options) {
  kv_store store = {};
  const auto put_u32 = [&](const std::string_view key, const uint32_t value) {
    if (key != options.skip_key) {
      store.u32(key, value);
    }
  };
  const auto put_string = [&](const std::string_view key,
                              const std::string_view value) {
    if (key != options.skip_key) {
      store.string(key, value);
    }
  };
  const auto put_bool = [&](const std::string_view key, const bool value) {
    if (key != options.skip_key) {
      store.boolean(key, value);
    }
  };

  put_string("moshi.component", "lm");
  put_u32("moshi.lm.card", 32u);
  put_u32("moshi.lm.n_q", options.n_q);
  put_u32("moshi.lm.dep_q", static_cast<uint32_t>(options.dep_q));
  put_u32("moshi.lm.text_card", 64u);
  put_u32("moshi.lm.existing_text_padding_id", 3u);
  put_u32("moshi.lm.dim", 32u);
  put_u32("moshi.lm.num_layers", 2u);
  put_u32("moshi.lm.num_heads", 2u);
  put_u32("moshi.lm.context", 64u);
  put_u32("moshi.lm.max_period", 10000u);
  put_u32("moshi.lm.dim_feedforward", 132u);
  put_string("moshi.lm.gating", options.gating);
  put_string("moshi.lm.norm", "rms_norm_f32");
  put_string("moshi.lm.positional_embedding", "rope");
  put_bool("moshi.lm.causal", true);
  put_bool("moshi.lm.cross_attention", false);
  put_bool("moshi.lm.demux_second_stream", false);
  if (options.skip_key != "moshi.lm.delays") {
    std::vector<int32_t> delays(options.delay_count, 1);
    if (!delays.empty()) {
      delays[0] = 0;
    }
    store.i32_array("moshi.lm.delays", std::span<const int32_t>{delays});
  }
  put_u32("moshi.lm.depformer.dim", 16u);
  put_u32("moshi.lm.depformer.num_heads", 2u);
  put_u32("moshi.lm.depformer.num_layers", 2u);
  put_u32("moshi.lm.depformer.dim_feedforward", 48u);
  put_u32("moshi.lm.depformer.context", 4u);
  put_u32("moshi.lm.depformer.max_period", 10000u);
  put_string("moshi.lm.depformer.gating", "silu");
  put_string("moshi.lm.depformer.pos_emb", "none");
  put_bool("moshi.lm.depformer.multi_linear", true);
  put_bool("moshi.lm.depformer.weights_per_step", true);
  return store;
}

bool load_lm_hparams(const kv_store &store, emel::model::data &model) {
  const auto binding = store.binding();
  const emel::model::detail::hparam_loader loader{binding};
  return moshi::load_hparams(loader, model);
}

struct mimi_kv_options {
  float frame_rate = 12.5f;
  uint32_t dim = 16u;
};

kv_store make_mimi_kv(const mimi_kv_options &options) {
  kv_store store = {};
  store.string("moshi.component", "mimi");
  store.u32("moshi.mimi.sample_rate", 24000u);
  store.f32("moshi.mimi.frame_rate", options.frame_rate);
  store.u32("moshi.mimi.n_q", 4u);
  store.u32("moshi.mimi.card", 32u);
  store.u32("moshi.mimi.dim", options.dim);
  store.u32("moshi.mimi.semantic_n_q", 1u);
  store.u32("moshi.mimi.codebook_dim", 8u);
  store.u32("moshi.mimi.transformer.num_layers", 2u);
  store.u32("moshi.mimi.transformer.num_heads", 2u);
  store.u32("moshi.mimi.transformer.context", 8u);
  store.u32("moshi.mimi.transformer.max_period", 10000u);
  return store;
}

constexpr emel::error::type k_none =
    emel::error::cast(emel::model::loader::error::none);

emel::model::data::tensor_record *
find_mutable_tensor(emel::model::data &model, const std::string_view target) {
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    auto &tensor = model.tensors[index];
    if (emel::model::tensor_name_view(model, tensor) == target) {
      return &tensor;
    }
  }
  return nullptr;
}

uint64_t q4_k_storage_bytes(const int64_t cols, const int64_t rows) noexcept {
  const auto blocks = static_cast<uint64_t>(cols) /
                      emel::kernel::detail::quant::QK_K *
                      static_cast<uint64_t>(rows);
  return blocks * sizeof(emel::kernel::detail::quant::block_q4_k);
}

uint64_t dense_storage_bytes(const int64_t cols, const int64_t rows,
                             const uint64_t element_bytes) noexcept {
  return static_cast<uint64_t>(cols) * static_cast<uint64_t>(rows) *
         element_bytes;
}

uint64_t quant_storage_bytes(const int64_t cols, const int64_t rows,
                             const uint64_t block_elements,
                             const uint64_t block_bytes) noexcept {
  const auto blocks = static_cast<uint64_t>(cols) / block_elements *
                      static_cast<uint64_t>(rows);
  return blocks * block_bytes;
}

void set_matrix_tensor(emel::model::data &model, const std::string_view name,
                       const int32_t dtype, const int64_t cols,
                       const int64_t rows, const uint64_t data_size) {
  auto *tensor = find_mutable_tensor(model, name);
  REQUIRE(tensor != nullptr);
  REQUIRE(cols > 0);
  REQUIRE(rows > 0);
  tensor->type = dtype;
  tensor->n_dims = 2;
  tensor->dims = {cols, rows, 1, 1};
  tensor->data_size = data_size;
}

void set_q4_k_matrix(emel::model::data &model, const std::string_view name,
                     const int64_t cols, const int64_t rows) {
  REQUIRE(cols % static_cast<int64_t>(emel::kernel::detail::quant::QK_K) == 0);
  set_matrix_tensor(model, name, emel::kernel::detail::dtype_q4_k, cols, rows,
                    q4_k_storage_bytes(cols, rows));
}

void configure_q4_k_lm_contract(emel::model::data &model) {
  constexpr int32_t k_dim = 256;
  constexpr int32_t k_text_card = 255;
  constexpr int32_t k_audio_card = 255;

  model.moshi_lm.dim = k_dim;
  model.moshi_lm.text_card = k_text_card;
  model.moshi_lm.card = k_audio_card;
  model.params.n_embd = k_dim;
  model.params.n_embd_out = k_dim;
  model.params.n_vocab = k_text_card;
  set_q4_k_matrix(model, "lm.text_emb.weight", k_dim, k_text_card + 1);
  set_q4_k_matrix(model, "lm.text_linear.weight", k_dim, k_text_card);
  set_q4_k_matrix(model, "lm.emb.0.weight", k_dim, k_audio_card + 1);
}

} // namespace

TEST_CASE("moshi architecture is registered") {
  const auto *architecture = emel::model::resolve_architecture(
      "moshi", emel::model::default_architecture_span());
  REQUIRE(architecture != nullptr);
  CHECK(architecture->load_hparams == &moshi::load_hparams);
  CHECK(architecture->validate_data == &moshi::validate_execution_contract);
  CHECK(moshi::is_execution_architecture("moshi"));
  CHECK_FALSE(moshi::is_execution_architecture("moshiko"));
  CHECK(emel::model::is_moshi_execution_architecture("moshi"));
  CHECK(emel::model::is_supported_execution_architecture("moshi"));
}

TEST_CASE("enriched moshi lm fixture loads hparams, vocab, and contract") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  const auto &model = *loaded.model;
  CHECK(emel::model::architecture_name_view(model) == "moshi");
  CHECK(model.moshi_component_id == emel::model::data::moshi_component::lm);
  CHECK(model.moshi_lm.card == 32);
  CHECK(model.moshi_lm.n_q == 4);
  CHECK(model.moshi_lm.dep_q == 4);
  CHECK(model.moshi_lm.text_card == 64);
  CHECK(model.moshi_lm.text_padding_id == 3);
  CHECK(model.moshi_lm.delay_count == 5u);
  CHECK(model.moshi_lm.delays[0] == 0);
  CHECK(model.moshi_lm.delays[2] == 1);
  CHECK(model.moshi_lm.depformer_dim == 16);
  CHECK(model.moshi_lm.depformer_num_layers == 2);
  CHECK(model.moshi_lm.causal);
  CHECK(model.moshi_lm.depformer_multi_linear);
  CHECK_FALSE(model.moshi_lm.cross_attention);
  CHECK(model.params.n_embd == 32);
  CHECK(model.params.n_layer == 2);
  CHECK(model.params.n_head == 2);
  CHECK(model.params.n_ctx == 64);
  CHECK(model.params.n_vocab == 64);
  CHECK(model.params.n_ff == 132);
  CHECK(model.params.n_rot == 16);
  CHECK(model.params.decoder_block_count == 2);
  CHECK(model.params.rope_freq_base == doctest::Approx(10000.0f));

  auto vocab = std::make_unique<emel::model::data::vocab>();
  REQUIRE(emel::model::detail::load_vocab_from_gguf(loaded.binding(), *vocab));
  CHECK(vocab->n_tokens == 64u);
  CHECK(vocab->tokenizer_model_id == emel::model::data::tokenizer_model::SPM);
  CHECK(vocab->unk_id == 0);
  CHECK(vocab->bos_id == 1);
  CHECK(vocab->eos_id == 2);
  CHECK(vocab->pad_id == 3);

  moshi::execution_contract contract = {};
  REQUIRE(moshi::build_execution_contract(model, contract) == k_none);
  CHECK(contract.component == emel::model::data::moshi_component::lm);
  CHECK(contract.lm.audio_emb.tensor_count == 4u);
  CHECK(contract.lm.transformer.tensor_count > 0u);
  CHECK(contract.lm.depformer.tensor_count > 0u);
  CHECK(contract.lm.linears.tensor_count == 4u);
}

TEST_CASE("enriched mimi fixture loads hparams and contract") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  const auto &model = *loaded.model;
  CHECK(model.moshi_component_id == emel::model::data::moshi_component::mimi);
  CHECK(model.mimi.sample_rate == 24000);
  CHECK(model.mimi.frame_rate == doctest::Approx(12.5f));
  CHECK(model.mimi.n_q == 4);
  CHECK(model.mimi.card == 32);
  CHECK(model.mimi.dim == 16);
  CHECK(model.mimi.codebook_dim == 8);
  CHECK(model.mimi.semantic_n_q == 1);
  CHECK(model.mimi.transformer_num_layers == 2);
  CHECK(model.params.n_features == 4);

  moshi::execution_contract contract = {};
  REQUIRE(moshi::build_execution_contract(model, contract) == k_none);
  CHECK(contract.component == emel::model::data::moshi_component::mimi);
  CHECK(contract.mimi.encoder.tensor_count > 0u);
  CHECK(contract.mimi.decoder.tensor_count > 0u);
  CHECK(contract.mimi.quantizer.tensor_count > 0u);
  CHECK(contract.mimi.upsample.tensor_count > 0u);
  CHECK(contract.mimi.downsample.tensor_count > 0u);
}

TEST_CASE("enriched voice fixture loads hparams and contract") {
  auto loaded = load_fixture_or_skip("moshi-tiny-voice.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  const auto &model = *loaded.model;
  CHECK(model.moshi_component_id == emel::model::data::moshi_component::voice);

  moshi::execution_contract contract = {};
  REQUIRE(moshi::build_execution_contract(model, contract) == k_none);
  CHECK(contract.component == emel::model::data::moshi_component::voice);
  CHECK(contract.voice.embeddings.tensor != nullptr);
  CHECK(contract.voice.cache.tensor != nullptr);
}

TEST_CASE("raw moshi.cpp gguf without metadata is rejected explicitly") {
  const auto fixture_path = moshi_fixture_path("moshi-tiny-lm-raw.gguf");
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping raw moshi fixture test because fixture is missing: "
            << fixture_path.string());
    return;
  }

  auto model = std::make_unique<emel::model::data>();
  const auto file_bytes = read_binary_file(fixture_path);
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  // The raw cache is structurally valid GGUF, so parsing succeeds ...
  REQUIRE(parse_gguf_binding(file_bytes, kv_arena, kv_entries, *model));
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
  };
  // ... but hparam binding refuses it: no general.architecture, no moshi.*
  // contract. Conversion via tools/bench/moshi_gguf_convert.py is required.
  CHECK_FALSE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(model->moshi_component_id == emel::model::data::moshi_component::none);
}

TEST_CASE("moshi lm hparams require the full metadata contract") {
  auto model = std::make_unique<emel::model::data>();

  SUBCASE("complete contract loads") {
    CHECK(load_lm_hparams(make_lm_kv({}), *model));
    CHECK(model->moshi_component_id == emel::model::data::moshi_component::lm);
  }

  SUBCASE("missing component key fails") {
    CHECK_FALSE(
        load_lm_hparams(make_lm_kv({.skip_key = "moshi.component"}), *model));
  }

  SUBCASE("unknown component fails") {
    kv_store store = {};
    store.string("moshi.component", "codec");
    CHECK_FALSE(load_lm_hparams(store, *model));
  }

  SUBCASE("missing dim fails") {
    CHECK_FALSE(
        load_lm_hparams(make_lm_kv({.skip_key = "moshi.lm.dim"}), *model));
  }

  SUBCASE("missing card fails") {
    CHECK_FALSE(
        load_lm_hparams(make_lm_kv({.skip_key = "moshi.lm.card"}), *model));
  }

  SUBCASE("unsupported gating fails") {
    CHECK_FALSE(load_lm_hparams(make_lm_kv({.gating = "gelu"}), *model));
  }

  SUBCASE("delay count must equal n_q plus one") {
    CHECK_FALSE(load_lm_hparams(make_lm_kv({.delay_count = 4u}), *model));
  }

  SUBCASE("dep_q above n_q fails") {
    CHECK_FALSE(load_lm_hparams(make_lm_kv({.dep_q = 5}), *model));
  }

  SUBCASE("missing depformer keys fail when dep_q is positive") {
    CHECK_FALSE(load_lm_hparams(
        make_lm_kv({.skip_key = "moshi.lm.depformer.dim"}), *model));
  }

  SUBCASE("n_q at the integer limit fails without overflowing the delay slot") {
    // dep_q <= n_q passes first, so the delay-slot bound must reject the
    // count without evaluating n_q + 1 in signed arithmetic.
    CHECK_FALSE(load_lm_hparams(
        make_lm_kv({.n_q = static_cast<uint32_t>(INT32_MAX)}), *model));
  }
}

TEST_CASE("moshi mimi hparams require finite frame rates") {
  auto model = std::make_unique<emel::model::data>();

  SUBCASE("complete contract loads") {
    CHECK(load_lm_hparams(make_mimi_kv({}), *model));
    CHECK(model->moshi_component_id ==
          emel::model::data::moshi_component::mimi);
  }

  SUBCASE("nan frame rate fails") {
    CHECK_FALSE(load_lm_hparams(
        make_mimi_kv({.frame_rate = std::numeric_limits<float>::quiet_NaN()}),
        *model));
  }

  SUBCASE("infinite frame rate fails") {
    CHECK_FALSE(load_lm_hparams(
        make_mimi_kv({.frame_rate = std::numeric_limits<float>::infinity()}),
        *model));
  }

  SUBCASE("odd attention head size fails") {
    // dim 6 over 2 heads divides evenly but leaves head_dim 3; the codec
    // rotary kernel halves head_dim, so the hparam gate must reject it.
    CHECK_FALSE(load_lm_hparams(make_mimi_kv({.dim = 6u}), *model));
  }
}

TEST_CASE("moshi contract validation rejects inconsistent models") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  SUBCASE("component none is invalid") {
    loaded.model->moshi_component_id = emel::model::data::moshi_component::none;
    CHECK(moshi::validate_execution_contract(*loaded.model) != k_none);
  }

  SUBCASE("hparam/tensor shape mismatch is invalid") {
    loaded.model->moshi_lm.text_card += 1;
    CHECK(moshi::validate_execution_contract(*loaded.model) != k_none);
  }

  SUBCASE("missing transformer block count is invalid") {
    loaded.model->moshi_lm.num_layers += 1;
    CHECK(moshi::validate_execution_contract(*loaded.model) != k_none);
  }

  SUBCASE("non-moshi architecture name is invalid") {
    loaded.model->architecture_name[0] = 'x';
    CHECK(moshi::validate_execution_contract(*loaded.model) != k_none);
  }

  SUBCASE("tensor storage smaller than its dtype payload is invalid") {
    // Runtime binding reads the full dtype-sized payload; exact-shape
    // metadata over a one-byte buffer must fail the contract.
    for (uint32_t index = 0u; index < loaded.model->n_tensors; ++index) {
      loaded.model->tensors[index].data_size = 1u;
    }
    CHECK(moshi::validate_execution_contract(*loaded.model) != k_none);
  }
}

TEST_CASE("moshi contract rejects tensor element products that overflow") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // Malformed positive dimensions can wrap the unchecked element product (or
  // the dtype byte count computed from it) to a small value: the storage
  // probe then marked a tiny payload as fully backed and later consumers
  // would read far past it. The checked products must reject the tensor (and
  // with it the contract) instead.
  auto &model = *loaded.model;
  bool corrupted = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != "lm.out_norm.alpha") {
      continue;
    }
    SUBCASE("element product wraps") {
      // Row extents beyond dims[0] multiply past uint64_t capacity before
      // dtype byte accounting can decide whether the payload is backed.
      tensor.n_dims = 4;
      tensor.dims = {1u, uint64_t{1} << 32, uint64_t{1} << 32, 2u};
    }
    SUBCASE("dtype byte count wraps") {
      // 2^62 elements are representable but the f32 byte count is 2^64.
      tensor.n_dims = 2;
      tensor.dims = {int64_t{1} << 31, int64_t{1} << 31, 1, 1};
    }
    corrupted = true;
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("moshi lm contract accepts q4_k block-backed tensors") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  auto &model = *loaded.model;
  configure_q4_k_lm_contract(model);

  SUBCASE("q4_k matrices with exact block storage are valid") {
    CHECK(moshi::validate_execution_contract(model) == k_none);
  }

  SUBCASE("q4_k matrices one byte under their block storage are invalid") {
    auto *tensor = find_mutable_tensor(model, "lm.text_emb.weight");
    REQUIRE(tensor != nullptr);
    REQUIRE(tensor->data_size > 0u);
    tensor->data_size -= 1u;
    CHECK(moshi::validate_execution_contract(model) != k_none);
  }
}

TEST_CASE("moshi lm contract accounts for supported dense tensor byte widths") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  struct dtype_case {
    const char *name = "";
    int32_t dtype = 0;
    uint64_t element_bytes = 0u;
  };
  const std::array cases{
      dtype_case{"f32", emel::kernel::detail::dtype_f32, sizeof(uint32_t)},
  };

  for (const auto &entry : cases) {
    CAPTURE(entry.name);
    auto model = std::make_unique<emel::model::data>(*loaded.model);
    configure_q4_k_lm_contract(*model);
    set_matrix_tensor(*model, "lm.text_emb.weight", entry.dtype,
                      model->moshi_lm.dim, model->moshi_lm.text_card + 1,
                      dense_storage_bytes(model->moshi_lm.dim,
                                          model->moshi_lm.text_card + 1,
                                          entry.element_bytes));
    CHECK(moshi::validate_execution_contract(*model) == k_none);

    auto *tensor = find_mutable_tensor(*model, "lm.text_emb.weight");
    REQUIRE(tensor != nullptr);
    REQUIRE(tensor->data_size > 0u);
    tensor->data_size -= 1u;
    CHECK(moshi::validate_execution_contract(*model) != k_none);
  }
}

TEST_CASE("moshi lm contract accepts supported gguf tensor layouts") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  struct dtype_case {
    const char *name = "";
    int32_t dtype = 0;
  };
  const std::array cases{
      dtype_case{"f32", 0},   dtype_case{"q4_0", 2},  dtype_case{"q8_0", 8},
      dtype_case{"q2_k", 10}, dtype_case{"q3_k", 11}, dtype_case{"q4_k", 12},
      dtype_case{"q6_k", 14},
  };

  for (const auto &entry : cases) {
    CAPTURE(entry.name);
    auto model = std::make_unique<emel::model::data>(*loaded.model);
    configure_q4_k_lm_contract(*model);
    const auto layout = emel::gguf::loader::detail::ggml_layout(
        static_cast<uint32_t>(entry.dtype));
    REQUIRE(layout.block_size > 0u);
    REQUIRE(layout.type_size > 0u);
    REQUIRE(static_cast<uint64_t>(model->moshi_lm.dim) % layout.block_size ==
            0u);
    set_matrix_tensor(*model, "lm.text_emb.weight", entry.dtype,
                      model->moshi_lm.dim, model->moshi_lm.text_card + 1,
                      quant_storage_bytes(model->moshi_lm.dim,
                                          model->moshi_lm.text_card + 1,
                                          layout.block_size, layout.type_size));
    CHECK(moshi::validate_execution_contract(*model) == k_none);

    auto *tensor = find_mutable_tensor(*model, "lm.text_emb.weight");
    REQUIRE(tensor != nullptr);
    REQUIRE(tensor->data_size > 0u);
    tensor->data_size -= 1u;
    CHECK(moshi::validate_execution_contract(*model) != k_none);
  }
}

TEST_CASE("moshi lm contract rejects unsupported known gguf tensor layouts") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  struct dtype_case {
    const char *name = "";
    int32_t dtype = 0;
  };
  const std::array cases{
      dtype_case{"f16", 1},     dtype_case{"q4_1", 3},
      dtype_case{"q5_0", 6},    dtype_case{"q5_1", 7},
      dtype_case{"q8_1", 9},    dtype_case{"q5_k", 13},
      dtype_case{"q8_k", 15},   dtype_case{"iq2_xxs", 16},
      dtype_case{"iq2_xs", 17}, dtype_case{"iq3_xxs", 18},
      dtype_case{"iq1_s", 19},  dtype_case{"iq4_nl", 20},
      dtype_case{"iq3_s", 21},  dtype_case{"iq2_s", 22},
      dtype_case{"iq4_xs", 23}, dtype_case{"i8", 24},
      dtype_case{"i16", 25},    dtype_case{"i32", 26},
      dtype_case{"i64", 27},    dtype_case{"f64", 28},
      dtype_case{"iq1_m", 29},  dtype_case{"bf16", 30},
      dtype_case{"tq1_0", 34},  dtype_case{"tq2_0", 35},
      dtype_case{"mxfp4", 39},
  };

  for (const auto &entry : cases) {
    CAPTURE(entry.name);
    auto model = std::make_unique<emel::model::data>(*loaded.model);
    configure_q4_k_lm_contract(*model);
    const auto layout = emel::gguf::loader::detail::ggml_layout(
        static_cast<uint32_t>(entry.dtype));
    REQUIRE(layout.block_size > 0u);
    REQUIRE(layout.type_size > 0u);
    REQUIRE(static_cast<uint64_t>(model->moshi_lm.dim) % layout.block_size ==
            0u);
    set_matrix_tensor(*model, "lm.text_emb.weight", entry.dtype,
                      model->moshi_lm.dim, model->moshi_lm.text_card + 1,
                      quant_storage_bytes(model->moshi_lm.dim,
                                          model->moshi_lm.text_card + 1,
                                          layout.block_size, layout.type_size));
    CHECK(moshi::validate_execution_contract(*model) != k_none);
  }
}

TEST_CASE("moshi lm contract rejects partial quantized tensor rows") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  namespace quant = emel::kernel::detail::quant;
  struct dtype_case {
    const char *name = "";
    int32_t dtype = 0;
    uint64_t block_elements = 0u;
    uint64_t block_bytes = 0u;
  };
  const std::array cases{
      dtype_case{"q4_0", emel::kernel::detail::dtype_q4_0, quant::QK4_0,
                 sizeof(quant::block_q4_0)},
      dtype_case{"q4_1", emel::kernel::detail::dtype_q4_1, quant::QK4_1,
                 sizeof(quant::block_q4_1)},
      dtype_case{"q5_0", emel::kernel::detail::dtype_q5_0, quant::QK5_0,
                 sizeof(quant::block_q5_0)},
      dtype_case{"q5_1", emel::kernel::detail::dtype_q5_1, quant::QK5_1,
                 sizeof(quant::block_q5_1)},
      dtype_case{"q8_0", emel::kernel::detail::dtype_q8_0, quant::QK8_0,
                 sizeof(quant::block_q8_0)},
      dtype_case{"q2_k", emel::kernel::detail::dtype_q2_k, quant::QK_K,
                 sizeof(quant::block_q2_k)},
      dtype_case{"q3_k", emel::kernel::detail::dtype_q3_k, quant::QK_K,
                 sizeof(quant::block_q3_k)},
      dtype_case{"q5_k", emel::kernel::detail::dtype_q5_k, quant::QK_K,
                 sizeof(quant::block_q5_k)},
      dtype_case{"q6_k", emel::kernel::detail::dtype_q6_k, quant::QK_K,
                 sizeof(quant::block_q6_k)},
      dtype_case{"q8_k", emel::kernel::detail::dtype_q8_k, quant::QK_K,
                 sizeof(quant::block_q8_k)},
  };

  for (const auto &entry : cases) {
    CAPTURE(entry.name);
    auto model = std::make_unique<emel::model::data>(*loaded.model);
    configure_q4_k_lm_contract(*model);
    set_matrix_tensor(
        *model, "lm.text_emb.weight", entry.dtype, model->moshi_lm.dim + 1,
        model->moshi_lm.text_card + 1,
        quant_storage_bytes(model->moshi_lm.dim, model->moshi_lm.text_card + 1,
                            entry.block_elements, entry.block_bytes));
    CHECK(moshi::validate_execution_contract(*model) != k_none);
  }
}

TEST_CASE("moshi lm contract rejects unknown tensor dtypes") {
  auto loaded = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  const std::array cases{
      int32_t{40},
      int32_t{999},
  };

  for (const int32_t dtype : cases) {
    CAPTURE(dtype);
    auto model = std::make_unique<emel::model::data>(*loaded.model);
    configure_q4_k_lm_contract(*model);
    set_matrix_tensor(*model, "lm.text_emb.weight", dtype, model->moshi_lm.dim,
                      model->moshi_lm.text_card + 1,
                      dense_storage_bytes(model->moshi_lm.dim,
                                          model->moshi_lm.text_card + 1, 1u));
    CHECK(moshi::validate_execution_contract(*model) != k_none);
  }
}

TEST_CASE("mimi contract validation requires every transformer layer") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // plan_transformer consumes every configured layer's norm1/in_proj/linear1
  // with dim-consistent shapes during codec bind, so a family-non-empty probe
  // alone let a truncated or wrong-shaped transformer family validate here
  // and then fail the codec planner.
  auto &model = *loaded.model;
  const auto corrupt_named = [&model](const std::string_view target,
                                      const bool bump_layer_digit) {
    for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
      auto &tensor = model.tensors[idx];
      const std::string_view name{
          model.name_storage.data() + tensor.name_offset, tensor.name_length};
      if (name != target) {
        continue;
      }
      if (bump_layer_digit) {
        // Bump the layer index digit so the tensor stays inside the family
        // but configured layer 1 no longer resolves.
        const size_t digit = name.find("layers.1") + 7u;
        model.name_storage[tensor.name_offset + digit] = '9';
      } else {
        tensor.dims[1] += 1;
      }
      return true;
    }
    return false;
  };

  SUBCASE("missing intermediate layer") {
    REQUIRE(corrupt_named(
        "mimi.encoder_transformer.transformer.layers.1.norm1.weight", true));
  }
  SUBCASE("wrong attention projection shape") {
    REQUIRE(corrupt_named("mimi.decoder_transformer.transformer.layers.1."
                          "self_attn.in_projs.0.weight",
                          false));
  }
  SUBCASE("missing layer norm2 weight") {
    // bind_transformer consumes both norms per layer, not just norm1.
    REQUIRE(corrupt_named(
        "mimi.encoder_transformer.transformer.layers.1.norm2.weight", true));
  }
  SUBCASE("missing attention out projection") {
    REQUIRE(corrupt_named("mimi.decoder_transformer.transformer.layers.1."
                          "self_attn.out_projs.0.weight",
                          true));
  }
  SUBCASE("missing layer scale") {
    REQUIRE(corrupt_named(
        "mimi.encoder_transformer.transformer.layers.1.layer_scale_2.scale",
        true));
  }
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

namespace {

// Appends a clone of an existing tensor record under a new name (sharing the
// original's storage), so contract tests can declare configurations larger
// than the tiny fixture actually carries.
bool append_cloned_tensor(emel::model::data &model,
                          const std::string_view source_name,
                          const std::string &clone_name) {
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    const auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != source_name) {
      continue;
    }
    auto &clone = model.tensors[model.n_tensors];
    clone = tensor;
    clone.name_offset = model.name_bytes_used;
    clone.name_length = static_cast<uint32_t>(clone_name.size());
    std::memcpy(model.name_storage.data() + model.name_bytes_used,
                clone_name.data(), clone_name.size());
    model.name_bytes_used += static_cast<uint32_t>(clone_name.size());
    model.n_tensors += 1;
    return true;
  }
  return false;
}

} // namespace

TEST_CASE(
    "mimi contract validation rejects rvq split counts past the codec cap") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // bind_rvq_split stores each split's levels in fixed 32-entry arrays; a
  // GGUF declaring more levels with a codebook tensor for every one of them
  // validated pre-cap and then failed quantizer bind. Clone the semantic
  // level-0 codebook into levels 1..39 so every probed tensor exists.
  auto &model = *loaded.model;
  for (int level = 1; level < 40; ++level) {
    REQUIRE(append_cloned_tensor(
        model, "mimi.quantizer.rvq_first.vq.layers.0._codebook.embedding",
        "mimi.quantizer.rvq_first.vq.layers." + std::to_string(level) +
            "._codebook.embedding"));
  }
  model.mimi.semantic_n_q = 40;
  model.mimi.n_q = 43;
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE(
    "mimi contract validation rejects transformer layers past the codec cap") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // plan_transformer stores layers in fixed 16-entry arrays; a GGUF
  // declaring more layers with tensors for every one of them validated
  // pre-cap and then failed codec bind. Clone layer 1's full tensor set into
  // layers 2..16 for both families so every probed tensor exists.
  auto &model = *loaded.model;
  static constexpr const char *k_layer_tensors[] = {
      "norm1.weight",
      "norm1.bias",
      "norm2.weight",
      "norm2.bias",
      "layer_scale_1.scale",
      "layer_scale_2.scale",
      "self_attn.in_projs.0.weight",
      "self_attn.out_projs.0.weight",
      "linear1.weight",
      "linear2.weight",
  };
  for (const char *family : {"encoder_transformer", "decoder_transformer"}) {
    for (int layer = 2; layer <= 16; ++layer) {
      for (const char *suffix : k_layer_tensors) {
        const std::string source =
            std::string("mimi.") + family + ".transformer.layers.1." + suffix;
        const std::string clone = std::string("mimi.") + family +
                                  ".transformer.layers." +
                                  std::to_string(layer) + "." + suffix;
        REQUIRE(append_cloned_tensor(model, source, clone));
      }
    }
  }
  model.mimi.transformer_num_layers = 17;
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation rejects non-float non-q8 projections") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // The binder's float path accepts only f32/f16 (prepare_matrix_raw /
  // prepare_linear); a bf16 projection with enough bytes validated as
  // "non-q8" and then failed codec initialization.
  auto &model = *loaded.model;
  constexpr int32_t k_dtype_bf16 = 30;
  bool corrupted = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != "mimi.decoder_transformer.transformer.layers.1.self_attn."
                "out_projs.0.weight" &&
        name != "mimi.quantizer.rvq_rest.input_proj.weight") {
      continue;
    }
    SUBCASE("bf16 transformer projection") {
      if (name.ends_with("out_projs.0.weight")) {
        tensor.type = k_dtype_bf16;
        corrupted = true;
      }
    }
    SUBCASE("bf16 rvq projection") {
      if (name.ends_with("input_proj.weight")) {
        tensor.type = k_dtype_bf16;
        corrupted = true;
      }
    }
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation rejects a mixed f16 conv class") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // plan_codec selects the f16 conv operand class from the first encoder
  // conv and bind_conv then requires raw f16 taps for every non-transposed
  // SEANet/downsample conv; flipping only the first conv to f16 produced a
  // mixed artifact that validated and then failed codec initialization.
  auto &model = *loaded.model;
  constexpr int32_t k_dtype_f16 = 1;
  bool corrupted = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != "mimi.encoder.model.0.conv.conv.weight") {
      continue;
    }
    tensor.type = k_dtype_f16;
    corrupted = true;
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE(
    "mimi contract validation requires f16 rvq projections in the f16 class") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // bind_rvq_split prepares raw f16 projection copies whenever the f16 conv
  // operand class is selected and the split is not q8; an f16-class model
  // with f32 RVQ projections validated here and then failed initialization.
  // Flip every non-transposed conv (and the downsample) to f16 so the class
  // selection and the per-conv f16 rule both hold; the f32 projections stay.
  auto &model = *loaded.model;
  constexpr int32_t k_dtype_f16 = 1;
  uint32_t flipped = 0;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    const bool non_transposed_conv =
        (name.starts_with("mimi.encoder.") ||
         name.starts_with("mimi.decoder.") ||
         name.starts_with("mimi.downsample.")) &&
        name.ends_with("conv.conv.weight") &&
        name.find("convtr") == std::string_view::npos;
    if (!non_transposed_conv) {
      continue;
    }
    tensor.type = k_dtype_f16;
    flipped += 1;
  }
  REQUIRE(flipped >= 25u); // 14 encoder + 10 decoder + downsample
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation rejects oversized conv geometry extents") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // The codec caps every resolved conv extent at 2^16 to keep its arena
  // sizing representable; a dividing-but-huge geometry (declared via
  // metadata, storage claim included) validated pre-cap and then failed
  // codec initialization.
  auto &model = *loaded.model;
  bool corrupted = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != "mimi.downsample.conv.conv.conv.weight") {
      continue;
    }
    tensor.n_dims = 3;
    tensor.dims = {int64_t{1} << 17, 16, 16, 0};
    tensor.data_size = (uint64_t{1} << 17) * 16u * 16u * sizeof(float);
    corrupted = true;
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE(
    "mimi contract validation rejects a frame length off the stride chain") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // The fixed encoder/downsample stride chain reduces exactly 1920 samples
  // to one token; a positive finite frame rate producing any other frame
  // length validated here and then failed plan_seanet/plan_codec.
  loaded.model->mimi.frame_rate = 25.0f; // 24000 / 25 = 960 samples
  CHECK(moshi::validate_execution_contract(*loaded.model) != k_none);
}

TEST_CASE(
    "mimi contract validation rejects codebook extents past the codec cap") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // plan_codec caps codebook_dim/card at 2^16 to keep the prepared and
  // search-table sizing representable; matching oversized codebook tensors
  // (metadata claim included) validated pre-cap and then failed initialize.
  auto &model = *loaded.model;
  model.mimi.card = int32_t{1} << 17;
  uint32_t corrupted = 0;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name.find("_codebook.embedding") == std::string_view::npos) {
      continue;
    }
    tensor.dims[1] = int64_t{1} << 17;
    tensor.data_size = static_cast<uint64_t>(tensor.dims[0]) *
                       (uint64_t{1} << 17) * sizeof(float);
    corrupted += 1;
  }
  REQUIRE(corrupted == 4u);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE(
    "mimi contract validation rejects non-float tensors the binder consumes") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // prepare_vector, prepare_conv_gemm/prepare_conv_transpose, and the RVQ
  // codebook bind all accept only f32/f16 (tensor_is_float); tensors with
  // enough stored bytes in another dtype validated here and then failed
  // codec initialization.
  auto &model = *loaded.model;
  constexpr int32_t k_dtype_bf16 = 30;
  constexpr int32_t k_dtype_q8_0 = 8;
  const auto flip_type = [&model](const std::string_view target,
                                  const int32_t type) {
    for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
      auto &tensor = model.tensors[idx];
      const std::string_view name{
          model.name_storage.data() + tensor.name_offset, tensor.name_length};
      if (name != target) {
        continue;
      }
      tensor.type = type;
      return true;
    }
    return false;
  };

  SUBCASE("bf16 transformer vector") {
    REQUIRE(
        flip_type("mimi.encoder_transformer.transformer.layers.1.norm2.bias",
                  k_dtype_bf16));
  }
  SUBCASE("bf16 seanet conv in the f32 class") {
    REQUIRE(flip_type("mimi.decoder.model.0.conv.conv.weight", k_dtype_bf16));
  }
  SUBCASE("q8_0 rvq codebook") {
    REQUIRE(
        flip_type("mimi.quantizer.rvq_first.vq.layers.0._codebook.embedding",
                  k_dtype_q8_0));
  }
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation rejects mixed transformer mlp widths") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // The codec planner stores one MLP width per transformer family and the
  // bind uses it for every layer's linear1/linear2; per-layer-consistent but
  // family-mixed widths validated here and failed codec initialization.
  auto &model = *loaded.model;
  bool narrowed = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name ==
        "mimi.encoder_transformer.transformer.layers.1.linear1.weight") {
      tensor.dims[1] = 1; // storage stays over-sized, width shrinks
      narrowed = true;
    }
    if (name ==
        "mimi.encoder_transformer.transformer.layers.1.linear2.weight") {
      tensor.dims[0] = 1; // keep linear2 consistent with the narrowed layer
    }
  }
  REQUIRE(narrowed);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation rejects malformed seanet geometry") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // plan_seanet resolves every conv against the running channel chain and
  // requires resnet blocks to return to their input width through a k1
  // conv; a name-presence probe let a geometrically broken conv (same
  // element count, same storage) validate and then fail codec bind.
  auto &model = *loaded.model;
  bool corrupted = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != "mimi.decoder.model.9.block.3.conv.conv.weight") {
      continue;
    }
    // Swap taps and the second extent: the product (and storage) stay the
    // same, but the k1 resnet output conv now declares multi-tap geometry.
    REQUIRE(tensor.n_dims >= 2);
    std::swap(tensor.dims[0], tensor.dims[1]);
    corrupted = tensor.dims[0] != 1;
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation requires the fixed seanet topology") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // plan_seanet walks the fixed mimi_v0_1 module topology by exact tensor
  // name; a family-non-empty probe let a component missing a fixed-topology
  // conv validate and then fail codec initialization.
  auto &model = *loaded.model;
  const auto corrupt_named = [&model](const std::string_view target,
                                      const size_t digit_offset) {
    for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
      auto &tensor = model.tensors[idx];
      const std::string_view name{
          model.name_storage.data() + tensor.name_offset, tensor.name_length};
      if (name != target) {
        continue;
      }
      model.name_storage[tensor.name_offset + digit_offset] = 'x';
      return true;
    }
    return false;
  };

  SUBCASE("missing encoder seanet conv") {
    const std::string_view name = "mimi.encoder.model.12.conv.conv.weight";
    REQUIRE(corrupt_named(name, name.find("conv")));
  }
  SUBCASE("missing decoder resnet conv") {
    const std::string_view name =
        "mimi.decoder.model.9.block.3.conv.conv.weight";
    REQUIRE(corrupt_named(name, name.find("block")));
  }
  SUBCASE("missing downsample conv") {
    const std::string_view name = "mimi.downsample.conv.conv.conv.weight";
    REQUIRE(corrupt_named(name, name.find("weight")));
  }
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE(
    "mimi contract validation rejects q8 classes with misaligned widths") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // The q8 matvec kernels need block-aligned contraction widths (k % 32);
  // the tiny fixture's dim = 16 is aggregate-aligned but row-misaligned, so
  // a uniform q8 projection class must reject at the contract.
  auto &model = *loaded.model;
  uint32_t flipped = 0;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    const bool projection =
        name.find("self_attn.in_projs.0.weight") != std::string_view::npos ||
        name.find("self_attn.out_projs.0.weight") != std::string_view::npos ||
        name.find("linear1.weight") != std::string_view::npos ||
        name.find("linear2.weight") != std::string_view::npos ||
        name.find("_proj.weight") != std::string_view::npos;
    if (projection && name.starts_with("mimi.")) {
      tensor.type = 8; // q8_0
      flipped += 1;
    }
  }
  REQUIRE(flipped >= 20u);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation rejects mixed projection dtype classes") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // plan_codec selects the q8-vs-float bind path from the first projection
  // of each family and the bind then requires every projection to match; a
  // single projection in the other class validated here and failed codec
  // initialization.
  auto &model = *loaded.model;
  constexpr int32_t k_dtype_q8_0 = 8;
  const auto flip_to_q8 = [&model](const std::string_view target) {
    for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
      auto &tensor = model.tensors[idx];
      const std::string_view name{
          model.name_storage.data() + tensor.name_offset, tensor.name_length};
      if (name != target) {
        continue;
      }
      tensor.type = k_dtype_q8_0;
      return true;
    }
    return false;
  };

  SUBCASE("mixed transformer projection class") {
    REQUIRE(flip_to_q8(
        "mimi.decoder_transformer.transformer.layers.1.linear2.weight"));
  }
  SUBCASE("mixed rvq projection class") {
    REQUIRE(flip_to_q8("mimi.quantizer.rvq_rest.output_proj.weight"));
  }
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation requires the rvq projections") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // bind_rvq_split consumes the input/output 1x1 projections of both splits
  // before any encode/decode; a codebooks-only probe let a component missing
  // or mis-sizing a projection validate and then fail quantizer bind.
  auto &model = *loaded.model;
  bool corrupted = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != "mimi.quantizer.rvq_rest.input_proj.weight") {
      continue;
    }
    SUBCASE("missing projection") {
      // Swap the underscore so the tensor stays inside the quantizer family
      // but the projection name no longer resolves.
      const size_t underscore = name.find("input_proj") + 5u;
      model.name_storage[tensor.name_offset + underscore] = 'x';
    }
    SUBCASE("wrong projection element count") { tensor.dims[0] += 1; }
    corrupted = true;
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}

TEST_CASE("mimi contract validation requires every semantic rvq codebook") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // Declare a second semantic level the fixture does not carry (and keep the
  // acoustic count at the fixture's three): an existence probe of layers.0
  // alone would still validate while bind_rvq_split later consumes semantic
  // levels 0..semantic_n_q-1 and fails initialization.
  loaded.model->mimi.semantic_n_q = 2;
  loaded.model->mimi.n_q = 5;
  CHECK(moshi::validate_execution_contract(*loaded.model) != k_none);
}

TEST_CASE("mimi contract validation requires every acoustic rvq codebook") {
  auto loaded = load_fixture_or_skip("mimi-tiny.gguf");
  if (loaded.model == nullptr) {
    return;
  }

  // bind_rvq_split consumes every acoustic level 0..n_q-semantic_n_q-1 with
  // the full {codebook_dim, card} shape; corrupt the intermediate one (level
  // 1 of 0..2 in the tiny fixture) so a last-index-only or existence-only
  // probe would still pass while initialization fails.
  auto &model = *loaded.model;
  bool corrupted = false;
  for (uint32_t idx = 0; idx < model.n_tensors; ++idx) {
    auto &tensor = model.tensors[idx];
    const std::string_view name{model.name_storage.data() + tensor.name_offset,
                                tensor.name_length};
    if (name != "mimi.quantizer.rvq_rest.vq.layers.1._codebook.embedding") {
      continue;
    }
    SUBCASE("missing intermediate level") {
      // Bump the layer index digit so the tensor stays inside the quantizer
      // family but level 1 no longer resolves.
      const size_t digit = name.find("layers.1") + 7u;
      model.name_storage[tensor.name_offset + digit] = '9';
    }
    SUBCASE("wrong intermediate codebook shape") { tensor.dims[1] += 1; }
    corrupted = true;
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}
