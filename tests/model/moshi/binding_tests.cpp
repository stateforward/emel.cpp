#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <climits>
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
    SUBCASE("wrong intermediate codebook shape") {
      tensor.dims[1] += 1;
    }
    corrupted = true;
  }
  REQUIRE(corrupted);
  CHECK(moshi::validate_execution_contract(model) != k_none);
}
