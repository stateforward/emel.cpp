#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <type_traits>
#include <vector>

#include "doctest/doctest.h"

#include "emel/cact/loader/any.hpp"
#include "emel/cact/loader/sm.hpp"
#include "emel/machines.hpp"
#include "emel/model/needle/detail.hpp"
#include "emel/model/needle/guards.hpp"

namespace {

// Pinned fixture geometry (tests/models/route-w4-qat.cact), verified against
// the Python exporter (`needle/model/export.py`).
constexpr uint32_t k_fixture_num_tensors = 405u;
constexpr uint32_t k_fixture_vocab = 8192u;
constexpr uint32_t k_fixture_d_model = 512u;
constexpr uint32_t k_fixture_num_layers = 27u;
constexpr uint32_t k_fixture_head_dim = 64u;
constexpr uint32_t k_fixture_hada_n = 512u;
constexpr uint32_t k_fixture_mhc_lanes = 4u;
constexpr uint32_t k_fixture_engram_slots = 8192u;
constexpr uint32_t k_fixture_engram_sub_dim = 128u;
constexpr uint32_t k_fixture_num_engram_tables = 4u;

struct binder_state {
  uint32_t done_count = 0u;
  uint32_t error_count = 0u;
  emel::error::type err = emel::error::cast(emel::model::needle::error::none);
};

binder_state *g_binder_state = nullptr;

struct binder_scope {
  explicit binder_scope(binder_state &state) noexcept {
    g_binder_state = &state;
  }

  ~binder_scope() { g_binder_state = nullptr; }
};

void on_bind_done(const emel::model::needle::events::bind_done &) noexcept {
  if (g_binder_state != nullptr) {
    ++g_binder_state->done_count;
  }
}

void on_bind_error(const emel::model::needle::events::bind_error &ev) noexcept {
  if (g_binder_state == nullptr) {
    return;
  }
  ++g_binder_state->error_count;
  g_binder_state->err = ev.err;
}

const emel::model::needle::event::bind_done_fn k_bind_done_cb =
    emel::model::needle::event::bind_done_fn::from<&on_bind_done>();
const emel::model::needle::event::bind_error_fn k_bind_error_cb =
    emel::model::needle::event::bind_error_fn::from<&on_bind_error>();

void on_loader_probe_done(
    const emel::cact::loader::events::probe_done &) noexcept {}
void on_loader_probe_error(
    const emel::cact::loader::events::probe_error &) noexcept {}
void on_loader_bind_done(
    const emel::cact::loader::events::bind_done &) noexcept {}
void on_loader_bind_error(
    const emel::cact::loader::events::bind_error &) noexcept {}
void on_loader_parse_done(
    const emel::cact::loader::events::parse_done &) noexcept {}
void on_loader_parse_error(
    const emel::cact::loader::events::parse_error &) noexcept {}

const emel::cact::loader::event::probe_done_fn k_loader_probe_done_cb =
    emel::cact::loader::event::probe_done_fn::from<&on_loader_probe_done>();
const emel::cact::loader::event::probe_error_fn k_loader_probe_error_cb =
    emel::cact::loader::event::probe_error_fn::from<&on_loader_probe_error>();
const emel::cact::loader::event::bind_done_fn k_loader_bind_done_cb =
    emel::cact::loader::event::bind_done_fn::from<&on_loader_bind_done>();
const emel::cact::loader::event::bind_error_fn k_loader_bind_error_cb =
    emel::cact::loader::event::bind_error_fn::from<&on_loader_bind_error>();
const emel::cact::loader::event::parse_done_fn k_loader_parse_done_cb =
    emel::cact::loader::event::parse_done_fn::from<&on_loader_parse_done>();
const emel::cact::loader::event::parse_error_fn k_loader_parse_error_cb =
    emel::cact::loader::event::parse_error_fn::from<&on_loader_parse_error>();

std::vector<uint8_t> read_file_bytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());

  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  REQUIRE(size > 0);
  input.seekg(0, std::ios::beg);

  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(input.good());
  return bytes;
}

// Runs the maintained cact loader chain (probe/bind/parse) on the pinned
// fixture, filling `geometry_out` and `tensors_out`.
void load_fixture_tensors(
    const std::vector<uint8_t> &file_bytes,
    emel::cact::loader::geometry &geometry_out,
    std::vector<emel::cact::loader::tensor_view> &tensors_out) {
  emel::cact::loader::sm loader{};

  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry_out,
      k_loader_probe_done_cb,
      k_loader_probe_error_cb,
  };
  REQUIRE(loader.process_event(probe));
  REQUIRE(geometry_out.num_tensors == k_fixture_num_tensors);

  tensors_out.resize(geometry_out.num_tensors);
  const emel::cact::loader::event::bind_storage bind{
      std::span<emel::cact::loader::tensor_view>{tensors_out},
      k_loader_bind_done_cb,
      k_loader_bind_error_cb,
  };
  REQUIRE(loader.process_event(bind));

  const emel::cact::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes},
      k_loader_parse_done_cb,
      k_loader_parse_error_cb,
  };
  REQUIRE(loader.process_event(parse));
}

std::filesystem::path fixture_model_path() {
  return std::filesystem::path{EMEL_TEST_REPO_ROOT} /
         "tests/models/route-w4-qat.cact";
}

bool same_view(const emel::model::needle::tensor_view &named,
               const emel::cact::loader::tensor_view &positional) {
  return named.data == positional.data && named.offset == positional.offset &&
         named.nbytes == positional.nbytes && named.dtype == positional.dtype &&
         named.ndim == positional.ndim && named.shape == positional.shape &&
         named.group == positional.group && named.bits == positional.bits;
}

void check_needle_binder_state_bound(emel::model::needle::sm &machine) {
  CHECK(machine.is(stateforward::sml::state<emel::model::needle::state_bound>));
  std::size_t visited_states = 0u;
  bool saw_bound = false;
  machine.visit_current_states([&](auto state) noexcept {
    ++visited_states;
    using state_t = typename decltype(state)::type;
    if constexpr (std::is_same_v<state_t, emel::model::needle::state_bound>) {
      saw_bound = true;
    }
  });
  CHECK(visited_states == 1u);
  CHECK(saw_bound);
}

} // namespace

TEST_CASE("machine aggregate exports maintained Needle aliases") {
  static_assert(std::is_same_v<emel::CactLoader, emel::cact::loader::sm>);
  static_assert(std::is_same_v<emel::NeedleBinder, emel::model::needle::sm>);
  static_assert(
      std::is_same_v<emel::NeedleGraph, emel::model::needle::graph::sm>);
  static_assert(std::is_same_v<emel::NeedleTokenizerLoader,
                               emel::text::tokenizer::needle::sm>);
}

TEST_CASE("needle binder maps the pinned route-w4-qat fixture to named "
          "tensor roles") {
  const std::vector<uint8_t> file_bytes = read_file_bytes(fixture_model_path());

  emel::cact::loader::geometry geometry = {};
  std::vector<emel::cact::loader::tensor_view> tensors;
  load_fixture_tensors(file_bytes, geometry, tensors);

  emel::model::needle::sm machine{};
  binder_state state = {};
  binder_scope scope{state};

  emel::model::needle::contract contract = {};
  const emel::model::needle::event::bind bind{
      geometry,
      std::span<const emel::cact::loader::tensor_view>{tensors},
      contract,
      k_bind_done_cb,
      k_bind_error_cb,
  };

  CHECK(machine.process_event(bind));
  CHECK(state.done_count == 1u);
  CHECK(state.error_count == 0u);
  check_needle_binder_state_bound(machine);

  // Positional canon order from export.py `_tensors()`:
  // embedding=0; layer i occupies [1 + i*14, 1 + i*14 + 13]; mHC=379..387;
  // engram site s occupies [388 + s*4, 388 + s*4 + 3]; final_norm=396;
  // heads.manifest=397 + two head triples; tokenizer=404.
  CHECK(same_view(contract.embedding, tensors[0]));
  CHECK(contract.embedding.shape[0] == k_fixture_vocab);
  CHECK(contract.embedding.shape[1] == k_fixture_d_model);

  CHECK(contract.layer_count == k_fixture_num_layers);
  const emel::model::needle::layer_views &layer0 = contract.layers[0];
  CHECK(same_view(layer0.norm_in, tensors[1]));
  CHECK(same_view(layer0.q_proj, tensors[2]));
  CHECK(same_view(layer0.k_proj, tensors[3]));
  CHECK(same_view(layer0.v_proj, tensors[4]));
  CHECK(same_view(layer0.q_norm, tensors[5]));
  CHECK(same_view(layer0.k_norm, tensors[6]));
  CHECK(same_view(layer0.gate_proj, tensors[7]));
  CHECK(same_view(layer0.out_proj, tensors[8]));
  CHECK(same_view(layer0.post_norm, tensors[9]));
  CHECK(same_view(layer0.attn_gate, tensors[10]));
  CHECK(same_view(layer0.pre_hada, tensors[11]));
  CHECK(same_view(layer0.d1, tensors[12]));
  CHECK(same_view(layer0.d2, tensors[13]));
  CHECK(same_view(layer0.d3, tensors[14]));
  CHECK(layer0.q_norm.shape[0] == k_fixture_head_dim);
  CHECK(layer0.d1.shape[0] == k_fixture_hada_n);

  const uint32_t last = k_fixture_num_layers - 1u;
  const size_t last_base = 1u + static_cast<size_t>(last) * 14u;
  const emel::model::needle::layer_views &layer26 = contract.layers[last];
  CHECK(same_view(layer26.norm_in, tensors[last_base]));
  CHECK(same_view(layer26.d3, tensors[last_base + 13u]));

  const size_t mhc_base = 1u + static_cast<size_t>(k_fixture_num_layers) * 14u;
  CHECK(same_view(contract.mhc.a_pre, tensors[mhc_base]));
  CHECK(same_view(contract.mhc.b_res, tensors[mhc_base + 5u]));
  CHECK(same_view(contract.mhc.phi_pre, tensors[mhc_base + 6u]));
  CHECK(same_view(contract.mhc.phi_res, tensors[mhc_base + 8u]));
  CHECK(contract.mhc.a_pre.shape[0] == k_fixture_num_layers);
  CHECK(contract.mhc.b_res.shape[1] == k_fixture_mhc_lanes);
  CHECK(contract.mhc.phi_pre.shape[0] ==
        k_fixture_num_layers * k_fixture_mhc_lanes);
  CHECK(contract.mhc.phi_res.shape[0] ==
        k_fixture_num_layers * k_fixture_mhc_lanes * k_fixture_mhc_lanes);

  CHECK(contract.engram_site_count == 2u);
  const size_t engram_base = mhc_base + 9u;
  CHECK(same_view(contract.engram_sites[0].tables, tensors[engram_base]));
  CHECK(same_view(contract.engram_sites[0].taps, tensors[engram_base + 3u]));
  CHECK(same_view(contract.engram_sites[1].tables, tensors[engram_base + 4u]));
  CHECK(same_view(contract.engram_sites[1].taps, tensors[engram_base + 7u]));
  CHECK(contract.engram_sites[0].tables.shape[0] ==
        k_fixture_num_engram_tables * k_fixture_engram_slots);
  CHECK(contract.engram_sites[0].tables.shape[1] == k_fixture_engram_sub_dim);

  const size_t final_norm_index = engram_base + 8u;
  CHECK(same_view(contract.final_norm, tensors[final_norm_index]));
  CHECK(contract.final_norm.shape[0] == k_fixture_d_model);

  CHECK(same_view(contract.head_manifest, tensors[final_norm_index + 1u]));
  CHECK(contract.head_count == 2u);
  CHECK(contract.heads[0].code == emel::model::needle::k_head_code_contrastive);
  CHECK(same_view(contract.heads[0].probes, tensors[final_norm_index + 2u]));
  CHECK(same_view(contract.heads[0].proj, tensors[final_norm_index + 3u]));
  CHECK(same_view(contract.heads[0].bias, tensors[final_norm_index + 4u]));
  CHECK(contract.heads[0].probes.shape[0] == 4u);
  CHECK(contract.heads[0].proj.shape[0] == 128u);
  CHECK(contract.heads[1].code == emel::model::needle::k_head_code_confidence);
  CHECK(same_view(contract.heads[1].probes, tensors[final_norm_index + 5u]));
  CHECK(same_view(contract.heads[1].proj, tensors[final_norm_index + 6u]));
  CHECK(same_view(contract.heads[1].bias, tensors[final_norm_index + 7u]));
  CHECK(contract.heads[1].probes.shape[0] == 8u);
  CHECK(contract.heads[1].proj.shape[0] == 1u);

  CHECK(contract.has_tokenizer);
  CHECK(
      same_view(contract.tokenizer_blob, tensors[k_fixture_num_tensors - 1u]));
  CHECK(contract.tokenizer_blob.dtype ==
        emel::cact::loader::constants::dtype_raw);
}

TEST_CASE("needle binder rejects an empty tensor span as invalid_request") {
  emel::model::needle::sm machine{};
  binder_state state = {};
  binder_scope scope{state};

  emel::cact::loader::geometry geometry = {};
  emel::model::needle::contract contract = {};
  const emel::model::needle::event::bind bind{
      geometry,        std::span<const emel::cact::loader::tensor_view>{},
      contract,        k_bind_done_cb,
      k_bind_error_cb,
  };

  CHECK_FALSE(machine.process_event(bind));
  CHECK(state.done_count == 0u);
  CHECK(state.error_count == 1u);
  CHECK(state.err ==
        emel::error::cast(emel::model::needle::error::invalid_request));
  CHECK(
      machine.is(stateforward::sml::state<emel::model::needle::state_errored>));
}

TEST_CASE("needle binder handles empty public callbacks explicitly") {
  const emel::model::needle::event::bind_done_fn empty_done{};
  const emel::model::needle::event::bind_error_fn empty_error{};

  SUBCASE("a valid bind allows the optional error callback to be absent") {
    const std::vector<uint8_t> file_bytes =
        read_file_bytes(fixture_model_path());
    emel::cact::loader::geometry geometry = {};
    std::vector<emel::cact::loader::tensor_view> tensors;
    load_fixture_tensors(file_bytes, geometry, tensors);

    emel::model::needle::sm machine{};
    binder_state state = {};
    binder_scope scope{state};
    emel::model::needle::contract contract = {};
    const emel::model::needle::event::bind bind{
        geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
        contract, k_bind_done_cb, empty_error};

    CHECK(machine.process_event(bind));
    CHECK(state.done_count == 1u);
    CHECK(state.error_count == 0u);
    check_needle_binder_state_bound(machine);
  }

  SUBCASE("a missing required done callback reports invalid_request") {
    const std::vector<uint8_t> file_bytes =
        read_file_bytes(fixture_model_path());
    emel::cact::loader::geometry geometry = {};
    std::vector<emel::cact::loader::tensor_view> tensors;
    load_fixture_tensors(file_bytes, geometry, tensors);

    emel::model::needle::sm machine{};
    binder_state state = {};
    binder_scope scope{state};
    emel::model::needle::contract contract = {};
    const emel::model::needle::event::bind bind{
        geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
        contract, empty_done, k_bind_error_cb};

    CHECK_FALSE(machine.process_event(bind));
    CHECK(state.done_count == 0u);
    CHECK(state.error_count == 1u);
    CHECK(state.err ==
          emel::error::cast(emel::model::needle::error::invalid_request));
    CHECK(machine.is(
        stateforward::sml::state<emel::model::needle::state_errored>));
  }

  SUBCASE("an invalid bind allows the optional error callback to be absent") {
    emel::model::needle::sm machine{};
    emel::model::needle::contract contract = {};
    emel::cact::loader::geometry geometry = {};
    const emel::model::needle::event::bind bind{
        geometry, std::span<const emel::cact::loader::tensor_view>{}, contract,
        k_bind_done_cb, empty_error};

    CHECK_FALSE(machine.process_event(bind));
    CHECK(machine.is(
        stateforward::sml::state<emel::model::needle::state_errored>));
  }
}

TEST_CASE("needle binder classifies malformed positional tables") {
  const std::vector<uint8_t> file_bytes = read_file_bytes(fixture_model_path());

  emel::cact::loader::geometry geometry = {};
  std::vector<emel::cact::loader::tensor_view> tensors;
  load_fixture_tensors(file_bytes, geometry, tensors);

  SUBCASE("truncated tensor table maps to tensor_count_mismatch") {
    emel::model::needle::sm machine{};
    binder_state state = {};
    binder_scope scope{state};

    emel::model::needle::contract contract = {};
    const emel::model::needle::event::bind bind{
        geometry,
        std::span<const emel::cact::loader::tensor_view>{tensors.data(),
                                                         tensors.size() - 1u},
        contract,
        k_bind_done_cb,
        k_bind_error_cb,
    };
    CHECK_FALSE(machine.process_event(bind));
    CHECK(state.err ==
          emel::error::cast(emel::model::needle::error::tensor_count_mismatch));
    CHECK(machine.is(
        stateforward::sml::state<emel::model::needle::state_errored>));
  }

  SUBCASE("wrong embedding dtype maps to tensor_dtype_mismatch") {
    std::vector<emel::cact::loader::tensor_view> corrupted = tensors;
    corrupted[0].dtype = emel::cact::loader::constants::dtype_fp16;

    emel::model::needle::sm machine{};
    binder_state state = {};
    binder_scope scope{state};

    emel::model::needle::contract contract = {};
    const emel::model::needle::event::bind bind{
        geometry,
        std::span<const emel::cact::loader::tensor_view>{corrupted},
        contract,
        k_bind_done_cb,
        k_bind_error_cb,
    };
    CHECK_FALSE(machine.process_event(bind));
    CHECK(state.err ==
          emel::error::cast(emel::model::needle::error::tensor_dtype_mismatch));
    CHECK(machine.is(
        stateforward::sml::state<emel::model::needle::state_errored>));
  }

  SUBCASE("wrong layer q_proj shape maps to tensor_shape_mismatch") {
    std::vector<emel::cact::loader::tensor_view> corrupted = tensors;
    corrupted[2].shape[0] += 1u;

    emel::model::needle::sm machine{};
    binder_state state = {};
    binder_scope scope{state};

    emel::model::needle::contract contract = {};
    const emel::model::needle::event::bind bind{
        geometry,
        std::span<const emel::cact::loader::tensor_view>{corrupted},
        contract,
        k_bind_done_cb,
        k_bind_error_cb,
    };
    CHECK_FALSE(machine.process_event(bind));
    CHECK(state.err ==
          emel::error::cast(emel::model::needle::error::tensor_shape_mismatch));
    CHECK(machine.is(
        stateforward::sml::state<emel::model::needle::state_errored>));
  }

  SUBCASE("unknown head manifest code maps to head_manifest_invalid") {
    // Head manifest is the fp16 vector after final_norm; 3.0 is not a
    // canonical head code.
    std::vector<uint8_t> corrupted_bytes = file_bytes;
    const size_t manifest_index =
        static_cast<size_t>(geometry.num_tensors) - 1u - 2u * 3u - 1u;
    const uint64_t manifest_offset = tensors[manifest_index].offset;
    corrupted_bytes[static_cast<size_t>(manifest_offset)] = 0x00u;
    corrupted_bytes[static_cast<size_t>(manifest_offset) + 1u] =
        0x42u; // 3.0f16

    emel::cact::loader::geometry corrupt_geometry = {};
    std::vector<emel::cact::loader::tensor_view> corrupt_tensors;
    load_fixture_tensors(corrupted_bytes, corrupt_geometry, corrupt_tensors);

    emel::model::needle::sm machine{};
    binder_state state = {};
    binder_scope scope{state};

    emel::model::needle::contract contract = {};
    const emel::model::needle::event::bind bind{
        corrupt_geometry,
        std::span<const emel::cact::loader::tensor_view>{corrupt_tensors},
        contract,
        k_bind_done_cb,
        k_bind_error_cb,
    };
    CHECK_FALSE(machine.process_event(bind));
    CHECK(state.err ==
          emel::error::cast(emel::model::needle::error::head_manifest_invalid));
    CHECK(machine.is(
        stateforward::sml::state<emel::model::needle::state_errored>));
  }

  SUBCASE("zero d_model geometry maps to geometry_invalid") {
    emel::cact::loader::geometry corrupt_geometry = geometry;
    corrupt_geometry.d_model = 0u;

    emel::model::needle::sm machine{};
    binder_state state = {};
    binder_scope scope{state};

    emel::model::needle::contract contract = {};
    const emel::model::needle::event::bind bind{
        corrupt_geometry,
        std::span<const emel::cact::loader::tensor_view>{tensors},
        contract,
        k_bind_done_cb,
        k_bind_error_cb,
    };
    CHECK_FALSE(machine.process_event(bind));
    CHECK(state.err ==
          emel::error::cast(emel::model::needle::error::geometry_invalid));
    CHECK(machine.is(
        stateforward::sml::state<emel::model::needle::state_errored>));
  }
}

TEST_CASE("needle binder allows re-binding after an error") {
  const std::vector<uint8_t> file_bytes = read_file_bytes(fixture_model_path());

  emel::cact::loader::geometry geometry = {};
  std::vector<emel::cact::loader::tensor_view> tensors;
  load_fixture_tensors(file_bytes, geometry, tensors);

  emel::model::needle::sm machine{};
  binder_state state = {};
  binder_scope scope{state};

  emel::model::needle::contract contract = {};
  const emel::model::needle::event::bind bad_bind{
      geometry,        std::span<const emel::cact::loader::tensor_view>{},
      contract,        k_bind_done_cb,
      k_bind_error_cb,
  };
  CHECK_FALSE(machine.process_event(bad_bind));
  CHECK(
      machine.is(stateforward::sml::state<emel::model::needle::state_errored>));

  const emel::model::needle::event::bind good_bind{
      geometry,
      std::span<const emel::cact::loader::tensor_view>{tensors},
      contract,
      k_bind_done_cb,
      k_bind_error_cb,
  };
  CHECK(machine.process_event(good_bind));
  CHECK(state.done_count == 1u);
  check_needle_binder_state_bound(machine);
}

TEST_CASE(
    "needle binder rejects geometry capacity and engram inconsistencies") {
  const std::vector<uint8_t> file_bytes = read_file_bytes(fixture_model_path());
  emel::cact::loader::geometry geometry = {};
  std::vector<emel::cact::loader::tensor_view> tensors;
  load_fixture_tensors(file_bytes, geometry, tensors);

  const auto check_geometry_error =
      [&](const emel::cact::loader::geometry &bad) {
        emel::model::needle::sm machine{};
        binder_state state = {};
        binder_scope scope{state};
        emel::model::needle::contract contract = {};
        const emel::model::needle::event::bind bind{
            bad, std::span<const emel::cact::loader::tensor_view>{tensors},
            contract, k_bind_done_cb, k_bind_error_cb};
        CHECK_FALSE(machine.process_event(bind));
        CHECK(state.done_count == 0u);
        CHECK(state.error_count == 1u);
        CHECK(state.err ==
              emel::error::cast(emel::model::needle::error::geometry_invalid));
        CHECK(contract.layer_count == 0u);
      };

  SUBCASE("too many layers") {
    auto bad = geometry;
    bad.num_layers = emel::model::needle::k_max_layers + 1u;
    check_geometry_error(bad);
  }
  SUBCASE("too many engram sites") {
    auto bad = geometry;
    bad.num_engram_sites = emel::model::needle::k_max_engram_sites + 1u;
    check_geometry_error(bad);
  }
  SUBCASE("engram geometry requires nonzero storage dimensions") {
    auto bad = geometry;
    bad.engram_slots = 0u;
    check_geometry_error(bad);
  }

  SUBCASE("derived attention geometry must not overflow") {
    auto bad = geometry;
    bad.num_heads = UINT32_MAX;
    bad.head_dim = 2u;
    check_geometry_error(bad);
  }
  SUBCASE("derived mHC geometry must not overflow") {
    auto bad = geometry;
    bad.mhc_lanes = UINT32_MAX;
    check_geometry_error(bad);
  }
  SUBCASE("derived engram geometry must not overflow") {
    auto bad = geometry;
    bad.num_engram_tables = UINT32_MAX;
    bad.engram_slots = 2u;
    check_geometry_error(bad);
  }
  SUBCASE("dilated engram history extent must not overflow") {
    auto bad = geometry;
    bad.engram_conv_taps = UINT32_MAX;
    bad.engram_conv_dilation = UINT32_MAX;
    check_geometry_error(bad);
  }
  SUBCASE("engram order must fit the checked hash window") {
    auto bad = geometry;
    bad.engram_orders[0] = UINT32_MAX;
    check_geometry_error(bad);
  }
  SUBCASE("engram site layer must be in range") {
    auto bad = geometry;
    bad.engram_sites[0] = bad.num_layers;
    check_geometry_error(bad);
  }
  SUBCASE("engram site layer assignments must be unique") {
    auto bad = geometry;
    REQUIRE(bad.num_engram_sites >= 2u);
    bad.engram_sites[1] = bad.engram_sites[0];
    check_geometry_error(bad);
  }
}

TEST_CASE("needle binder classifies failures in later positional sections") {
  const std::vector<uint8_t> file_bytes = read_file_bytes(fixture_model_path());
  emel::cact::loader::geometry geometry = {};
  std::vector<emel::cact::loader::tensor_view> tensors;
  load_fixture_tensors(file_bytes, geometry, tensors);
  const size_t mhc_base = 1u + static_cast<size_t>(geometry.num_layers) *
                                   emel::model::needle::k_layer_tensor_count;
  const size_t engram_base = mhc_base + emel::model::needle::k_mhc_tensor_count;
  const size_t final_norm_index =
      engram_base + static_cast<size_t>(geometry.num_engram_sites) *
                        emel::model::needle::k_engram_site_tensor_count;
  const size_t manifest_index = final_norm_index + 1u;

  const auto check_bind_error =
      [&](std::vector<emel::cact::loader::tensor_view> bad,
          const emel::model::needle::error expected) {
        emel::model::needle::sm machine{};
        binder_state state = {};
        binder_scope scope{state};
        emel::model::needle::contract contract = {};
        const emel::model::needle::event::bind bind{
            geometry, std::span<const emel::cact::loader::tensor_view>{bad},
            contract, k_bind_done_cb, k_bind_error_cb};
        CHECK_FALSE(machine.process_event(bind));
        CHECK(state.done_count == 0u);
        CHECK(state.error_count == 1u);
        CHECK(state.err == emel::error::cast(expected));
        CHECK(machine.is(
            stateforward::sml::state<emel::model::needle::state_errored>));
      };

  SUBCASE("mHC shape mismatch") {
    auto bad = tensors;
    bad[mhc_base].ndim = 2u;
    check_bind_error(std::move(bad),
                     emel::model::needle::error::tensor_shape_mismatch);
  }
  SUBCASE("engram shape mismatch") {
    auto bad = tensors;
    bad[engram_base].shape[0] += 1u;
    check_bind_error(std::move(bad),
                     emel::model::needle::error::tensor_shape_mismatch);
  }
  SUBCASE("final norm dtype mismatch") {
    auto bad = tensors;
    bad[final_norm_index].dtype = emel::cact::loader::constants::dtype_cq;
    check_bind_error(std::move(bad),
                     emel::model::needle::error::tensor_dtype_mismatch);
  }
  SUBCASE("manifest payload is required") {
    auto bad = tensors;
    bad[manifest_index].data = nullptr;
    check_bind_error(std::move(bad),
                     emel::model::needle::error::head_manifest_invalid);
  }
  SUBCASE("probe rows must be nonzero") {
    auto bad = tensors;
    bad[manifest_index + 1u].shape[0] = 0u;
    check_bind_error(std::move(bad),
                     emel::model::needle::error::tensor_shape_mismatch);
  }
  SUBCASE("projection rows must be nonzero") {
    auto bad = tensors;
    bad[manifest_index + 2u].shape[0] = 0u;
    check_bind_error(std::move(bad),
                     emel::model::needle::error::tensor_shape_mismatch);
  }
  SUBCASE("bias shape follows projection rows") {
    auto bad = tensors;
    bad[manifest_index + 3u].shape[0] += 1u;
    check_bind_error(std::move(bad),
                     emel::model::needle::error::tensor_shape_mismatch);
  }
}

TEST_CASE(
    "needle binder accepts the base contract without heads or tokenizer") {
  const std::vector<uint8_t> file_bytes = read_file_bytes(fixture_model_path());
  emel::cact::loader::geometry geometry = {};
  std::vector<emel::cact::loader::tensor_view> tensors;
  load_fixture_tensors(file_bytes, geometry, tensors);
  const size_t base_count =
      1u +
      static_cast<size_t>(geometry.num_layers) *
          emel::model::needle::k_layer_tensor_count +
      emel::model::needle::k_mhc_tensor_count +
      static_cast<size_t>(geometry.num_engram_sites) *
          emel::model::needle::k_engram_site_tensor_count +
      1u;
  tensors.resize(base_count);
  geometry.num_tensors = static_cast<uint32_t>(tensors.size());

  emel::model::needle::sm machine{};
  binder_state state = {};
  binder_scope scope{state};
  emel::model::needle::contract contract = {};
  const emel::model::needle::event::bind bind{
      geometry, std::span<const emel::cact::loader::tensor_view>{tensors},
      contract, k_bind_done_cb, k_bind_error_cb};
  REQUIRE(machine.process_event(bind));
  CHECK(state.done_count == 1u);
  CHECK(state.error_count == 0u);
  CHECK(contract.head_count == 0u);
  CHECK_FALSE(contract.has_tokenizer);
  check_needle_binder_state_bound(machine);
}
