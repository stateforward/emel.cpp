#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "doctest/doctest.h"

#include "emel/cact/loader/sm.hpp"
#include "emel/model/needle/sm.hpp"
#include "emel/text/tokenizer/needle/detail.hpp"
#include "emel/text/tokenizer/needle/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

struct loader_state {
  uint32_t done_count = 0u;
  uint32_t error_count = 0u;
  emel::error::type err =
      emel::error::cast(emel::text::tokenizer::needle::error::none);
};

loader_state *g_loader_state = nullptr;

struct loader_scope {
  explicit loader_scope(loader_state &state) noexcept {
    g_loader_state = &state;
  }

  ~loader_scope() { g_loader_state = nullptr; }
};

void on_load_done(const emel::text::tokenizer::needle::events::load_done &) {
  if (g_loader_state != nullptr) {
    ++g_loader_state->done_count;
  }
}

void on_load_error(
    const emel::text::tokenizer::needle::events::load_error &ev) {
  if (g_loader_state == nullptr) {
    return;
  }
  ++g_loader_state->error_count;
  g_loader_state->err = ev.err;
}

const emel::text::tokenizer::needle::event::load_done_fn k_load_done_cb =
    emel::text::tokenizer::needle::event::load_done_fn::from<&on_load_done>();
const emel::text::tokenizer::needle::event::load_error_fn k_load_error_cb =
    emel::text::tokenizer::needle::event::load_error_fn::from<&on_load_error>();

void on_cact_probe_done(const emel::cact::loader::events::probe_done &) {}
void on_cact_probe_error(const emel::cact::loader::events::probe_error &) {}
void on_cact_bind_done(const emel::cact::loader::events::bind_done &) {}
void on_cact_bind_error(const emel::cact::loader::events::bind_error &) {}
void on_cact_parse_done(const emel::cact::loader::events::parse_done &) {}
void on_cact_parse_error(const emel::cact::loader::events::parse_error &) {}

const emel::cact::loader::event::probe_done_fn k_cact_probe_done_cb =
    emel::cact::loader::event::probe_done_fn::from<&on_cact_probe_done>();
const emel::cact::loader::event::probe_error_fn k_cact_probe_error_cb =
    emel::cact::loader::event::probe_error_fn::from<&on_cact_probe_error>();
const emel::cact::loader::event::bind_done_fn k_cact_bind_done_cb =
    emel::cact::loader::event::bind_done_fn::from<&on_cact_bind_done>();
const emel::cact::loader::event::bind_error_fn k_cact_bind_error_cb =
    emel::cact::loader::event::bind_error_fn::from<&on_cact_bind_error>();
const emel::cact::loader::event::parse_done_fn k_cact_parse_done_cb =
    emel::cact::loader::event::parse_done_fn::from<&on_cact_parse_done>();
const emel::cact::loader::event::parse_error_fn k_cact_parse_error_cb =
    emel::cact::loader::event::parse_error_fn::from<&on_cact_parse_error>();

void on_needle_bind_done(const emel::model::needle::events::bind_done &) {}
void on_needle_bind_error(const emel::model::needle::events::bind_error &) {}

const emel::model::needle::event::bind_done_fn k_needle_bind_done_cb =
    emel::model::needle::event::bind_done_fn::from<&on_needle_bind_done>();
const emel::model::needle::event::bind_error_fn k_needle_bind_error_cb =
    emel::model::needle::event::bind_error_fn::from<&on_needle_bind_error>();

std::filesystem::path repo_relative(const char *relative) {
  return std::filesystem::path{EMEL_TEST_REPO_ROOT} / relative;
}

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

// Extracts the trailing RAW tokenizer blob from the pinned fixture through
// the maintained cact loader + needle binder chain.
std::span<const uint8_t>
fixture_tokenizer_blob(const std::vector<uint8_t> &file_bytes,
                       std::vector<emel::cact::loader::tensor_view> &tensors,
                       emel::model::needle::contract &contract) {
  emel::cact::loader::sm loader{};
  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_cact_probe_done_cb,
      k_cact_probe_error_cb,
  };
  REQUIRE(loader.process_event(probe));

  tensors.resize(geometry.num_tensors);
  const emel::cact::loader::event::bind_storage bind_storage{
      std::span<emel::cact::loader::tensor_view>{tensors},
      k_cact_bind_done_cb,
      k_cact_bind_error_cb,
  };
  REQUIRE(loader.process_event(bind_storage));

  const emel::cact::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes},
      k_cact_parse_done_cb,
      k_cact_parse_error_cb,
  };
  REQUIRE(loader.process_event(parse));

  emel::model::needle::sm binder{};
  const emel::model::needle::event::bind bind{
      geometry,
      std::span<const emel::cact::loader::tensor_view>{tensors},
      contract,
      k_needle_bind_done_cb,
      k_needle_bind_error_cb,
  };
  REQUIRE(binder.process_event(bind));
  REQUIRE(contract.has_tokenizer);
  return std::span<const uint8_t>{
      contract.tokenizer_blob.data,
      static_cast<size_t>(contract.tokenizer_blob.nbytes)};
}

struct tokenizer_fixture_header {
  uint32_t n_pieces = 0u;
  int32_t pad_id = -1;
  int32_t eos_id = -1;
  int32_t bos_id = -1;
  int32_t unk_id = -1;
  uint32_t add_dummy_prefix = 0u;
  uint32_t byte_fallback = 0u;
};

struct tokenizer_fixture_piece {
  uint32_t id = 0u;
  float score = 0.0f;
  int32_t type = 0;
  std::string surface = {};
};

// Parses the committed parity CSV produced by
// scripts/gen_cact_tokenizer_csv.py from the pinned fixture (raw struct
// dump, same byte-level source of truth as the Python exporter).
void read_tokenizer_fixture(const std::filesystem::path &path,
                            tokenizer_fixture_header &header_out,
                            std::vector<tokenizer_fixture_piece> &pieces_out) {
  std::ifstream input(path);
  REQUIRE(input.good());

  std::string line;
  REQUIRE(static_cast<bool>(std::getline(input, line))); // header row
  REQUIRE(static_cast<bool>(std::getline(input, line)));
  REQUIRE(std::sscanf(line.c_str(), "%u,%d,%d,%d,%d,%u,%u",
                      &header_out.n_pieces, &header_out.pad_id,
                      &header_out.eos_id, &header_out.bos_id,
                      &header_out.unk_id, &header_out.add_dummy_prefix,
                      &header_out.byte_fallback) == 7);
  REQUIRE(static_cast<bool>(std::getline(input, line))); // piece header row

  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    tokenizer_fixture_piece piece = {};
    std::array<char, 512> surface_hex = {};
    REQUIRE(std::sscanf(line.c_str(), "%u,%f,%d,%511s", &piece.id, &piece.score,
                        &piece.type, surface_hex.data()) == 4);
    const std::string hex{surface_hex.data()};
    REQUIRE(hex.size() % 2u == 0u);
    for (size_t i = 0; i < hex.size(); i += 2u) {
      piece.surface.push_back(
          static_cast<char>(std::stoi(hex.substr(i, 2u), nullptr, 16)));
    }
    pieces_out.push_back(piece);
  }
  REQUIRE(!pieces_out.empty());
}

// Piece-type codes as stored in the shared vocab (GGUF convention).
constexpr int32_t k_vocab_type_from_blob[5] = {1, 2, 3, 4, 6};

std::string_view vocab_token_text(const emel::model::data::vocab &vocab,
                                  const uint32_t id) {
  return std::string_view{vocab.token_storage.data() +
                              vocab.entries[id].text_offset,
                          vocab.entries[id].text_length};
}

std::vector<int32_t> encode_text(emel::text::tokenizer::sm &tokenizer,
                                 const emel::model::data::vocab &vocab,
                                 const std::string_view text) {
  std::vector<int32_t> tokens(text.size() * 4u + 8u);
  int32_t count = 0;
  int32_t err =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  emel::text::tokenizer::event::tokenize tokenize = {};
  tokenize.vocab = &vocab;
  tokenize.text = text;
  tokenize.add_special = false;
  tokenize.parse_special = true;
  tokenize.token_ids_out = tokens.data();
  tokenize.token_capacity = static_cast<int32_t>(tokens.size());
  tokenize.token_count_out = &count;
  tokenize.error_out = &err;

  REQUIRE(tokenizer.process_event(tokenize));
  REQUIRE(err == emel::text::tokenizer::error_code(
                     emel::text::tokenizer::error::none));
  REQUIRE(count >= 0);
  tokens.resize(static_cast<size_t>(count));
  return tokens;
}

emel::text::tokenizer::sm &
bind_shared_spm_tokenizer(emel::text::tokenizer::sm &tokenizer,
                          const emel::model::data::vocab &vocab) {
  int32_t err =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  emel::text::tokenizer::event::bind bind = {};
  bind.vocab = &vocab;
  bind.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
  bind.encoder_variant = emel::text::encoders::encoder_kind::spm;
  bind.error_out = &err;
  REQUIRE(tokenizer.process_event(bind));
  REQUIRE(err == emel::text::tokenizer::error_code(
                     emel::text::tokenizer::error::none));
  return tokenizer;
}

uint64_t hash_token_ids(const std::span<const int32_t> ids,
                        uint64_t hash = 1469598103934665603ULL) noexcept {
  constexpr uint64_t k_fnv_prime = 1099511628211ULL;
  for (const int32_t id : ids) {
    const uint32_t value = static_cast<uint32_t>(id);
    for (uint32_t byte = 0u; byte < 4u; ++byte) {
      hash ^= static_cast<uint8_t>(value >> (byte * 8u));
      hash *= k_fnv_prime;
    }
  }
  return hash;
}

std::string decode_hex(const std::string_view hex) {
  REQUIRE(hex.size() % 2u == 0u);
  std::string decoded;
  decoded.reserve(hex.size() / 2u);
  for (size_t i = 0u; i < hex.size(); i += 2u) {
    decoded.push_back(static_cast<char>(
        std::stoi(std::string{hex.substr(i, 2u)}, nullptr, 16)));
  }
  return decoded;
}

struct heldout_row {
  std::vector<int32_t> reference_ids = {};
  std::string prompt = {};
};

std::vector<heldout_row> read_heldout_rows(const std::filesystem::path &path) {
  std::ifstream input(path);
  REQUIRE(input.good());

  std::vector<heldout_row> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const size_t first_tab = line.find('\t');
    const size_t second_tab = line.find('\t', first_tab + 1u);
    const size_t third_tab = line.find('\t', second_tab + 1u);
    REQUIRE(first_tab != std::string::npos);
    REQUIRE(second_tab != std::string::npos);
    REQUIRE(third_tab != std::string::npos);

    heldout_row row = {};
    const std::string ids_text =
        line.substr(second_tab + 1u, third_tab - second_tab - 1u);
    size_t cursor = 0u;
    while (cursor < ids_text.size()) {
      char *end = nullptr;
      const long value = std::strtol(ids_text.c_str() + cursor, &end, 10);
      REQUIRE(end != ids_text.c_str() + cursor);
      row.reference_ids.push_back(static_cast<int32_t>(value));
      cursor = static_cast<size_t>(end - ids_text.c_str());
      while (cursor < ids_text.size() && ids_text[cursor] == ' ') {
        ++cursor;
      }
    }
    row.prompt = decode_hex(std::string_view{line}.substr(third_tab + 1u));
    rows.push_back(std::move(row));
  }
  return rows;
}

std::unique_ptr<emel::model::data::vocab> load_pinned_needle_vocab() {
  const std::vector<uint8_t> file_bytes =
      read_file_bytes(repo_relative("tests/models/route-w4-qat.cact"));
  std::vector<emel::cact::loader::tensor_view> tensors;
  emel::model::needle::contract contract = {};
  const std::span<const uint8_t> blob =
      fixture_tokenizer_blob(file_bytes, tensors, contract);
  auto vocab = std::make_unique<emel::model::data::vocab>();
  emel::text::tokenizer::needle::sm loader{};
  loader_state state = {};
  loader_scope scope{state};
  REQUIRE(loader.process_event(emel::text::tokenizer::needle::event::load{
      blob, *vocab, k_load_done_cb, k_load_error_cb}));
  REQUIRE(state.done_count == 1u);
  REQUIRE(state.error_count == 0u);
  return vocab;
}

void check_reference_ids(emel::text::tokenizer::sm &tokenizer,
                         const emel::model::data::vocab &vocab,
                         const std::string_view text,
                         const std::span<const int32_t> expected) {
  const std::vector<int32_t> actual = encode_text(tokenizer, vocab, text);
  CAPTURE(text);
  REQUIRE(actual.size() == expected.size());
  for (size_t i = 0u; i < expected.size(); ++i) {
    CAPTURE(text);
    CAPTURE(i);
    CHECK(actual[i] == expected[i]);
  }
}

} // namespace

TEST_CASE("needle tokenizer loader parses the pinned fixture blob with "
          "piece parity") {
  const std::vector<uint8_t> file_bytes =
      read_file_bytes(repo_relative("tests/models/route-w4-qat.cact"));
  std::vector<emel::cact::loader::tensor_view> tensors;
  emel::model::needle::contract contract = {};
  const std::span<const uint8_t> blob =
      fixture_tokenizer_blob(file_bytes, tensors, contract);

  tokenizer_fixture_header expected_header = {};
  std::vector<tokenizer_fixture_piece> expected_pieces;
  read_tokenizer_fixture(
      repo_relative("tests/fixtures/cact/route-w4-qat.tokenizer.csv"),
      expected_header, expected_pieces);

  emel::text::tokenizer::needle::sm machine{};
  loader_state state = {};
  loader_scope scope{state};

  auto vocab = std::make_unique<emel::model::data::vocab>();
  const emel::text::tokenizer::needle::event::load load{
      blob,
      *vocab,
      k_load_done_cb,
      k_load_error_cb,
  };

  CHECK(machine.process_event(load));
  CHECK(state.done_count == 1u);
  CHECK(state.error_count == 0u);
  CHECK(machine.is(
      stateforward::sml::state<emel::text::tokenizer::needle::state_loaded>));
  std::size_t visited_states = 0u;
  bool saw_loaded = false;
  machine.visit_current_states([&](auto state_id) noexcept {
    ++visited_states;
    using state_t = typename decltype(state_id)::type;
    if constexpr (std::is_same_v<state_t,
                                 emel::text::tokenizer::needle::state_loaded>) {
      saw_loaded = true;
    }
  });
  CHECK(visited_states == 1u);
  CHECK(saw_loaded);

  CHECK(vocab->n_tokens == expected_header.n_pieces);
  CHECK(vocab->pad_id == expected_header.pad_id);
  CHECK(vocab->eos_id == expected_header.eos_id);
  CHECK(vocab->bos_id == expected_header.bos_id);
  CHECK(vocab->unk_id == expected_header.unk_id);
  CHECK(vocab->add_space_prefix == (expected_header.add_dummy_prefix != 0u));
  CHECK(vocab->tokenizer_model_id == emel::model::data::tokenizer_model::SPM);
  CHECK(vocab->add_bos == false);
  CHECK(vocab->add_eos == false);
  CHECK(vocab->tokenizer_pre_id == emel::model::data::tokenizer_pre::NEEDLE);

  for (const tokenizer_fixture_piece &piece : expected_pieces) {
    CAPTURE(piece.id);
    REQUIRE(piece.id < vocab->n_tokens);
    CHECK(vocab_token_text(*vocab, piece.id) == piece.surface);
    CHECK(vocab->entries[piece.id].score == doctest::Approx(piece.score));
    REQUIRE(piece.type >= 0);
    REQUIRE(piece.type < 5);
    CHECK(vocab->entries[piece.id].type == k_vocab_type_from_blob[piece.type]);
  }
}

TEST_CASE("needle tokenizer loader drives the shared SPM tokenizer machine") {
  const std::vector<uint8_t> file_bytes =
      read_file_bytes(repo_relative("tests/models/route-w4-qat.cact"));
  std::vector<emel::cact::loader::tensor_view> tensors;
  emel::model::needle::contract contract = {};
  const std::span<const uint8_t> blob =
      fixture_tokenizer_blob(file_bytes, tensors, contract);

  emel::text::tokenizer::needle::sm machine{};
  loader_state state = {};
  loader_scope scope{state};

  auto vocab = std::make_unique<emel::model::data::vocab>();
  const emel::text::tokenizer::needle::event::load load{
      blob,
      *vocab,
      k_load_done_cb,
      k_load_error_cb,
  };
  REQUIRE(machine.process_event(load));

  emel::text::tokenizer::sm tokenizer{};
  int32_t bind_err =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  emel::text::tokenizer::event::bind bind_ev = {};
  bind_ev.vocab = vocab.get();
  bind_ev.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
  bind_ev.encoder_variant = emel::text::encoders::encoder_kind::spm;
  bind_ev.error_out = &bind_err;
  CHECK(tokenizer.process_event(bind_ev));
  CHECK(bind_err ==
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none));

  // Expected ids computed with the reference encoder
  // (`needle/model/export.py` RefTokenizer) on the pinned fixture blob:
  // "hello world" -> [323, 636, 8048, 2328] (pieces "_h", "ell", "o",
  // "_world" with SentencePiece meta-space).
  std::array<int32_t, 16> tokens = {};
  int32_t count = 0;
  int32_t tok_err =
      emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
  emel::text::tokenizer::event::tokenize tok_ev = {};
  tok_ev.vocab = vocab.get();
  tok_ev.text = std::string_view("hello world");
  tok_ev.add_special = false;
  tok_ev.parse_special = false;
  tok_ev.token_ids_out = tokens.data();
  tok_ev.token_capacity = static_cast<int32_t>(tokens.size());
  tok_ev.token_count_out = &count;
  tok_ev.error_out = &tok_err;

  CHECK(tokenizer.process_event(tok_ev));
  CHECK(tok_err ==
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none));
  CHECK(count == 4);
  CHECK(tokens[0] == 323);
  CHECK(tokens[1] == 636);
  CHECK(tokens[2] == 8048);
  CHECK(tokens[3] == 2328);

  // Round-trip: the emitted pieces reassemble the input text under the
  // SentencePiece meta-space convention.
  std::string round_trip;
  for (int32_t i = 0; i < count; ++i) {
    const std::string_view piece =
        vocab_token_text(*vocab, static_cast<uint32_t>(tokens[i]));
    round_trip.append(piece);
  }
  std::string expected = "\xE2\x96\x81hello\xE2\x96\x81world";
  CHECK(round_trip == expected);
}

TEST_CASE(
    "needle tokenizer matches RefTokenizer across chat marker boundaries") {
  const auto vocab = load_pinned_needle_vocab();
  emel::text::tokenizer::sm tokenizer{};
  bind_shared_spm_tokenizer(tokenizer, *vocab);

  const std::array<int32_t, 4> start_user = {8042, 4, 573, 24};
  check_reference_ids(tokenizer, *vocab, "<|im_start|>user\n", start_user);

  const std::array<int32_t, 5> marker_join = {8042, 5, 24, 4, 612};
  check_reference_ids(tokenizer, *vocab, "<|im_end|>\n<|im_start|>assistant",
                      marker_join);

  const std::array<int32_t, 6> leading_spaces = {8042, 8042, 323,
                                                 636,  8048, 2328};
  check_reference_ids(tokenizer, *vocab, "  hello world", leading_spaces);

  const std::array<int32_t, 12> rendered = {8042, 4, 573, 24, 8056, 636,
                                            8048, 5, 24,  4,  612,  24};
  check_reference_ids(
      tokenizer, *vocab,
      "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", rendered);

  const std::array<int32_t, 7> utf8_fallback = {281, 698, 2018, 8042,
                                                240, 166, 145};
  check_reference_ids(tokenizer, *vocab, "snowman \xE2\x98\x83", utf8_fallback);

  const std::array<int32_t, 2> nul_fallback = {8042, 14};
  check_reference_ids(tokenizer, *vocab, std::string_view{"\0", 1u},
                      nul_fallback);
}

TEST_CASE("needle tokenizer matches RefTokenizer for all heldout prompts") {
  auto heldout_path = repo_relative("build/needle_eval/heldout_prompts.tsv");
  if (!std::filesystem::exists(heldout_path)) {
    heldout_path =
        repo_relative("tests/fixtures/cact/needle-heldout-prompts.tsv");
  }
  const auto vocab = load_pinned_needle_vocab();
  emel::text::tokenizer::sm tokenizer{};
  bind_shared_spm_tokenizer(tokenizer, *vocab);
  const std::vector<heldout_row> rows = read_heldout_rows(heldout_path);
  REQUIRE(rows.size() == 287u);

  uint64_t reference_hash = 1469598103934665603ULL;
  uint64_t native_hash = 1469598103934665603ULL;
  for (size_t i = 0u; i < rows.size(); ++i) {
    CAPTURE(i);
    const std::vector<int32_t> actual =
        encode_text(tokenizer, *vocab, rows[i].prompt);
    CHECK(actual == rows[i].reference_ids);
    reference_hash = hash_token_ids(rows[i].reference_ids, reference_hash);
    native_hash = hash_token_ids(actual, native_hash);
  }
  CHECK(reference_hash == 0xafe7916cf70b4801ULL);
  CHECK(native_hash == reference_hash);
}

TEST_CASE("needle tokenizer loader rejects malformed blobs") {
  const std::vector<uint8_t> file_bytes =
      read_file_bytes(repo_relative("tests/models/route-w4-qat.cact"));
  std::vector<emel::cact::loader::tensor_view> tensors;
  emel::model::needle::contract contract = {};
  const std::span<const uint8_t> blob =
      fixture_tokenizer_blob(file_bytes, tensors, contract);

  SUBCASE("empty blob maps to invalid_request") {
    emel::text::tokenizer::needle::sm machine{};
    loader_state state = {};
    loader_scope scope{state};

    auto vocab = std::make_unique<emel::model::data::vocab>();
    const emel::text::tokenizer::needle::event::load load{
        std::span<const uint8_t>{},
        *vocab,
        k_load_done_cb,
        k_load_error_cb,
    };
    CHECK_FALSE(machine.process_event(load));
    CHECK(state.err ==
          emel::error::cast(
              emel::text::tokenizer::needle::error::invalid_request));
    CHECK(machine.is(stateforward::sml::state<
                     emel::text::tokenizer::needle::state_errored>));
  }

  SUBCASE("truncated record stream maps to parse_failed") {
    const std::vector<uint8_t> truncated{blob.begin(),
                                         blob.begin() + blob.size() / 2u};

    emel::text::tokenizer::needle::sm machine{};
    loader_state state = {};
    loader_scope scope{state};

    auto vocab = std::make_unique<emel::model::data::vocab>();
    const emel::text::tokenizer::needle::event::load load{
        std::span<const uint8_t>{truncated},
        *vocab,
        k_load_done_cb,
        k_load_error_cb,
    };
    CHECK_FALSE(machine.process_event(load));
    CHECK(state.err == emel::error::cast(
                           emel::text::tokenizer::needle::error::parse_failed));
    CHECK(machine.is(stateforward::sml::state<
                     emel::text::tokenizer::needle::state_errored>));
  }

  SUBCASE("out-of-range special id maps to model_invalid") {
    std::vector<uint8_t> corrupted{blob.begin(), blob.end()};
    // unk_id field is bytes [16, 20) of the header.
    corrupted[16] = 0xffu;
    corrupted[17] = 0xffu;
    corrupted[18] = 0xffu;
    corrupted[19] = 0xffu;

    emel::text::tokenizer::needle::sm machine{};
    loader_state state = {};
    loader_scope scope{state};

    auto vocab = std::make_unique<emel::model::data::vocab>();
    const emel::text::tokenizer::needle::event::load load{
        std::span<const uint8_t>{corrupted},
        *vocab,
        k_load_done_cb,
        k_load_error_cb,
    };
    CHECK_FALSE(machine.process_event(load));
    CHECK(
        state.err ==
        emel::error::cast(emel::text::tokenizer::needle::error::model_invalid));
    CHECK(machine.is(stateforward::sml::state<
                     emel::text::tokenizer::needle::state_errored>));
  }
}
