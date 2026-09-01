// Heldout accuracy eval for the native needle runtime.
//
// Drives the maintained chain end to end: cact loader (probe/bind/parse) ->
// needle binder -> needle tokenizer blob loader -> shared SPM tokenizer ->
// native graph (public init/prefill/decode events only). For every heldout
// row it consumes the prompt text plus reference tokenizer IDs produced by
// scripts/gen_needle_heldout_eval.py, asserts native parity, and feeds the
// native IDs to the graph.
//
// Prediction protocol matches the authoritative heldout generation path that
// produced the published 0.840 domain / 0.760 effort numbers: CQ4 weights,
// f32 activations, greedy decode up to 80 new tokens, stop at EOS (id 1),
// detokenize, and parse the first <tool_call>{...}</tool_call> arguments object.
// The probe heads bound in the phase-2 contract (head manifest codes
// 1 contrastive / 2 confidence) are NOT part of this accuracy path in the
// maintained reference: the contrastive head embeds tool schemas and the
// confidence head is uncalibrated for tuned weights (train/REPORT.md), so
// domain/effort accuracy is a text-generation metric.
//
// Usage:
//   needle_eval <model.cact> <prompts.tsv> [row_begin row_end]
//               [--activation-route a8|f32]
// Output: one "row i=..." line per example plus a final summary line.
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/cact/loader/sm.hpp"
#include "emel/io/mmap/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/needle/graph/sm.hpp"
#include "emel/model/needle/sm.hpp"
#include "emel/text/tokenizer/needle/sm.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

namespace cact_loader = emel::cact::loader;
namespace needle = emel::model::needle;

constexpr uint32_t k_max_new_tokens = 80u;
constexpr uint64_t k_max_model_bytes = 256u * 1024u * 1024u;
constexpr uint64_t k_max_tsv_bytes = 16u * 1024u * 1024u;
constexpr size_t k_max_tsv_rows = 4096u;
constexpr size_t k_max_tsv_line_bytes = 64u * 1024u;
constexpr size_t k_max_reference_ids_per_row = 4096u;
constexpr size_t k_max_reference_ids_total = 1u * 1024u * 1024u;
constexpr int32_t k_model_mapping_tensor_id = 1;
constexpr uint32_t k_accuracy_comparison_scale = 1000u;
enum class activation_route { a8, f32 };

struct eval_options {
  double min_domain_accuracy = 0.840;
  double min_effort_accuracy = 0.760;
  size_t max_no_parse = 1u;
  activation_route route = activation_route::f32;
  size_t row_begin = 0u;
  size_t row_end = std::numeric_limits<size_t>::max();
};

struct evaluator_error {
  const char *message = nullptr;
};

[[noreturn]] void die(const char *what) { throw evaluator_error{what}; }

struct map_owner_state {
  bool done = false;
  bool error = false;
  uint32_t handle = emel::io::mmap::k_invalid_mapping_handle;
  const uint8_t *buffer = nullptr;
  uint64_t buffer_bytes = 0u;
};

struct release_owner_state {
  bool done = false;
  bool error = false;
};

void on_map_done(void *object,
                 const emel::io::mmap::events::map_tensor_done &ev) noexcept {
  auto &owner = *static_cast<map_owner_state *>(object);
  owner.done = true;
  owner.handle = ev.handle;
  owner.buffer = static_cast<const uint8_t *>(ev.buffer);
  owner.buffer_bytes = ev.buffer_bytes;
}

void on_map_error(void *object,
                  const emel::io::mmap::events::map_tensor_error &) noexcept {
  static_cast<map_owner_state *>(object)->error = true;
}

void on_release_done(
    void *object,
    const emel::io::mmap::events::release_mapping_done &) noexcept {
  static_cast<release_owner_state *>(object)->done = true;
}

void on_release_error(
    void *object,
    const emel::io::mmap::events::release_mapping_error &) noexcept {
  static_cast<release_owner_state *>(object)->error = true;
}

void on_probe_done(const cact_loader::events::probe_done &) {}
void on_probe_error(const cact_loader::events::probe_error &) {}
void on_bind_done(const cact_loader::events::bind_done &) {}
void on_bind_error(const cact_loader::events::bind_error &) {}
void on_parse_done(const cact_loader::events::parse_done &) {}
void on_parse_error(const cact_loader::events::parse_error &) {}
void on_needle_done(const needle::events::bind_done &) {}
void on_needle_error(const needle::events::bind_error &) {}
void on_tok_load_done(const emel::text::tokenizer::needle::events::load_done &) {}
void on_tok_load_error(const emel::text::tokenizer::needle::events::load_error &) {}

uint64_t bounded_file_size(const char *path, const uint64_t max_bytes,
                           const char *open_error, const char *size_error,
                           const char *oversize_error) {
  std::error_code ec;
  const auto status = std::filesystem::status(path, ec);
  if (ec || !std::filesystem::is_regular_file(status)) die(open_error);
  const uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec || size == 0u) die(size_error);
  if (size > max_bytes) die(oversize_error);
  return static_cast<uint64_t>(size);
}

bool ascii_space(const char ch) noexcept {
  return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
         ch == '\f' || ch == '\v';
}

bool parse_reference_ids(const std::string_view text,
                         std::vector<int32_t> &ids_out) {
  const char *cursor = text.data();
  const char *const end = text.data() + text.size();
  while (cursor < end) {
    while (cursor < end && ascii_space(*cursor)) ++cursor;
    if (cursor == end) return true;
    if (ids_out.size() == k_max_reference_ids_per_row || *cursor < '0' ||
        *cursor > '9')
      return false;

    const char *token_end = cursor;
    while (token_end < end && !ascii_space(*token_end)) ++token_end;
    errno = 0;
    char *parsed_end = nullptr;
    const unsigned long value = std::strtoul(cursor, &parsed_end, 10);
    if (errno == ERANGE || parsed_end != token_end ||
        value > static_cast<unsigned long>(std::numeric_limits<int32_t>::max()))
      return false;
    ids_out.push_back(static_cast<int32_t>(value));
    cursor = token_end;
  }
  return true;
}

int hex_digit(const char ch) noexcept {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return 10 + ch - 'a';
  if (ch >= 'A' && ch <= 'F') return 10 + ch - 'A';
  return -1;
}

uint32_t argmax(const std::span<const float> logits) {
  uint32_t best = 0u;
  for (uint32_t i = 1u; i < logits.size(); ++i)
    best = logits[i] > logits[best] ? i : best;
  return best;
}

struct eval_row {
  std::string gold_domain;
  std::string gold_effort;
  std::vector<int32_t> ref_ids;
  std::string prompt;
};

std::vector<eval_row> read_rows(const char *path) {
  (void)bounded_file_size(path, k_max_tsv_bytes, "open prompts tsv",
                          "prompts tsv size", "prompts tsv too large");
  std::ifstream input(path);
  if (!input.good()) die("open prompts tsv");

  std::vector<eval_row> rows;
  size_t total_reference_ids = 0u;
  std::string line;
  while (std::getline(input, line)) {
    if (line.size() > k_max_tsv_line_bytes) die("prompts tsv line too large");
    if (line.empty()) continue;
    if (rows.size() == k_max_tsv_rows) die("too many prompts tsv rows");

    const size_t a = line.find('\t');
    const size_t b = a == std::string::npos ? std::string::npos
                                            : line.find('\t', a + 1u);
    const size_t c = b == std::string::npos ? std::string::npos
                                            : line.find('\t', b + 1u);
    if (a == std::string::npos || b == std::string::npos ||
        c == std::string::npos || line.find('\t', c + 1u) != std::string::npos)
      die("malformed tsv row");

    eval_row row;
    row.gold_domain = line.substr(0u, a);
    row.gold_effort = line.substr(a + 1u, b - a - 1u);
    const std::string ids_text = line.substr(b + 1u, c - b - 1u);
    if (!parse_reference_ids(ids_text, row.ref_ids) || row.ref_ids.empty())
      die("invalid reference IDs");
    if (row.ref_ids.size() > k_max_reference_ids_total - total_reference_ids)
      die("too many reference IDs");
    total_reference_ids += row.ref_ids.size();

    const std::string_view hex{line.data() + c + 1u, line.size() - c - 1u};
    if (hex.size() % 2u != 0u) die("odd prompt hex");
    row.prompt.reserve(hex.size() / 2u);
    for (size_t i = 0u; i < hex.size(); i += 2u) {
      const int high = hex_digit(hex[i]);
      const int low = hex_digit(hex[i + 1u]);
      if (high < 0 || low < 0) die("invalid prompt hex");
      row.prompt.push_back(static_cast<char>((high << 4) | low));
    }
    rows.push_back(std::move(row));
  }
  if (input.bad()) die("read prompts tsv");
  return rows;
}

std::string_view vocab_piece(const emel::model::data::vocab &vocab,
                             const uint32_t id) {
  return std::string_view{vocab.token_storage.data() +
                              vocab.entries[id].text_offset,
                          vocab.entries[id].text_length};
}

// Detokenize per export.py RefTokenizer.decode: byte pieces append the raw
// byte, CONTROL/UNKNOWN skipped, everything else appends the surface; the
// SentencePiece meta-space becomes ' '. Piece types use the shared vocab
// (GGUF) codes: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 6 BYTE.
std::string detokenize(const emel::model::data::vocab &vocab,
                       const std::span<const int32_t> ids) {
  std::string out;
  for (const int32_t id : ids) {
    if (id < 0 || static_cast<uint32_t>(id) >= vocab.n_tokens) continue;
    const int32_t type = vocab.entries[static_cast<uint32_t>(id)].type;
    if (type == 2 || type == 3) continue;
    const std::string_view piece =
        vocab_piece(vocab, static_cast<uint32_t>(id));
    if (type == 6 && piece.size() == 6u && piece.rfind("<0x", 0u) == 0u) {
      out.push_back(static_cast<char>(
          std::stoi(std::string{piece.substr(3u, 2u)}, nullptr, 16)));
      continue;
    }
    out += piece;
  }
  // UTF-8 meta space U+2581 (e2 96 81) -> ' '.
  std::string replaced;
  replaced.reserve(out.size());
  for (size_t i = 0; i < out.size();) {
    if (i + 3u <= out.size() && static_cast<uint8_t>(out[i]) == 0xE2u &&
        static_cast<uint8_t>(out[i + 1u]) == 0x96u &&
        static_cast<uint8_t>(out[i + 2u]) == 0x81u) {
      replaced.push_back(' ');
      i += 3u;
      continue;
    }
    replaced.push_back(out[i]);
    ++i;
  }
  return replaced;
}

// Extracts the string value of `key` inside the first <tool_call>...</tool_call>
// block. Returns empty when absent.
std::string extract_call_value(const std::string &text, const char *key) {
  const size_t call_begin = text.find("<tool_call>");
  if (call_begin == std::string::npos) return {};
  const size_t call_end = text.find("</tool_call>", call_begin);
  const std::string_view body{
      text.data() + call_begin,
      (call_end == std::string::npos ? text.size() : call_end) - call_begin};
  const std::string quoted_key = std::string{"\""} + key + "\"";
  const size_t key_pos = body.find(quoted_key);
  if (key_pos == std::string_view::npos) return {};
  size_t cursor = key_pos + quoted_key.size();
  while (cursor < body.size() &&
         (body[cursor] == ' ' || body[cursor] == ':' || body[cursor] == '\t'))
    ++cursor;
  if (cursor >= body.size() || body[cursor] != '"') return {};
  ++cursor;
  const size_t value_end = body.find('"', cursor);
  if (value_end == std::string_view::npos) return {};

  return std::string{body.substr(cursor, value_end - cursor)};
}

int effort_rank(const std::string &effort) {
  if (effort == "low") return 0;
  if (effort == "medium") return 1;
  if (effort == "high") return 2;
  if (effort == "xhigh") return 3;
  return -1;
}

bool parse_fraction(const char *text, double &value_out) {
  if (text == nullptr || *text == '\0') return false;
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(text, &end);
  if (errno == ERANGE || end == text || *end != '\0' ||
      !std::isfinite(value) || value < 0.0 || value > 1.0)
    return false;
  value_out = value;
  return true;
}

bool parse_count(const char *text, size_t &value_out) {
  if (text == nullptr || *text == '\0' || *text == '-') return false;
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
      value > std::numeric_limits<size_t>::max())
    return false;
  value_out = static_cast<size_t>(value);
  return true;
}
constexpr uint32_t rounded_accuracy_units(const double accuracy) {
  return static_cast<uint32_t>(accuracy * k_accuracy_comparison_scale + 0.5);
}

constexpr bool accuracy_meets_threshold(const double accuracy,
                                        const double threshold) {
  return static_cast<double>(rounded_accuracy_units(accuracy)) >=
         threshold * k_accuracy_comparison_scale;
}

constexpr double comparison_accuracy(const double accuracy) {
  return static_cast<double>(rounded_accuracy_units(accuracy)) /
         k_accuracy_comparison_scale;
}

static_assert(!accuracy_meets_threshold(0.8394, 0.840));
static_assert(accuracy_meets_threshold(0.8395, 0.840));
static_assert(!accuracy_meets_threshold(0.7594, 0.760));
static_assert(accuracy_meets_threshold(0.7595, 0.760));
static_assert(eval_options{}.max_no_parse == 1u);
static_assert(eval_options{}.route == activation_route::f32);

bool parse_activation_route(const char *text, activation_route &route_out) {
  if (text == nullptr) return false;
  const std::string_view value{text};
  if (value == "a8") {
    route_out = activation_route::a8;
    return true;
  }
  if (value == "f32") {
    route_out = activation_route::f32;
    return true;
  }
  return false;
}

const char *activation_route_name(const activation_route route) {
  return route == activation_route::a8 ? "a8" : "f32";
}

bool take_option_value(int argc, char **argv, int &index, const char *name,
                       const char *&value_out) {
  const std::string_view arg{argv[index]};
  const std::string prefix = std::string{name} + "=";
  if (arg.rfind(prefix, 0u) == 0u) {
    value_out = argv[index] + prefix.size();
    return true;
  }
  if (arg == name && index + 1 < argc) {
    value_out = argv[++index];
    return true;
  }
  return false;
}

bool parse_options(int argc, char **argv, eval_options &options) {
  bool have_range = false;
  for (int index = 3; index < argc; ++index) {
    const char *value = nullptr;
    if (take_option_value(argc, argv, index, "--min-domain-accuracy", value)) {
      if (!parse_fraction(value, options.min_domain_accuracy)) return false;
    } else if (take_option_value(argc, argv, index, "--min-effort-accuracy",
                                 value)) {
      if (!parse_fraction(value, options.min_effort_accuracy)) return false;
    } else if (take_option_value(argc, argv, index, "--max-no-parse", value)) {
      if (!parse_count(value, options.max_no_parse)) return false;
    } else if (take_option_value(argc, argv, index, "--activation-route",
                                 value)) {
      if (!parse_activation_route(value, options.route)) return false;
    } else if (!have_range && index + 1 < argc &&
               parse_count(argv[index], options.row_begin) &&
               parse_count(argv[index + 1], options.row_end)) {
      ++index;
      have_range = true;
    } else {
      return false;
    }
  }
  return true;
}

int run_eval(const char *model_path, const std::vector<eval_row> &rows,
             const eval_options &options) {
  const uint64_t model_bytes =
      bounded_file_size(model_path, k_max_model_bytes, "open model",
                        "model size", "model too large");
  const size_t row_begin = options.row_begin;
  const size_t row_end = options.row_end == std::numeric_limits<size_t>::max()
                             ? rows.size()
                             : options.row_end;
  if (rows.empty() || row_begin >= rows.size() || row_end > rows.size() ||
      row_begin >= row_end)
    die("bad row range or empty prompts");

  emel::io::mmap::sm mapping{};
  map_owner_state map_owner{};
  const emel::io::mmap::event::map_tensor_request map_request_data{
      .tensor_id = k_model_mapping_tensor_id,
      .file_index = 0u,
      .file_offset = 0u,
      .byte_size = model_bytes,
      .file_path = model_path,
  };
  emel::io::mmap::event::map_tensor map_request{map_request_data};
  map_request.on_done = {&map_owner, on_map_done};
  map_request.on_error = {&map_owner, on_map_error};
  if (!mapping.process_event(map_request) || !map_owner.done || map_owner.error ||
      map_owner.buffer == nullptr || map_owner.buffer_bytes != model_bytes)
    die("map model");

  int result = 0;
  {
    const std::span<const uint8_t> file_image{map_owner.buffer,
                                              static_cast<size_t>(model_bytes)};
    cact_loader::sm loader{};
    cact_loader::geometry geometry = {};
    if (!loader.process_event(cact_loader::event::probe{
            file_image, geometry,
            cact_loader::event::probe_done_fn::from<&on_probe_done>(),
            cact_loader::event::probe_error_fn::from<&on_probe_error>()}))
      die("loader probe");
    std::vector<cact_loader::tensor_view> tensors(geometry.num_tensors);
    if (!loader.process_event(cact_loader::event::bind_storage{
            std::span<cact_loader::tensor_view>{tensors},
            cact_loader::event::bind_done_fn::from<&on_bind_done>(),
            cact_loader::event::bind_error_fn::from<&on_bind_error>()}))
      die("loader bind");
    if (!loader.process_event(cact_loader::event::parse{
            file_image,
            cact_loader::event::parse_done_fn::from<&on_parse_done>(),
            cact_loader::event::parse_error_fn::from<&on_parse_error>()}))
      die("loader parse");

    needle::sm binder{};
    needle::contract contract = {};
    if (!binder.process_event(needle::event::bind{
            geometry, std::span<const cact_loader::tensor_view>{tensors},
            contract, needle::event::bind_done_fn::from<&on_needle_done>(),
            needle::event::bind_error_fn::from<&on_needle_error>()}))
      die("needle bind");
    if (!contract.has_tokenizer) die("fixture has no tokenizer blob");

    auto vocab = std::make_unique<emel::model::data::vocab>();
    emel::text::tokenizer::needle::sm blob_loader{};
    if (!blob_loader.process_event(emel::text::tokenizer::needle::event::load{
            std::span<const uint8_t>{
                contract.tokenizer_blob.data,
                static_cast<size_t>(contract.tokenizer_blob.nbytes)},
            *vocab,
            emel::text::tokenizer::needle::event::load_done_fn::from<
                &on_tok_load_done>(),
            emel::text::tokenizer::needle::event::load_error_fn::from<
                &on_tok_load_error>()}))
      die("tokenizer blob load");
    if (vocab->bos_id < 0 || vocab->eos_id < 0 ||
        static_cast<uint32_t>(vocab->bos_id) >= vocab->n_tokens ||
        static_cast<uint32_t>(vocab->eos_id) >= vocab->n_tokens)
      die("tokenizer blob has invalid BOS/EOS IDs");

    emel::text::tokenizer::sm tokenizer{};
    int32_t bind_err =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
    emel::text::tokenizer::event::bind bind_ev = {};
    bind_ev.vocab = vocab.get();
    bind_ev.preprocessor_variant =
        emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
    bind_ev.encoder_variant = emel::text::encoders::encoder_kind::spm;
    bind_ev.error_out = &bind_err;
    if (!tokenizer.process_event(bind_ev)) die("tokenizer bind");

    needle::graph::sm graph{contract};
    std::vector<float> logits(contract.geo.vocab_size);
    std::vector<int32_t> token_buffer(contract.geo.max_seq_len);
    std::vector<int32_t> prefill_ids;
    std::vector<int32_t> generated;
    prefill_ids.reserve(contract.geo.max_seq_len);
    generated.reserve(k_max_new_tokens);

    size_t evaluated = 0u, domain_ok = 0u, effort_ok = 0u, joint_ok = 0u;
    size_t no_parse = 0u, within1 = 0u, tokenizer_mismatch = 0u;

    for (size_t index = row_begin; index < row_end; ++index) {
      const eval_row &row = rows[index];

      int32_t native_count = 0;
      int32_t tok_err =
          emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
      emel::text::tokenizer::event::tokenize tok_ev = {};
      tok_ev.vocab = vocab.get();
      tok_ev.text = std::string_view{row.prompt};
      tok_ev.add_special = false;
      tok_ev.parse_special = true;
      tok_ev.token_ids_out = token_buffer.data();
      tok_ev.token_capacity = static_cast<int32_t>(token_buffer.size());
      tok_ev.token_count_out = &native_count;
      tok_ev.error_out = &tok_err;
      if (!tokenizer.process_event(tok_ev)) die("tokenize");

      bool ids_match =
          static_cast<size_t>(native_count) == row.ref_ids.size();
      for (int32_t i = 0; ids_match && i < native_count; ++i)
        ids_match = token_buffer[static_cast<size_t>(i)] ==
                    row.ref_ids[static_cast<size_t>(i)];
      if (!ids_match) {
        ++tokenizer_mismatch;
        die("native tokenizer ids differ from reference");
      }

      prefill_ids.clear();
      prefill_ids.push_back(vocab->bos_id);
      const size_t prompt_cap =
          contract.geo.max_seq_len > k_max_new_tokens
              ? contract.geo.max_seq_len - k_max_new_tokens
              : 1u;
      for (int32_t i = 0; i < native_count && prefill_ids.size() < prompt_cap;
           ++i)
        prefill_ids.push_back(token_buffer[static_cast<size_t>(i)]);

      if (!graph.process_event(needle::graph::event::init{
              .activation_quant = options.route == activation_route::a8}))
        die("graph init");
      if (!graph.process_event(needle::graph::event::prefill{
              std::span<const int32_t>{prefill_ids},
              std::span<float>{logits}}))
        die("graph prefill");

      generated.clear();
      for (uint32_t step = 0u; step < k_max_new_tokens; ++step) {
        const int32_t next = static_cast<int32_t>(
            argmax(std::span<const float>{logits.data(), logits.size()}));
        if (next == vocab->eos_id) break;
        generated.push_back(next);
        if (step + 1u < k_max_new_tokens) {
          if (!graph.process_event(needle::graph::event::decode{
                  next, std::span<float>{logits}}))
            die("graph decode");
        }
      }

      const std::string text =
          detokenize(*vocab, std::span<const int32_t>{generated});
      const std::string pred_domain = extract_call_value(text, "domain");
      const std::string pred_effort = extract_call_value(text, "effort");

      ++evaluated;
      if (pred_domain.empty() && pred_effort.empty()) {
        ++no_parse;
      } else {
        if (pred_domain == row.gold_domain) ++domain_ok;
        if (pred_effort == row.gold_effort) ++effort_ok;
        if (pred_domain == row.gold_domain && pred_effort == row.gold_effort)
          ++joint_ok;
        const int gold_rank = effort_rank(row.gold_effort);
        const int pred_rank = effort_rank(pred_effort);
        if (gold_rank >= 0 && pred_rank >= 0 &&
            (gold_rank - pred_rank <= 1 && pred_rank - gold_rank <= 1))
          ++within1;
      }
      std::printf(
          "row i=%zu gold=%s/%s pred=%s/%s ids_match=%d new_tokens=%zu\n",
          index, row.gold_domain.c_str(), row.gold_effort.c_str(),
          pred_domain.empty() ? "-" : pred_domain.c_str(),
          pred_effort.empty() ? "-" : pred_effort.c_str(), ids_match ? 1 : 0,
          generated.size());
      std::fflush(stdout);
    }

    const double domain_accuracy =
        evaluated ? static_cast<double>(domain_ok) / evaluated : 0.0;
    const double effort_accuracy =
        evaluated ? static_cast<double>(effort_ok) / evaluated : 0.0;
    const double joint_accuracy =
        evaluated ? static_cast<double>(joint_ok) / evaluated : 0.0;
    const double effort_within1 =
        evaluated ? static_cast<double>(within1) / evaluated : 0.0;
    std::printf("needle_eval_summary rows=%zu no_parse=%zu domain_acc=%.4f "
                "effort_acc=%.4f domain_compare=%.3f effort_compare=%.3f "
                "joint_acc=%.4f effort_within1=%.4f "
                "tokenizer_id_mismatch_rows=%zu activation_route=%s "
                "thresholds=%.3f/%.3f/%zu comparison_precision=3dp\n",
                evaluated, no_parse, domain_accuracy, effort_accuracy,
                comparison_accuracy(domain_accuracy),
                comparison_accuracy(effort_accuracy), joint_accuracy,
                effort_within1, tokenizer_mismatch,
                activation_route_name(options.route),
                options.min_domain_accuracy, options.min_effort_accuracy,
                options.max_no_parse);
    if (tokenizer_mismatch != 0u) {
      std::fprintf(stderr, "error: tokenizer ID mismatch rows=%zu\n",
                   tokenizer_mismatch);
      result = 1;
    } else if (no_parse > options.max_no_parse ||
               !accuracy_meets_threshold(domain_accuracy,
                                         options.min_domain_accuracy) ||
               !accuracy_meets_threshold(effort_accuracy,
                                         options.min_effort_accuracy)) {
      std::fprintf(
          stderr,
          "error: accuracy thresholds unmet domain=%.4f (3dp %.3f, min %.3f) "
          "effort=%.4f (3dp %.3f, min %.3f) no_parse=%zu (max %zu)\n",
          domain_accuracy, comparison_accuracy(domain_accuracy),
          options.min_domain_accuracy, effort_accuracy,
          comparison_accuracy(effort_accuracy), options.min_effort_accuracy,
          no_parse, options.max_no_parse);
      result = 1;
    }
  }

  release_owner_state release_owner{};
  emel::io::mmap::event::release_mapping release_request{
      k_model_mapping_tensor_id, map_owner.handle};
  release_request.on_done = {&release_owner, on_release_done};
  release_request.on_error = {&release_owner, on_release_error};
  if (!mapping.process_event(release_request) || !release_owner.done ||
      release_owner.error)
    die("release model mapping");
  return result;
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: needle_eval <model.cact> <prompts.tsv> "
                 "[row_begin row_end] [--activation-route a8|f32] "
                 "[--min-domain-accuracy VALUE] "
                 "[--min-effort-accuracy VALUE] [--max-no-parse COUNT]\n");
    return 2;
  }
  try {
    eval_options options = {};
    if (!parse_options(argc, argv, options)) {
      std::fprintf(stderr, "error: needle_eval: invalid CLI option or value\n");
      return 2;
    }
    return run_eval(argv[1], read_rows(argv[2]), options);
  } catch (const evaluator_error &error) {
    std::fprintf(stderr, "error: needle_eval: %s\n", error.message);
    return 1;
  } catch (const std::bad_alloc &) {
    std::fprintf(stderr, "error: needle_eval: allocation failed\n");
    return 1;
  } catch (const std::length_error &) {
    std::fprintf(stderr, "error: needle_eval: allocation size invalid\n");
    return 1;
  } catch (const std::exception &) {
    std::fprintf(stderr, "error: needle_eval: unexpected failure\n");
    return 1;
  }
}
